use anyhow::Result;
use cutile::cuda_async::device_operation::{DeviceOp, value};
use cutile::cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::core::f16;
use cutile::tensor::{IntoPartition, PartitionMut as _, ToHostVec};
use cutile::tile_kernel::TileKernel;
use grout::kernels::add_2d_f16;
use std::sync::Arc;

#[test]
fn add_2d_kernel_compiles_and_executes() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        Ok(_) => {
            eprintln!("skipping CUDA kernel integration test: no CUDA devices found");
            return Ok(());
        }
        Err(err) => {
            eprintln!("skipping CUDA kernel integration test: CUDA unavailable: {err:?}");
            return Ok(());
        }
    }

    const BLOCK: usize = 4;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let lhs_host = Arc::new(vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ]);
    let rhs_host = Arc::new(vec![
        f16::from_f32(10.0),
        f16::from_f32(20.0),
        f16::from_f32(30.0),
        f16::from_f32(40.0),
    ]);

    let lhs = Arc::new(
        api::copy_host_vec_to_device(&lhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let rhs = Arc::new(
        api::copy_host_vec_to_device(&rhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let out = api::zeros::<f16>(&[1, BLOCK]).sync_on(&stream)?;

    let result = add_2d_f16(value(out.partition([1, BLOCK])), value(lhs), value(rhs))
        .generics(vec![BLOCK.to_string()])
        .sync_on(&stream)?;
    let out = result.0.unpartition();
    let actual = out.to_host_vec().sync_on(&stream)?;

    let actual: Vec<f32> = actual.into_iter().map(|x| x.to_f32()).collect();
    assert_eq!(actual, vec![11.0, 22.0, 33.0, 44.0]);
    Ok(())
}

#[test]
fn add_rms_norm_decode_bounded_matches_raw() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        _ => {
            eprintln!("skipping: no CUDA device");
            return Ok(());
        }
    }
    use grout::kernels::add_rms_norm_decode_bounded_f16;

    const N: usize = 2560;
    const BS: usize = 512;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    // Deterministic pseudo-random inputs spanning sign/magnitude range.
    let gen_vals = |seed: u32| -> Arc<Vec<f16>> {
        let mut v = Vec::with_capacity(N);
        let mut x = seed;
        for _ in 0..N {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(f16::from_f32(((x >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0));
        }
        Arc::new(v)
    };
    let residual = Arc::new(
        api::copy_host_vec_to_device(&gen_vals(1))
            .reshape(&[1, N])
            .sync_on(&stream)?,
    );
    let x = Arc::new(
        api::copy_host_vec_to_device(&gen_vals(2))
            .reshape(&[1, N])
            .sync_on(&stream)?,
    );
    let w = Arc::new(
        api::copy_host_vec_to_device(&gen_vals(3))
            .reshape(&[N])
            .sync_on(&stream)?,
    );
    let eps = 1e-6f32;

    // Host reference (f32 mirror of the kernel math; GPU reduction order
    // differs, so compare with a small tolerance).
    let host_r = gen_vals(1);
    let host_x = gen_vals(2);
    let host_w = gen_vals(3);
    let mut combined = vec![0f32; N];
    let mut ssq = 0f64;
    for i in 0..N {
        let c = host_r[i].to_f32() + host_x[i].to_f32();
        combined[i] = c;
        ssq += (c as f64) * (c as f64);
    }
    let inv_rms = (1.0 / ((ssq / N as f64) + eps as f64).sqrt()) as f32;
    let expect_out: Vec<f32> = (0..N)
        .map(|i| combined[i] * inv_rms * host_w[i].to_f32())
        .collect();

    // Bounded kernel.
    let out_b = api::zeros::<f16>(&[1, N]).sync_on(&stream)?;
    let res_b = api::zeros::<f16>(&[1, N]).sync_on(&stream)?;
    let result = add_rms_norm_decode_bounded_f16(
        &residual,
        &x,
        &w,
        value(out_b.partition([1usize, N])),
        value(res_b.partition([1usize, N])),
        eps,
    )
    .generics(vec![N.to_string(), BS.to_string()])
    .grid((1u32, 1u32, 1u32))
    .sync_on(&stream)?;
    let out_b = result.3.unpartition();
    let res_b = result.4.unpartition();

    let ob = out_b.to_host_vec().sync_on(&stream)?;
    let rb = res_b.to_host_vec().sync_on(&stream)?;
    let mut bad = 0;
    for i in 0..N {
        let got_out = ob[i].to_f32();
        let got_res = rb[i].to_f32();
        let ref_res = f16::from_f32(combined[i]).to_f32();
        let tol = 1e-2f32 * expect_out[i].abs().max(0.05);
        if (got_out - expect_out[i]).abs() > tol || got_res != ref_res {
            if bad < 5 {
                eprintln!(
                    "mismatch @{i}: out got={got_out} want~{} | res got={got_res} want={ref_res}",
                    expect_out[i]
                );
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad}/{N} elements differ");
    Ok(())
}

/// Regression test for the derived-fact placement of the grid-rowed form:
/// the fully safe row-wise kernel (row from `get_tile_block_id`, columns
/// from `num_tiles` ranges) must JIT and launch cleanly. This was the
/// owned-axis acceptance spec; the derived-fact design closed it.
#[test]
fn rowwise_bounded_spec_jit_error() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        _ => return Ok(()),
    }
    use grout::kernels::add_rms_norm_rows_bounded_spec_f16;
    const N: usize = 2560;
    const BS: usize = 512;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let z = |shape: &[usize]| api::zeros::<f16>(shape);
    let residual = Arc::new(z(&[4, N]).sync_on(&stream)?);
    let x = Arc::new(z(&[4, N]).sync_on(&stream)?);
    let w = Arc::new(z(&[N]).sync_on(&stream)?);
    let out = z(&[4, N]).sync_on(&stream)?;
    let r = add_rms_norm_rows_bounded_spec_f16(
        &residual,
        &x,
        &w,
        value(out.partition([1usize, N])),
        1e-6f32,
    )
    .generics(vec![N.to_string(), BS.to_string()])
    .grid((4u32, 1u32, 1u32))
    .sync_on(&stream);
    r.map(|_| ())
        .map_err(|e| anyhow::anyhow!("grid-rowed safe kernel must JIT+launch clean: {e:?}"))
}

/// Elementwise A/B: the safe fused decode qk_norm+rope+kv kernel must match
/// the raw kernel exactly on random data across the Q, K-cache and V-cache
/// outputs (and leave unwritten cache slots untouched).
#[test]
fn qk_norm_rope_kv_decode_safe_matches_reference() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        _ => return Ok(()),
    }
    use grout::kernels::qk_norm_rope_kv_decode_f16;
    const D: usize = 128;
    const HALF_D: usize = 64;
    const MAX_SEQ: usize = 32;
    const NQ: usize = 4;
    const NKV: usize = 2;
    const POS: u32 = 5;
    let total = NQ + NKV;
    let qkv_len = (NQ + 2 * NKV) * D;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let gen_f16 = |seed: u32, n: usize| -> Arc<Vec<f16>> {
        let mut v = Vec::with_capacity(n);
        let mut x = seed;
        for _ in 0..n {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(f16::from_f32(((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0));
        }
        Arc::new(v)
    };
    let inv_host: Arc<Vec<f32>> = Arc::new(
        (0..HALF_D)
            .map(|i| 1.0f32 / 10000f32.powf(2.0 * i as f32 / D as f32))
            .collect(),
    );

    let qkv = Arc::new(api::copy_host_vec_to_device(&gen_f16(1, qkv_len)).reshape(&[qkv_len]).sync_on(&stream)?);
    let qw = Arc::new(api::copy_host_vec_to_device(&gen_f16(2, D)).reshape(&[D]).sync_on(&stream)?);
    let kw = Arc::new(api::copy_host_vec_to_device(&gen_f16(3, D)).reshape(&[D]).sync_on(&stream)?);
    let inv = Arc::new(api::copy_host_vec_to_device(&inv_host).reshape(&[HALF_D]).sync_on(&stream)?);
    let pos = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![POS]))
            .reshape(&[1])
            .sync_on(&stream)?,
    );

    let mk_outs = || -> Result<_> {
        Ok((
            api::zeros::<f16>(&[total, D]).sync_on(&stream)?,
            api::zeros::<f16>(&[NKV, MAX_SEQ, D]).sync_on(&stream)?,
            api::zeros::<f16>(&[NKV, MAX_SEQ, D]).sync_on(&stream)?,
        ))
    };
    let (q_safe, k_safe, v_safe) = mk_outs()?;

    let generics = vec![D.to_string(), HALF_D.to_string(), MAX_SEQ.to_string()];
    qk_norm_rope_kv_decode_f16(
        &qkv,
        &qw,
        &kw,
        &inv,
        &q_safe,
        &k_safe,
        &v_safe,
        &pos,
        1e-6f32,
        NQ as i32,
        NKV as i32,
    )
    .generics(generics)
    .grid((total as u32, 2u32, 1u32))
    .sync_on(&stream)?;

    // Host reference in f32. Device cos/sin and reduction order differ in
    // rounding, so q/k compare with tolerance; v is a pure copy (exact) and
    // untouched cache slots must stay exactly zero.
    let qkv_h = gen_f16(1, qkv_len);
    let qw_h = gen_f16(2, D);
    let kw_h = gen_f16(3, D);
    let q_got = q_safe.to_host_vec().sync_on(&stream)?;
    let k_got = k_safe.to_host_vec().sync_on(&stream)?;
    let v_got = v_safe.to_host_vec().sync_on(&stream)?;
    let mut bad = 0usize;
    let tol = |x: f32| 2e-2f32 * x.abs().max(0.05);
    for h in 0..total {
        let is_q = h < NQ;
        let local = if is_q { h } else { h - NQ };
        let base = if is_q { local * D } else { NQ * D + local * D };
        let lo: Vec<f32> = (0..HALF_D).map(|i| qkv_h[base + i].to_f32()).collect();
        let hi: Vec<f32> = (0..HALF_D).map(|i| qkv_h[base + HALF_D + i].to_f32()).collect();
        let w = if is_q { &qw_h } else { &kw_h };
        let mut ss = 0f64;
        for i in 0..HALF_D {
            ss += (lo[i] as f64) * (lo[i] as f64) + (hi[i] as f64) * (hi[i] as f64);
        }
        let inv = (1.0 / ((ss / D as f64) + 1e-6).sqrt()) as f32;
        for i in 0..HALF_D {
            let nl = lo[i] * inv * w[i].to_f32();
            let nh = hi[i] * inv * w[HALF_D + i].to_f32();
            let theta = POS as f32 * inv_host[i];
            let (sn, cs) = theta.sin_cos();
            let ylo = nl * cs - nh * sn;
            let yhi = nh * cs + nl * sn;
            let (g_lo, g_hi) = if is_q {
                (
                    q_got[local * D + i].to_f32(),
                    q_got[local * D + HALF_D + i].to_f32(),
                )
            } else {
                let o = local * MAX_SEQ * D + (POS as usize) * D;
                (k_got[o + i].to_f32(), k_got[o + HALF_D + i].to_f32())
            };
            if (g_lo - ylo).abs() > tol(ylo) || (g_hi - yhi).abs() > tol(yhi) {
                if bad < 5 {
                    eprintln!("h={h} i={i}: got ({g_lo},{g_hi}) want ({ylo},{yhi})");
                }
                bad += 1;
            }
        }
        if !is_q {
            let vbase = (NQ + NKV) * D + local * D;
            let o = local * MAX_SEQ * D + (POS as usize) * D;
            for i in 0..D {
                if v_got[o + i].to_f32() != qkv_h[vbase + i].to_f32() {
                    bad += 1;
                }
                if k_got[local * MAX_SEQ * D + i].to_f32() != 0.0 {
                    bad += 1;
                }
            }
        }
    }
    assert_eq!(bad, 0, "{bad} mismatches vs host reference");
    Ok(())
}

/// Host-reference check for the safe fused prefill qk_norm+rope+kv kernel:
/// multiple sequence rows at a nonzero position offset. Q/K compare with
/// tolerance (device cos/sin rounding); V is a pure copy (exact); cache
/// slots outside the written range stay exactly zero.
#[test]
fn qk_norm_rope_kv_prefill_safe_matches_reference() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        _ => return Ok(()),
    }
    use grout::kernels::{
        k_norm_rope_v_prefill_f16, k_norm_rope_v_prefill_wide_f16, q_norm_rope_prefill_f16,
        q_norm_rope_prefill_wide_f16,
    };
    const D: usize = 128;
    const HALF_D: usize = 64;
    const MAX_SEQ: usize = 32;
    const NQ: usize = 4;
    const NKV: usize = 2;
    const SEQ: usize = 5;
    const POS0: usize = 4;
    let total = NQ + NKV;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let gen_f16 = |seed: u32, n: usize| -> Arc<Vec<f16>> {
        let mut v = Vec::with_capacity(n);
        let mut x = seed;
        for _ in 0..n {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(f16::from_f32(((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0));
        }
        Arc::new(v)
    };
    let inv_host: Arc<Vec<f32>> = Arc::new(
        (0..HALF_D)
            .map(|i| 1.0f32 / 10000f32.powf(2.0 * i as f32 / D as f32))
            .collect(),
    );
    let q_h = gen_f16(11, SEQ * NQ * D);
    let k_h = gen_f16(12, SEQ * NKV * D);
    let v_h = gen_f16(13, SEQ * NKV * D);
    let qw_h = gen_f16(2, D);
    let kw_h = gen_f16(3, D);

    let q = Arc::new(api::copy_host_vec_to_device(&q_h).reshape(&[SEQ, NQ, D]).sync_on(&stream)?);
    let k = Arc::new(api::copy_host_vec_to_device(&k_h).reshape(&[SEQ, NKV, D]).sync_on(&stream)?);
    let v = Arc::new(api::copy_host_vec_to_device(&v_h).reshape(&[SEQ, NKV, D]).sync_on(&stream)?);
    let qw = Arc::new(api::copy_host_vec_to_device(&qw_h).reshape(&[D]).sync_on(&stream)?);
    let kw = Arc::new(api::copy_host_vec_to_device(&kw_h).reshape(&[D]).sync_on(&stream)?);
    let inv = Arc::new(api::copy_host_vec_to_device(&inv_host).reshape(&[HALF_D]).sync_on(&stream)?);
    let mut q_out = api::zeros::<f16>(&[SEQ, NQ, D]).sync_on(&stream)?;
    let mut k_cache = api::zeros::<f16>(&[NKV, MAX_SEQ, D]).sync_on(&stream)?;
    let mut v_cache = api::zeros::<f16>(&[NKV, MAX_SEQ, D]).sync_on(&stream)?;

    let map_generics = || ["1".to_string(), "1".to_string(), "2".to_string()];
    // Wide bulk over the BM-aligned prefix, per-row subrange kernel over
    // the tail — the same split the engine host performs.
    const BM: usize = 2;
    const BULK: usize = SEQ - SEQ % BM;
    const TAIL: usize = SEQ - BULK;
    // Prefix coverage: grid covers only the BULK rows of q_out's 3-block
    // grid — exercises the partial-coverage &mut binding (cutile-rs
    // e90f9b8).
    q_norm_rope_prefill_wide_f16(
        &q,
        &qw,
        &inv,
        (&mut q_out).partition([BM, 1, D]).prefix(),
        1e-6f32,
        POS0 as i32,
    )
    .generics(vec![D.to_string(), HALF_D.to_string(), BM.to_string()])
    .grid(((BULK / BM) as u32, NQ as u32, 1u32))
    .sync_on(&stream)?;
    q_norm_rope_prefill_f16(
        (&mut q_out)
            .partition([1, 1, HALF_D])
            .map([1, 1, 2], (TAIL * NQ) as u32),
        &q,
        &qw,
        &inv,
        1e-6f32,
        POS0 as i32,
        BULK as i32,
        TAIL as i32,
    )
    .generics(
        [D.to_string(), HALF_D.to_string()]
            .into_iter()
            .chain(map_generics())
            .collect(),
    )
    .sync_on(&stream)?;

    // POS0=4 is BM-aligned, so the KV side takes the same wide-bulk +
    // per-row-tail split the engine host performs.
    k_norm_rope_v_prefill_wide_f16(&k, &v, &kw, &inv, &k_cache, &v_cache, 1e-6f32, POS0 as i32)
        .generics(vec![D.to_string(), HALF_D.to_string(), BM.to_string()])
        .grid(((BULK / BM) as u32, NKV as u32, 1u32))
        .sync_on(&stream)?;
    k_norm_rope_v_prefill_f16(
        (&mut k_cache)
            .partition([1, 1, HALF_D])
            .map([1, 1, 2], (NKV * TAIL) as u32),
        (&mut v_cache)
            .partition([1, 1, HALF_D])
            .map([1, 1, 2], (NKV * TAIL) as u32),
        &k,
        &v,
        &kw,
        &inv,
        1e-6f32,
        POS0 as i32,
        (POS0 + BULK) as i32,
        TAIL as i32,
    )
    .generics(
        [D.to_string(), HALF_D.to_string()]
            .into_iter()
            .chain(map_generics())
            .collect(),
    )
    .sync_on(&stream)?;

    let q_got = q_out.to_host_vec().sync_on(&stream)?;
    let k_got = k_cache.to_host_vec().sync_on(&stream)?;
    let v_got = v_cache.to_host_vec().sync_on(&stream)?;
    let tol = |x: f32| 2e-2f32 * x.abs().max(0.05);
    let mut bad = 0usize;
    for s_i in 0..SEQ {
        for h in 0..total {
            let is_q = h < NQ;
            let local = if is_q { h } else { h - NQ };
            let (src, w): (&Arc<Vec<f16>>, &Arc<Vec<f16>>) =
                if is_q { (&q_h, &qw_h) } else { (&k_h, &kw_h) };
            let heads = if is_q { NQ } else { NKV };
            let base = s_i * heads * D + local * D;
            let lo: Vec<f32> = (0..HALF_D).map(|i| src[base + i].to_f32()).collect();
            let hi: Vec<f32> = (0..HALF_D).map(|i| src[base + HALF_D + i].to_f32()).collect();
            let mut ss = 0f64;
            for i in 0..HALF_D {
                ss += (lo[i] as f64) * (lo[i] as f64) + (hi[i] as f64) * (hi[i] as f64);
            }
            let inv_rms = (1.0 / ((ss / D as f64) + 1e-6).sqrt()) as f32;
            for i in 0..HALF_D {
                let nl = lo[i] * inv_rms * w[i].to_f32();
                let nh = hi[i] * inv_rms * w[HALF_D + i].to_f32();
                let theta = (POS0 + s_i) as f32 * inv_host[i];
                let (sn, cs) = theta.sin_cos();
                let ylo = nl * cs - nh * sn;
                let yhi = nh * cs + nl * sn;
                let (g_lo, g_hi) = if is_q {
                    let o = s_i * NQ * D + local * D;
                    (q_got[o + i].to_f32(), q_got[o + HALF_D + i].to_f32())
                } else {
                    let o = local * MAX_SEQ * D + (POS0 + s_i) * D;
                    (k_got[o + i].to_f32(), k_got[o + HALF_D + i].to_f32())
                };
                if (g_lo - ylo).abs() > tol(ylo) || (g_hi - yhi).abs() > tol(yhi) {
                    if bad < 5 {
                        eprintln!("s={s_i} h={h} i={i}: got ({g_lo},{g_hi}) want ({ylo},{yhi})");
                    }
                    bad += 1;
                }
            }
            if !is_q {
                let o = local * MAX_SEQ * D + (POS0 + s_i) * D;
                let vbase = s_i * NKV * D + local * D;
                for i in 0..D {
                    if v_got[o + i].to_f32() != v_h[vbase + i].to_f32() {
                        bad += 1;
                    }
                }
            }
        }
    }
    // untouched cache rows: positions 0..POS0 and POS0+SEQ..MAX_SEQ must be zero
    for h in 0..NKV {
        for p in (0..POS0).chain(POS0 + SEQ..MAX_SEQ) {
            for i in 0..D {
                if k_got[h * MAX_SEQ * D + p * D + i].to_f32() != 0.0 {
                    bad += 1;
                }
            }
        }
    }
    assert_eq!(bad, 0, "{bad} mismatches vs host reference");
    Ok(())
}

/// Prefix-coverage semantics for the safe wide Q kernel: launching over
/// only the first 2 of 3 row-blocks must write exactly those rows and
/// leave the uncovered block untouched; a full-coverage launch over the
/// same inputs must agree bitwise on the covered prefix.
#[test]
fn q_norm_rope_prefill_wide_prefix_coverage_semantics() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        _ => return Ok(()),
    }
    use grout::kernels::q_norm_rope_prefill_wide_f16;
    const D: usize = 128;
    const HALF_D: usize = 64;
    const NQ: usize = 4;
    const SEQ: usize = 6;
    const BM: usize = 2;
    const POS0: usize = 5;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let gen_f16 = |seed: u32, n: usize| -> Arc<Vec<f16>> {
        let mut v = Vec::with_capacity(n);
        let mut x = seed;
        for _ in 0..n {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(f16::from_f32(((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0));
        }
        Arc::new(v)
    };
    let inv_host: Arc<Vec<f32>> = Arc::new(
        (0..HALF_D)
            .map(|i| 1.0f32 / 10000f32.powf(2.0 * i as f32 / D as f32))
            .collect(),
    );
    let q_h = gen_f16(31, SEQ * NQ * D);
    let qw_h = gen_f16(32, D);
    let q = Arc::new(api::copy_host_vec_to_device(&q_h).reshape(&[SEQ, NQ, D]).sync_on(&stream)?);
    let qw = Arc::new(api::copy_host_vec_to_device(&qw_h).reshape(&[D]).sync_on(&stream)?);
    let inv = Arc::new(api::copy_host_vec_to_device(&inv_host).reshape(&[HALF_D]).sync_on(&stream)?);
    let mut out_full = api::zeros::<f16>(&[SEQ, NQ, D]).sync_on(&stream)?;
    let mut out_pre = api::zeros::<f16>(&[SEQ, NQ, D]).sync_on(&stream)?;

    q_norm_rope_prefill_wide_f16(
        &q,
        &qw,
        &inv,
        (&mut out_full).partition([BM, 1, D]),
        1e-6f32,
        POS0 as i32,
    )
    .generics(vec![D.to_string(), HALF_D.to_string(), BM.to_string()])
    .grid(((SEQ / BM) as u32, NQ as u32, 1u32))
    .sync_on(&stream)?;
    const COVERED: usize = 4; // 2 of 3 row-blocks
    q_norm_rope_prefill_wide_f16(
        &q,
        &qw,
        &inv,
        (&mut out_pre).partition([BM, 1, D]).prefix(),
        1e-6f32,
        POS0 as i32,
    )
    .generics(vec![D.to_string(), HALF_D.to_string(), BM.to_string()])
    .grid(((COVERED / BM) as u32, NQ as u32, 1u32))
    .sync_on(&stream)?;

    let a = out_full.to_host_vec().sync_on(&stream)?;
    let b = out_pre.to_host_vec().sync_on(&stream)?;
    let row = NQ * D;
    let mut bad = 0usize;
    for i in 0..SEQ * row {
        let want = if i < COVERED * row { a[i].to_bits() } else { 0u16 };
        if b[i].to_bits() != want {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} elements violate prefix-coverage semantics");
    Ok(())
}

/// LPT stride invariance: the checked kernel's output on a padded KV cache
/// (rows > kv_len — the engine's real layout) must equal its output on a
/// contiguous cache. The deleted raw kernel failed exactly this (kv_len*D
/// head stride against the cache's max_seq*D — 50% of elements wrong at
/// kv_heads=2), which is the bug this test pins closed.
#[test]
fn fmha_prefill_lpt_checked_matches_and_fixes_strides() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        _ => return Ok(()),
    }
    use grout::kernels::fmha_prefill_gqa_lpt_checked;
    const D: usize = 128;
    const BM: usize = 16;
    const BN: usize = 64;
    const GROUP: usize = 4;
    const M_EFF: usize = BM * GROUP;
    const QLEN: usize = 64;
    const QHEADS: usize = 8;
    const KVHEADS: usize = 2;
    const KVLEN: usize = 64;
    const PAD: usize = 128;
    let qgs = (QHEADS / KVHEADS) as i32; // 4
    let num_q_blocks = (QLEN / BM) as i32; // 4
    let num_head_groups = (QHEADS / GROUP) as i32; // 2
    let grid_x = (num_q_blocks * num_head_groups) as u32;
    let generics: Vec<String> = vec![
        BM.to_string(), BN.to_string(), D.to_string(), GROUP.to_string(),
        M_EFF.to_string(), "1".into(), "1".into(), "2".into(), "1".into(), "0".into(),
    ];

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let gen_f16 = |seed: u32, n: usize| -> Arc<Vec<f16>> {
        let mut v = Vec::with_capacity(n);
        let mut x = seed;
        for _ in 0..n {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(f16::from_f32(((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0));
        }
        Arc::new(v)
    };
    let q_h = gen_f16(21, QLEN * QHEADS * D);
    let k_h = gen_f16(22, KVHEADS * KVLEN * D);
    let v_h = gen_f16(23, KVHEADS * KVLEN * D);
    // Padded copies: same rows 0..KVLEN, garbage-free zeros beyond.
    let pad_copy = |src: &Arc<Vec<f16>>| -> Arc<Vec<f16>> {
        let mut p = vec![f16::from_f32(0.0); KVHEADS * PAD * D];
        for h in 0..KVHEADS {
            for r in 0..KVLEN {
                for d0 in 0..D {
                    p[h * PAD * D + r * D + d0] = src[h * KVLEN * D + r * D + d0];
                }
            }
        }
        Arc::new(p)
    };
    let kp_h = pad_copy(&k_h);
    let vp_h = pad_copy(&v_h);

    let q = Arc::new(api::copy_host_vec_to_device(&q_h).reshape(&[QLEN, QHEADS, D]).sync_on(&stream)?);
    let k_c = Arc::new(api::copy_host_vec_to_device(&k_h).reshape(&[KVHEADS, KVLEN, D]).sync_on(&stream)?);
    let v_c = Arc::new(api::copy_host_vec_to_device(&v_h).reshape(&[KVHEADS, KVLEN, D]).sync_on(&stream)?);
    let k_p = Arc::new(api::copy_host_vec_to_device(&kp_h).reshape(&[KVHEADS, PAD, D]).sync_on(&stream)?);
    let v_p = Arc::new(api::copy_host_vec_to_device(&vp_h).reshape(&[KVHEADS, PAD, D]).sync_on(&stream)?);
    let scale = 1.0f32 / (D as f32).sqrt();

    // 1) checked on the contiguous cache = ground truth
    let out_raw = api::zeros::<f16>(&[QLEN, QHEADS, D]).sync_on(&stream)?;
    fmha_prefill_gqa_lpt_checked(
        &q, &k_c, &v_c, &out_raw,
        value(scale), value(qgs), value(KVLEN as i32), value(0i32),
        value(num_q_blocks), value(num_head_groups),
        value(1i32), value(2i32), value(1i32),
    )
    .generics(generics.clone()).grid((grid_x, 1, 1)).sync_on(&stream)?;

    // 2) checked on the PADDED cache (engine layout) — must equal truth
    let out_chk = api::zeros::<f16>(&[QLEN, QHEADS, D]).sync_on(&stream)?;
    fmha_prefill_gqa_lpt_checked(
        &q, &k_p, &v_p, &out_chk,
        value(scale), value(qgs), value(KVLEN as i32), value(0i32),
        value(num_q_blocks), value(num_head_groups),
        value(1i32), value(2i32), value(1i32),
    )
    .generics(generics.clone()).grid((grid_x, 1, 1)).sync_on(&stream)?;

    // 1b) resurrected RAW kernel — bug confirmation. On the CONTIGUOUS
    // cache its kv_len-derived strides agree with the layout, so it must
    // match truth; on the PADDED cache (the engine's real layout) the
    // strides are wrong for kv heads >= 1, so it must diverge.
    use grout::kernels::fmha_prefill_gqa_lpt_raw_resurrected;
    let out_raw_c = api::zeros::<f16>(&[QLEN, QHEADS, D]).sync_on(&stream)?;
    unsafe {
        fmha_prefill_gqa_lpt_raw_resurrected(
            q.device_pointer().clone(),
            k_c.device_pointer().clone(),
            v_c.device_pointer().clone(),
            out_raw_c.device_pointer().clone(),
            value(scale), value(qgs), value(QLEN as i32), value(KVLEN as i32), value(0i32),
            value(num_q_blocks), value(num_head_groups),
            value(1i32), value(2i32), value(1i32),
        )
        .generics(generics.clone()).grid((grid_x, 1, 1)).sync_on(&stream)?;
    }
    let out_raw_p = api::zeros::<f16>(&[QLEN, QHEADS, D]).sync_on(&stream)?;
    unsafe {
        fmha_prefill_gqa_lpt_raw_resurrected(
            q.device_pointer().clone(),
            k_p.device_pointer().clone(),
            v_p.device_pointer().clone(),
            out_raw_p.device_pointer().clone(),
            value(scale), value(qgs), value(QLEN as i32), value(KVLEN as i32), value(0i32),
            value(num_q_blocks), value(num_head_groups),
            value(1i32), value(2i32), value(1i32),
        )
        .generics(generics.clone()).grid((grid_x, 1, 1)).sync_on(&stream)?;
    }

    let truth = out_raw.to_host_vec().sync_on(&stream)?;
    let chk = out_chk.to_host_vec().sync_on(&stream)?;
    let mut chk_bad = 0usize;
    for i in 0..truth.len() {
        if truth[i].to_f32() != chk[i].to_f32() {
            if chk_bad < 4 {
                eprintln!("checked mismatch @{i}: truth={} chk={}", truth[i].to_f32(), chk[i].to_f32());
            }
            chk_bad += 1;
        }
    }
    assert_eq!(chk_bad, 0, "checked LPT diverges from contiguous truth: {chk_bad}");

    // raw-kernel bug confirmation
    let raw_c = out_raw_c.to_host_vec().sync_on(&stream)?;
    let raw_p = out_raw_p.to_host_vec().sync_on(&stream)?;
    let mut raw_c_bad = 0usize;
    let mut raw_p_bad = 0usize;
    for i in 0..truth.len() {
        if truth[i].to_f32() != raw_c.get(i).map(|v| v.to_f32()).unwrap_or(f32::NAN) {
            raw_c_bad += 1;
        }
        if truth[i].to_f32() != raw_p[i].to_f32() {
            raw_p_bad += 1;
        }
    }
    eprintln!(
        "raw resurrected: contiguous mismatches={raw_c_bad}, padded mismatches={raw_p_bad} of {}",
        truth.len()
    );
    assert_eq!(raw_c_bad, 0, "raw kernel must match truth on a contiguous cache");
    assert!(
        raw_p_bad > 0,
        "raw kernel unexpectedly CORRECT on padded cache — stride bug not reproduced"
    );
    Ok(())
}
