use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use clap::Parser;
use cutile::cuda_async::device_operation::{value, with_context, DeviceOp};
use cutile::cuda_core::Stream;
use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use cutile::{api, core::f16};

use grout::kernels::{
    fmha_prefill_causal_mapped,
    fmha_prefill_gqa_lpt_checked, fmha_prefill_gqa_lpt_split, fmha_prefill_gqa_mapped,
    prefill_splitk_reduce_merge,
};

#[derive(Parser, Debug)]
struct Args {
    /// Attention kernel: causal or gqa.
    #[arg(long, default_value = "causal")]
    mode: String,

    #[arg(long, default_value_t = 2048)]
    q_len: usize,

    #[arg(long, default_value_t = 64)]
    q_heads: usize,

    #[arg(long, default_value_t = 8)]
    kv_heads: usize,

    #[arg(long, default_value_t = 128)]
    head_dim: usize,

    #[arg(long, default_value_t = 32)]
    bm: usize,

    #[arg(long, default_value_t = 16)]
    bn: usize,

    /// Q heads per grouped CTA for --mode gqa. Defaults to q_heads / kv_heads.
    #[arg(long)]
    group: Option<usize>,

    /// Head-group swizzle for --mode gqa-lpt. Defaults to the TileGym L2-fit heuristic.
    #[arg(long)]
    swizzle: Option<usize>,

    /// LPT scheduler: 0=swizzled q-block-major reverse, 1=q-block-major reverse,
    /// 2=head-group-major reverse, 3=swizzled q-block-major forward.
    #[arg(long, default_value_t = 1)]
    schedule: usize,

    /// Split full causal K tiles from masked boundary tiles in the LPT kernel.
    #[arg(long, default_value_t = 0)]
    mask_split: usize,

    /// Number of K splits for --mode gqa-lpt-split.
    #[arg(long, default_value_t = 4)]
    kv_splits: usize,

    /// D chunk for --mode gqa-lpt-split merge.
    #[arg(long, default_value_t = 16)]
    merge_chunk_d: usize,

    #[arg(long, default_value_t = 3)]
    latency: usize,

    #[arg(long, default_value_t = 2)]
    occupancy: usize,

    #[arg(long, default_value_t = 20)]
    iters: usize,

    #[arg(long, default_value_t = 5)]
    warmup_iters: usize,

    #[arg(long, default_value = "current")]
    label: String,

    #[arg(long)]
    no_header: bool,

    /// Check output against a CPU reference. Intended for small shapes.
    #[arg(long)]
    check: bool,

    /// Print an FNV-1a hash of the output tensor bytes (stderr) for
    /// cross-arm bitwise comparison; inputs are deterministic by index.
    #[arg(long, default_value_t = false)]
    output_hash: bool,
}

struct Buffers {
    q: Arc<Tensor<f16>>,
    k: Arc<Tensor<f16>>,
    v: Arc<Tensor<f16>>,
    out: Tensor<f16>,
    att_partial: Option<Tensor<f16>>,
    lse_partial: Option<Tensor<f32>>,
    q_host: Option<Vec<f32>>,
    k_host: Option<Vec<f32>>,
    v_host: Option<Vec<f32>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.iters > 0, "--iters must be positive");
    ensure!(args.q_len > 0, "--q-len must be positive");
    ensure!(args.q_heads > 0, "--q-heads must be positive");
    ensure!(args.kv_heads > 0, "--kv-heads must be positive");
    ensure!(
        args.q_heads % args.kv_heads == 0,
        "q_heads must be divisible by kv_heads"
    );
    ensure!(args.head_dim > 0, "--head-dim must be positive");
    ensure!(args.bm > 0 && args.bn > 0, "--bm and --bn must be positive");
    ensure!(args.schedule <= 3, "--schedule must be in 0..=3");
    ensure!(args.mask_split <= 1, "--mask-split must be 0 or 1");
    ensure!(args.kv_splits > 0, "--kv-splits must be positive");
    ensure!(
        args.merge_chunk_d > 0 && args.head_dim % args.merge_chunk_d == 0,
        "--merge-chunk-d must be positive and divide --head-dim"
    );

    let mode = args.mode.trim().to_ascii_lowercase();
    if mode != "causal"
        && mode != "gqa"
        && mode != "gqa-lpt"
        && mode != "gqa-lpt-split"
    {
        bail!(
            "unknown --mode `{}`; expected causal, gqa, gqa-lpt, or gqa-lpt-split",
            args.mode
        );
    }

    let stream = with_context(|ctx| value(ctx.get_cuda_stream().clone()))
        .await
        .map_err(|e| anyhow!("failed to get CUDA stream: {e:?}"))?;
    stream
        .device()
        .bind_to_thread()
        .map_err(|e| anyhow!("failed to bind CUDA context: {e:?}"))?;

    let mut buffers = alloc_buffers(&stream, &args, &mode)?;

    if args.check {
        let out = std::mem::replace(
            &mut buffers.out,
            alloc_zeros(&stream, &[args.q_len, args.q_heads, args.head_dim], "out")?,
        );
        let checked_out = launch_attention(
            &stream,
            &args,
            &mode,
            out,
            &buffers.q,
            &buffers.k,
            &buffers.v,
            buffers.att_partial.as_mut(),
            buffers.lse_partial.as_mut(),
        )?;
        check_attention(&stream, &args, &buffers, checked_out)?;
        buffers.out = alloc_zeros(&stream, &[args.q_len, args.q_heads, args.head_dim], "out")?;
    }
    if args.output_hash {
        let out = std::mem::replace(
            &mut buffers.out,
            alloc_zeros(&stream, &[args.q_len, args.q_heads, args.head_dim], "out")?,
        );
        let hashed_out = launch_attention(
            &stream,
            &args,
            &mode,
            out,
            &buffers.q,
            &buffers.k,
            &buffers.v,
            buffers.att_partial.as_mut(),
            buffers.lse_partial.as_mut(),
        )?;
        let host: Vec<f16> = hashed_out.to_host_vec().sync_on(&stream)?;
        let mut h: u64 = 0xcbf29ce484222325;
        for v in &host {
            for b in v.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        eprintln!("output_hash={h:016x}");
        buffers.out = alloc_zeros(&stream, &[args.q_len, args.q_heads, args.head_dim], "out")?;
    }

    if !args.no_header {
        println!(
            "label,mode,q_len,q_heads,kv_heads,head_dim,bm,bn,group,swizzle,schedule,mask_split,kv_splits,merge_chunk_d,latency,occupancy,iters,total_ms,avg_us,calls_per_sec"
        );
    }

    let total_ms = time_attention(&stream, &args, &mode, buffers)?;
    let avg_us = total_ms * 1000.0 / args.iters as f64;
    let calls_per_sec = 1.0e6 / avg_us;
    let group = args.group.unwrap_or(args.q_heads / args.kv_heads);
    println!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.3},{:.3}",
        args.label,
        mode,
        args.q_len,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.bm,
        args.bn,
        group,
        lpt_swizzle(&args, args.q_heads / group),
        args.schedule,
        args.mask_split,
        args.kv_splits,
        args.merge_chunk_d,
        args.latency,
        args.occupancy,
        args.iters,
        total_ms,
        avg_us,
        calls_per_sec,
    );

    Ok(())
}

fn alloc_zeros(stream: &Arc<Stream>, shape: &[usize], name: &str) -> Result<Tensor<f16>> {
    api::zeros::<f16>(shape)
        .sync_on(stream)
        .map_err(|e| anyhow!("alloc/init {name} failed: {e:?}"))
}

fn pattern_value(i: usize, salt: usize) -> f32 {
    let raw = ((i.wrapping_mul(37) + salt.wrapping_mul(17)) % 101) as f32;
    (raw - 50.0) / 80.0
}

fn host_pattern(len: usize, salt: usize) -> (Vec<f16>, Vec<f32>) {
    let h16: Vec<f16> = (0..len)
        .map(|i| f16::from_f32(pattern_value(i, salt)))
        .collect();
    let h32: Vec<f32> = h16.iter().map(|v| v.to_f32()).collect();
    (h16, h32)
}

fn alloc_pattern(
    stream: &Arc<Stream>,
    len: usize,
    shape: &[usize],
    salt: usize,
    name: &str,
) -> Result<(Tensor<f16>, Vec<f32>)> {
    let (host_f16, host_f32) = host_pattern(len, salt);
    let tensor = api::copy_host_vec_to_device(&Arc::new(host_f16))
        .sync_on(stream)
        .map_err(|e| anyhow!("copy {name} failed: {e:?}"))?
        .reshape(shape)
        .map_err(|e| anyhow!("reshape {name} failed: {e:?}"))?;
    Ok((tensor, host_f32))
}

fn split_scratch_shapes(args: &Args) -> Result<(usize, usize)> {
    let qgs = args.q_heads / args.kv_heads;
    let group = args.group.unwrap_or(qgs);
    ensure!(
        group >= 1 && qgs % group == 0,
        "group must divide q_group_size={qgs}"
    );
    ensure!(
        args.q_heads % group == 0,
        "group must divide q_heads={}",
        args.q_heads
    );
    let num_q_blocks = args.q_len.div_ceil(args.bm);
    let num_head_groups = args.q_heads / group;
    let total_tiles = num_q_blocks * num_head_groups;
    let ns_m = args.kv_splits * args.bm * group;
    Ok((total_tiles, ns_m))
}

fn alloc_split_scratch(
    stream: &Arc<Stream>,
    args: &Args,
    mode: &str,
) -> Result<(Option<Tensor<f16>>, Option<Tensor<f32>>)> {
    if mode != "gqa-lpt-split" {
        return Ok((None, None));
    }
    let (total_tiles, ns_m) = split_scratch_shapes(args)?;
    let att = alloc_zeros(stream, &[total_tiles, ns_m, args.head_dim], "att_partial")?;
    let lse = api::zeros::<f32>(&[total_tiles, ns_m])
        .sync_on(stream)
        .map_err(|e| anyhow!("alloc/init lse_partial failed: {e:?}"))?;
    Ok((Some(att), Some(lse)))
}

fn alloc_buffers(stream: &Arc<Stream>, args: &Args, mode: &str) -> Result<Buffers> {
    let (att_partial, lse_partial) = alloc_split_scratch(stream, args, mode)?;
    // The deterministic input pattern is required by --check AND by
    // --output-hash: over zero-filled Q/K/V every attention kernel writes an
    // all-zero output, so a hash of it compares nothing (found the hard way —
    // the hash matched the FNV-1a of an all-zero buffer).
    if args.check || args.output_hash {
        let q_len = args.q_len * args.q_heads * args.head_dim;
        let k_len = args.kv_heads * args.q_len * args.head_dim;
        let v_len = k_len;
        let (q, q_host) = alloc_pattern(
            stream,
            q_len,
            &[args.q_len, args.q_heads, args.head_dim],
            1,
            "q",
        )?;
        let (k, k_host) = alloc_pattern(
            stream,
            k_len,
            &[args.kv_heads, args.q_len, args.head_dim],
            2,
            "k",
        )?;
        let (v, v_host) = alloc_pattern(
            stream,
            v_len,
            &[args.kv_heads, args.q_len, args.head_dim],
            3,
            "v",
        )?;
        Ok(Buffers {
            q: Arc::new(q),
            k: Arc::new(k),
            v: Arc::new(v),
            out: alloc_zeros(stream, &[args.q_len, args.q_heads, args.head_dim], "out")?,
            att_partial,
            lse_partial,
            q_host: Some(q_host),
            k_host: Some(k_host),
            v_host: Some(v_host),
        })
    } else {
        Ok(Buffers {
            q: Arc::new(alloc_zeros(
                stream,
                &[args.q_len, args.q_heads, args.head_dim],
                "q",
            )?),
            k: Arc::new(alloc_zeros(
                stream,
                &[args.kv_heads, args.q_len, args.head_dim],
                "k",
            )?),
            v: Arc::new(alloc_zeros(
                stream,
                &[args.kv_heads, args.q_len, args.head_dim],
                "v",
            )?),
            out: alloc_zeros(stream, &[args.q_len, args.q_heads, args.head_dim], "out")?,
            att_partial,
            lse_partial,
            q_host: None,
            k_host: None,
            v_host: None,
        })
    }
}

fn floor_power_of_two_le(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p.saturating_mul(2) <= n {
        p *= 2;
    }
    p
}

fn lpt_swizzle(args: &Args, num_head_groups: usize) -> usize {
    if let Some(swizzle) = args.swizzle {
        return swizzle.min(num_head_groups).max(1);
    }
    let bytes_per_kv_head = args
        .q_len
        .saturating_mul(args.head_dim.saturating_mul(2))
        .saturating_mul(std::mem::size_of::<f16>());
    let fit = if bytes_per_kv_head == 0 {
        1
    } else {
        (50 * 1024 * 1024 / bytes_per_kv_head).max(1)
    };
    floor_power_of_two_le(fit).min(num_head_groups).max(1)
}

fn time_attention(
    stream: &Arc<Stream>,
    args: &Args,
    mode: &str,
    mut buffers: Buffers,
) -> Result<f64> {
    for _ in 0..args.warmup_iters {
        buffers.out = launch_attention(
            stream,
            args,
            mode,
            buffers.out,
            &buffers.q,
            &buffers.k,
            &buffers.v,
            buffers.att_partial.as_mut(),
            buffers.lse_partial.as_mut(),
        )?;
    }

    let start = Instant::now();
    for _ in 0..args.iters {
        buffers.out = launch_attention(
            stream,
            args,
            mode,
            buffers.out,
            &buffers.q,
            &buffers.k,
            &buffers.v,
            buffers.att_partial.as_mut(),
            buffers.lse_partial.as_mut(),
        )?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn launch_attention(
    stream: &Arc<Stream>,
    args: &Args,
    mode: &str,
    out: Tensor<f16>,
    q: &Arc<Tensor<f16>>,
    k: &Arc<Tensor<f16>>,
    v: &Arc<Tensor<f16>>,
    att_partial: Option<&mut Tensor<f16>>,
    lse_partial: Option<&mut Tensor<f32>>,
) -> Result<Tensor<f16>> {
    let qk_scale = 1.0f32 / (args.head_dim as f32).sqrt();
    let qgs = args.q_heads / args.kv_heads;
    let even_k = if args.q_len % args.bn == 0 { 1 } else { 0 };

    if mode == "gqa" || mode == "gqa-lpt" || mode == "gqa-lpt-split" {
        let group = args.group.unwrap_or(qgs);
        ensure!(
            group >= 1 && qgs % group == 0,
            "group must divide q_group_size={qgs}"
        );
        let m_eff = args.bm * group;
        if mode == "gqa-lpt-split" {
            ensure!(
                args.q_heads % group == 0,
                "group must divide q_heads={}",
                args.q_heads
            );
            let num_q_blocks = args.q_len.div_ceil(args.bm);
            let num_head_groups = args.q_heads / group;
            let total_tiles = num_q_blocks * num_head_groups;
            let ns_m = args.kv_splits * m_eff;
            let swizzle = lpt_swizzle(args, num_head_groups);
            let num_hb_quotient = num_head_groups / swizzle;
            let num_hb_remainder = (num_head_groups % swizzle).max(1);
            let att_partial = att_partial.context("missing att_partial for gqa-lpt-split mode")?;
            let lse_partial = lse_partial.context("missing lse_partial for gqa-lpt-split mode")?;
            unsafe {
                fmha_prefill_gqa_lpt_split(
                    q.device_pointer().clone(),
                    k.device_pointer().clone(),
                    v.device_pointer().clone(),
                    att_partial.device_pointer().clone(),
                    lse_partial.device_pointer().clone(),
                    value(qk_scale),
                    value(qgs as i32),
                    value(args.q_len as i32),
                    value(args.q_len as i32),
                    value(0i32),
                    value(num_q_blocks as i32),
                    value(num_head_groups as i32),
                    value(swizzle as i32),
                    value(num_hb_quotient as i32),
                    value(num_hb_remainder as i32),
                )
            }
            .generics(vec![
                args.bm.to_string(),
                args.bn.to_string(),
                args.head_dim.to_string(),
                group.to_string(),
                m_eff.to_string(),
                even_k.to_string(),
                args.kv_splits.to_string(),
                ns_m.to_string(),
                args.latency.to_string(),
                args.schedule.to_string(),
            ])
            .grid((total_tiles as u32, args.kv_splits as u32, 1))
            .compile_options(
                cutile::tile_kernel::CompileOptions::default().occupancy(args.occupancy as i32),
            )
            .sync_on(stream)
            .map_err(|e| anyhow!("fmha_prefill_gqa_lpt_split failed: {e:?}"))?;

            unsafe {
                prefill_splitk_reduce_merge(
                    att_partial.device_pointer().clone(),
                    lse_partial.device_pointer().clone(),
                    out.device_pointer().clone(),
                    value(args.q_len as i32),
                    value(num_q_blocks as i32),
                    value(num_head_groups as i32),
                    value(swizzle as i32),
                    value(num_hb_quotient as i32),
                    value(num_hb_remainder as i32),
                )
            }
            .generics(vec![
                args.bm.to_string(),
                group.to_string(),
                args.head_dim.to_string(),
                m_eff.to_string(),
                args.merge_chunk_d.to_string(),
                args.kv_splits.to_string(),
                ns_m.to_string(),
                args.schedule.to_string(),
                args.latency.to_string(),
            ])
            .grid((
                total_tiles as u32,
                (args.head_dim / args.merge_chunk_d) as u32,
                1,
            ))
            .compile_options(
                cutile::tile_kernel::CompileOptions::default().occupancy(args.occupancy as i32),
            )
            .sync_on(stream)
            .map_err(|e| anyhow!("prefill_splitk_reduce_merge failed: {e:?}"))?;
            return Ok(out);
        }
        if mode == "gqa-lpt" {
            ensure!(
                args.q_heads % group == 0,
                "group must divide q_heads={}",
                args.q_heads
            );
            let num_q_blocks = args.q_len.div_ceil(args.bm);
            let num_head_groups = args.q_heads / group;
            let swizzle = lpt_swizzle(args, num_head_groups);
            let num_hb_quotient = num_head_groups / swizzle;
            let num_hb_remainder = (num_head_groups % swizzle).max(1);
            fmha_prefill_gqa_lpt_checked(
                &*q,
                &*k,
                &*v,
                &out,
                value(qk_scale),
                value(qgs as i32),
                value(args.q_len as i32),
                value(0i32),
                value(num_q_blocks as i32),
                value(num_head_groups as i32),
                value(swizzle as i32),
                value(num_hb_quotient as i32),
                value(num_hb_remainder as i32),
            )
            .generics(vec![
                args.bm.to_string(),
                args.bn.to_string(),
                args.head_dim.to_string(),
                group.to_string(),
                m_eff.to_string(),
                1.to_string(),
                even_k.to_string(),
                args.latency.to_string(),
                args.schedule.to_string(),
                args.mask_split.to_string(),
            ])
            .grid(((num_q_blocks * num_head_groups) as u32, 1, 1))
            .compile_options(
                cutile::tile_kernel::CompileOptions::default().occupancy(args.occupancy as i32),
            )
            .sync_on(stream)
            .map_err(|e| anyhow!("fmha_prefill_gqa_lpt_checked failed: {e:?}"))?;
            return Ok(out);
        }
        ensure!(
            args.q_heads % group == 0,
            "group must divide q_heads={}",
            args.q_heads
        );
        let ntb = (args.q_len.div_ceil(args.bm) * (args.q_heads / group)) as u32;
        let out_part = out
            .partition([args.bm, group, args.head_dim])
            .map([1, 1, 1], ntb);
        let result = fmha_prefill_gqa_mapped(
            value(out_part),
            value(q.clone()),
            value(k.clone()),
            value(v.clone()),
            value(qk_scale),
            value(qgs as i32),
            value(args.q_len as i32),
            value(0i32),
        )
        .generics(vec![
            args.bm.to_string(),
            args.bn.to_string(),
            args.head_dim.to_string(),
            group.to_string(),
            m_eff.to_string(),
            1.to_string(),
            even_k.to_string(),
            args.latency.to_string(),
            "1".to_string(),
            "1".to_string(),
            "1".to_string(),
        ])
        .compile_options(
            cutile::tile_kernel::CompileOptions::default().occupancy(args.occupancy as i32),
        )
        .sync_on(stream)
        .map_err(|e| anyhow!("fmha_prefill_gqa_mapped failed: {e:?}"))?;
        Ok(result.0.unpartition())
    } else {
        let ntb = (args.q_len.div_ceil(args.bm) * args.q_heads) as u32;
        let out_part = out
            .partition([args.bm, 1, args.head_dim])
            .map([1, 1, 1], ntb);
        let result = fmha_prefill_causal_mapped(
            value(out_part),
            value(q.clone()),
            value(k.clone()),
            value(v.clone()),
            value(qk_scale),
            value(qgs as i32),
            value(args.q_len as i32),
            value(0i32),
        )
        .generics(vec![
            args.bm.to_string(),
            args.bn.to_string(),
            args.head_dim.to_string(),
            1.to_string(),
            even_k.to_string(),
            args.latency.to_string(),
            "1".to_string(),
            "1".to_string(),
            "1".to_string(),
        ])
        .compile_options(
            cutile::tile_kernel::CompileOptions::default().occupancy(args.occupancy as i32),
        )
        .sync_on(stream)
        .map_err(|e| anyhow!("fmha_prefill_causal_mapped failed: {e:?}"))?;
        Ok(result.0.unpartition())
    }
}

fn check_attention(
    stream: &Arc<Stream>,
    args: &Args,
    buffers: &Buffers,
    out: Tensor<f16>,
) -> Result<()> {
    let q = buffers
        .q_host
        .as_ref()
        .context("missing q host data for check")?;
    let k = buffers
        .k_host
        .as_ref()
        .context("missing k host data for check")?;
    let v = buffers
        .v_host
        .as_ref()
        .context("missing v host data for check")?;
    let actual_f16: Vec<f16> = out
        .to_host_vec()
        .sync_on(stream)
        .map_err(|e| anyhow!("copy output for check failed: {e:?}"))?;
    let actual: Vec<f32> = actual_f16.iter().map(|x| x.to_f32()).collect();
    let expected = cpu_reference(args, q, k, v);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs = (a - e).abs();
        let rel = abs / e.abs().max(1.0e-3);
        if abs > max_abs {
            max_abs = abs;
            max_rel = rel;
            max_idx = i;
        }
    }
    println!(
        "check: max_abs={max_abs:.6}, max_rel={max_rel:.6}, idx={max_idx}, actual={:.6}, expected={:.6}",
        actual[max_idx], expected[max_idx]
    );
    ensure!(
        max_abs < 0.08 || max_rel < 0.08,
        "attention check failed: max_abs={max_abs}, max_rel={max_rel}, idx={max_idx}"
    );
    Ok(())
}

fn cpu_reference(args: &Args, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; args.q_len * args.q_heads * args.head_dim];
    let qgs = args.q_heads / args.kv_heads;
    let scale = 1.0f32 / (args.head_dim as f32).sqrt();
    for m in 0..args.q_len {
        for qh in 0..args.q_heads {
            let kvh = qh / qgs;
            let mut scores = vec![0.0f32; m + 1];
            let mut max_score = f32::NEG_INFINITY;
            for n in 0..=m {
                let mut dot = 0.0f32;
                for d in 0..args.head_dim {
                    let q_idx = (m * args.q_heads + qh) * args.head_dim + d;
                    let k_idx = (kvh * args.q_len + n) * args.head_dim + d;
                    dot += q[q_idx] * k[k_idx];
                }
                let score = dot * scale;
                scores[n] = score;
                max_score = max_score.max(score);
            }
            let mut denom = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            for d in 0..args.head_dim {
                let mut acc = 0.0f32;
                for n in 0..=m {
                    let v_idx = (kvh * args.q_len + n) * args.head_dim + d;
                    acc += scores[n] / denom * v[v_idx];
                }
                let out_idx = (m * args.q_heads + qh) * args.head_dim + d;
                out[out_idx] = acc;
            }
        }
    }
    out
}
