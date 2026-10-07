use std::mem::MaybeUninit;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::Parser;
use cutile::cuda_async::device_operation::{DeviceOp, value, with_context};
use cutile::cuda_core::{IntoResult, Stream, sys as cu_sys};
use cutile::tensor::{DeviceVec, Reshape, Tensor, ToHostVec};
use cutile::tile_kernel::{CompileOptions, TileKernel};
use cutile::{api, core::f16};

use grout::kernels::group_gemm_f16_nt_desc;

#[derive(Parser, Debug)]
struct Args {
    /// Homogeneous M for the default benchmark shape.
    #[arg(long, default_value_t = 1024)]
    m: usize,

    /// Homogeneous N for the default benchmark shape.
    #[arg(long, default_value_t = 1024)]
    n: usize,

    /// Homogeneous K for the default benchmark shape.
    #[arg(long, default_value_t = 1024)]
    k: usize,

    /// Number of GEMM groups when --shapes is not supplied.
    #[arg(long, default_value_t = 16)]
    groups: usize,

    /// Semicolon-separated per-group shapes, e.g. "1024x1024x1024;2048x1024x1024".
    #[arg(long)]
    shapes: Option<String>,

    /// Use a small heterogeneous divisible-shape set.
    #[arg(long)]
    heterogeneous: bool,

    /// Tile config: 128x128x128, 256x256x64, or auto.
    #[arg(long, default_value = "auto")]
    tile_config: String,

    /// Timed benchmark iterations.
    #[arg(long, default_value_t = 100)]
    iters: usize,

    /// Untimed warmup iterations.
    #[arg(long, default_value_t = 10)]
    warmup_iters: usize,

    /// Validate outputs against the all-ones reference C = K.
    #[arg(long, default_value_t = true)]
    validate: bool,

    /// Suppress CSV header.
    #[arg(long)]
    no_header: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Problem {
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TileConfig {
    bm: usize,
    bn: usize,
    bk: usize,
    num_ctas: usize,
    occupancy: usize,
}

struct GroupBucket {
    cfg: TileConfig,
    problems: Vec<Problem>,
    a: DeviceVec<Tensor<f16>>,
    b: DeviceVec<Tensor<f16>>,
    c: DeviceVec<Tensor<f16>>,
    a_metas: Arc<Tensor<i32>>,
    b_metas: Arc<Tensor<i32>>,
    c_metas: Arc<Tensor<i32>>,
}

struct CudaEvent {
    event: cu_sys::CUevent,
}

impl CudaEvent {
    fn new() -> Result<Self> {
        let mut event = MaybeUninit::<cu_sys::CUevent>::uninit();
        unsafe {
            cu_sys::cuEventCreate(
                event.as_mut_ptr(),
                cu_sys::CUevent_flags_enum_CU_EVENT_DEFAULT,
            )
            .result()
            .map_err(|e| anyhow!("cuEventCreate failed: {e:?}"))?;
            Ok(Self {
                event: event.assume_init(),
            })
        }
    }

    fn record(&self, stream: &Stream) -> Result<()> {
        unsafe {
            cu_sys::cuEventRecord(self.event, stream.cu_stream())
                .result()
                .map_err(|e| anyhow!("cuEventRecord failed: {e:?}"))
        }
    }

    fn synchronize(&self) -> Result<()> {
        unsafe {
            cu_sys::cuEventSynchronize(self.event)
                .result()
                .map_err(|e| anyhow!("cuEventSynchronize failed: {e:?}"))
        }
    }

    fn elapsed_ms_since(&self, start: &CudaEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        unsafe {
            cu_sys::cuEventElapsedTime_v2(&mut ms, start.event, self.event)
                .result()
                .map_err(|e| anyhow!("cuEventElapsedTime_v2 failed: {e:?}"))?;
        }
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = cu_sys::cuEventDestroy_v2(self.event);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.iters > 0, "--iters must be positive");
    ensure!(args.warmup_iters > 0, "--warmup-iters must be positive");
    ensure!(args.groups > 0, "--groups must be positive");

    let stream = with_context(|ctx| value(ctx.get_cuda_stream().clone()))
        .await
        .map_err(|e| anyhow!("failed to get CUDA stream: {e:?}"))?;
    stream
        .device()
        .bind_to_thread()
        .map_err(|e| anyhow!("failed to bind CUDA context: {e:?}"))?;

    let physical_sms = num_sms(&stream)?;
    let problems = problems_from_args(&args)?;
    let buckets = build_buckets(&stream, &args, &problems)?;

    if args.validate {
        for bucket in &buckets {
            launch_bucket(&stream, bucket, physical_sms)?;
        }
        unsafe {
            stream
                .synchronize()
                .map_err(|e| anyhow!("validation synchronize failed: {e:?}"))?;
        }
        validate_buckets(&stream, &buckets)?;
    }

    for _ in 0..args.warmup_iters {
        for bucket in &buckets {
            launch_bucket(&stream, bucket, physical_sms)?;
        }
    }
    unsafe {
        stream
            .synchronize()
            .map_err(|e| anyhow!("warmup synchronize failed: {e:?}"))?;
    }

    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    start.record(&stream)?;
    for _ in 0..args.iters {
        for bucket in &buckets {
            launch_bucket(&stream, bucket, physical_sms)?;
        }
    }
    end.record(&stream)?;
    end.synchronize()?;
    let total_ms = end.elapsed_ms_since(&start)?;

    let flops_per_iter: f64 = problems
        .iter()
        .map(|p| 2.0 * p.m as f64 * p.n as f64 * p.k as f64)
        .sum();
    let elapsed_s = total_ms as f64 / 1000.0;
    let tflops = flops_per_iter * args.iters as f64 / elapsed_s / 1.0e12;
    let calls_per_sec = args.iters as f64 / elapsed_s;

    if !args.no_header {
        println!(
            "groups,buckets,tile_config,iters,warmup_iters,total_ms,avg_us,tflops,calls_per_sec,physical_sms"
        );
    }
    println!(
        "{},{},{},{},{},{:.6},{:.3},{:.3},{:.3},{}",
        problems.len(),
        buckets.len(),
        args.tile_config,
        args.iters,
        args.warmup_iters,
        total_ms,
        total_ms * 1000.0 / args.iters as f32,
        tflops,
        calls_per_sec,
        physical_sms,
    );

    Ok(())
}

fn problems_from_args(args: &Args) -> Result<Vec<Problem>> {
    if let Some(spec) = &args.shapes {
        return parse_shapes(spec);
    }

    if args.heterogeneous {
        let base = [
            Problem {
                m: 1024,
                n: 1024,
                k: 1024,
            },
            Problem {
                m: 2048,
                n: 1024,
                k: 1024,
            },
            Problem {
                m: 1024,
                n: 2048,
                k: 1024,
            },
            Problem {
                m: 2048,
                n: 2048,
                k: 1024,
            },
        ];
        let mut out = Vec::with_capacity(args.groups);
        for i in 0..args.groups {
            out.push(base[i % base.len()]);
        }
        return Ok(out);
    }

    Ok(vec![
        Problem {
            m: args.m,
            n: args.n,
            k: args.k,
        };
        args.groups
    ])
}

fn parse_shapes(spec: &str) -> Result<Vec<Problem>> {
    let mut out = Vec::new();
    for raw in spec.split(';') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let parts: Vec<_> = raw
            .split(['x', 'X', ','])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        ensure!(parts.len() == 3, "shape `{raw}` must be MxNxK or M,N,K");
        out.push(Problem {
            m: parts[0]
                .parse()
                .with_context(|| format!("bad M in `{raw}`"))?,
            n: parts[1]
                .parse()
                .with_context(|| format!("bad N in `{raw}`"))?,
            k: parts[2]
                .parse()
                .with_context(|| format!("bad K in `{raw}`"))?,
        });
    }
    ensure!(!out.is_empty(), "--shapes produced no problems");
    Ok(out)
}

fn build_buckets(
    stream: &Arc<Stream>,
    args: &Args,
    problems: &[Problem],
) -> Result<Vec<GroupBucket>> {
    struct BucketBuild {
        cfg: TileConfig,
        problems: Vec<Problem>,
        a: Vec<Tensor<f16>>,
        b: Vec<Tensor<f16>>,
        c: Vec<Tensor<f16>>,
    }

    let mut buckets: Vec<BucketBuild> = Vec::new();
    for &problem in problems {
        let cfg = choose_tile_config(&args.tile_config, problem)?;
        ensure_tile_divisible(problem, cfg)?;
        let b_shape = vec![problem.k, problem.n];
        let a = api::ones::<f16>(&[problem.m, problem.k])
            .sync_on(stream)
            .map_err(|e| anyhow!("alloc/init A failed: {e:?}"))?;
        let b = api::ones::<f16>(&b_shape)
            .sync_on(stream)
            .map_err(|e| anyhow!("alloc/init B failed: {e:?}"))?;
        let c = api::zeros::<f16>(&[problem.m, problem.n])
            .sync_on(stream)
            .map_err(|e| anyhow!("alloc/init C failed: {e:?}"))?;

        let idx = buckets.iter().position(|b| b.cfg == cfg);
        let idx = match idx {
            Some(i) => i,
            None => {
                buckets.push(BucketBuild {
                    cfg,
                    problems: Vec::new(),
                    a: Vec::new(),
                    b: Vec::new(),
                    c: Vec::new(),
                });
                buckets.len() - 1
            }
        };
        buckets[idx].problems.push(problem);
        buckets[idx].a.push(a);
        buckets[idx].b.push(b);
        buckets[idx].c.push(c);
    }

    let mut out = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        let a_meta_host = Arc::new(meta_rows(&bucket.problems, TensorRole::A));
        let b_meta_host = Arc::new(meta_rows(&bucket.problems, TensorRole::B));
        let c_meta_host = Arc::new(meta_rows(&bucket.problems, TensorRole::C));
        let group_count = bucket.problems.len();
        out.push(GroupBucket {
            cfg: bucket.cfg,
            problems: bucket.problems,
            a: DeviceVec::from(bucket.a),
            b: DeviceVec::from(bucket.b),
            c: DeviceVec::from(bucket.c),
            a_metas: Arc::new(
                api::copy_host_vec_to_device(&a_meta_host)
                    .sync_on(stream)
                    .map_err(|e| anyhow!("copy a_metas failed: {e:?}"))?
                    .reshape(&[group_count, 8])?,
            ),
            b_metas: Arc::new(
                api::copy_host_vec_to_device(&b_meta_host)
                    .sync_on(stream)
                    .map_err(|e| anyhow!("copy b_metas failed: {e:?}"))?
                    .reshape(&[group_count, 8])?,
            ),
            c_metas: Arc::new(
                api::copy_host_vec_to_device(&c_meta_host)
                    .sync_on(stream)
                    .map_err(|e| anyhow!("copy c_metas failed: {e:?}"))?
                    .reshape(&[group_count, 8])?,
            ),
        });
    }
    Ok(out)
}

#[derive(Clone, Copy)]
enum TensorRole {
    A,
    B,
    C,
}

fn meta_rows(problems: &[Problem], role: TensorRole) -> Vec<i32> {
    let mut rows = Vec::with_capacity(problems.len() * 8);
    for problem in problems {
        let (rows_dim, cols_dim, stride0) = match role {
            TensorRole::A => (problem.m, problem.k, problem.k),
            TensorRole::B => (problem.k, problem.n, problem.n),
            TensorRole::C => (problem.m, problem.n, problem.n),
        };
        rows.push(rows_dim as i32);
        rows.push(cols_dim as i32);
        rows.push(stride0 as i32);
        rows.push(0);
        rows.push(0);
        rows.push(0);
        rows.push(0);
        rows.push(0);
    }
    rows
}

fn choose_tile_config(spec: &str, problem: Problem) -> Result<TileConfig> {
    let sm100_large = TileConfig {
        bm: 256,
        bn: 256,
        bk: 64,
        num_ctas: 2,
        occupancy: 1,
    };
    let baseline = TileConfig {
        bm: 128,
        bn: 128,
        bk: 128,
        num_ctas: 1,
        occupancy: 1,
    };
    match spec.trim().to_ascii_lowercase().as_str() {
        "auto" => {
            if problem.m >= 2048 && problem.n >= 2048 && problem.k >= 512 {
                Ok(sm100_large)
            } else {
                Ok(baseline)
            }
        }
        "128x128x128" | "128,128,128" => Ok(baseline),
        "256x256x64" | "256,256,64" => Ok(sm100_large),
        other => {
            bail!("unknown --tile-config `{other}`; expected auto, 128x128x128, or 256x256x64")
        }
    }
}

fn ensure_tile_divisible(problem: Problem, cfg: TileConfig) -> Result<()> {
    ensure!(
        problem.m % cfg.bm == 0 && problem.n % cfg.bn == 0 && problem.k % cfg.bk == 0,
        "fast group_gemm requires tile-divisible shapes for now: problem={:?}, tile={:?}",
        problem,
        cfg
    );
    Ok(())
}

fn launch_bucket(stream: &Arc<Stream>, bucket: &GroupBucket, physical_sms: usize) -> Result<()> {
    let cfg = bucket.cfg;
    let grid_x = (physical_sms / cfg.num_ctas * cfg.occupancy).max(1) as u32;
    let compile_options = CompileOptions::default()
        .occupancy(cfg.occupancy as i32)
        .num_cta_in_cga(cfg.num_ctas as i32)
        .max_divisibility(16);
    unsafe {
        let a_ptrs = value(bucket.a.inner().clone());
        let b_ptrs = value(bucket.b.inner().clone());
        let c_ptrs = value(bucket.c.inner().clone());
        group_gemm_f16_nt_desc(
            a_ptrs,
            b_ptrs,
            c_ptrs,
            value(bucket.a_metas.clone()),
            value(bucket.b_metas.clone()),
            value(bucket.c_metas.clone()),
            value(bucket.problems.len() as i32),
        )
        .generics(vec![
            cfg.bm.to_string(),
            cfg.bn.to_string(),
            cfg.bk.to_string(),
            grid_x.to_string(),
        ])
        .const_grid((grid_x, 1, 1))
        .compile_options(compile_options)
        .async_on(stream)
        .map_err(|e| anyhow!("group_gemm launch failed: {e:?}"))?;
    }
    Ok(())
}

fn validate_buckets(stream: &Arc<Stream>, buckets: &[GroupBucket]) -> Result<()> {
    for bucket in buckets {
        for (group_idx, problem) in bucket.problems.iter().enumerate() {
            let host = (&bucket.c[group_idx])
                .to_host_vec()
                .sync_on(stream)
                .map_err(|e| anyhow!("copy C[{group_idx}] to host failed: {e:?}"))?;
            let expected = problem.k as f32;
            let mut max_abs = 0.0f32;
            for &x in host.iter().take(1024) {
                max_abs = max_abs.max((x.to_f32() - expected).abs());
            }
            ensure!(
                max_abs <= 1.0,
                "validation failed for group {group_idx}: max_abs={max_abs}, expected={expected}"
            );
        }
    }
    Ok(())
}

fn num_sms(stream: &Arc<Stream>) -> Result<usize> {
    let mut value = 0i32;
    unsafe {
        cu_sys::cuDeviceGetAttribute(
            &mut value as *mut i32,
            cu_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            stream.device().cu_device(),
        )
        .result()
        .map_err(|e| anyhow!("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed: {e:?}"))?;
    }
    ensure!(value > 0, "CUDA reported invalid SM count {value}");
    Ok(value as usize)
}
