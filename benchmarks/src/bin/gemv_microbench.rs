use std::mem::MaybeUninit;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use cutile::cuda_async::device_operation::{DeviceOp, value, with_context};
use cutile::cuda_core::{IntoResult, Stream, sys as cu_sys};
use cutile::{api, core::f16, tensor::Tensor};

#[path = "../../../src/cublas.rs"]
#[allow(dead_code)]
mod cublas;

#[derive(Parser, Debug)]
struct Args {
    /// Comma-separated shape list: all,qkv,o_proj,gate_up,down,lm_head.
    /// 32B also supports prefill slices: q,k,v,gate_only,up_only.
    #[arg(long, default_value = "all")]
    shape: String,

    /// Operation to benchmark: gemv uses n=1, gemm uses --n rows.
    #[arg(long, default_value = "gemv")]
    mode: String,

    /// RHS row count for --mode gemm, corresponding to prompt length.
    #[arg(long, default_value_t = 2048)]
    n: usize,

    /// Model shape set: 4b or 32b.
    #[arg(long, default_value = "4b")]
    model_size: String,

    /// Label written to the CSV output. Use this for env/algorithm sweeps.
    #[arg(long, default_value = "current")]
    label: String,

    /// Timed GEMV launches per shape.
    #[arg(long, default_value_t = 200)]
    iters: usize,

    /// Untimed GEMV launches per shape before timing.
    #[arg(long, default_value_t = 20)]
    warmup_iters: usize,

    /// Independent matrix/vector/output copies to cycle through during timing.
    #[arg(long, default_value_t = 1)]
    copies: usize,

    /// Suppress CSV header. Useful when appending sweep cases.
    #[arg(long)]
    no_header: bool,
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
}

struct Buffers {
    matrix: Tensor<f16>,
    rhs: Tensor<f16>,
    out: Tensor<f16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchMode {
    Gemv,
    Gemm,
}

const SHAPES_4B: &[Shape] = &[
    Shape {
        name: "qkv",
        m: 6144,
        k: 2560,
    },
    Shape {
        name: "o_proj",
        m: 2560,
        k: 4096,
    },
    Shape {
        name: "gate_up",
        m: 19456,
        k: 2560,
    },
    Shape {
        name: "down",
        m: 2560,
        k: 9728,
    },
    Shape {
        name: "lm_head",
        m: 151936,
        k: 2560,
    },
];

const SHAPES_32B: &[Shape] = &[
    Shape {
        name: "qkv",
        m: 10240,
        k: 5120,
    },
    Shape {
        name: "o_proj",
        m: 5120,
        k: 8192,
    },
    Shape {
        name: "gate_up",
        m: 51200,
        k: 5120,
    },
    Shape {
        name: "down",
        m: 5120,
        k: 25600,
    },
    Shape {
        name: "lm_head",
        m: 151936,
        k: 5120,
    },
];

const SHAPES_32B_PREFILL_SLICES: &[Shape] = &[
    Shape {
        name: "q",
        m: 8192,
        k: 5120,
    },
    Shape {
        name: "k",
        m: 1024,
        k: 5120,
    },
    Shape {
        name: "v",
        m: 1024,
        k: 5120,
    },
    Shape {
        name: "gate_only",
        m: 25600,
        k: 5120,
    },
    Shape {
        name: "up_only",
        m: 25600,
        k: 5120,
    },
];

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
    if args.iters == 0 {
        bail!("--iters must be positive");
    }
    if args.copies == 0 {
        bail!("--copies must be positive");
    }
    if args.n == 0 {
        bail!("--n must be positive");
    }

    let shapes = selected_shapes(&args.model_size, &args.shape)?;
    let mode = selected_mode(&args.mode)?;
    let stream = with_context(|ctx| value(ctx.get_cuda_stream().clone()))
        .await
        .map_err(|e| anyhow!("failed to get CUDA stream: {e:?}"))?;

    stream
        .device()
        .bind_to_thread()
        .map_err(|e| anyhow!("failed to bind CUDA context: {e:?}"))?;

    if !args.no_header {
        println!(
            "label,mode,shape,m,n,k,iters,copies,total_ms,avg_us,weight_gb,io_gb,tflops,effective_weight_gbps,effective_io_gbps,calls_per_sec"
        );
    }

    for shape in shapes {
        let n = if mode == BenchMode::Gemv { 1 } else { args.n };
        let buffers = alloc_buffers(&stream, shape, n, args.copies)?;

        let total_ms = time_op(
            &stream,
            mode,
            shape,
            n,
            &buffers,
            args.warmup_iters,
            args.iters,
        )?;
        let avg_ms = total_ms / args.iters as f32;
        let weight_gb = (shape.m * shape.k * size_of::<f16>()) as f64 / 1.0e9;
        let io_gb =
            ((shape.m * shape.k + n * shape.k + n * shape.m) * size_of::<f16>()) as f64 / 1.0e9;
        let elapsed_s = total_ms as f64 / 1000.0;
        let tflops = (2.0 * shape.m as f64 * n as f64 * shape.k as f64 * args.iters as f64)
            / elapsed_s
            / 1.0e12;
        let effective_weight_gbps = weight_gb * args.iters as f64 / elapsed_s;
        let effective_io_gbps = io_gb * args.iters as f64 / elapsed_s;
        let calls_per_sec = 1000.0 / avg_ms as f64;

        println!(
            "{},{:?},{},{},{},{},{},{},{:.6},{:.3},{:.9},{:.9},{:.3},{:.3},{:.3},{:.3}",
            args.label,
            mode,
            shape.name,
            shape.m,
            n,
            shape.k,
            args.iters,
            args.copies,
            total_ms,
            avg_ms * 1000.0,
            weight_gb,
            io_gb,
            tflops,
            effective_weight_gbps,
            effective_io_gbps,
            calls_per_sec,
        );
    }

    Ok(())
}

fn selected_mode(mode: &str) -> Result<BenchMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "gemv" => Ok(BenchMode::Gemv),
        "gemm" => Ok(BenchMode::Gemm),
        other => bail!("unknown --mode `{other}`; expected gemv or gemm"),
    }
}

fn selected_shapes(model_size: &str, spec: &str) -> Result<Vec<Shape>> {
    let model_size = model_size.trim().to_ascii_lowercase();
    let shape_set = match model_size.as_str() {
        "4b" | "qwen3_4b" | "qwen3-4b" => SHAPES_4B,
        "32b" | "qwen3_32b" | "qwen3-32b" => SHAPES_32B,
        other => bail!("unknown --model-size `{other}`; expected 4b or 32b"),
    };

    let spec = spec.trim();
    if spec.is_empty() || spec == "all" {
        return Ok(shape_set.to_vec());
    }

    let mut out = Vec::new();
    for raw in spec.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        let shape = match name {
            "qkv" | "q_proj" | "k_proj" | "v_proj" => shape_set[0],
            "o_proj" | "o" => shape_set[1],
            "gate_up" | "gate" | "up" | "gate_proj" | "up_proj" => shape_set[2],
            "down" | "down_proj" => shape_set[3],
            "lm_head" | "lm" | "head" => shape_set[4],
            "q" | "q_only"
                if model_size == "32b"
                    || model_size == "qwen3_32b"
                    || model_size == "qwen3-32b" =>
            {
                SHAPES_32B_PREFILL_SLICES[0]
            }
            "k" | "k_only"
                if model_size == "32b"
                    || model_size == "qwen3_32b"
                    || model_size == "qwen3-32b" =>
            {
                SHAPES_32B_PREFILL_SLICES[1]
            }
            "v" | "v_only"
                if model_size == "32b"
                    || model_size == "qwen3_32b"
                    || model_size == "qwen3-32b" =>
            {
                SHAPES_32B_PREFILL_SLICES[2]
            }
            "gate_only"
                if model_size == "32b"
                    || model_size == "qwen3_32b"
                    || model_size == "qwen3-32b" =>
            {
                SHAPES_32B_PREFILL_SLICES[3]
            }
            "up_only"
                if model_size == "32b"
                    || model_size == "qwen3_32b"
                    || model_size == "qwen3-32b" =>
            {
                SHAPES_32B_PREFILL_SLICES[4]
            }
            _ => bail!("unknown shape `{name}`; expected all,qkv,o_proj,gate_up,down,lm_head"),
        };
        out.push(shape);
    }

    if out.is_empty() {
        bail!("no shapes selected");
    }
    Ok(out)
}

fn alloc_zeros(stream: &Arc<Stream>, shape: &[usize], name: &str) -> Result<Tensor<f16>> {
    api::zeros::<f16>(shape)
        .sync_on(stream)
        .map_err(|e| anyhow!("alloc/init {name} failed: {e:?}"))
}

fn alloc_buffers(
    stream: &Arc<Stream>,
    shape: Shape,
    n: usize,
    copies: usize,
) -> Result<Vec<Buffers>> {
    let mut buffers = Vec::with_capacity(copies);
    for copy in 0..copies {
        buffers.push(Buffers {
            matrix: alloc_zeros(stream, &[shape.m, shape.k], &format!("matrix[{copy}]"))?,
            rhs: alloc_zeros(stream, &[n, shape.k], &format!("rhs[{copy}]"))?,
            out: alloc_zeros(stream, &[n, shape.m], &format!("out[{copy}]"))?,
        });
    }
    Ok(buffers)
}

fn time_op(
    stream: &Arc<Stream>,
    mode: BenchMode,
    shape: Shape,
    n: usize,
    buffers: &[Buffers],
    warmup_iters: usize,
    iters: usize,
) -> Result<f32> {
    for i in 0..warmup_iters {
        let buffers = &buffers[i % buffers.len()];
        launch_op(stream, mode, shape, n, buffers)?;
    }
    unsafe {
        stream
            .synchronize()
            .map_err(|e| anyhow!("warmup stream synchronize failed: {e:?}"))?;
    }

    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    start.record(stream)?;
    for i in 0..iters {
        let buffers = &buffers[i % buffers.len()];
        launch_op(stream, mode, shape, n, buffers)?;
    }
    end.record(stream)?;
    end.synchronize()?;
    end.elapsed_ms_since(&start)
}

fn launch_op(
    stream: &Arc<Stream>,
    mode: BenchMode,
    shape: Shape,
    n: usize,
    buffers: &Buffers,
) -> Result<()> {
    match mode {
        BenchMode::Gemv => unsafe {
            cublas::GemvInPlace {
                matrix: &buffers.matrix,
                vector: &buffers.rhs,
                out: &buffers.out,
                m: shape.m as i32,
                k: shape.k as i32,
            }
            .async_on(stream)
            .map_err(|e| anyhow!("GEMV launch failed for {}: {e:?}", shape.name))
        },
        BenchMode::Gemm => unsafe {
            cublas::GemmInPlace {
                matrix: &buffers.matrix,
                rhs: &buffers.rhs,
                out: &buffers.out,
                m: shape.m as i32,
                n: n as i32,
                k: shape.k as i32,
            }
            .async_on(stream)
            .map_err(|e| anyhow!("GEMM launch failed for {}: {e:?}", shape.name))
        },
    }
}
