use std::mem::MaybeUninit;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail, ensure};
use clap::Parser;
use cutile::cuda_async::device_operation::{DeviceOp, value, with_context};
use cutile::cuda_core::{IntoResult, Stream, memcpy_dtoh_async, sys as cu_sys};
use cutile::tensor::Tensor;
use cutile::tile_kernel::TileKernel;
use cutile::{api, core::f16};

use grout::kernels::gemm_persistent_f16;

#[path = "../../../src/cublas.rs"]
#[allow(dead_code)]
mod cublas;

#[derive(Parser, Debug)]
struct Args {
    /// Qwen3-32B projection: qkv, o, gate_up, or down.
    #[arg(long)]
    shape: String,

    /// Logical row count. M=1 is padded to the selected BM for cuTile only.
    #[arg(long)]
    m: usize,

    #[arg(long, default_value = "16,32,64,128")]
    bm: String,

    #[arg(long, default_value = "64,128,256,512")]
    bn: String,

    #[arg(long, default_value = "32,64,128")]
    bk: String,

    /// Number of paired samples; order alternates for every sample.
    #[arg(long, default_value_t = 5)]
    samples: usize,

    /// Launches enclosed by each CUDA-event sample.
    #[arg(long, default_value_t = 5)]
    iters: usize,

    #[arg(long, default_value_t = 3)]
    warmup_iters: usize,
}

#[derive(Clone, Copy)]
struct Projection {
    name: &'static str,
    n: usize,
    k: usize,
}

const PROJECTIONS: &[Projection] = &[
    Projection {
        name: "qkv",
        n: 10240,
        k: 5120,
    },
    Projection {
        name: "o",
        n: 5120,
        k: 8192,
    },
    Projection {
        name: "gate_up",
        n: 51200,
        k: 5120,
    },
    Projection {
        name: "down",
        n: 5120,
        k: 25600,
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
        unsafe { cu_sys::cuEventRecord(self.event, stream.cu_stream()) }
            .result()
            .map_err(|e| anyhow!("cuEventRecord failed: {e:?}"))
    }

    fn synchronize(&self) -> Result<()> {
        unsafe { cu_sys::cuEventSynchronize(self.event) }
            .result()
            .map_err(|e| anyhow!("cuEventSynchronize failed: {e:?}"))
    }

    fn elapsed_us_since(&self, start: &CudaEvent, iters: usize) -> Result<f64> {
        let mut ms = 0.0f32;
        unsafe { cu_sys::cuEventElapsedTime_v2(&mut ms, start.event, self.event) }
            .result()
            .map_err(|e| anyhow!("cuEventElapsedTime failed: {e:?}"))?;
        Ok(ms as f64 * 1000.0 / iters as f64)
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
    ensure!(args.m > 0, "--m must be positive");
    ensure!(args.samples > 0, "--samples must be positive");
    ensure!(args.iters > 0, "--iters must be positive");
    let projection = projection(&args.shape)?;
    let bms = parse_list(&args.bm, "BM")?;
    let bns = parse_list(&args.bn, "BN")?;
    let bks = parse_list(&args.bk, "BK")?;

    let stream = with_context(|ctx| value(ctx.get_cuda_stream().clone()))
        .await
        .map_err(|e| anyhow!("failed to get CUDA stream: {e:?}"))?;
    stream
        .device()
        .bind_to_thread()
        .map_err(|e| anyhow!("failed to bind CUDA context: {e:?}"))?;
    let sms = num_sms(&stream)?;

    let cublas_matrix = api::ones::<f16>(&[projection.n, projection.k]).sync_on(&stream)?;
    let cublas_rhs = api::ones::<f16>(&[args.m, projection.k]).sync_on(&stream)?;
    let cublas_out = api::zeros::<f16>(&[args.m, projection.n]).sync_on(&stream)?;
    let tile_y = api::ones::<f16>(&[projection.k, projection.n]).sync_on(&stream)?;

    println!("shape,m,n,k,bm,bn,bk,grid_x,cublas_us,persistent_us,ratio,status");
    for bm in bms {
        let tile_m = args.m.div_ceil(bm) * bm;
        let tile_x = api::ones::<f16>(&[tile_m, projection.k]).sync_on(&stream)?;
        let mut tile_out = api::zeros::<f16>(&[tile_m, projection.n]).sync_on(&stream)?;

        for &bn in &bns {
            for &bk in &bks {
                if projection.n % bn != 0 || projection.k % bk != 0 {
                    println!(
                        "{},{},{},{},{},{},{},0,0,0,0,not_divisible",
                        projection.name, args.m, projection.n, projection.k, bm, bn, bk
                    );
                    continue;
                }
                let total_tiles = (tile_m / bm) * (projection.n / bn);
                let grid_x = persistent_grid(total_tiles, sms);
                let generics = vec![
                    bm.to_string(),
                    bn.to_string(),
                    bk.to_string(),
                    "8".to_string(),
                    "1".to_string(),
                ];

                let candidate = run_candidate(
                    &stream,
                    projection,
                    args.m,
                    bm,
                    bn,
                    bk,
                    grid_x,
                    &generics,
                    &cublas_matrix,
                    &cublas_rhs,
                    &cublas_out,
                    &tile_x,
                    &tile_y,
                    &mut tile_out,
                    args.warmup_iters,
                    args.iters,
                    args.samples,
                );
                match candidate {
                    Ok((cublas_us, tile_us)) => println!(
                        "{},{},{},{},{},{},{},{},{:.3},{:.3},{:.4},ok",
                        projection.name,
                        args.m,
                        projection.n,
                        projection.k,
                        bm,
                        bn,
                        bk,
                        grid_x,
                        cublas_us,
                        tile_us,
                        tile_us / cublas_us,
                    ),
                    Err(error) => println!(
                        "{},{},{},{},{},{},{},{},0,0,0,error:{:?}",
                        projection.name,
                        args.m,
                        projection.n,
                        projection.k,
                        bm,
                        bn,
                        bk,
                        grid_x,
                        error,
                    ),
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_candidate(
    stream: &Arc<Stream>,
    projection: Projection,
    logical_m: usize,
    bm: usize,
    bn: usize,
    bk: usize,
    grid_x: u32,
    generics: &[String],
    cublas_matrix: &Tensor<f16>,
    cublas_rhs: &Tensor<f16>,
    cublas_out: &Tensor<f16>,
    tile_x: &Tensor<f16>,
    tile_y: &Tensor<f16>,
    tile_out: &mut Tensor<f16>,
    warmup_iters: usize,
    iters: usize,
    samples: usize,
) -> Result<(f64, f64)> {
    for _ in 0..warmup_iters {
        launch_cublas(
            stream,
            projection,
            logical_m,
            cublas_matrix,
            cublas_rhs,
            cublas_out,
        )?;
        launch_persistent(
            stream, bm, bn, bk, grid_x, generics, tile_x, tile_y, tile_out,
        )?;
    }
    unsafe { stream.synchronize() }.map_err(|e| anyhow!("warmup synchronization failed: {e:?}"))?;

    verify_candidate(
        stream,
        projection,
        logical_m,
        cublas_matrix,
        cublas_rhs,
        cublas_out,
        bm,
        bn,
        bk,
        grid_x,
        generics,
        tile_x,
        tile_y,
        tile_out,
    )?;

    let mut cublas_samples = Vec::with_capacity(samples);
    let mut tile_samples = Vec::with_capacity(samples);
    for sample in 0..samples {
        if sample % 2 == 0 {
            cublas_samples.push(time_cublas(
                stream,
                projection,
                logical_m,
                cublas_matrix,
                cublas_rhs,
                cublas_out,
                iters,
            )?);
            tile_samples.push(time_persistent(
                stream, bm, bn, bk, grid_x, generics, tile_x, tile_y, tile_out, iters,
            )?);
        } else {
            tile_samples.push(time_persistent(
                stream, bm, bn, bk, grid_x, generics, tile_x, tile_y, tile_out, iters,
            )?);
            cublas_samples.push(time_cublas(
                stream,
                projection,
                logical_m,
                cublas_matrix,
                cublas_rhs,
                cublas_out,
                iters,
            )?);
        }
    }
    Ok((median(&mut cublas_samples), median(&mut tile_samples)))
}

fn launch_cublas(
    stream: &Arc<Stream>,
    projection: Projection,
    m: usize,
    matrix: &Tensor<f16>,
    rhs: &Tensor<f16>,
    out: &Tensor<f16>,
) -> Result<()> {
    unsafe {
        cublas::GemmInPlace {
            matrix,
            rhs,
            out,
            m: projection.n as i32,
            n: m as i32,
            k: projection.k as i32,
        }
        .async_on(stream)
        .map_err(|e| anyhow!("cuBLAS launch failed: {e:?}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_persistent(
    stream: &Arc<Stream>,
    bm: usize,
    bn: usize,
    _bk: usize,
    grid_x: u32,
    generics: &[String],
    x: &Tensor<f16>,
    y: &Tensor<f16>,
    z: &mut Tensor<f16>,
) -> Result<()> {
    let mapped_z = cutile::tensor::PartitionMut::partition(z, [bm, bn]).map([8, 1], grid_x);
    unsafe {
        gemm_persistent_f16(mapped_z, x, y)
            .generics(generics.to_vec())
            .async_on(stream)
            .map_err(|e| anyhow!("persistent GEMM launch failed: {e:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_candidate(
    stream: &Arc<Stream>,
    projection: Projection,
    logical_m: usize,
    cublas_matrix: &Tensor<f16>,
    cublas_rhs: &Tensor<f16>,
    cublas_out: &Tensor<f16>,
    bm: usize,
    bn: usize,
    bk: usize,
    grid_x: u32,
    generics: &[String],
    tile_x: &Tensor<f16>,
    tile_y: &Tensor<f16>,
    tile_out: &mut Tensor<f16>,
) -> Result<()> {
    launch_cublas(
        stream,
        projection,
        logical_m,
        cublas_matrix,
        cublas_rhs,
        cublas_out,
    )?;
    launch_persistent(
        stream, bm, bn, bk, grid_x, generics, tile_x, tile_y, tile_out,
    )?;
    unsafe { stream.synchronize() }
        .map_err(|e| anyhow!("correctness synchronization failed: {e:?}"))?;

    let indices = sample_indices(logical_m, projection.n);
    let cublas_values = read_elements(stream, cublas_out, &indices)?;
    let tile_values = read_elements(stream, tile_out, &indices)?;
    for (index, (reference, actual)) in cublas_values.iter().zip(&tile_values).enumerate() {
        let reference = reference.to_f32();
        let actual = actual.to_f32();
        let relative = (actual - reference).abs() / reference.abs().max(1.0);
        ensure!(
            relative <= 1.0e-2,
            "correctness failed at sample {}: persistent={} cublas={} rel={}",
            index,
            actual,
            reference,
            relative,
        );
    }
    Ok(())
}

fn time_cublas(
    stream: &Arc<Stream>,
    projection: Projection,
    m: usize,
    matrix: &Tensor<f16>,
    rhs: &Tensor<f16>,
    out: &Tensor<f16>,
    iters: usize,
) -> Result<f64> {
    time_launches(stream, iters, || {
        launch_cublas(stream, projection, m, matrix, rhs, out)
    })
}

#[allow(clippy::too_many_arguments)]
fn time_persistent(
    stream: &Arc<Stream>,
    bm: usize,
    bn: usize,
    bk: usize,
    grid_x: u32,
    generics: &[String],
    x: &Tensor<f16>,
    y: &Tensor<f16>,
    z: &mut Tensor<f16>,
    iters: usize,
) -> Result<f64> {
    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    start.record(stream)?;
    for _ in 0..iters {
        launch_persistent(stream, bm, bn, bk, grid_x, generics, x, y, z)?;
    }
    end.record(stream)?;
    end.synchronize()?;
    end.elapsed_us_since(&start, iters)
}

fn time_launches<F>(stream: &Arc<Stream>, iters: usize, mut launch: F) -> Result<f64>
where
    F: FnMut() -> Result<()>,
{
    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    start.record(stream)?;
    for _ in 0..iters {
        launch()?;
    }
    end.record(stream)?;
    end.synchronize()?;
    end.elapsed_us_since(&start, iters)
}

fn read_elements(
    stream: &Arc<Stream>,
    tensor: &Tensor<f16>,
    indices: &[usize],
) -> Result<Vec<f16>> {
    let base = tensor.device_pointer().cu_deviceptr();
    let mut host = vec![f16::ZERO; indices.len()];
    for (slot, index) in host.iter_mut().zip(indices) {
        unsafe {
            memcpy_dtoh_async(
                slot as *mut f16,
                base + (*index * std::mem::size_of::<f16>()) as u64,
                1,
                stream,
            )
        }
        .map_err(|e| anyhow!("sample D2H failed: {e:?}"))?;
    }
    unsafe { stream.synchronize() }
        .map_err(|e| anyhow!("sample D2H synchronization failed: {e:?}"))?;
    Ok(host)
}

fn sample_indices(m: usize, n: usize) -> Vec<usize> {
    let mut indices = vec![0, n - 1, (m - 1) * n, m * n - 1];
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn projection(name: &str) -> Result<Projection> {
    let normalized = name.trim().to_ascii_lowercase();
    PROJECTIONS
        .iter()
        .copied()
        .find(|projection| projection.name == normalized)
        .ok_or_else(|| anyhow!("unknown --shape `{name}`; expected qkv,o,gate_up,down"))
}

fn parse_list(raw: &str, label: &str) -> Result<Vec<usize>> {
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| anyhow!("invalid {label} value `{value}`: {error}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        bail!("{label} list must not be empty");
    }
    if values.contains(&0) {
        bail!("{label} values must be positive");
    }
    Ok(values)
}

fn persistent_grid(total_tiles: usize, sms: usize) -> u32 {
    let max_programs = (sms / 2).max(1);
    total_tiles.min(max_programs).max(1) as u32
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn num_sms(stream: &Arc<Stream>) -> Result<usize> {
    let mut sms = 0i32;
    unsafe {
        cu_sys::cuDeviceGetAttribute(
            &mut sms,
            cu_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            stream.device().cu_device(),
        )
    }
    .result()
    .map_err(|e| anyhow!("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed: {e:?}"))?;
    Ok(sms as usize)
}
