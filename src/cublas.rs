use anyhow::{Result, anyhow, ensure};
use cutile::cuda_async::device_operation::{DeviceOp, GraphNode, value, with_context};
use cutile::cuda_async::error::DeviceError;
use cutile::cuda_core::{IntoResult, sys as cu_sys};
use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cutile::core::f16;
use cutile::tensor::Tensor;
use std::cell::RefCell;
use std::ffi::c_void;
use std::mem::{MaybeUninit, size_of};
use std::sync::Arc;

type CublasHandle = usize;
type StreamKey = (usize, usize);

#[derive(Clone, Copy)]
struct DeviceGemmScalars {
    alpha_f32: cu_sys::CUdeviceptr,
    beta_f32: cu_sys::CUdeviceptr,
    alpha_f16: cu_sys::CUdeviceptr,
    beta_f16: cu_sys::CUdeviceptr,
}

const ALPHA_F32_VAL: f32 = 1.0;
const BETA_F32_VAL: f32 = 0.0;
const ALPHA_F16_VAL: f16 = f16::from_bits(0x3c00);
const BETA_F16_VAL: f16 = f16::from_bits(0x0000);

thread_local! {
    static CUBLAS_HANDLE_CACHE: RefCell<Option<(StreamKey, CublasHandle)>> = const { RefCell::new(None) };
    static CUBLAS_FAST_GEMM_OK: RefCell<Option<(StreamKey, bool)>> = const { RefCell::new(None) };
    static CUBLAS_DEVICE_SCALARS: RefCell<Option<(StreamKey, DeviceGemmScalars)>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy)]
struct GemmDispatch {
    compute_type: cublas_sys::cublasComputeType_t,
    algo: cublas_sys::cublasGemmAlgo_t,
}

fn device_compute_capability(device_id: usize) -> Option<(i32, i32)> {
    unsafe {
        let mut dev = MaybeUninit::<cu_sys::CUdevice>::uninit();
        cu_sys::cuDeviceGet(dev.as_mut_ptr(), device_id as i32)
            .result()
            .ok()?;
        let dev = dev.assume_init();
        let mut major = MaybeUninit::<i32>::uninit();
        cu_sys::cuDeviceGetAttribute(
            major.as_mut_ptr(),
            cu_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            dev,
        )
        .result()
        .ok()?;
        let mut minor = MaybeUninit::<i32>::uninit();
        cu_sys::cuDeviceGetAttribute(
            minor.as_mut_ptr(),
            cu_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            dev,
        )
        .result()
        .ok()?;
        Some((major.assume_init(), minor.assume_init()))
    }
}

fn selected_fast_compute_type(
    device_id: usize,
    m: i32,
    n: i32,
    k: i32,
) -> cublas_sys::cublasComputeType_t {
    let default_ty = cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F;
    let compute16_env = std::env::var("GROUT_CUBLAS_COMPUTE16").ok();
    let compute16_enabled = match compute16_env.as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => {
            match device_compute_capability(device_id) {
                // B200/sm_100 retune, 2026-05-01:
                // - 4B decode GEMVs prefer 32F_FAST_16F across the board.
                // - 32B decode GEMVs mostly prefer 32F_FAST_16F, except
                //   gate_up (m=51200,k=5120,n=1), where compute16 wins.
                // Keep the old compute16 default on sm_120/RTX 5090.
                Some((10, 0)) => n == 1 && m == 51200 && k == 5120,
                // Every other arch takes the fp16-accumulate path: sm_120,
                // sm_121 (GB10 / DGX Spark, never retuned), and anything not
                // yet measured. Only sm_100 has been measured to prefer
                // 32F_FAST_16F for most decode GEMVs.
                _ => true,
            }
        }
    };
    if !compute16_enabled {
        return default_ty;
    }
    let max_m_for_f16 = std::env::var("GROUT_CUBLAS_COMPUTE16_MAX_M")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    if (m as usize) <= max_m_for_f16 {
        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F
    } else {
        default_ty
    }
}

fn parse_gemm_algo(raw: &str) -> cublas_sys::cublasGemmAlgo_t {
    match raw.trim().to_ascii_lowercase().as_str() {
        "default" => cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        "default_tensor_op" | "tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP
        }
        "0" | "algo0" | "algo0_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0_TENSOR_OP
        }
        "1" | "algo1" | "algo1_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO1_TENSOR_OP
        }
        "2" | "algo2" | "algo2_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO2_TENSOR_OP
        }
        "3" | "algo3" | "algo3_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO3_TENSOR_OP
        }
        "4" | "algo4" | "algo4_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO4_TENSOR_OP
        }
        "5" | "algo5" | "algo5_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO5_TENSOR_OP
        }
        "6" | "algo6" | "algo6_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO6_TENSOR_OP
        }
        "7" | "algo7" | "algo7_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO7_TENSOR_OP
        }
        "8" | "algo8" | "algo8_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO8_TENSOR_OP
        }
        "9" | "algo9" | "algo9_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO9_TENSOR_OP
        }
        "10" | "algo10" | "algo10_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO10_TENSOR_OP
        }
        "11" | "algo11" | "algo11_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO11_TENSOR_OP
        }
        "12" | "algo12" | "algo12_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO12_TENSOR_OP
        }
        "13" | "algo13" | "algo13_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO13_TENSOR_OP
        }
        "14" | "algo14" | "algo14_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO14_TENSOR_OP
        }
        "15" | "algo15" | "algo15_tensor_op" => {
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO15_TENSOR_OP
        }
        _ => cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    }
}

fn selected_fast_algo() -> cublas_sys::cublasGemmAlgo_t {
    let Some(raw) = std::env::var("GROUT_CUBLAS_FAST_ALGO").ok() else {
        return cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    };
    parse_gemm_algo(&raw)
}

fn selected_gemm_dispatch(device_id: usize, m: i32, n: i32, k: i32) -> GemmDispatch {
    GemmDispatch {
        compute_type: selected_fast_compute_type(device_id, m, n, k),
        algo: selected_fast_algo(),
    }
}

unsafe fn get_or_create_handle(
    device_id: usize,
    stream: cublas_sys::cudaStream_t,
) -> Result<cublas_sys::cublasHandle_t> {
    let key: StreamKey = (device_id, stream as usize);
    CUBLAS_HANDLE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((cached_key, handle)) = *cache
            && cached_key == key
        {
            return Ok(handle as cublas_sys::cublasHandle_t);
        }

        let handle =
            cublas_result::create_handle().map_err(|e| anyhow!("cublasCreate_v2 failed: {e:?}"))?;
        unsafe {
            cublas_result::set_stream(handle, stream)
                .map_err(|e| anyhow!("cublasSetStream_v2 failed: {e:?}"))?;
            cublas_sys::cublasSetPointerMode_v2(
                handle,
                cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
            )
            .result()
            .map_err(|e| anyhow!("cublasSetPointerMode_v2 failed: {e:?}"))?;
        }
        *cache = Some((key, handle as usize));
        Ok(handle)
    })
}

unsafe fn alloc_device_scalar<T: Copy>(
    stream: cublas_sys::cudaStream_t,
    value: T,
) -> Result<cu_sys::CUdeviceptr> {
    let stream = stream as cu_sys::CUstream;
    let mut dptr = MaybeUninit::<cu_sys::CUdeviceptr>::uninit();
    unsafe {
        cu_sys::cuMemAllocAsync(dptr.as_mut_ptr(), size_of::<T>(), stream)
            .result()
            .map_err(|e| anyhow!("cuMemAllocAsync for scalar failed: {e:?}"))?;
        let dptr = dptr.assume_init();
        cu_sys::cuMemcpyHtoDAsync_v2(
            dptr,
            (&value as *const T).cast::<c_void>(),
            size_of::<T>(),
            stream,
        )
        .result()
        .map_err(|e| anyhow!("cuMemcpyHtoDAsync_v2 for scalar failed: {e:?}"))?;
        Ok(dptr)
    }
}

unsafe fn get_or_create_device_scalars(
    key: StreamKey,
    stream: cublas_sys::cudaStream_t,
) -> Result<DeviceGemmScalars> {
    CUBLAS_DEVICE_SCALARS.with(|scalars| {
        let mut scalars = scalars.borrow_mut();
        if let Some((cached_key, existing)) = *scalars
            && cached_key == key
        {
            return Ok(existing);
        }
        let created = unsafe {
            DeviceGemmScalars {
                alpha_f32: alloc_device_scalar(stream, ALPHA_F32_VAL)?,
                beta_f32: alloc_device_scalar(stream, BETA_F32_VAL)?,
                alpha_f16: alloc_device_scalar(stream, ALPHA_F16_VAL)?,
                beta_f16: alloc_device_scalar(stream, BETA_F16_VAL)?,
            }
        };
        *scalars = Some((key, created));
        Ok(created)
    })
}

/// Core cuBLAS GEMM launch using raw device pointers.
///
/// Computes C = op(A) * op(B) where A is transposed (row-major weight)
/// and B is not transposed.  `a_ptr` / `b_ptr` / `c_ptr` are device
/// pointers that may include byte offsets from a base tensor.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_f16_ptrs(
    device_id: usize,
    stream: cublas_sys::cudaStream_t,
    a_ptr: *const c_void,
    b_ptr: *const c_void,
    c_ptr: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
) -> Result<()> {
    let handle = unsafe { get_or_create_handle(device_id, stream)? };
    let key: StreamKey = (device_id, stream as usize);
    let scalars = unsafe { get_or_create_device_scalars(key, stream)? };
    let run = |compute_type: cublas_sys::cublasComputeType_t,
               algo: cublas_sys::cublasGemmAlgo_t| unsafe {
        let (alpha_ptr, beta_ptr): (*const c_void, *const c_void) =
            if compute_type == cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F {
                (
                    (scalars.alpha_f16 as usize as *const c_void),
                    (scalars.beta_f16 as usize as *const c_void),
                )
            } else {
                (
                    (scalars.alpha_f32 as usize as *const c_void),
                    (scalars.beta_f32 as usize as *const c_void),
                )
            };
        cublas_result::gemm_ex(
            handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            alpha_ptr,
            a_ptr,
            cublas_sys::cudaDataType_t::CUDA_R_16F,
            k,
            b_ptr,
            cublas_sys::cudaDataType_t::CUDA_R_16F,
            k,
            beta_ptr,
            c_ptr,
            cublas_sys::cudaDataType_t::CUDA_R_16F,
            m,
            compute_type,
            algo,
        )
    };

    let fast_known_ok = CUBLAS_FAST_GEMM_OK.with(|m| {
        if let Some((cached_key, ok)) = *m.borrow()
            && cached_key == key
        {
            return Some(ok);
        }
        None
    });
    if fast_known_ok == Some(false) {
        return Err(anyhow!(
            "cublasGemmEx fast path previously failed (m={m}, n={n}, k={k}); refusing to silently fall back"
        ));
    }

    let dispatch = selected_gemm_dispatch(device_id, m, n, k);
    let fast_res = run(dispatch.compute_type, dispatch.algo);
    match fast_res {
        Ok(()) => {
            CUBLAS_FAST_GEMM_OK.with(|m| {
                *m.borrow_mut() = Some((key, true));
            });
            Ok(())
        }
        Err(e_fast) => {
            CUBLAS_FAST_GEMM_OK.with(|m| {
                *m.borrow_mut() = Some((key, false));
            });
            Err(anyhow!(
                "cublasGemmEx fast path failed (m={m}, n={n}, k={k}, compute={:?}, err={e_fast:?})",
                dispatch.compute_type
            ))
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_f16(
    device_id: usize,
    stream: cublas_sys::cudaStream_t,
    matrix: &Tensor<f16>,
    rhs: &Tensor<f16>,
    out: &Tensor<f16>,
    m: i32,
    n: i32,
    k: i32,
) -> Result<()> {
    let a_ptr = matrix.device_pointer().cu_deviceptr() as usize as *const c_void;
    let b_ptr = rhs.device_pointer().cu_deviceptr() as usize as *const c_void;
    let c_ptr = out.device_pointer().cu_deviceptr() as usize as *mut c_void;

    let handle = unsafe { get_or_create_handle(device_id, stream)? };
    let key: StreamKey = (device_id, stream as usize);
    let scalars = unsafe { get_or_create_device_scalars(key, stream)? };
    let run = |compute_type: cublas_sys::cublasComputeType_t,
               algo: cublas_sys::cublasGemmAlgo_t| unsafe {
        let (alpha_ptr, beta_ptr): (*const c_void, *const c_void) =
            if compute_type == cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F {
                (
                    (scalars.alpha_f16 as usize as *const c_void),
                    (scalars.beta_f16 as usize as *const c_void),
                )
            } else {
                (
                    (scalars.alpha_f32 as usize as *const c_void),
                    (scalars.beta_f32 as usize as *const c_void),
                )
            };
        cublas_result::gemm_ex(
            handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            alpha_ptr,
            a_ptr,
            cublas_sys::cudaDataType_t::CUDA_R_16F,
            k,
            b_ptr,
            cublas_sys::cudaDataType_t::CUDA_R_16F,
            k,
            beta_ptr,
            c_ptr,
            cublas_sys::cudaDataType_t::CUDA_R_16F,
            m,
            compute_type,
            algo,
        )
    };

    let fast_known_ok = CUBLAS_FAST_GEMM_OK.with(|m| {
        if let Some((cached_key, ok)) = *m.borrow()
            && cached_key == key
        {
            return Some(ok);
        }
        None
    });
    if fast_known_ok == Some(false) {
        return Err(anyhow!(
            "cublasGemmEx fast path previously failed (m={m}, n={n}, k={k}); refusing to silently fall back"
        ));
    }

    let dispatch = selected_gemm_dispatch(device_id, m, n, k);
    let fast_res = run(dispatch.compute_type, dispatch.algo);
    match fast_res {
        Ok(()) => {
            CUBLAS_FAST_GEMM_OK.with(|m| {
                *m.borrow_mut() = Some((key, true));
            });
            Ok(())
        }
        Err(e_fast) => {
            CUBLAS_FAST_GEMM_OK.with(|m| {
                *m.borrow_mut() = Some((key, false));
            });
            Err(anyhow!(
                "cublasGemmEx fast path failed (m={m}, n={n}, k={k}, compute={:?}, err={e_fast:?})",
                dispatch.compute_type
            ))
        }
    }
}

pub fn gemv_f16_op(
    matrix: Arc<Tensor<f16>>,
    vector: Arc<Tensor<f16>>,
    out: Tensor<f16>,
    m: usize,
    k: usize,
) -> Result<impl DeviceOp<Output = Result<Tensor<f16>>>> {
    ensure!(
        m > 0 && k > 0,
        "gemv requires positive dims, got m={m}, k={k}"
    );
    ensure!(m <= i32::MAX as usize, "gemv m too large for cuBLAS: {m}");
    ensure!(k <= i32::MAX as usize, "gemv k too large for cuBLAS: {k}");

    Ok(with_context(move |ctx| {
        let launch_status = (|| {
            ctx.device()
                .bind_to_thread()
                .map_err(|e| anyhow!("failed to bind CUDA context: {e:?}"))?;
            let stream = ctx.get_cuda_stream().cu_stream() as cublas_sys::cudaStream_t;
            unsafe {
                launch_gemm_f16(
                    ctx.get_device_id(),
                    stream,
                    &matrix,
                    &vector,
                    &out,
                    m as i32,
                    1,
                    k as i32,
                )?;
            }
            Ok(())
        })();
        value((matrix, vector, out, launch_status))
    })
    .then(|(_matrix, _vector, out, launch_status)| value(launch_status.map(|()| out))))
}

pub fn gemm_f16_op(
    matrix: Arc<Tensor<f16>>,
    rhs: Arc<Tensor<f16>>,
    out: Tensor<f16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<impl DeviceOp<Output = Result<Tensor<f16>>>> {
    ensure!(
        m > 0 && n > 0 && k > 0,
        "gemm requires positive dims, got m={m}, n={n}, k={k}"
    );
    ensure!(m <= i32::MAX as usize, "gemm m too large for cuBLAS: {m}");
    ensure!(n <= i32::MAX as usize, "gemm n too large for cuBLAS: {n}");
    ensure!(k <= i32::MAX as usize, "gemm k too large for cuBLAS: {k}");

    Ok(with_context(move |ctx| {
        let launch_status = (|| {
            ctx.device()
                .bind_to_thread()
                .map_err(|e| anyhow!("failed to bind CUDA context: {e:?}"))?;
            let stream = ctx.get_cuda_stream().cu_stream() as cublas_sys::cudaStream_t;
            unsafe {
                launch_gemm_f16(
                    ctx.get_device_id(),
                    stream,
                    &matrix,
                    &rhs,
                    &out,
                    m as i32,
                    n as i32,
                    k as i32,
                )?;
            }
            Ok(())
        })();
        value((matrix, rhs, out, launch_status))
    })
    .then(|(_matrix, _rhs, out, launch_status)| value(launch_status.map(|()| out))))
}

/// Row-sliced GEMM: uses rows [row_offset..row_offset+m) of the weight matrix.
///
/// The weight matrix is [M_total, K] row-major.  This computes
/// out = weight[row_offset..row_offset+m, :] × rhs^T  without copying the
/// weight sub-matrix — it just offsets the device pointer passed to cuBLAS.
pub fn gemm_f16_row_slice_op(
    matrix: Arc<Tensor<f16>>,
    row_offset: usize,
    rhs: Arc<Tensor<f16>>,
    out: Tensor<f16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<impl DeviceOp<Output = Result<Tensor<f16>>>> {
    ensure!(
        m > 0 && n > 0 && k > 0,
        "gemm requires positive dims, got m={m}, n={n}, k={k}"
    );
    ensure!(m <= i32::MAX as usize, "gemm m too large for cuBLAS: {m}");
    ensure!(n <= i32::MAX as usize, "gemm n too large for cuBLAS: {n}");
    ensure!(k <= i32::MAX as usize, "gemm k too large for cuBLAS: {k}");

    let row_byte_offset = (row_offset * k * size_of::<f16>()) as u64;

    Ok(with_context(move |ctx| {
        let launch_status = (|| {
            ctx.device()
                .bind_to_thread()
                .map_err(|e| anyhow!("failed to bind CUDA context: {e:?}"))?;
            let stream = ctx.get_cuda_stream().cu_stream() as cublas_sys::cudaStream_t;
            let a_ptr = (matrix.device_pointer().cu_deviceptr() + row_byte_offset) as usize
                as *const c_void;
            unsafe {
                launch_gemm_f16_ptrs(
                    ctx.get_device_id(),
                    stream,
                    a_ptr,
                    rhs.device_pointer().cu_deviceptr() as usize as *const c_void,
                    out.device_pointer().cu_deviceptr() as usize as *mut c_void,
                    m as i32,
                    n as i32,
                    k as i32,
                )?;
            }
            Ok(())
        })();
        value((matrix, rhs, out, launch_status))
    })
    .then(|(_matrix, _rhs, out, launch_status)| value(launch_status.map(|()| out))))
}

// ── Graph-compatible cuBLAS ops (for CudaGraph::scope) ─────────────────────
//
// These write into a pre-allocated output tensor and implement GraphNode,
// so they can be used with `s.record()` inside a CudaGraph::scope closure.

/// cuBLAS GEMV that writes into a pre-allocated output. Implements GraphNode.
pub struct GemvInPlace<'a> {
    pub matrix: &'a Tensor<f16>,
    pub vector: &'a Tensor<f16>,
    pub out: &'a Tensor<f16>,
    pub m: i32,
    pub k: i32,
}

impl GraphNode for GemvInPlace<'_> {}

impl DeviceOp for GemvInPlace<'_> {
    type Output = ();
    unsafe fn execute(
        self,
        ctx: &cutile::cuda_async::device_operation::ExecutionContext,
    ) -> Result<(), DeviceError> {
        unsafe {
            ctx.device()
                .bind_to_thread()
                .map_err(|e| DeviceError::Internal(format!("bind failed: {e:?}")))?;
            let stream = ctx.get_cuda_stream().cu_stream() as cublas_sys::cudaStream_t;
            launch_gemm_f16(
                ctx.get_device_id(),
                stream,
                self.matrix,
                self.vector,
                self.out,
                self.m,
                1,
                self.k,
            )
            .map_err(|e| DeviceError::Internal(format!("cuBLAS gemv failed: {e:?}")))?;
            Ok(())
        }
    }
}

impl std::future::IntoFuture for GemvInPlace<'_> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = cutile::cuda_async::device_future::DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        cutile::cuda_async::device_future::DeviceFuture::failed(DeviceError::Internal(
            "GemvInPlace is only for graph capture, not standalone execution".into(),
        ))
    }
}

/// cuBLAS GEMM that writes into a pre-allocated output. Implements GraphNode.
#[allow(dead_code)]
pub struct GemmInPlace<'a> {
    pub matrix: &'a Tensor<f16>,
    pub rhs: &'a Tensor<f16>,
    pub out: &'a Tensor<f16>,
    pub m: i32,
    pub n: i32,
    pub k: i32,
}

impl GraphNode for GemmInPlace<'_> {}

impl DeviceOp for GemmInPlace<'_> {
    type Output = ();
    unsafe fn execute(
        self,
        ctx: &cutile::cuda_async::device_operation::ExecutionContext,
    ) -> Result<(), DeviceError> {
        unsafe {
            ctx.device()
                .bind_to_thread()
                .map_err(|e| DeviceError::Internal(format!("bind failed: {e:?}")))?;
            let stream = ctx.get_cuda_stream().cu_stream() as cublas_sys::cudaStream_t;
            launch_gemm_f16(
                ctx.get_device_id(),
                stream,
                self.matrix,
                self.rhs,
                self.out,
                self.m,
                self.n,
                self.k,
            )
            .map_err(|e| DeviceError::Internal(format!("cuBLAS gemm failed: {e:?}")))?;
            Ok(())
        }
    }
}

impl std::future::IntoFuture for GemmInPlace<'_> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = cutile::cuda_async::device_future::DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        cutile::cuda_async::device_future::DeviceFuture::failed(DeviceError::Internal(
            "GemmInPlace is only for graph capture, not standalone execution".into(),
        ))
    }
}
