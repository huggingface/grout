use crate::config::{GenerationConfig, Qwen3Config};
use crate::cublas;
use crate::flash_decode::attention_decode_kernel_grouped;
use crate::kernels::{
    KernelKind, TILE_KERNEL_KINDS, add_2d_f16, add_rms_norm_decode_raw_f16, add_rms_norm_f16,
    argmax_blocks_f16, argmax_reduce_blocks_to_u32, embedding_batch_f16,
    flash_attn_causal_seq_dynpos_f16, flash_attn_causal_seq_f16, fmha_causal,
    fmha_decode_gqa_split, fmha_prefill_causal, fmha_prefill_gqa, fmha_prefill_gqa_lpt,
    gather_row_f16, kv_cache_update_seq_dynpos_f16, kv_cache_update_seq_f16,
    lm_head_argmax_blocks_f16, qk_norm_f16, qk_norm_rope_kv_decode_raw_f16,
    qk_norm_rope_kv_prefill_raw_f16, qk_rope_dynpos_f16, rms_norm_f16, rope_seq_dynpos_f16,
    rope_seq_f16, silu_mul_2d_f16, splitk_reduce_merge,
};
use crate::loader::WeightLoader;
use anyhow::{Context, Result, bail, ensure};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext, GraphNode, value, with_context};
use cuda_async::error::DeviceError;
use cuda_core::{
    IntoResult, memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async, sys as cu_sys,
};
use cutile::api;
use cutile::core::f16;
use cutile::tensor::{
    IntoPartition, IntoPartitionArc, Partition, PartitionMut, Reshape, Tensor, ToHostVec,
};
use cutile::tile_kernel::{CompileOptions, TileKernel};
use rand::Rng;
use std::cmp::{Reverse, min};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::mem::{MaybeUninit, size_of};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// Wraps a closure as a graph-capturable operation for CudaGraph::scope.
///
/// Use with `s.record(KernelGraphOp(|ctx| { ... }))` to record
/// raw-pointer kernel ops that don't go through the &Tensor partition API.
struct KernelGraphOp<F>(F);

impl<F: FnOnce(&ExecutionContext) -> Result<(), DeviceError> + Send> GraphNode
    for KernelGraphOp<F>
{
}

impl<F: FnOnce(&ExecutionContext) -> Result<(), DeviceError> + Send> DeviceOp for KernelGraphOp<F> {
    type Output = ();
    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        (self.0)(ctx)
    }
}

impl<F: FnOnce(&ExecutionContext) -> Result<(), DeviceError> + Send> std::future::IntoFuture
    for KernelGraphOp<F>
{
    type Output = Result<(), DeviceError>;
    type IntoFuture = cuda_async::device_future::DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        cuda_async::device_future::DeviceFuture::failed(DeviceError::Internal(
            "KernelGraphOp is only for graph capture, not standalone execution".into(),
        ))
    }
}

// VEC_BLOCK: tiles head_dim (= 128) in kv_cache_update / gather_row.
// Structurally capped at head_dim — kernel fails if BLOCK_SIZE > D. Not
// independently tunable.
const VEC_BLOCK: usize = 128;
// BM_S: seq_len chunking for kv_cache_update_seq_f16. Grid becomes
// (num_kv_heads, ceil(seq_len/BM_S), 1). Pre-refactor this kernel ran
// on (num_kv_heads, 1, 1) = 8 CTAs with a seq_len-iteration inner loop,
// taking 31 ms at pp=2048. BM_S=16 gives 1024 CTAs at pp=2048, tuned
// to amortize launch overhead while saturating SMs. Override with
// GROUT_KV_CACHE_BM_S.
const KV_CACHE_BM_S_DEFAULT: usize = 16;
// EMBED_BLOCK: tiles hidden_size (= 2560) in embedding_batch_f16. Picked
// from the 2026-04-20 sweep: 1024 wins (138.8 t/s decode) over 128
// (126.1 t/s) by ~10%. 512 is effectively tied with 1024; 2048
// regresses slightly. Ceiling div: 2560/1024 = 3 CTAs per lookup with
// partial overhang on the last tile (tile IR masks).
// Tunable via GROUT_EMBED_BLOCK.
const EMBED_BLOCK: usize = 1024;
const POINTWISE_BLOCK: usize = 1024;
// BLOCK_SIZE for the plain `rms_norm_f16` and `qk_norm_f16` kernels
// when invoked at small N (head_dim = 128 for Q/K norm). BLOCK_SIZE
// must be ≤ N or cutile's bounded-assume check fails at JIT time.
// 128 divides both 128 (head_dim) and 2560 (hidden_size) cleanly.
const RMS_BLOCK: usize = 128;
// BLOCK_SIZE for plain `rms_norm_f16` at N=hidden_size=2560. Same
// "512 is the tuned sweet spot" argument as add_rms_norm_f16 — closes
// BS=128's perf gap to the cutile rmsnorm reference benchmark. 512
// divides 2560 exactly (num_tiles=5, no overhang).
const RMS_BLOCK_HIDDEN: usize = 512;
// Default BLOCK_SIZE for the generic `add_rms_norm_f16` path. Picked
// empirically via the BLOCK_SIZE × max_divisibility sweep on 2026-04-20
// (Qwen3-4B, N=hidden_size=2560, max_divisibility=8). Winner:
//   BS=2048: kernel median 2016 ns, decode 154.9 t/s (best end-to-end)
//   vs BS=128 (old default): kernel median 2272 ns, decode 151.1 t/s
// At BS=2048, num_tiles=ceil(2560/2048)=2 with a partial last tile
// (512 valid / 1536 overhang). Tile IR masks the overhang on
// load/store; the OOB sum-of-squares contribution is zero.
const ADD_RMS_BLOCK: usize = 2048;
// Decode CUDA graphs use `add_rms_norm_decode_raw_f16`, a contiguous raw
// pointer variant. The 2026-04-29 sm_120 retry found BS=4096 best for that
// kernel (median 1376 ns vs 3232 ns for the old generic decode path).
// Override with GROUT_RMS_BLOCK for further ablation.
const ADD_RMS_DECODE_BLOCK: usize = 4096;
const ROPE_BLOCK: usize = 128;
// ARGMAX_BLOCK: tiles vocab (= 151936). Never swept. Current default 128
// produces 1188 CTAs (ceil 151936/128); larger tiles → fewer CTAs but
// more work each. Tunable via GROUT_ARGMAX_BLOCK.
const ARGMAX_BLOCK: usize = 128;
// ── Attention tile constants — DECODE path ─────────────────────────
// Decode has q_len=1 by construction, so BM is structurally pinned to
// 1 (one query row per CTA). No env override — changing this would
// require kernel rewrites.
const ATTN_BM_DECODE: usize = 1;
// KV-seq tile size for the decode attention kernel. Default 32 as a
// reasonable across-workload default for commercial hardware: the
// 2026-04-20 tile sweep found {16,32,64,128} within ~1% at kv_len≈54 and
// 32 the common-case winner for kv_len ≳ 128; very short kv (≲64) is a
// hair faster at 16 but within run-to-run noise. BN=256 regressed 13%
// (lane overhang). The paper sweep overrides this per-pp via
// GROUT_ATTN_BN_DECODE (benchmarks/sweep_tg_tile.sh is the tile search).
const ATTN_BN_DECODE: usize = 32;
// Split-K decode parallelism: kv_len split into NUM_KV_SPLITS chunks,
// each handled by a separate CTA. Default 16: the universal winner for
// kv_len ≳ 164 in the tg tile sweep, and the best single compromise
// across prefill and decode. Very short kv prefers fewer (4, avoids empty
// splits) and very long kv prefers more (32 at pp=8192); the paper sweep
// picks those per-pp via GROUT_FMHA_NUM_KV_SPLITS.
const FMHA_NUM_KV_SPLITS_DEFAULT: usize = 16;
// Software-pipelining depth + occupancy for fmha_decode_gqa_split.
// Tuned via 2D (LAT × OCC) sweep on sm_120 at pp=18 tg=128:
//   - Whole grid within 1.4% (compute-bound, like prefill).
//   - Minimum at (LAT=4, OCC=2) = 764.8 ms, vs default (LAT=2, OCC=1)
//     = 771.0 ms — marginal 0.8% edge, below measurement noise, but
//     consistently the best cell across the run.
// Override with GROUT_FMHA_DECODE_LATENCY / GROUT_FMHA_DECODE_OCCUPANCY.
const FMHA_DECODE_LATENCY_DEFAULT: usize = 4;
const FMHA_DECODE_OCCUPANCY_DEFAULT: usize = 2;
// CHUNK_D for splitk_reduce_merge's expanded grid. Grid becomes
// (kv_heads, 1, D/CHUNK_D). At head_dim=128: CHUNK_D=16 → 8 D-chunks
// per kv_head × 8 kv_heads = 64 CTAs (matches Blackwell's 64 SMs).
// Previously grid was (kv_heads, 1, 1) = 8 CTAs → 12.5% SM
// utilization. Override with GROUT_FMHA_MERGE_CHUNK_D. Must divide
// head_dim.
const FMHA_MERGE_CHUNK_D_DEFAULT: usize = 16;
// Pipeline depth for load_from_view inside splitk_reduce_merge.
// Override via GROUT_FMHA_MERGE_LATENCY.
const FMHA_MERGE_LATENCY_DEFAULT: usize = 2;
// Pipeline depth + occupancy + CTA clustering for qk_rope_dynpos_f16.
// On sm_120 num_cta_in_cga is binary — either unset or 2. Use
// GROUT_QK_ROPE_CGA=1 to enable clustering (adds `.num_cta_in_cga(2)`
// to CompileOptions). LATENCY + OCCUPANCY default tuning below.
const QK_ROPE_LATENCY_DEFAULT: usize = 2;
const QK_ROPE_OCCUPANCY_DEFAULT: usize = 1;
const QK_ROPE_CGA_DEFAULT: bool = false;
// D-chunk size for the decode kv_cache_update path. Grid expands from
// (kv_heads, 1, 1) = 8 CTAs to (kv_heads, 1, head_dim/CHUNK_D).
// Probed 2026-04-22 at pp=18 tg=128 (all within 0.8% noise):
//   CHUNK_D=128 → 8 CTAs:  796.7
//   CHUNK_D=64  → 16 CTAs: 792.4
//   CHUNK_D=32  → 32 CTAs: 790.8 ← min
//   CHUNK_D=16  → 64 CTAs: 791.9
//   CHUNK_D=8   → 128 CTAs: 791.7
// Override with GROUT_KV_CACHE_DYN_CHUNK_D. Must divide head_dim.
const KV_CACHE_DYN_CHUNK_D_DEFAULT: usize = 32;

// ── Attention tile constants — PREFILL path ────────────────────────
// Static defaults tuned for short-pp (pp=18) from 2026-04-20 sweep.
// At long pp these are no longer optimal; override via
// GROUT_ATTN_BM_PREFILL / GROUT_ATTN_BN_PREFILL. The benchmark wrappers
// also set per-architecture overrides for long prompt lengths.
const ATTN_BM_PREFILL: usize = 16;
const ATTN_BN_PREFILL: usize = 32;
// Software-pipelining depth and occupancy for fmha_prefill_causal. Tuned
// via 2D sweep on sm_120 (RTX 5090) at pp=2048 tg=36:
//   - OCC=2 wins; OCC=1 is 4-15% slower, OCC=4 is ~2× slower
//     (SMEM/register thrashing).
//   - Within OCC=2, LATENCY ∈ {1..4} all tied at ~116 ms (within noise);
//     LAT=2 picked as midpoint of the tied band. LATENCY ≥5 regresses
//     2-3%.
// Supplying CompileOptions (even when occupancy matches the entry-level
// hint) takes a different compiler path that's ~6% faster than relying
// on the sm_120 entry hint alone — always pass it.
// Override with GROUT_FMHA_PREFILL_LATENCY / GROUT_FMHA_PREFILL_OCCUPANCY.
const FMHA_PREFILL_LATENCY_DEFAULT: usize = 2;
const FMHA_PREFILL_OCCUPANCY_DEFAULT: usize = 2;

fn env_usize_or(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn env_bool_or(var: &str, default: bool) -> bool {
    std::env::var(var).ok().map(|v| v != "0").unwrap_or(default)
}

fn env_usize_hint_or(var: &str, default: usize) -> Option<usize> {
    let Ok(raw) = std::env::var(var) else {
        return Some(default);
    };
    let raw = raw.trim();
    if raw.eq_ignore_ascii_case("default")
        || raw.eq_ignore_ascii_case("none")
        || raw.eq_ignore_ascii_case("unset")
        || raw == "0"
    {
        return None;
    }
    raw.parse::<usize>()
        .ok()
        .filter(|v| *v > 0)
        .or(Some(default))
}

fn device_is_sm100(device_id: usize) -> bool {
    unsafe {
        let mut dev = MaybeUninit::<cu_sys::CUdevice>::uninit();
        if cu_sys::cuDeviceGet(dev.as_mut_ptr(), device_id as i32)
            .result()
            .is_err()
        {
            return false;
        }
        let dev = dev.assume_init();
        let mut major = MaybeUninit::<i32>::uninit();
        let mut minor = MaybeUninit::<i32>::uninit();
        if cu_sys::cuDeviceGetAttribute(
            major.as_mut_ptr(),
            cu_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            dev,
        )
        .result()
        .is_err()
        {
            return false;
        }
        if cu_sys::cuDeviceGetAttribute(
            minor.as_mut_ptr(),
            cu_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            dev,
        )
        .result()
        .is_err()
        {
            return false;
        }
        (major.assume_init(), minor.assume_init()) == (10, 0)
    }
}

fn env_bool_hint_or(var: &str, default: bool) -> Option<bool> {
    let Ok(raw) = std::env::var(var) else {
        return Some(default);
    };
    let raw = raw.trim();
    if raw.eq_ignore_ascii_case("default")
        || raw.eq_ignore_ascii_case("none")
        || raw.eq_ignore_ascii_case("unset")
    {
        return None;
    }
    Some(raw != "0")
}

fn compile_options_with_occupancy(occupancy: Option<usize>) -> CompileOptions {
    match occupancy {
        Some(occupancy) => CompileOptions::default().occupancy(occupancy as i32),
        None => CompileOptions::default(),
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

fn prefill_lpt_swizzle(q_len: usize, head_dim: usize, head_groups: usize) -> usize {
    // TileGym's prefill LPT path groups nearby head/batch work while K/V for
    // one head fit in L2. With Qwen-style equal QK/VO head dims:
    //   bytes_per_kv_head = seq_len * (head_dim_qk + head_dim_vo) * sizeof(f16)
    let bytes_per_kv_head = q_len
        .saturating_mul(head_dim.saturating_mul(2))
        .saturating_mul(size_of::<f16>());
    let l2_budget = env_usize_or("GROUT_FMHA_PREFILL_LPT_L2_BYTES", 50 * 1024 * 1024);
    let fit = if bytes_per_kv_head == 0 {
        1
    } else {
        (l2_budget / bytes_per_kv_head).max(1)
    };
    floor_power_of_two_le(fit).min(head_groups.max(1)).max(1)
}

// Smallest power-of-two tile size that covers `num_blocks` entries for the
// single-CTA `argmax_reduce_blocks_to_u32` kernel (cutile requires pow-2).
fn argmax_reduce_block_size(num_blocks: usize) -> usize {
    let mut s = 1usize;
    while s < num_blocks.max(1) {
        s *= 2;
    }
    s
}

// SAFETY: CudaGraph contains raw CUDA driver handles (CUgraph, CUgraphExec)
// which are opaque pointers safe to send/share between threads.
unsafe impl Send for DecodeCudaGraphRunner {}
unsafe impl Sync for DecodeCudaGraphRunner {}

struct DecodeCudaGraphRunner {
    graph: CudaGraph<()>,
    token_host: [u32; 1],
    position_host: [u32; 1],
    s_kv_host: [i32; 1],
    token_ids_device: Tensor<u32>,
    position_device: Tensor<u32>,
    s_kv_device: Tensor<i32>,
    /// Shared alias of bufs.logits — the graph writes into it on each launch.
    logits: Arc<Tensor<f16>>,
    logits_valid: bool,
    // Keep the buffers alive — the graph replays into their device pointers.
    _bufs: DecodeBuffers,
    // KV caches owned by the runner — the graph replays into these device pointers.
    // Temporarily moved to layer state during prefill, then moved back.
    kv_caches: Vec<(Tensor<f16>, Tensor<f16>)>,
}

impl DecodeCudaGraphRunner {
    /// Seed token_ids_device[0] via H2D. Called once before the first decode
    /// step so the graph's embedding can read the starting token; after that,
    /// the in-graph argmax writes subsequent tokens in place.
    fn seed_token(&mut self, token_id: u32) -> Result<()> {
        self.token_host[0] = token_id;
        unsafe {
            memcpy_htod_async(
                self.token_ids_device.device_pointer().cu_deviceptr(),
                self.token_host.as_ptr(),
                1,
                self.graph.stream(),
            );
        }
        Ok(())
    }

    /// Run one decode step. Reads the current token from `token_ids_device`
    /// (either seeded or left over from the previous step's in-graph argmax),
    /// produces logits, and picks the next token — which is written back to
    /// `token_ids_device[0]` by the graph. Returns the newly-selected token
    /// via a 4-byte D2H copy.
    fn launch_step(&mut self, position_start: usize) -> Result<u32> {
        ensure!(
            position_start <= u32::MAX as usize,
            "position_start {} exceeds u32 range",
            position_start
        );
        self.position_host[0] = position_start as u32;
        unsafe {
            memcpy_htod_async(
                self.position_device.device_pointer().cu_deviceptr(),
                self.position_host.as_ptr(),
                1,
                self.graph.stream(),
            );
            // s_kv copy only needed when flash_decode graph is active
            if env_bool_or("GROUT_FLASH_DECODE", false) {
                self.s_kv_host[0] = (position_start + 1) as i32;
                memcpy_htod_async(
                    self.s_kv_device.device_pointer().cu_deviceptr(),
                    self.s_kv_host.as_ptr(),
                    1,
                    self.graph.stream(),
                );
            }
        }
        self.graph
            .launch()
            .sync_on(self.graph.stream())
            .map_err(|e| anyhow::anyhow!("graph launch failed: {e:?}"))?;
        unsafe {
            memcpy_dtoh_async(
                self.token_host.as_mut_ptr(),
                self.token_ids_device.device_pointer().cu_deviceptr(),
                1,
                self.graph.stream(),
            );
        }
        unsafe { self.graph.stream().synchronize() }
            .map_err(|e| anyhow::anyhow!("sync after token d2h failed: {e:?}"))?;
        Ok(self.token_host[0])
    }

    /// Legacy path: return logits tensor (used by fallback callers that want
    /// to run host-side argmax or sampling). The graph still writes
    /// `token_ids_device[0]` as a side effect; callers using this path
    /// simply ignore it.
    #[allow(dead_code)]
    fn launch_step_with_logits(
        &mut self,
        token_id: u32,
        position_start: usize,
    ) -> Result<Arc<Tensor<f16>>> {
        ensure!(
            self.logits_valid,
            "decode graph was captured with fused lm_head_argmax and does not materialize logits"
        );
        self.seed_token(token_id)?;
        let _ = self.launch_step(position_start)?;
        // logits tensor is written in-place by the graph. The Arc aliases
        // the same device memory as _bufs.logits, so the data is fresh.
        Ok(self.logits.clone())
    }

    fn synchronize(&self) -> Result<()> {
        // launch().sync_on() already synchronizes.
        Ok(())
    }

    /// Zero all KV caches in-place. The device pointers are unchanged,
    /// so the captured graph remains valid.
    fn zero_kv_caches(&self) -> Result<()> {
        let stream = self.graph.stream();
        for (k, v) in &self.kv_caches {
            let k_bytes =
                k.shape().iter().map(|d| *d as usize).product::<usize>() * size_of::<f16>();
            let v_bytes =
                v.shape().iter().map(|d| *d as usize).product::<usize>() * size_of::<f16>();
            unsafe {
                cu_sys::cuMemsetD8Async(
                    k.device_pointer().cu_deviceptr(),
                    0,
                    k_bytes,
                    stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("zero k_cache failed: {e:?}"))?;
                cu_sys::cuMemsetD8Async(
                    v.device_pointer().cu_deviceptr(),
                    0,
                    v_bytes,
                    stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("zero v_cache failed: {e:?}"))?;
            }
        }
        unsafe { stream.synchronize() }
            .map_err(|e| anyhow::anyhow!("zero_kv_caches sync failed: {e:?}"))?;
        Ok(())
    }

    /// Move KV caches into layer state so the StepGraph prefill can write to them.
    fn lend_kv_caches_to_layers(&mut self, layers: &mut [Layer]) {
        for (layer_idx, (k, v)) in self.kv_caches.drain(..).enumerate() {
            layers[layer_idx].state.k_cache = Some(Arc::new(k));
            layers[layer_idx].state.v_cache = Some(Arc::new(v));
        }
    }

    /// Take KV caches back from layer state after prefill.
    fn reclaim_kv_caches_from_layers(&mut self, layers: &mut [Layer]) -> Result<()> {
        for layer in layers.iter_mut() {
            let k_arc = layer
                .state
                .k_cache
                .take()
                .context("missing k_cache after prefill")?;
            let v_arc = layer
                .state
                .v_cache
                .take()
                .context("missing v_cache after prefill")?;
            let k = Arc::try_unwrap(k_arc)
                .map_err(|_| anyhow::anyhow!("k_cache Arc has multiple owners after prefill"))?;
            let v = Arc::try_unwrap(v_arc)
                .map_err(|_| anyhow::anyhow!("v_cache Arc has multiple owners after prefill"))?;
            self.kv_caches.push((k, v));
        }
        Ok(())
    }
}

/// Pre-allocated tensors for the decode forward pass (seqlen=1).
/// All tensors are allocated once and reused across graph replays.
struct DecodeBuffers {
    hidden: Tensor<f16>,            // [1, hidden_size]
    normed: Tensor<f16>,            // [1, hidden_size]
    qkv: Tensor<f16>,               // [1, qkv_width]
    qk_norm_flat: Tensor<f16>,      // [num_heads + num_kv_heads, head_dim]
    qk_rope: Tensor<f16>,           // [1, num_heads + num_kv_heads, head_dim]
    attn_out: Tensor<f16>, // split-K: [kv_heads, group, head_dim], otherwise [1, num_heads, head_dim]
    attn_proj: Tensor<f16>, // [1, hidden_size]
    ff_normed: Tensor<f16>, // [1, hidden_size]
    hidden_after_attn: Tensor<f16>, // [1, hidden_size]
    gate_up: Tensor<f16>,  // [1, 2*inter_size]
    ff: Tensor<f16>,       // [1, inter_size]
    ff_down: Tensor<f16>,  // [1, hidden_size]
    logits: Tensor<f16>,   // [vocab_size]
    lse_scratch: Tensor<f32>, // [num_heads] — scratch for flash_decode LSE output
    argmax_block_max: Tensor<f32>, // [num_blocks] — stage-1 argmax per-block max
    argmax_block_idx: Tensor<u32>, // [num_blocks] — stage-1 argmax per-block argmax
    // Split-K decode attention scratch. Default ON as of 2026-04-20:
    // fmha_decode_gqa_split + splitk_reduce_merge replace the old
    // flash_attn_causal_seq_dynpos_f16 path. Opt out with GROUT_FMHA_SPLIT_KV=0.
    // [kv_heads, NUM_KV_SPLITS * group, head_dim] f16 per-split partial acc.
    fmha_att_partial: Tensor<f16>,
    // [kv_heads, NUM_KV_SPLITS * group] f32 per-split LSE.
    fmha_lse_partial: Tensor<f32>,
}

#[allow(dead_code)]
enum TokenInput<'a> {
    Host(&'a [u32]),
    Device(Arc<Tensor<u32>>),
}

#[derive(Clone)]
enum PositionInput {
    Host(usize),
    Device(Arc<Tensor<u32>>),
}

#[allow(dead_code)]
enum FinalLogitsPolicy<'a> {
    Allocate,
    Preallocated(&'a mut Option<Tensor<f16>>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum GraphDType {
    F16,
}

impl GraphDType {
    fn size_in_bytes(self) -> usize {
        match self {
            Self::F16 => size_of::<f16>(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct TensorSpec {
    dtype: GraphDType,
    shape: Vec<usize>,
}

impl TensorSpec {
    fn f16(shape: Vec<usize>) -> Self {
        Self {
            dtype: GraphDType::F16,
            shape,
        }
    }

    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn bytes(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct ValueId(usize);

impl ValueId {
    fn idx(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LayerWeightSlot {
    InputLayerNorm,
    PostAttentionLayerNorm,
    QNorm,
    KNorm,
    QkvProj,
    OProj,
    GateUpProj,
    DownProj,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WeightRef {
    LmHead,
    Norm,
    Layer {
        layer_idx: usize,
        slot: LayerWeightSlot,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorRef {
    Value(ValueId),
    Weight(WeightRef),
}

impl TensorRef {
    fn as_value(self) -> Option<ValueId> {
        match self {
            Self::Value(v) => Some(v),
            Self::Weight(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
enum GraphOp {
    EmbeddingBatch {
        out: ValueId,
    },
    MatMul {
        matrix: TensorRef,
        rhs: TensorRef,
        out: ValueId,
    },
    MatVec {
        matrix: TensorRef,
        vector: TensorRef,
        out: ValueId,
    },
    Add {
        lhs: TensorRef,
        rhs: TensorRef,
        out: ValueId,
    },
    SiluMul {
        gate: TensorRef,
        up: TensorRef,
        out: ValueId,
    },
    RmsNorm {
        x: TensorRef,
        weight: TensorRef,
        n: usize,
        out: ValueId,
    },
    Reshape {
        input: TensorRef,
        shape: Vec<usize>,
        out: ValueId,
    },
    Rope {
        x: TensorRef,
        out: ValueId,
    },
    KvCacheUpdate {
        layer_idx: usize,
        new_k: TensorRef,
        new_v: TensorRef,
    },
    QkNormRopeKvPrefill {
        layer_idx: usize,
        q: TensorRef,
        k: TensorRef,
        v: TensorRef,
        out: ValueId,
    },
    Attention {
        layer_idx: usize,
        q: TensorRef,
        out: ValueId,
    },
    GatherRow {
        src: TensorRef,
        row_idx: usize,
        out: ValueId,
    },
    /// Copies a column-slice of a 2D tensor.
    /// For an input of shape [rows, total_cols], copies columns
    /// [col_offset .. col_offset + out_cols) from each row into the output [rows, out_cols].
    SliceCols {
        input: TensorRef,
        col_offset: usize,
        out_cols: usize,
        out: ValueId,
    },
    /// Row-sliced MatMul: uses rows [row_offset..row_offset+out_features) of the weight matrix.
    /// Equivalent to MatMul with weight[row_offset..row_offset+out_features, :] without copying.
    MatMulSlice {
        matrix: TensorRef,
        row_offset: usize,
        out_features: usize,
        rhs: TensorRef,
        out: ValueId,
    },
    /// Fused residual + x add followed by RMS norm.
    /// Produces two outputs: the normed result (out) and the combined residual (residual_out).
    AddRmsNorm {
        residual: TensorRef,
        x: TensorRef,
        weight: TensorRef,
        n: usize,
        out: ValueId,
        residual_out: ValueId,
    },
}

impl GraphOp {
    fn output(&self) -> Option<ValueId> {
        match self {
            Self::EmbeddingBatch { out, .. }
            | Self::MatMul { out, .. }
            | Self::MatVec { out, .. }
            | Self::Add { out, .. }
            | Self::SiluMul { out, .. }
            | Self::RmsNorm { out, .. }
            | Self::Reshape { out, .. }
            | Self::Rope { out, .. }
            | Self::QkNormRopeKvPrefill { out, .. }
            | Self::Attention { out, .. }
            | Self::GatherRow { out, .. }
            | Self::SliceCols { out, .. }
            | Self::MatMulSlice { out, .. } => Some(*out),
            Self::AddRmsNorm { out, .. } => Some(*out),
            Self::KvCacheUpdate { .. } => None,
        }
    }

    /// Returns additional outputs beyond the primary one (for multi-output ops).
    fn extra_outputs(&self) -> Vec<ValueId> {
        match self {
            Self::AddRmsNorm { residual_out, .. } => vec![*residual_out],
            _ => vec![],
        }
    }

    fn value_inputs(&self) -> Vec<ValueId> {
        fn maybe_push(values: &mut Vec<ValueId>, input: TensorRef) {
            if let Some(v) = input.as_value() {
                values.push(v);
            }
        }

        let mut values = Vec::new();
        match self {
            Self::EmbeddingBatch { .. } => {}
            Self::MatMul { matrix, rhs, .. } => {
                maybe_push(&mut values, *matrix);
                maybe_push(&mut values, *rhs);
            }
            Self::MatVec { matrix, vector, .. } => {
                maybe_push(&mut values, *matrix);
                maybe_push(&mut values, *vector);
            }
            Self::Add { lhs, rhs, .. } => {
                maybe_push(&mut values, *lhs);
                maybe_push(&mut values, *rhs);
            }
            Self::SiluMul { gate, up, .. } => {
                maybe_push(&mut values, *gate);
                maybe_push(&mut values, *up);
            }
            Self::RmsNorm { x, weight, .. } => {
                maybe_push(&mut values, *x);
                maybe_push(&mut values, *weight);
            }
            Self::Reshape { input, .. } => {
                maybe_push(&mut values, *input);
            }
            Self::Rope { x, .. } => {
                maybe_push(&mut values, *x);
            }
            Self::KvCacheUpdate { new_k, new_v, .. } => {
                maybe_push(&mut values, *new_k);
                maybe_push(&mut values, *new_v);
            }
            Self::QkNormRopeKvPrefill { q, k, v, .. } => {
                maybe_push(&mut values, *q);
                maybe_push(&mut values, *k);
                maybe_push(&mut values, *v);
            }
            Self::Attention { q, .. } => {
                maybe_push(&mut values, *q);
            }
            Self::GatherRow { src, .. } => {
                maybe_push(&mut values, *src);
            }
            Self::SliceCols { input, .. } => {
                maybe_push(&mut values, *input);
            }
            Self::MatMulSlice { matrix, rhs, .. } => {
                maybe_push(&mut values, *matrix);
                maybe_push(&mut values, *rhs);
            }
            Self::AddRmsNorm {
                residual,
                x,
                weight,
                ..
            } => {
                maybe_push(&mut values, *residual);
                maybe_push(&mut values, *x);
                maybe_push(&mut values, *weight);
            }
        }
        values
    }
}

#[derive(Default, Clone)]
struct TensorPoolPlan {
    high_water_bytes: usize,
    high_water_live: HashMap<TensorSpec, usize>,
    max_live_by_spec: HashMap<TensorSpec, usize>,
}

#[derive(Clone)]
struct StepGraph {
    ops: Vec<GraphOp>,
    specs: Vec<TensorSpec>,
    use_counts: Vec<usize>,
    final_value: ValueId,
    pool_plan: TensorPoolPlan,
}

impl StepGraph {
    fn new(ops: Vec<GraphOp>, specs: Vec<TensorSpec>, final_value: ValueId) -> Result<Self> {
        ensure!(
            final_value.idx() < specs.len(),
            "final value {} is out of range (values={})",
            final_value.idx(),
            specs.len()
        );
        let mut use_counts = vec![0usize; specs.len()];
        for op in &ops {
            for v in op.value_inputs() {
                use_counts[v.idx()] += 1;
            }
        }
        ensure!(
            use_counts[final_value.idx()] == 0,
            "final value {} has {} consumers",
            final_value.idx(),
            use_counts[final_value.idx()]
        );

        // Reshape nodes are modeled as view-moves, so their input must not have other consumers.
        for op in &ops {
            if let GraphOp::Reshape {
                input: TensorRef::Value(v),
                ..
            } = op
            {
                ensure!(
                    use_counts[v.idx()] == 1,
                    "reshape input value {} has {} consumers (expected exactly 1)",
                    v.idx(),
                    use_counts[v.idx()]
                );
            }
        }

        let pool_plan = Self::compute_pool_plan(&ops, &specs, &use_counts, final_value)?;
        Ok(Self {
            ops,
            specs,
            use_counts,
            final_value,
            pool_plan,
        })
    }

    fn spec(&self, value: ValueId) -> &TensorSpec {
        &self.specs[value.idx()]
    }

    fn compute_pool_plan(
        ops: &[GraphOp],
        specs: &[TensorSpec],
        use_counts: &[usize],
        final_value: ValueId,
    ) -> Result<TensorPoolPlan> {
        let mut remaining_uses = use_counts.to_vec();
        let mut live_spec_by_value: Vec<Option<TensorSpec>> = vec![None; specs.len()];
        let mut live_counts: HashMap<TensorSpec, usize> = HashMap::new();
        let mut max_live_by_spec: HashMap<TensorSpec, usize> = HashMap::new();
        let mut total_live_bytes = 0usize;
        let mut high_water_bytes = 0usize;
        let mut high_water_live: HashMap<TensorSpec, usize> = HashMap::new();

        for op in ops {
            match op {
                GraphOp::Reshape {
                    input: TensorRef::Value(input),
                    out,
                    ..
                } => {
                    let src_spec = live_spec_by_value[input.idx()]
                        .take()
                        .context("planner expected reshape source to be live")?;
                    let dst_spec = specs[out.idx()].clone();
                    ensure!(
                        src_spec.numel() == dst_spec.numel(),
                        "reshape numel mismatch {:?} -> {:?}",
                        src_spec.shape,
                        dst_spec.shape
                    );
                    decrement_live_count(&mut live_counts, &src_spec)?;
                    if *out != final_value {
                        increment_live_count(&mut live_counts, &mut max_live_by_spec, &dst_spec);
                        live_spec_by_value[out.idx()] = Some(dst_spec);
                    }
                }
                GraphOp::Reshape { .. } => {
                    bail!("reshape input must be a value");
                }
                _ => {
                    // Track primary output.
                    if let Some(out) = op.output()
                        && out != final_value
                    {
                        let out_spec = specs[out.idx()].clone();
                        increment_live_count(&mut live_counts, &mut max_live_by_spec, &out_spec);
                        live_spec_by_value[out.idx()] = Some(out_spec.clone());
                        total_live_bytes += out_spec.bytes();
                        if total_live_bytes > high_water_bytes {
                            high_water_bytes = total_live_bytes;
                            high_water_live = live_counts.clone();
                        }
                    }
                    // Track extra outputs (e.g., AddRmsNorm::residual_out).
                    for extra_out in op.extra_outputs() {
                        if extra_out != final_value {
                            let extra_spec = specs[extra_out.idx()].clone();
                            increment_live_count(
                                &mut live_counts,
                                &mut max_live_by_spec,
                                &extra_spec,
                            );
                            live_spec_by_value[extra_out.idx()] = Some(extra_spec.clone());
                            total_live_bytes += extra_spec.bytes();
                            if total_live_bytes > high_water_bytes {
                                high_water_bytes = total_live_bytes;
                                high_water_live = live_counts.clone();
                            }
                        }
                    }
                }
            }

            for input in op.value_inputs() {
                let idx = input.idx();
                ensure!(
                    remaining_uses[idx] > 0,
                    "invalid use-count state for value {idx}"
                );
                remaining_uses[idx] -= 1;
                if remaining_uses[idx] == 0
                    && input != final_value
                    && let Some(spec) = live_spec_by_value[idx].take()
                {
                    decrement_live_count(&mut live_counts, &spec)?;
                    total_live_bytes = total_live_bytes.saturating_sub(spec.bytes());
                }
            }
        }

        Ok(TensorPoolPlan {
            high_water_bytes,
            high_water_live,
            max_live_by_spec,
        })
    }
}

#[derive(Default)]
struct TensorPool {
    free_exact: HashMap<TensorSpec, Vec<Tensor<f16>>>,
    cache_caps: HashMap<TensorSpec, usize>,
}

impl TensorPool {
    fn from_plan_ctx(ctx: &ExecutionContext, plan: &TensorPoolPlan) -> Result<Self> {
        let mut pool = Self {
            free_exact: HashMap::new(),
            cache_caps: plan.max_live_by_spec.clone(),
        };
        let mut preallocated_bytes = 0usize;
        for (spec, count) in &plan.high_water_live {
            let bin = pool.free_exact.entry(spec.clone()).or_default();
            for _ in 0..*count {
                bin.push(alloc_f16_ctx(ctx, &spec.shape)?);
                preallocated_bytes += spec.bytes();
            }
        }
        ensure!(
            preallocated_bytes == plan.high_water_bytes,
            "pool preallocation mismatch: expected {} bytes, got {} bytes",
            plan.high_water_bytes,
            preallocated_bytes
        );
        Ok(pool)
    }

    fn checkout(&mut self, ctx: &ExecutionContext, spec: &TensorSpec) -> Result<Tensor<f16>> {
        if let Some(bin) = self.free_exact.get_mut(spec)
            && let Some(t) = bin.pop()
        {
            return Ok(t
                .reshape(&spec.shape)
                .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?);
        }
        if let Some(t) = self.take_compatible(spec) {
            return Ok(t
                .reshape(&spec.shape)
                .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?);
        }
        if std::env::var("GROUT_DEBUG_POOL_ALLOC").ok().as_deref() == Some("1") {
            eprintln!(
                "debug: tensor-pool fallback alloc for shape {:?}",
                spec.shape
            );
        }
        alloc_f16_ctx(ctx, &spec.shape)
    }

    fn checkin(&mut self, tensor: Tensor<f16>, spec: &TensorSpec) -> Result<()> {
        let numel = tensor
            .shape()
            .iter()
            .map(|d| *d as usize)
            .product::<usize>();
        ensure!(
            numel == spec.numel(),
            "pool checkin numel mismatch: tensor shape {:?}, expected {:?}",
            tensor.shape(),
            spec.shape
        );

        let cap = self.cache_caps.get(spec).copied().unwrap_or(usize::MAX);
        let bin = self.free_exact.entry(spec.clone()).or_default();
        if bin.len() < cap {
            bin.push(
                tensor
                    .reshape(&spec.shape)
                    .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?,
            );
        }
        Ok(())
    }

    fn take_compatible(&mut self, spec: &TensorSpec) -> Option<Tensor<f16>> {
        let target_numel = spec.numel();
        let mut donor_key = None;
        for (candidate, bin) in &self.free_exact {
            if candidate.dtype == spec.dtype && candidate.numel() == target_numel && !bin.is_empty()
            {
                donor_key = Some(candidate.clone());
                break;
            }
        }
        donor_key.and_then(|key| self.free_exact.get_mut(&key)?.pop())
    }
}

fn alloc_f16_ctx(ctx: &ExecutionContext, shape: &[usize]) -> Result<Tensor<f16>> {
    let out = unsafe { api::zeros::<f16>(shape).execute(ctx)? };
    Ok(out)
}

fn push_value(specs: &mut Vec<TensorSpec>, shape: Vec<usize>) -> ValueId {
    let id = ValueId(specs.len());
    specs.push(TensorSpec::f16(shape));
    id
}

fn increment_live_count(
    live_counts: &mut HashMap<TensorSpec, usize>,
    max_live_by_spec: &mut HashMap<TensorSpec, usize>,
    spec: &TensorSpec,
) {
    let entry = live_counts.entry(spec.clone()).or_insert(0);
    *entry += 1;
    let cur = *entry;
    max_live_by_spec
        .entry(spec.clone())
        .and_modify(|v| *v = (*v).max(cur))
        .or_insert(cur);
}

fn decrement_live_count(
    live_counts: &mut HashMap<TensorSpec, usize>,
    spec: &TensorSpec,
) -> Result<()> {
    let count = live_counts
        .get_mut(spec)
        .context("live-count underflow in planner")?;
    ensure!(*count > 0, "live-count underflow for spec {:?}", spec.shape);
    *count -= 1;
    if *count == 0 {
        live_counts.remove(spec);
    }
    Ok(())
}

#[derive(Default)]
struct RunProfile {
    prefill_steps: usize,
    decode_steps: usize,
    prefill_step_total: Duration,
    decode_step_total: Duration,
    op_profile_enabled: bool,
    op_profile_sync_enabled: bool,
    op_totals: HashMap<&'static str, (usize, Duration)>,
}

impl RunProfile {
    fn new() -> Self {
        Self {
            prefill_steps: 0,
            decode_steps: 0,
            prefill_step_total: Duration::ZERO,
            decode_step_total: Duration::ZERO,
            op_profile_enabled: std::env::var("GROUT_PROFILE_OPS").ok().as_deref() == Some("1"),
            op_profile_sync_enabled: std::env::var("GROUT_PROFILE_SYNC_OPS").ok().as_deref()
                == Some("1"),
            op_totals: HashMap::new(),
        }
    }

    fn add_step(&mut self, dur: Duration, is_decode: bool) {
        if is_decode {
            self.decode_steps += 1;
            self.decode_step_total += dur;
        } else {
            self.prefill_steps += 1;
            self.prefill_step_total += dur;
        }
    }

    fn add_op(&mut self, op_name: &'static str, dur: Duration) {
        if !self.op_profile_enabled {
            return;
        }
        let entry = self.op_totals.entry(op_name).or_insert((0, Duration::ZERO));
        entry.0 += 1;
        entry.1 += dur;
    }

    fn render(&self) -> String {
        let mut out = String::new();
        let prefill_avg_ms = if self.prefill_steps > 0 {
            1.0e3 * self.prefill_step_total.as_secs_f64() / (self.prefill_steps as f64)
        } else {
            0.0
        };
        let decode_avg_ms = if self.decode_steps > 0 {
            1.0e3 * self.decode_step_total.as_secs_f64() / (self.decode_steps as f64)
        } else {
            0.0
        };
        let _ = writeln!(out, "perf profile");
        let _ = writeln!(
            out,
            "steps: prefill={} decode={} | avg_ms: prefill={:.3} decode={:.3}",
            self.prefill_steps, self.decode_steps, prefill_avg_ms, decode_avg_ms
        );
        if self.op_profile_enabled && !self.op_totals.is_empty() {
            if self.op_profile_sync_enabled {
                let _ = writeln!(out, "op profile uses stream synchronization after each op");
            }
            let mut totals: Vec<(&'static str, usize, Duration)> = self
                .op_totals
                .iter()
                .map(|(name, (count, total))| (*name, *count, *total))
                .collect();
            totals.sort_by_key(|x| Reverse(x.2));
            let _ = writeln!(out, "op profile (total_ms, avg_us, calls):");
            for (name, count, total) in totals {
                let total_ms = 1.0e3 * total.as_secs_f64();
                let avg_us = if count > 0 {
                    1.0e6 * total.as_secs_f64() / (count as f64)
                } else {
                    0.0
                };
                let _ = writeln!(
                    out,
                    "  {name:<14} total_ms={total_ms:>8.3} avg_us={avg_us:>8.2} calls={count}"
                );
            }
        }
        out
    }
}

struct LayerWeights {
    input_layernorm: Arc<Tensor<f16>>,
    post_attention_layernorm: Arc<Tensor<f16>>,
    q_norm: Arc<Tensor<f16>>,
    k_norm: Arc<Tensor<f16>>,
    qkv_proj: Arc<Tensor<f16>>,
    o_proj: Arc<Tensor<f16>>,
    gate_up_proj: Arc<Tensor<f16>>,
    down_proj: Arc<Tensor<f16>>,
}

struct LayerState {
    k_cache: Option<Arc<Tensor<f16>>>,
    v_cache: Option<Arc<Tensor<f16>>>,
}

struct Layer {
    weights: LayerWeights,
    state: LayerState,
}

#[derive(Default)]
struct KernelWarmRegistry {
    warmed: [bool; KernelKind::COUNT],
}

impl KernelWarmRegistry {
    fn is_warmed(&self, kind: KernelKind) -> bool {
        self.warmed[kind.idx()]
    }

    fn mark_warmed(&mut self, kind: KernelKind) {
        self.warmed[kind.idx()] = true;
    }

    fn all_warmed(&self) -> bool {
        self.warmed.iter().copied().all(|w| w)
    }
}

pub struct GenerationOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_elapsed: Duration,
    pub decode_elapsed: Duration,
    pub total_elapsed: Duration,
    pub profile_report: Option<String>,
}

impl GenerationOutput {
    #[allow(dead_code)]
    pub fn prompt_tps(&self) -> f64 {
        let secs = self.prompt_elapsed.as_secs_f64().max(1.0e-9);
        self.prompt_tokens as f64 / secs
    }

    pub fn decode_phase_tps(&self) -> f64 {
        let secs = self.decode_elapsed.as_secs_f64().max(1.0e-9);
        self.generated_tokens as f64 / secs
    }

    pub fn request_gen_tps(&self) -> f64 {
        let secs = self.total_elapsed.as_secs_f64().max(1.0e-9);
        self.generated_tokens as f64 / secs
    }

    pub fn total_tps(&self) -> f64 {
        let secs = self.total_elapsed.as_secs_f64().max(1.0e-9);
        (self.prompt_tokens + self.generated_tokens) as f64 / secs
    }
}

pub struct Qwen3Engine {
    cfg: Qwen3Config,
    tokenizer: Tokenizer,
    model_dir: std::path::PathBuf,
    embed_tokens: Arc<Tensor<f16>>,
    lm_head: Arc<Tensor<f16>>,
    norm: Arc<Tensor<f16>>,
    inv_freq: Arc<Tensor<f32>>,
    layers: Vec<Layer>,
    max_seq_len: usize,
    eos_token_ids: Vec<u32>,
    do_sample: bool,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    use_chat_template: bool,
    use_device_argmax: bool,
    add_rms_block: usize,
    rms_hidden_block: usize,
    profile_enabled: bool,
    active_profile: Option<RunProfile>,
    kernel_warm_registry: KernelWarmRegistry,
    step_graph_cache: HashMap<usize, Arc<StepGraph>>,
    step_pool_cache: HashMap<usize, TensorPool>,
    decode_runner: Option<DecodeCudaGraphRunner>,
}

impl Qwen3Engine {
    pub async fn load(model_dir: &Path, max_seq_len: Option<usize>) -> Result<Self> {
        model_dir.try_exists()?;
        let cfg = Qwen3Config::from_model_dir(model_dir)?;
        let generation_cfg = GenerationConfig::from_model_dir(model_dir)?;
        ensure!(
            !cfg.use_sliding_window,
            "sliding-window attention is not supported in this engine"
        );
        ensure!(cfg.head_dim == ROPE_BLOCK, "expected head_dim={ROPE_BLOCK}");
        // Note: cutile-rs handles non-divisible tile shapes via boundary
        // masking. No divisibility checks needed for tile block sizes.

        let max_seq_len = min(max_seq_len.unwrap_or(4096), cfg.max_position_embeddings);

        let loader = WeightLoader::new(model_dir)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;
        let vocab = tokenizer.get_vocab(true);
        let use_chat_template =
            vocab.contains_key("<|im_start|>") && vocab.contains_key("<|im_end|>");

        let mut eos_token_ids = vec![cfg.eos_token_id];
        let mut do_sample = false;
        let mut temperature = 1.0f32;
        let mut top_k = 0usize;
        let mut top_p = 1.0f32;
        if let Some(gen_cfg) = generation_cfg {
            if let Some(eos) = gen_cfg.eos_token_id {
                eos_token_ids = eos.into_vec();
            }
            do_sample = gen_cfg.do_sample.unwrap_or(false);
            temperature = gen_cfg.temperature.unwrap_or(1.0).max(1.0e-5);
            top_k = gen_cfg.top_k.unwrap_or(0);
            top_p = gen_cfg.top_p.unwrap_or(1.0).clamp(0.0, 1.0);
        }
        // Device argmax is default-on (avoids 300 KB logits D2H copy per
        // decode token). Use --host-argmax on the CLI to opt out.
        let use_device_argmax = true;
        let add_rms_block = env_usize_or("GROUT_ADD_RMS_BLOCK", ADD_RMS_BLOCK);
        let rms_hidden_candidate = env_usize_or("GROUT_RMS_HIDDEN_BLOCK", RMS_BLOCK_HIDDEN);
        let rms_hidden_block = if rms_hidden_candidate.is_power_of_two() {
            rms_hidden_candidate
        } else {
            RMS_BLOCK_HIDDEN
        };
        let profile_enabled = std::env::var("GROUT_PROFILE")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(false);

        // Get a stream for synchronous weight loading.
        let stream = with_context(|ctx| value(ctx.get_cuda_stream().clone())).await?;

        let embed_tokens = loader
            .load_device_f16("model.embed_tokens.weight", &stream)
            .context("failed to load model.embed_tokens.weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            loader
                .load_device_f16("lm_head.weight", &stream)
                .context("failed to load lm_head.weight")?
        };
        let norm = loader
            .load_device_f16("model.norm.weight", &stream)
            .context("failed to load model.norm.weight")?;

        let inv_freq = build_inv_freq(&stream, &cfg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let q_proj =
                load_layer_weight(&loader, &stream, i, "self_attn.q_proj.weight", "q_proj")?;
            let k_proj =
                load_layer_weight(&loader, &stream, i, "self_attn.k_proj.weight", "k_proj")?;
            let v_proj =
                load_layer_weight(&loader, &stream, i, "self_attn.v_proj.weight", "v_proj")?;
            let qkv_proj = concat_weight_rows_2d(&stream, &[&q_proj, &k_proj, &v_proj])?;

            let gate_proj =
                load_layer_weight(&loader, &stream, i, "mlp.gate_proj.weight", "gate_proj")?;
            let up_proj = load_layer_weight(&loader, &stream, i, "mlp.up_proj.weight", "up_proj")?;
            let gate_up_proj = concat_weight_rows_2d(&stream, &[&gate_proj, &up_proj])?;

            let weights = LayerWeights {
                input_layernorm: load_layer_weight(
                    &loader,
                    &stream,
                    i,
                    "input_layernorm.weight",
                    "input layernorm",
                )?,
                post_attention_layernorm: load_layer_weight(
                    &loader,
                    &stream,
                    i,
                    "post_attention_layernorm.weight",
                    "post-attention layernorm",
                )?,
                q_norm: load_layer_weight(
                    &loader,
                    &stream,
                    i,
                    "self_attn.q_norm.weight",
                    "q_norm",
                )?,
                k_norm: load_layer_weight(
                    &loader,
                    &stream,
                    i,
                    "self_attn.k_norm.weight",
                    "k_norm",
                )?,
                qkv_proj,
                o_proj: load_layer_weight(
                    &loader,
                    &stream,
                    i,
                    "self_attn.o_proj.weight",
                    "o_proj",
                )?,
                gate_up_proj,
                down_proj: load_layer_weight(
                    &loader,
                    &stream,
                    i,
                    "mlp.down_proj.weight",
                    "down_proj",
                )?,
            };

            let k_cache = api::zeros::<f16>(&[cfg.num_key_value_heads, max_seq_len, cfg.head_dim])
                .sync_on(&stream)
                .map_err(|e| anyhow::anyhow!("alloc k_cache failed: {e:?}"))?;
            let v_cache = api::zeros::<f16>(&[cfg.num_key_value_heads, max_seq_len, cfg.head_dim])
                .sync_on(&stream)
                .map_err(|e| anyhow::anyhow!("alloc v_cache failed: {e:?}"))?;
            layers.push(Layer {
                weights,
                state: LayerState {
                    k_cache: Some(Arc::new(k_cache)),
                    v_cache: Some(Arc::new(v_cache)),
                },
            });
        }

        Ok(Self {
            cfg,
            tokenizer,
            model_dir: loader.model_dir().to_path_buf(),
            embed_tokens,
            lm_head,
            norm,
            inv_freq,
            layers,
            max_seq_len,
            eos_token_ids,
            do_sample,
            temperature,
            top_k,
            top_p,
            use_chat_template,
            use_device_argmax,
            add_rms_block,
            rms_hidden_block,
            profile_enabled,
            active_profile: None,
            kernel_warm_registry: KernelWarmRegistry::default(),
            step_graph_cache: HashMap::new(),
            step_pool_cache: HashMap::new(),
            decode_runner: None,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn set_sampling_enabled(&mut self, enabled: bool) {
        self.do_sample = enabled;
    }

    pub fn set_chat_template_enabled(&mut self, enabled: bool) {
        self.use_chat_template = enabled;
    }

    pub fn set_device_argmax_enabled(&mut self, enabled: bool) {
        self.use_device_argmax = enabled;
    }

    pub fn set_profile_enabled(&mut self, enabled: bool) {
        self.profile_enabled = enabled;
    }

    /// Disable EOS-based early termination so decode always runs for exactly
    /// `max_new_tokens` steps. Used by paper benchmarks where we need a fixed
    /// decode window to match `min_tokens==max_tokens` / `ignore_eos=true` on
    /// the Python engines.
    pub fn set_ignore_eos(&mut self, ignore: bool) {
        if ignore {
            self.eos_token_ids.clear();
        }
    }

    fn profile_step(&mut self, dur: Duration, is_decode: bool) {
        if let Some(profile) = self.active_profile.as_mut() {
            profile.add_step(dur, is_decode);
        }
    }

    fn profile_op(&mut self, op_name: &'static str, dur: Duration) {
        if let Some(profile) = self.active_profile.as_mut() {
            profile.add_op(op_name, dur);
        }
    }

    pub async fn warm_all_kernels(&mut self) -> Result<()> {
        if self.kernel_warm_registry.all_warmed() {
            return Ok(());
        }

        self.reset_cache().await?;
        with_context(|ctx| value(self.warm_tile_kernels_ctx(ctx))).await??;

        self.reset_cache().await?;
        Ok(())
    }

    fn warm_tile_kernels_ctx(&mut self, ctx: &ExecutionContext) -> Result<()> {
        for kind in TILE_KERNEL_KINDS {
            self.warm_tile_kernel_ctx(ctx, kind)?;
        }
        // CUDA graph decode is the fast path — default on. Opt out with
        // GROUT_CUDA_GRAPH_DECODE=0 for diagnostic / StepGraph comparison.
        if env_bool_or("GROUT_CUDA_GRAPH_DECODE", true) {
            self.warm_decode_graph_kernels_ctx(ctx)?;
        }

        Ok(())
    }

    fn warm_decode_graph_kernels_ctx(&mut self, ctx: &ExecutionContext) -> Result<()> {
        let pos = Arc::new(unsafe { api::zeros::<u32>(&[1]).execute(ctx)? });
        let x = Arc::new(unsafe {
            api::zeros::<f16>(&[1, self.cfg.num_attention_heads, self.cfg.head_dim]).execute(ctx)?
        });
        let x_out = alloc_f16_ctx(ctx, &[1, self.cfg.num_attention_heads, self.cfg.head_dim])?;
        let _ = self.rope_seq_arc_into_ctx_device_pos(ctx, x, pos.clone(), x_out)?;

        let new_k = Arc::new(unsafe {
            api::zeros::<f16>(&[1, self.cfg.num_key_value_heads, self.cfg.head_dim]).execute(ctx)?
        });
        let new_v = Arc::new(unsafe {
            api::zeros::<f16>(&[1, self.cfg.num_key_value_heads, self.cfg.head_dim]).execute(ctx)?
        });
        self.kv_cache_update_seq_arc_ctx_device_pos(ctx, 0, new_k, new_v, pos.clone())?;

        let q = Arc::new(unsafe {
            api::zeros::<f16>(&[1, self.cfg.num_attention_heads, self.cfg.head_dim]).execute(ctx)?
        });
        let q_out = alloc_f16_ctx(ctx, &[1, self.cfg.num_attention_heads, self.cfg.head_dim])?;
        let _ = self.attend_seq_arc_into_ctx_device_pos(ctx, 0, q, pos, q_out)?;
        Ok(())
    }

    fn warm_tile_kernel_ctx(&mut self, ctx: &ExecutionContext, kind: KernelKind) -> Result<()> {
        if self.kernel_warm_registry.is_warmed(kind) {
            return Ok(());
        }

        match kind {
            KernelKind::EmbeddingBatch => {
                let max_tok = self.cfg.vocab_size.saturating_sub(1) as u32;
                let warm_tok = self
                    .eos_token_ids
                    .first()
                    .copied()
                    .unwrap_or(0)
                    .min(max_tok);
                let warm_ids = [warm_tok];
                let _ = self.embedding_batch_ctx(ctx, &warm_ids)?;
            }
            KernelKind::Gemm => {
                let x = unsafe { api::zeros::<f16>(&[1, self.cfg.hidden_size]).execute(ctx)? };
                let x = Arc::new(x);
                let qkv_proj = self.layers[0].weights.qkv_proj.clone();
                let _ = self.gemm_ctx(ctx, qkv_proj, x)?;
            }
            KernelKind::Gemv => {
                let v = unsafe { api::zeros::<f16>(&[self.cfg.hidden_size]).execute(ctx)? };
                let v = Arc::new(v);
                let _ = self.gemv_ctx(ctx, self.embed_tokens.clone(), v)?;
            }
            KernelKind::RmsNorm => {
                let hidden = unsafe { api::zeros::<f16>(&[1, self.cfg.hidden_size]).execute(ctx)? };
                let _ = self.rms_norm_ctx(ctx, hidden, self.norm.clone(), self.cfg.hidden_size)?;

                let q_norm = self.layers[0].weights.q_norm.clone();
                let head = unsafe { api::zeros::<f16>(&[1, self.cfg.head_dim]).execute(ctx)? };
                let _ = self.rms_norm_ctx(ctx, head, q_norm, self.cfg.head_dim)?;
            }
            KernelKind::RopeSeq => {
                let q = unsafe {
                    api::zeros::<f16>(&[1, self.cfg.num_attention_heads, self.cfg.head_dim])
                        .execute(ctx)?
                };
                let _ = self.rope_seq_ctx(ctx, q, 0)?;
            }
            KernelKind::KvCacheUpdateSeq => {
                let new_k = unsafe {
                    api::zeros::<f16>(&[1, self.cfg.num_key_value_heads, self.cfg.head_dim])
                        .execute(ctx)?
                };
                let new_v = unsafe {
                    api::zeros::<f16>(&[1, self.cfg.num_key_value_heads, self.cfg.head_dim])
                        .execute(ctx)?
                };
                self.kv_cache_update_seq_ctx(ctx, 0, new_k, new_v, 0)?;
            }
            KernelKind::FlashAttnCausalSeq => {
                // Warm both decode (q_len=1 => ATTN_BN_DECODE) and prefill (q_len>1 => ATTN_BN_PREFILL).
                for q_len in [1usize, 2usize] {
                    if q_len > self.max_seq_len {
                        continue;
                    }
                    let q = unsafe {
                        api::zeros::<f16>(&[q_len, self.cfg.num_attention_heads, self.cfg.head_dim])
                            .execute(ctx)?
                    };
                    let _ = self.attend_seq_ctx(ctx, 0, q, 0)?;
                }
            }
            KernelKind::AddVec => {
                let lhs = Arc::new(unsafe {
                    api::zeros::<f16>(&[1, self.cfg.hidden_size]).execute(ctx)?
                });
                let rhs = Arc::new(unsafe {
                    api::zeros::<f16>(&[1, self.cfg.hidden_size]).execute(ctx)?
                });
                let _ = self.add_2d_ctx(ctx, lhs, rhs)?;
            }
            KernelKind::SiluMul => {
                let gate = Arc::new(unsafe {
                    api::zeros::<f16>(&[1, self.cfg.intermediate_size]).execute(ctx)?
                });
                let up = Arc::new(unsafe {
                    api::zeros::<f16>(&[1, self.cfg.intermediate_size]).execute(ctx)?
                });
                let _ = self.silu_mul_2d_ctx(ctx, gate, up)?;
            }
            KernelKind::GatherRow => {
                let src = Arc::new(unsafe {
                    api::zeros::<f16>(&[1, self.cfg.hidden_size]).execute(ctx)?
                });
                let _ = self.gather_row_ctx(ctx, src, 0)?;
            }
            KernelKind::ArgmaxBlocks => {
                let logits =
                    Arc::new(unsafe { api::zeros::<f16>(&[self.cfg.vocab_size]).execute(ctx)? });
                let _ = self.argmax_blocks_ctx(ctx, logits, self.cfg.vocab_size)?;
            }
            KernelKind::AddRmsNorm => {
                let residual = Arc::new(unsafe {
                    api::zeros::<f16>(&[1, self.cfg.hidden_size]).execute(ctx)?
                });
                let x = Arc::new(unsafe {
                    api::zeros::<f16>(&[1, self.cfg.hidden_size]).execute(ctx)?
                });
                let weight = self.norm.clone();
                let out = alloc_f16_ctx(ctx, &[1, self.cfg.hidden_size])?;
                let residual_out = alloc_f16_ctx(ctx, &[1, self.cfg.hidden_size])?;
                let _ = self.add_rms_norm_into_ctx(
                    ctx,
                    residual,
                    x,
                    weight,
                    self.cfg.hidden_size,
                    out,
                    residual_out,
                )?;
            }
            KernelKind::QkNorm => {
                let attn_heads = self.cfg.num_attention_heads;
                let kv_heads = self.cfg.num_key_value_heads;
                let head_dim = self.cfg.head_dim;
                let q = unsafe { api::zeros::<f16>(&[attn_heads, head_dim]).execute(ctx)? };
                let k = unsafe { api::zeros::<f16>(&[kv_heads, head_dim]).execute(ctx)? };
                let q_w = self.layers[0].weights.q_norm.clone();
                let k_w = self.layers[0].weights.k_norm.clone();
                let mut out = alloc_f16_ctx(ctx, &[attn_heads + kv_heads, head_dim])?;
                unsafe {
                    qk_norm_f16(
                        &q,
                        &k,
                        &*q_w,
                        &*k_w,
                        (&mut out).partition([1, head_dim]),
                        self.cfg.rms_norm_eps,
                        attn_heads as i32,
                    )
                    .generics(vec![head_dim.to_string(), RMS_BLOCK.to_string()])
                    .execute(ctx)?
                };
            }
            KernelKind::QkRope => {
                let attn_heads = self.cfg.num_attention_heads;
                let kv_heads = self.cfg.num_key_value_heads;
                let head_dim = self.cfg.head_dim;
                let q = unsafe { api::zeros::<f16>(&[1, attn_heads, head_dim]).execute(ctx)? };
                let k = unsafe { api::zeros::<f16>(&[1, kv_heads, head_dim]).execute(ctx)? };
                let pos = unsafe { api::zeros::<u32>(&[1]).execute(ctx)? };
                let mut out = alloc_f16_ctx(ctx, &[1, attn_heads + kv_heads, head_dim])?;
                unsafe {
                    qk_rope_dynpos_f16(
                        &q,
                        &k,
                        &*self.inv_freq,
                        &pos,
                        (&mut out).partition([1, 1, head_dim / 2]),
                        attn_heads as i32,
                    )
                    .generics(vec![
                        head_dim.to_string(),
                        (head_dim / 2).to_string(),
                        QK_ROPE_LATENCY_DEFAULT.to_string(),
                    ])
                    .execute(ctx)?
                };
            }
            KernelKind::QkNormRopeKvPrefill => {
                if self.max_seq_len >= 2 {
                    let seq_len = 2usize;
                    let attn_heads = self.cfg.num_attention_heads;
                    let kv_heads = self.cfg.num_key_value_heads;
                    let head_dim = self.cfg.head_dim;
                    let q = Arc::new(unsafe {
                        api::zeros::<f16>(&[seq_len, attn_heads, head_dim]).execute(ctx)?
                    });
                    let k = Arc::new(unsafe {
                        api::zeros::<f16>(&[seq_len, kv_heads, head_dim]).execute(ctx)?
                    });
                    let v = Arc::new(unsafe {
                        api::zeros::<f16>(&[seq_len, kv_heads, head_dim]).execute(ctx)?
                    });
                    let out = alloc_f16_ctx(ctx, &[seq_len, attn_heads, head_dim])?;
                    let _ = self.execute_qk_norm_rope_kv_prefill_op_ctx(
                        ctx,
                        0,
                        q,
                        k,
                        v,
                        &PositionInput::Host(0),
                        out,
                    )?;
                }
            }
            KernelKind::QkNormRopeKvDecode => {
                let attn_heads = self.cfg.num_attention_heads;
                let kv_heads = self.cfg.num_key_value_heads;
                let head_dim = self.cfg.head_dim;
                let qkv_width = (attn_heads + 2 * kv_heads) * head_dim;
                let qkv = unsafe { api::zeros::<f16>(&[1, qkv_width]).execute(ctx)? };
                let q_out = unsafe {
                    api::zeros::<f16>(&[1, attn_heads + kv_heads, head_dim]).execute(ctx)?
                };
                let k_cache = unsafe {
                    api::zeros::<f16>(&[kv_heads, self.max_seq_len, head_dim]).execute(ctx)?
                };
                let v_cache = unsafe {
                    api::zeros::<f16>(&[kv_heads, self.max_seq_len, head_dim]).execute(ctx)?
                };
                let pos = unsafe { api::zeros::<u32>(&[1]).execute(ctx)? };
                let w = &self.layers[0].weights;
                unsafe {
                    qk_norm_rope_kv_decode_raw_f16(
                        qkv.device_pointer().clone(),
                        w.q_norm.device_pointer().clone(),
                        w.k_norm.device_pointer().clone(),
                        self.inv_freq.device_pointer().clone(),
                        q_out.device_pointer().clone(),
                        k_cache.device_pointer().clone(),
                        v_cache.device_pointer().clone(),
                        &pos,
                        self.cfg.rms_norm_eps,
                        attn_heads as i32,
                        kv_heads as i32,
                    )
                    .generics(vec![
                        head_dim.to_string(),
                        (head_dim / 2).to_string(),
                        self.max_seq_len.to_string(),
                    ])
                    .grid(((attn_heads + kv_heads) as u32, 2u32, 1u32))
                    .execute(ctx)?
                };
            }
            KernelKind::ArgmaxReduceBlocks => {
                let argmax_block = env_usize_or("GROUT_ARGMAX_BLOCK", ARGMAX_BLOCK);
                let num_blocks = (self.cfg.vocab_size + argmax_block - 1) / argmax_block;
                let reduce_block = argmax_reduce_block_size(num_blocks);
                let block_max = unsafe { api::zeros::<f32>(&[num_blocks]).execute(ctx)? };
                let block_idx = unsafe { api::zeros::<u32>(&[num_blocks]).execute(ctx)? };
                let mut out = unsafe { api::zeros::<u32>(&[1]).execute(ctx)? };
                unsafe {
                    argmax_reduce_blocks_to_u32(
                        &block_max,
                        &block_idx,
                        (&mut out).partition([1]),
                        num_blocks as i32,
                    )
                    .generics(vec![reduce_block.to_string()])
                    .execute(ctx)?
                };
            }
        }

        self.kernel_warm_registry.mark_warmed(kind);
        Ok(())
    }

    pub async fn reset_cache(&mut self) -> Result<()> {
        // Invalidate cached decode graph — new caches will have different pointers.
        self.decode_runner = None;
        for layer in &mut self.layers {
            layer.state.k_cache = Some(Arc::new(
                api::zeros::<f16>(&[
                    self.cfg.num_key_value_heads,
                    self.max_seq_len,
                    self.cfg.head_dim,
                ])
                .await
                .map_err(|e| anyhow::anyhow!("alloc k_cache failed: {e:?}"))?,
            ));
            layer.state.v_cache = Some(Arc::new(
                api::zeros::<f16>(&[
                    self.cfg.num_key_value_heads,
                    self.max_seq_len,
                    self.cfg.head_dim,
                ])
                .await
                .map_err(|e| anyhow::anyhow!("alloc v_cache failed: {e:?}"))?,
            ));
        }
        Ok(())
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub async fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<GenerationOutput> {
        let debug_logits = std::env::var("GROUT_DEBUG_LOGITS").ok().as_deref() == Some("1");
        if self.profile_enabled {
            self.active_profile = Some(RunProfile::new());
        } else {
            self.active_profile = None;
        }
        let prompt = self.maybe_apply_chat_template(prompt);
        let prompt_ids = self.encode_prompt(&prompt)?;
        if prompt_ids.is_empty() {
            bail!("prompt produced no tokens");
        }
        ensure!(
            prompt_ids.len() + max_new_tokens <= self.max_seq_len,
            "requested total sequence length exceeds max_seq_len={}",
            self.max_seq_len
        );

        self.warm_all_kernels().await?;

        let use_cuda_graph_decode =
            env_bool_or("GROUT_CUDA_GRAPH_DECODE", true) && max_new_tokens > 0;

        // Reset KV caches: if we have a cached decode graph, zero its caches
        // in-place and lend them to layer state for prefill. Otherwise allocate fresh.
        if let Some(runner) = &mut self.decode_runner {
            runner.zero_kv_caches()?;
            runner.lend_kv_caches_to_layers(&mut self.layers);
        } else {
            self.reset_cache().await?;
        }

        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let step_start = Instant::now();
        let mut logits = self.step_seq_await(&prompt_ids, 0).await?;
        self.profile_step(step_start.elapsed(), false);
        let prompt_elapsed = prompt_start.elapsed();
        if debug_logits {
            let logits_host = logits.clone().to_host_vec().await?;
            eprintln!(
                "prefill logits: {}",
                summarize_logits(&logits_host, &self.tokenizer)?
            );
        }

        let mut cur_pos = prompt_ids.len();

        // Reclaim caches from layer state and reuse cached graph, or build new.
        if self.decode_runner.is_some() {
            // Reclaim KV caches back from layer state into the runner.
            let runner = self.decode_runner.as_mut().unwrap();
            runner.reclaim_kv_caches_from_layers(&mut self.layers)?;
        } else if use_cuda_graph_decode {
            // First call: build the decode graph (captures KV cache pointers).
            let stream = with_context(|ctx| value(ctx.get_cuda_stream().clone())).await?;
            match self.build_decode_graph_scope(&stream, cur_pos) {
                Ok(runner) => {
                    self.decode_runner = Some(runner);
                }
                Err(err) => {
                    eprintln!(
                        "warning: failed to initialize CUDA decode graph ({err:#}); falling back"
                    );
                }
            }
        };

        let decode_start = Instant::now();
        let mut generated_ids: Vec<u32> = Vec::new();
        let mut rng = rand::thread_rng();

        // Greedy + graph path: the decode graph performs argmax in-graph and
        // writes the next token back into token_ids_device, eliminating the
        // per-step host roundtrip (host argmax reduce + H2D of token). Only
        // `position` is still copied H2D each step.
        let use_ingraph_argmax = !self.do_sample && !debug_logits && self.decode_runner.is_some();

        if use_ingraph_argmax {
            let first_next = self.argmax_device(logits.clone()).await? as u32;
            if !self.eos_token_ids.contains(&first_next) && max_new_tokens > 0 {
                generated_ids.push(first_next);
                {
                    let runner = self.decode_runner.as_mut().unwrap();
                    runner.seed_token(first_next)?;
                }
                let mut graph_ok = true;
                for _ in 1..max_new_tokens {
                    let step_start = Instant::now();
                    let step_res = self.decode_runner.as_mut().unwrap().launch_step(cur_pos);
                    let next = match step_res {
                        Ok(tok) => tok,
                        Err(err) => {
                            eprintln!(
                                "warning: CUDA decode graph launch failed at pos {} ({err:#}); falling back",
                                cur_pos
                            );
                            graph_ok = false;
                            break;
                        }
                    };
                    self.profile_step(step_start.elapsed(), true);
                    cur_pos += 1;
                    if self.eos_token_ids.contains(&next) {
                        break;
                    }
                    generated_ids.push(next);
                }
                if !graph_ok {
                    // Fallback: drop the graph and finish remaining tokens on
                    // the step-sequential path. Logits for continuation are
                    // not available (runner path discards them), so restart
                    // from the last generated token via step_seq.
                    self.decode_runner = None;
                    let last = *generated_ids.last().unwrap_or(&first_next);
                    let step_token = [last];
                    logits = self.step_seq_await(&step_token, cur_pos).await?;
                    cur_pos += 1;
                    let remaining = max_new_tokens.saturating_sub(generated_ids.len());
                    for _ in 0..remaining {
                        let next = if self.use_device_argmax {
                            self.argmax_device(logits.clone()).await? as u32
                        } else {
                            let lh = logits.clone().to_host_vec().await?;
                            argmax_f16(&lh) as u32
                        };
                        if self.eos_token_ids.contains(&next) {
                            break;
                        }
                        generated_ids.push(next);
                        let step_token = [next];
                        logits = self.step_seq_await(&step_token, cur_pos).await?;
                        cur_pos += 1;
                    }
                }
            }
        } else {
            for _ in 0..max_new_tokens {
                let next = if self.do_sample {
                    let logits_host = logits.clone().to_host_vec().await?;
                    self.sample_next(&logits_host, &mut rng)? as u32
                } else if self.use_device_argmax {
                    self.argmax_device(logits.clone()).await? as u32
                } else {
                    let logits_host = logits.clone().to_host_vec().await?;
                    argmax_f16(&logits_host) as u32
                };
                if self.eos_token_ids.contains(&next) {
                    break;
                }
                generated_ids.push(next);
                let step_start = Instant::now();
                if let Some(runner) = self.decode_runner.as_mut() {
                    let graph_res =
                        runner
                            .launch_step_with_logits(next, cur_pos)
                            .and_then(|new_logits| {
                                runner.synchronize()?;
                                Ok(new_logits)
                            });
                    match graph_res {
                        Ok(new_logits) => {
                            logits = new_logits;
                        }
                        Err(err) => {
                            eprintln!(
                                "warning: CUDA decode graph launch failed at pos {} ({err:#}); falling back",
                                cur_pos
                            );
                            self.decode_runner = None;
                            let step_token = [next];
                            logits = self.step_seq_await(&step_token, cur_pos).await?;
                        }
                    }
                } else {
                    let step_token = [next];
                    logits = self.step_seq_await(&step_token, cur_pos).await?;
                }
                self.profile_step(step_start.elapsed(), true);
                if debug_logits {
                    let logits_host = logits.clone().to_host_vec().await?;
                    eprintln!(
                        "decode@{} logits: {}",
                        cur_pos,
                        summarize_logits(&logits_host, &self.tokenizer)?
                    );
                }
                cur_pos += 1;
            }
        }
        let decode_elapsed = decode_start.elapsed();
        let total_elapsed = total_start.elapsed();
        let profile_report = self.active_profile.take().map(|p| p.render());

        let text = if generated_ids.is_empty() {
            String::new()
        } else {
            self.tokenizer
                .decode(&generated_ids, false)
                .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?
        };

        Ok(GenerationOutput {
            text,
            prompt_tokens: prompt_ids.len(),
            generated_tokens: generated_ids.len(),
            prompt_elapsed,
            decode_elapsed,
            total_elapsed,
            profile_report,
        })
    }

    fn maybe_apply_chat_template(&self, prompt: &str) -> String {
        if !self.use_chat_template || prompt.contains("<|im_start|>") {
            return prompt.to_string();
        }
        // Match Qwen3 tokenizer template with enable_thinking=false for cleaner direct answers.
        format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    }

    fn sample_next<R: Rng + ?Sized>(&self, logits: &[f16], rng: &mut R) -> Result<usize> {
        if logits.is_empty() {
            bail!("cannot sample from empty logits");
        }
        if self.temperature <= 1.0e-5 {
            return Ok(argmax_f16(logits));
        }

        let inv_temp = 1.0f32 / self.temperature;
        let mut scores: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.to_f32() * inv_temp))
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        if self.top_k > 0 && self.top_k < scores.len() {
            scores.truncate(self.top_k);
        }

        let max_score = scores
            .first()
            .map(|x| x.1)
            .context("scores unexpectedly empty")?;
        let mut probs: Vec<(usize, f64)> = scores
            .into_iter()
            .map(|(i, s)| (i, ((s - max_score) as f64).exp()))
            .collect();

        if self.top_p < 1.0 {
            probs.sort_by(|a, b| b.1.total_cmp(&a.1));
            let total: f64 = probs.iter().map(|x| x.1).sum();
            if total.is_finite() && total > 0.0 {
                let mut kept = Vec::new();
                let mut cum = 0.0f64;
                for (idx, p) in probs {
                    let pn = p / total;
                    cum += pn;
                    kept.push((idx, p));
                    if cum >= self.top_p as f64 && !kept.is_empty() {
                        break;
                    }
                }
                probs = kept;
            }
        }

        let sum_p: f64 = probs.iter().map(|x| x.1).sum();
        if !sum_p.is_finite() || sum_p <= 0.0 {
            return Ok(argmax_f16(logits));
        }
        let mut threshold = rng.r#gen::<f64>() * sum_p;
        for (idx, p) in probs {
            threshold -= p;
            if threshold <= 0.0 {
                return Ok(idx);
            }
        }
        Ok(argmax_f16(logits))
    }

    async fn step_seq_await(
        &mut self,
        token_ids: &[u32],
        position_start: usize,
    ) -> Result<Arc<Tensor<f16>>> {
        with_context(|ctx| value(self.step_seq_await_ctx(ctx, token_ids, position_start))).await?
    }

    fn step_seq_await_ctx(
        &mut self,
        ctx: &ExecutionContext,
        token_ids: &[u32],
        position_start: usize,
    ) -> Result<Arc<Tensor<f16>>> {
        ensure!(!token_ids.is_empty(), "step_seq expects at least one token");
        let seqlen = token_ids.len();
        ensure!(
            position_start + seqlen <= self.max_seq_len,
            "position range [{}..{}) exceeds max_seq_len {}",
            position_start,
            position_start + seqlen,
            self.max_seq_len
        );

        let graph = self.get_or_build_step_graph(seqlen)?;

        let mut pool = if let Some(pool) = self.step_pool_cache.remove(&seqlen) {
            pool
        } else {
            TensorPool::from_plan_ctx(ctx, &graph.pool_plan)?
        };

        let result =
            self.execute_step_graph_ctx(ctx, graph.as_ref(), &mut pool, token_ids, position_start);
        self.step_pool_cache.insert(seqlen, pool);
        result
    }

    fn get_or_build_step_graph(&mut self, seqlen: usize) -> Result<Arc<StepGraph>> {
        ensure!(seqlen > 0, "step_seq expects at least one token");
        if let Some(graph) = self.step_graph_cache.get(&seqlen) {
            return Ok(graph.clone());
        }
        let graph = Arc::new(self.build_step_graph(seqlen)?);
        self.step_graph_cache.insert(seqlen, graph.clone());
        Ok(graph)
    }

    // Legacy: StepGraph + CudaGraph::capture approach. Retained for reference.
    // The new scope-based approach is build_decode_graph_scope below.
    //
    // fn build_decode_graph_runner_ctx_legacy(
    //     &mut self,
    //     ctx: &ExecutionContext,
    //     position_start: usize,
    // ) -> Result<DecodeCudaGraphRunner> {
    //     cuda_async::device_context::with_global_device_context(ctx.get_device_id(), |_| ())?;
    //     ensure!(
    //         position_start <= u32::MAX as usize,
    //         "position_start {} exceeds u32 range",
    //         position_start
    //     );
    //     let seqlen = 1usize;
    //     let graph = self.get_or_build_step_graph(seqlen)?;
    //     let mut pool = TensorPool::from_plan_ctx(ctx, &graph.pool_plan)?;
    //     let token_host = [0u32; 1];
    //     let position_host = [position_start as u32; 1];
    //     let token_init = Arc::new(vec![token_host[0]]);
    //     let position_init = Arc::new(vec![position_host[0]]);
    //     let token_ids_device =
    //         Arc::new(unsafe { api::copy_host_vec_to_device(&token_init).execute(ctx)? });
    //     let position_device =
    //         Arc::new(unsafe { api::copy_host_vec_to_device(&position_init).execute(ctx)? });
    //     let mut prime_logits = Some(alloc_f16_ctx(ctx, &graph.spec(graph.final_value).shape)?);
    //     let _ = self.execute_step_graph_decode_capture_ctx(
    //         ctx, graph.as_ref(), &mut pool,
    //         token_ids_device.clone(), position_device.clone(), &mut prime_logits,
    //     )?;
    //     ctx.get_cuda_stream().synchronize()
    //         .map_err(|e| anyhow::anyhow!("stream synchronize failed: {e:?}"))?;
    //     let mut final_logits = Some(alloc_f16_ctx(ctx, &graph.spec(graph.final_value).shape)?);
    //     let stream = ctx.get_cuda_stream().clone();
    //     let decode_op = with_context(|cap_ctx| {
    //         let logits = self.execute_step_graph_decode_capture_ctx(
    //             cap_ctx, graph.as_ref(), &mut pool,
    //             token_ids_device.clone(), position_device.clone(), &mut final_logits,
    //         );
    //         value(logits)
    //     });
    //     let mut graph_exec = CudaGraph::capture(stream, decode_op)
    //         .map_err(|e| anyhow::anyhow!("CudaGraph capture failed: {e:?}"))?;
    //     let logits_result = graph_exec.take_output()
    //         .context("decode graph capture produced no output")?;
    //     let logits = logits_result?;
    //     Ok(DecodeCudaGraphRunner { graph: graph_exec, _pool_keepalive: pool,
    //         token_host, position_host, token_ids_device, position_device, logits })
    // }

    /// Build a decode CUDA graph using `CudaGraph::scope`.
    ///
    /// This replaces the old StepGraph IR executor + `CudaGraph::capture` approach
    /// with a direct imperative scope capture. All buffers are pre-allocated and
    /// the graph replays into the same device pointers.
    fn build_decode_graph_scope(
        &mut self,
        stream: &Arc<cuda_core::Stream>,
        position_start: usize,
    ) -> Result<DecodeCudaGraphRunner> {
        ensure!(
            position_start <= u32::MAX as usize,
            "position_start {} exceeds u32 range",
            position_start
        );

        let d = self.cfg.hidden_size;
        let attn_heads = self.cfg.num_attention_heads;
        let kv_heads = self.cfg.num_key_value_heads;
        let head_dim = self.cfg.head_dim;
        let inter_size = self.cfg.intermediate_size;
        let attn_width = attn_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let qkv_width = attn_width + 2 * kv_width;
        let vocab_size = self.cfg.vocab_size;
        let eps = self.cfg.rms_norm_eps;
        let num_layers = self.cfg.num_hidden_layers;
        let max_seq_len = self.max_seq_len;
        let qk_scale = 1.0f32 / (head_dim as f32).sqrt();
        let query_group_size = self.cfg.num_kv_groups() as i32;
        let attn_bn = env_usize_or("GROUT_ATTN_BN_DECODE", ATTN_BN_DECODE);
        let use_flash_decode = env_bool_or("GROUT_FLASH_DECODE", false);
        // BLOCK_SIZE ablation knob for decode add_rms_norm only (plain
        // rms_norm_f16 and qk_norm_f16 stay at RMS_BLOCK because they're
        // also invoked at N=head_dim=128).
        let rms_block = env_usize_or("GROUT_RMS_BLOCK", ADD_RMS_DECODE_BLOCK);
        // Tile-tuning knobs for kernels that fire in decode.
        let embed_block = env_usize_or("GROUT_EMBED_BLOCK", EMBED_BLOCK);

        // ── Allocate input tensors ─────────────────────────────────────────
        let mut token_ids: Tensor<u32> = api::zeros(&[1])
            .sync_on(stream)
            .map_err(|e| anyhow::anyhow!("alloc token_ids failed: {e:?}"))?;
        // The eager prime pass below runs real kernels against the same KV
        // caches populated by prefill. Seed the device position to the first
        // decode slot so kernel warmup does not overwrite prompt cache slot 0.
        let position_init = Arc::new(vec![position_start as u32]);
        let position: Tensor<u32> = api::copy_host_vec_to_device(&position_init)
            .sync_on(stream)
            .map_err(|e| anyhow::anyhow!("alloc position failed: {e:?}"))?;
        let s_kv_init = Arc::new(vec![(position_start + 1) as i32]);
        let s_kv_device: Tensor<i32> = api::copy_host_vec_to_device(&s_kv_init)
            .sync_on(stream)
            .map_err(|e| anyhow::anyhow!("alloc s_kv_device failed: {e:?}"))?;

        // ── Allocate decode buffers ────────────────────────────────────────
        let alloc = |shape: &[usize]| -> Result<Tensor<f16>> {
            api::zeros::<f16>(shape)
                .sync_on(stream)
                .map_err(|e| anyhow::anyhow!("alloc failed: {e:?}"))
        };
        let lm_head_argmax_rows = 64usize;
        let lm_head_argmax_block_k = 32usize;
        let debug_logits_enabled = std::env::var("GROUT_DEBUG_LOGITS").ok().as_deref() == Some("1");
        let use_fused_lm_head_argmax = env_bool_or("GROUT_FUSED_LM_HEAD_ARGMAX", false)
            && self.use_device_argmax
            && !self.do_sample
            && !debug_logits_enabled
            && vocab_size % lm_head_argmax_rows == 0
            && d % lm_head_argmax_block_k == 0;
        let argmax_block = if use_fused_lm_head_argmax {
            lm_head_argmax_rows
        } else {
            env_usize_or("GROUT_ARGMAX_BLOCK", ARGMAX_BLOCK)
        };
        let num_argmax_blocks = (vocab_size + argmax_block - 1) / argmax_block;
        let argmax_reduce_block = argmax_reduce_block_size(num_argmax_blocks);

        // Split-K decode attention config (GROUT_FMHA_SPLIT_KV=1). These are
        // used to size scratch buffers regardless of whether the split path is
        // active at capture time — allocating them unconditionally keeps the
        // DecodeBuffers layout stable and costs ~64 KiB of VRAM for Qwen3.
        let fmha_group_size = attn_heads / kv_heads; // Qwen3: 32/8 = 4
        // Default tuned at tg=512 (BN=32/NKS=16 and BN=32/NKS=8 within
        // noise; picked 8 to keep short-kv cases gentle). See
        // FMHA_NUM_KV_SPLITS_DEFAULT.
        let fmha_num_kv_splits =
            env_usize_or("GROUT_FMHA_NUM_KV_SPLITS", FMHA_NUM_KV_SPLITS_DEFAULT);
        let fmha_decode_latency =
            env_usize_or("GROUT_FMHA_DECODE_LATENCY", FMHA_DECODE_LATENCY_DEFAULT);
        let fmha_decode_occupancy =
            env_usize_hint_or("GROUT_FMHA_DECODE_OCCUPANCY", FMHA_DECODE_OCCUPANCY_DEFAULT);
        let use_fmha_split_kv = !use_flash_decode && env_bool_or("GROUT_FMHA_SPLIT_KV", true);
        let fmha_merge_chunk_d =
            env_usize_or("GROUT_FMHA_MERGE_CHUNK_D", FMHA_MERGE_CHUNK_D_DEFAULT);
        let fmha_merge_latency =
            env_usize_or("GROUT_FMHA_MERGE_LATENCY", FMHA_MERGE_LATENCY_DEFAULT);
        let qk_rope_latency = env_usize_or("GROUT_QK_ROPE_LATENCY", QK_ROPE_LATENCY_DEFAULT);
        let fuse_qk_rope_kv_decode = env_bool_or("GROUT_FUSED_QK_ROPE_KV_DECODE", true);
        let qk_rope_occupancy =
            env_usize_hint_or("GROUT_QK_ROPE_OCCUPANCY", QK_ROPE_OCCUPANCY_DEFAULT);
        let qk_rope_cga = env_bool_hint_or("GROUT_QK_ROPE_CGA", QK_ROPE_CGA_DEFAULT);
        let kv_cache_dyn_chunk_d =
            env_usize_or("GROUT_KV_CACHE_DYN_CHUNK_D", KV_CACHE_DYN_CHUNK_D_DEFAULT);
        // NOTE: on sm_120 `num_cta_in_cga` and `occupancy` appear to be
        // mutually exclusive — setting both makes the CGA hint a no-op.
        // When GROUT_QK_ROPE_CGA=1 we ONLY set num_cta_in_cga(2) and
        // drop occupancy; otherwise we set occupancy(qk_rope_occupancy).
        let qk_rope_compile_opts = || {
            if qk_rope_cga == Some(true) {
                CompileOptions::default().num_cta_in_cga(2)
            } else if let Some(occupancy) = qk_rope_occupancy {
                CompileOptions::default().occupancy(occupancy as i32)
            } else {
                CompileOptions::default()
            }
        };
        let mut bufs = DecodeBuffers {
            hidden: alloc(&[1, d])?,
            normed: alloc(&[1, d])?,
            qkv: alloc(&[1, qkv_width])?,
            qk_norm_flat: alloc(&[attn_heads + kv_heads, head_dim])?,
            qk_rope: alloc(&[1, attn_heads + kv_heads, head_dim])?,
            attn_out: if use_fmha_split_kv {
                alloc(&[kv_heads, fmha_group_size, head_dim])?
            } else {
                alloc(&[1, attn_heads, head_dim])?
            },
            attn_proj: alloc(&[1, d])?,
            ff_normed: alloc(&[1, d])?,
            hidden_after_attn: alloc(&[1, d])?,
            gate_up: alloc(&[1, 2 * inter_size])?,
            ff: alloc(&[1, inter_size])?,
            ff_down: alloc(&[1, d])?,
            logits: alloc(&[vocab_size])?,
            lse_scratch: api::zeros::<f32>(&[attn_heads])
                .sync_on(stream)
                .map_err(|e| anyhow::anyhow!("alloc lse_scratch failed: {e:?}"))?,
            argmax_block_max: api::zeros::<f32>(&[num_argmax_blocks])
                .sync_on(stream)
                .map_err(|e| anyhow::anyhow!("alloc argmax_block_max failed: {e:?}"))?,
            argmax_block_idx: api::zeros::<u32>(&[num_argmax_blocks])
                .sync_on(stream)
                .map_err(|e| anyhow::anyhow!("alloc argmax_block_idx failed: {e:?}"))?,
            // Flat layout [kv_heads, NUM_KV_SPLITS * GROUP, D] (att) and
            // [kv_heads, NUM_KV_SPLITS * GROUP] (lse) — split-major. See
            // comments in fmha_decode_gqa_split / splitk_reduce_merge.
            fmha_att_partial: api::zeros::<f16>(&[
                kv_heads,
                fmha_num_kv_splits * fmha_group_size,
                head_dim,
            ])
            .sync_on(stream)
            .map_err(|e| anyhow::anyhow!("alloc fmha_att_partial failed: {e:?}"))?,
            fmha_lse_partial: api::zeros::<f32>(&[kv_heads, fmha_num_kv_splits * fmha_group_size])
                .sync_on(stream)
                .map_err(|e| anyhow::anyhow!("alloc fmha_lse_partial failed: {e:?}"))?,
        };

        // ── Extract KV caches from layer state (Arc → owned Tensor) ───────
        // The scope needs &mut access to write into caches, so we unwrap
        // them from Arc. After capture, they stay alive in the runner.
        // On failure, we restore them back into layer state.
        let mut kv_caches: Vec<(Tensor<f16>, Tensor<f16>)> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let layer = &mut self.layers[layer_idx];
            let k_arc = layer
                .state
                .k_cache
                .take()
                .context("missing k_cache in layer state")?;
            let v_arc = layer
                .state
                .v_cache
                .take()
                .context("missing v_cache in layer state")?;
            let k = Arc::try_unwrap(k_arc).map_err(|_| {
                anyhow::anyhow!("k_cache Arc has multiple owners (layer {layer_idx})")
            })?;
            let v = Arc::try_unwrap(v_arc).map_err(|_| {
                anyhow::anyhow!("v_cache Arc has multiple owners (layer {layer_idx})")
            })?;
            kv_caches.push((k, v));
        }

        // Run prime pass + scope capture.
        // On failure, restore KV caches so the fallback path can use them.
        let scope_result: Result<CudaGraph<()>> = (|| -> Result<CudaGraph<()>> {
            // ── Prime pass (warm kernel caches, stream-local state) ───────────
            // Run the full forward pass eagerly once so cuBLAS handle creation,
            // tile-kernel compilation, etc. happen outside graph capture.
            {
                // Embedding (just prime with zeros — actual token loaded at launch)
                embedding_batch_f16(
                    &token_ids,
                    &*self.embed_tokens,
                    (&mut bufs.hidden).partition([1, embed_block]),
                )
                .generics(vec![d.to_string(), embed_block.to_string()])
                .sync_on(stream)
                .map_err(|e| anyhow::anyhow!("prime embedding failed: {e:?}"))?;

                for layer_idx in 0..num_layers {
                    let w = &self.layers[layer_idx].weights;
                    let (ref mut k_cache, ref mut v_cache) = kv_caches[layer_idx];

                    // Input norm (layer 0: plain RMS norm; layers 1+: fused add + RMS norm
                    // that folds in the previous layer's residual add)
                    if layer_idx == 0 {
                        let hidden_2d = bufs
                            .hidden
                            .view(&[1, d])
                            .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                        unsafe {
                            rms_norm_f16(
                                &hidden_2d,
                                &*w.input_layernorm,
                                (&mut bufs.normed).partition([1, d]),
                                eps,
                            )
                        }
                        .generics(vec![d.to_string(), RMS_BLOCK_HIDDEN.to_string()])
                        .sync_on(stream)
                        .map_err(|e| anyhow::anyhow!("prime rms_norm failed: {e:?}"))?;
                    } else {
                        unsafe {
                            add_rms_norm_decode_raw_f16(
                                bufs.hidden_after_attn.device_pointer().clone(),
                                bufs.ff_down.device_pointer().clone(),
                                w.input_layernorm.device_pointer().clone(),
                                bufs.normed.device_pointer().clone(),
                                bufs.hidden.device_pointer().clone(),
                                eps,
                            )
                        }
                        .generics(vec![d.to_string(), rms_block.to_string()])
                        .grid((1u32, 1u32, 1u32))
                        .sync_on(stream)
                        .map_err(|e| anyhow::anyhow!("prime add_rms_norm input failed: {e:?}"))?;
                    }

                    // QKV GEMV (cuBLAS uses raw device pointers + explicit m/k)
                    cublas::GemvInPlace {
                        matrix: &*w.qkv_proj,
                        vector: &bufs.normed,
                        out: &bufs.qkv,
                        m: qkv_width as i32,
                        k: d as i32,
                    }
                    .sync_on(stream)
                    .map_err(|e| anyhow::anyhow!("prime qkv gemv failed: {e:?}"))?;

                    // Slice Q, K, V — use 1D view so slices are contiguous
                    let qkv_1d = bufs
                        .qkv
                        .view(&[qkv_width])
                        .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                    let q_1d = qkv_1d
                        .slice(&[0..attn_width])
                        .map_err(|e| anyhow::anyhow!("slice failed: {e:?}"))?;
                    let k_1d = qkv_1d
                        .slice(&[attn_width..attn_width + kv_width])
                        .map_err(|e| anyhow::anyhow!("slice failed: {e:?}"))?;
                    let v_1d = qkv_1d
                        .slice(&[attn_width + kv_width..qkv_width])
                        .map_err(|e| anyhow::anyhow!("slice failed: {e:?}"))?;

                    if fuse_qk_rope_kv_decode {
                        unsafe {
                            qk_norm_rope_kv_decode_raw_f16(
                                bufs.qkv.device_pointer().clone(),
                                w.q_norm.device_pointer().clone(),
                                w.k_norm.device_pointer().clone(),
                                self.inv_freq.device_pointer().clone(),
                                bufs.qk_rope.device_pointer().clone(),
                                k_cache.device_pointer().clone(),
                                v_cache.device_pointer().clone(),
                                &position,
                                eps,
                                attn_heads as i32,
                                kv_heads as i32,
                            )
                        }
                        .generics(vec![
                            head_dim.to_string(),
                            (head_dim / 2).to_string(),
                            max_seq_len.to_string(),
                        ])
                        .grid(((attn_heads + kv_heads) as u32, 2u32, 1u32))
                        .sync_on(stream)
                        .map_err(|e| {
                            anyhow::anyhow!("prime fused qk/rope/kv decode failed: {e:?}")
                        })?;
                    } else {
                        // Q/K flat for per-head RMS norm
                        let q_flat = q_1d
                            .view(&[attn_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                        let k_flat = k_1d
                            .view(&[kv_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;

                        // Fused Q+K norm
                        unsafe {
                            qk_norm_f16(
                                &q_flat,
                                &k_flat,
                                &*w.q_norm,
                                &*w.k_norm,
                                (&mut bufs.qk_norm_flat).partition([1, head_dim]),
                                eps,
                                attn_heads as i32,
                            )
                        }
                        .generics(vec![head_dim.to_string(), RMS_BLOCK.to_string()])
                        .sync_on(stream)
                        .map_err(|e| anyhow::anyhow!("prime qk_norm failed: {e:?}"))?;

                        // Slice Q/K norm results and reshape to 3D for RoPE
                        let qk_norm_1d = bufs
                            .qk_norm_flat
                            .view(&[(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                        let q_norm_1d = qk_norm_1d
                            .slice(&[0..attn_heads * head_dim])
                            .map_err(|e| anyhow::anyhow!("slice failed: {e:?}"))?;
                        let k_norm_1d = qk_norm_1d
                            .slice(&[attn_heads * head_dim..(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("slice failed: {e:?}"))?;
                        let q_norm_3d = q_norm_1d
                            .view(&[1, attn_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                        let k_norm_3d = k_norm_1d
                            .view(&[1, kv_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;

                        // Fused Q+K RoPE
                        unsafe {
                            qk_rope_dynpos_f16(
                                &q_norm_3d,
                                &k_norm_3d,
                                &*self.inv_freq,
                                &position,
                                (&mut bufs.qk_rope).partition([1, 1, head_dim / 2]),
                                attn_heads as i32,
                            )
                        }
                        .generics(vec![
                            head_dim.to_string(),
                            (head_dim / 2).to_string(),
                            qk_rope_latency.to_string(),
                        ])
                        .compile_options(qk_rope_compile_opts())
                        .sync_on(stream)
                        .map_err(|e| anyhow::anyhow!("prime qk_rope failed: {e:?}"))?;

                        // Slice Q/K RoPE results
                        let qk_rope_1d = bufs
                            .qk_rope
                            .view(&[(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                        let rope_k_1d = qk_rope_1d
                            .slice(&[attn_heads * head_dim..(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("slice failed: {e:?}"))?;
                        let rope_k_ref = rope_k_1d
                            .view(&[1, kv_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;

                        // KV cache update (using sliced RoPE K)
                        let v_3d = v_1d
                            .view(&[1, kv_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                        unsafe {
                            kv_cache_update_seq_dynpos_f16(
                                &rope_k_ref,
                                &v_3d,
                                (k_cache).partition([1, max_seq_len, kv_cache_dyn_chunk_d]),
                                (v_cache).partition([1, max_seq_len, kv_cache_dyn_chunk_d]),
                                &position,
                                1i32,
                            )
                        }
                        .generics(vec![
                            head_dim.to_string(),
                            kv_cache_dyn_chunk_d.to_string(),
                            max_seq_len.to_string(),
                        ])
                        .sync_on(stream)
                        .map_err(|e| anyhow::anyhow!("prime kv_cache_update failed: {e:?}"))?;
                    }

                    // Attention (Q from fused RoPE output)
                    let qk_rope_1d_attn = bufs
                        .qk_rope
                        .view(&[(attn_heads + kv_heads) * head_dim])
                        .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                    let rope_q_1d_attn = qk_rope_1d_attn
                        .slice(&[0..attn_heads * head_dim])
                        .map_err(|e| anyhow::anyhow!("slice failed: {e:?}"))?;
                    let attn_q = rope_q_1d_attn
                        .view(&[1, attn_heads, head_dim])
                        .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                    if use_flash_decode {
                        // Flash decode: split-K grouped query attention
                        // Q grouped view: [1, attn_heads, head_dim] → [1, kv_heads, group_size, head_dim]
                        let group_size = attn_heads / kv_heads;
                        // Default 2: best in the 2026-04-20 sweep (140.6 t/s) at
                        // kv_len≈54. At longer contexts a higher split count may
                        // win — re-sweep for long-context workloads.
                        let num_kv_splits = env_usize_or("GROUT_ATTN_NUM_KV_SPLITS", 2);
                        // kv_len_per_split is a compile-time const (const generic)
                        // used as the max KV length each split will iterate over.
                        // Must cover max_seq_len / num_kv_splits worth of tokens.
                        let kv_len_per_split = (max_seq_len + num_kv_splits - 1) / num_kv_splits;
                        let tile_n = attn_bn;

                        // Q pointer from the qk_rope buffer (attn_q is a view into it)
                        let q_ptr = bufs.qk_rope.device_pointer().clone();
                        let q_str2 = head_dim as i32;
                        let q_str1 = (group_size * head_dim) as i32;
                        let q_str0 = (kv_heads * group_size * head_dim) as i32;

                        // K [kv_heads, max_seq_len, head_dim] viewed as [1, kv_heads, max_seq_len, head_dim]
                        let k_ptr = k_cache.device_pointer().clone();
                        let k_str2 = head_dim as i32;
                        let k_str1 = (max_seq_len * head_dim) as i32;
                        let k_str0 = (kv_heads * max_seq_len * head_dim) as i32;

                        // V same layout as K
                        let v_ptr = v_cache.device_pointer().clone();

                        // att_out [1, kv_heads, group_size, 1, head_dim] — same memory as attn_out [1, attn_heads, head_dim]
                        let att_ptr = bufs.attn_out.device_pointer().clone();
                        let att_str3 = head_dim as i32;
                        let att_str2 = (num_kv_splits * head_dim) as i32;
                        let att_str1 = (group_size * num_kv_splits * head_dim) as i32;
                        let att_str0 = (kv_heads * group_size * num_kv_splits * head_dim) as i32;

                        // lse_out [1, kv_heads, group_size, 1] f32 — small scratch
                        let lse_ptr = bufs.lse_scratch.device_pointer().clone();
                        let lse_str2 = num_kv_splits as i32;
                        let lse_str1 = (group_size * num_kv_splits) as i32;
                        let lse_str0 = (kv_heads * group_size * num_kv_splits) as i32;

                        // Write s_kv to device
                        let s_kv_val = (position_start + 1) as i32;
                        unsafe {
                            memcpy_htod_async(
                                s_kv_device.device_pointer().cu_deviceptr(),
                                &s_kv_val as *const i32,
                                1,
                                stream,
                            );
                        }
                        unsafe { stream.synchronize() }
                            .map_err(|e| anyhow::anyhow!("s_kv sync failed: {e:?}"))?;

                        let s_kv_dptr = s_kv_device.device_pointer().clone();
                        let grid = (1u32, kv_heads as u32, num_kv_splits as u32);
                        let generics = vec![
                            "f16".to_string(),
                            head_dim.to_string(),
                            tile_n.to_string(),
                            kv_len_per_split.to_string(),
                            group_size.to_string(),
                            group_size.to_string(), // QUERY_GROUP_TILE_SIZE = group_size
                            num_kv_splits.to_string(),
                        ];
                        unsafe {
                            attention_decode_kernel_grouped(
                                q_ptr,
                                1i32,
                                kv_heads as i32,
                                group_size as i32,
                                head_dim as i32,
                                q_str0,
                                q_str1,
                                q_str2,
                                k_ptr,
                                1i32,
                                kv_heads as i32,
                                max_seq_len as i32,
                                head_dim as i32,
                                k_str0,
                                k_str1,
                                k_str2,
                                v_ptr,
                                1i32,
                                kv_heads as i32,
                                max_seq_len as i32,
                                head_dim as i32,
                                k_str0,
                                k_str1,
                                k_str2,
                                att_ptr,
                                1i32,
                                kv_heads as i32,
                                group_size as i32,
                                num_kv_splits as i32,
                                head_dim as i32,
                                att_str0,
                                att_str1,
                                att_str2,
                                att_str3,
                                lse_ptr,
                                1i32,
                                kv_heads as i32,
                                group_size as i32,
                                num_kv_splits as i32,
                                lse_str0,
                                lse_str1,
                                lse_str2,
                                qk_scale,
                                s_kv_dptr,
                            )
                        }
                        .generics(generics)
                        .grid(grid)
                        .sync_on(stream)
                        .map_err(|e| anyhow::anyhow!("prime flash_decode failed: {e:?}"))?;
                    } else if use_fmha_split_kv {
                        // Prime split-K + merge so both JIT before scope capture.
                        // Q reshape: [1, attn_heads, head_dim] → [kv_heads, group, head_dim].
                        let rope_q_grouped = rope_q_1d_attn
                            .view(&[kv_heads, fmha_group_size, head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                        let qk_scale_f16 = f16::from_f32(qk_scale);
                        // Per-CTA partition shapes: [1, GROUP, 1, D] for att,
                        // [1, GROUP, 1] for lse. Grid = (kv_heads, num_splits).
                        unsafe {
                            fmha_decode_gqa_split(
                                &rope_q_grouped,
                                &*k_cache,
                                &*v_cache,
                                (&mut bufs.fmha_att_partial).partition([
                                    1,
                                    fmha_group_size,
                                    head_dim,
                                ]),
                                (&mut bufs.fmha_lse_partial).partition([1, fmha_group_size]),
                                qk_scale_f16,
                                &position,
                            )
                        }
                        .generics(vec![
                            fmha_group_size.to_string(),
                            attn_bn.to_string(),
                            head_dim.to_string(),
                            fmha_num_kv_splits.to_string(),
                            fmha_decode_latency.to_string(),
                        ])
                        .compile_options(compile_options_with_occupancy(fmha_decode_occupancy))
                        .sync_on(stream)
                        .map_err(|e| anyhow::anyhow!("prime fmha_split failed: {e:?}"))?;
                        unsafe {
                            splitk_reduce_merge(
                                &bufs.fmha_att_partial,
                                &bufs.fmha_lse_partial,
                                (&mut bufs.attn_out).partition([
                                    1,
                                    fmha_group_size,
                                    fmha_merge_chunk_d,
                                ]),
                            )
                        }
                        .generics(vec![
                            fmha_group_size.to_string(),
                            head_dim.to_string(),
                            fmha_merge_chunk_d.to_string(),
                            fmha_num_kv_splits.to_string(),
                            (fmha_num_kv_splits * fmha_group_size).to_string(),
                            fmha_merge_latency.to_string(),
                        ])
                        .sync_on(stream)
                        .map_err(|e| anyhow::anyhow!("prime fmha_merge failed: {e:?}"))?;
                    } else {
                        unsafe {
                            flash_attn_causal_seq_dynpos_f16(
                                &attn_q,
                                &*k_cache,
                                &*v_cache,
                                (&mut bufs.attn_out).partition([ATTN_BM_DECODE, 1, head_dim]),
                                qk_scale,
                                query_group_size,
                                &position,
                            )
                        }
                        .generics(vec![
                            ATTN_BM_DECODE.to_string(),
                            attn_bn.to_string(),
                            head_dim.to_string(),
                        ])
                        .sync_on(stream)
                        .map_err(|e| anyhow::anyhow!("prime attn failed: {e:?}"))?;
                    }

                    // O projection
                    cublas::GemvInPlace {
                        matrix: &*w.o_proj,
                        vector: &bufs.attn_out,
                        out: &bufs.attn_proj,
                        m: d as i32,
                        k: attn_width as i32,
                    }
                    .sync_on(stream)
                    .map_err(|e| anyhow::anyhow!("prime o_proj gemv failed: {e:?}"))?;

                    // Add + RMS norm
                    unsafe {
                        add_rms_norm_decode_raw_f16(
                            bufs.hidden.device_pointer().clone(),
                            bufs.attn_proj.device_pointer().clone(),
                            w.post_attention_layernorm.device_pointer().clone(),
                            bufs.ff_normed.device_pointer().clone(),
                            bufs.hidden_after_attn.device_pointer().clone(),
                            eps,
                        )
                    }
                    .generics(vec![d.to_string(), rms_block.to_string()])
                    .grid((1u32, 1u32, 1u32))
                    .sync_on(stream)
                    .map_err(|e| anyhow::anyhow!("prime add_rms_norm failed: {e:?}"))?;

                    // Gate+Up GEMV
                    cublas::GemvInPlace {
                        matrix: &*w.gate_up_proj,
                        vector: &bufs.ff_normed,
                        out: &bufs.gate_up,
                        m: (2 * inter_size) as i32,
                        k: d as i32,
                    }
                    .sync_on(stream)
                    .map_err(|e| anyhow::anyhow!("prime gate_up gemv failed: {e:?}"))?;

                    // Slice gate, up — 1D for contiguity, then reshape to 2D for kernel
                    let gu_1d = bufs
                        .gate_up
                        .view(&[2 * inter_size])
                        .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                    let gate_1d = gu_1d
                        .slice(&[0..inter_size])
                        .map_err(|e| anyhow::anyhow!("slice failed: {e:?}"))?;
                    let up_1d = gu_1d
                        .slice(&[inter_size..2 * inter_size])
                        .map_err(|e| anyhow::anyhow!("slice failed: {e:?}"))?;
                    let gate_2d = gate_1d
                        .view(&[1, inter_size])
                        .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;
                    let up_2d = up_1d
                        .view(&[1, inter_size])
                        .map_err(|e| anyhow::anyhow!("view failed: {e:?}"))?;

                    // SiLU * Up
                    silu_mul_2d_f16(
                        (&mut bufs.ff).partition([1, POINTWISE_BLOCK]),
                        &gate_2d,
                        &up_2d,
                    )
                    .generics(vec![POINTWISE_BLOCK.to_string()])
                    .sync_on(stream)
                    .map_err(|e| anyhow::anyhow!("prime silu_mul failed: {e:?}"))?;

                    // Down GEMV
                    cublas::GemvInPlace {
                        matrix: &*w.down_proj,
                        vector: &bufs.ff,
                        out: &bufs.ff_down,
                        m: d as i32,
                        k: inter_size as i32,
                    }
                    .sync_on(stream)
                    .map_err(|e| anyhow::anyhow!("prime down gemv failed: {e:?}"))?;

                    // Residual add is deferred to the next layer's input norm
                    // (or the final epilogue norm for the last layer).
                }

                // Final fused add + RMS norm: fold last layer's residual add into final norm
                unsafe {
                    add_rms_norm_decode_raw_f16(
                        bufs.hidden_after_attn.device_pointer().clone(),
                        bufs.ff_down.device_pointer().clone(),
                        self.norm.device_pointer().clone(),
                        bufs.normed.device_pointer().clone(),
                        bufs.hidden.device_pointer().clone(),
                        eps,
                    )
                }
                .generics(vec![d.to_string(), rms_block.to_string()])
                .grid((1u32, 1u32, 1u32))
                .sync_on(stream)
                .map_err(|e| anyhow::anyhow!("prime final add_rms_norm failed: {e:?}"))?;

                if use_fused_lm_head_argmax {
                    unsafe {
                        lm_head_argmax_blocks_f16(
                            &*self.lm_head,
                            &bufs.normed,
                            (&mut bufs.argmax_block_max).partition([1]),
                            (&mut bufs.argmax_block_idx).partition([1]),
                            vocab_size as i32,
                        )
                    }
                    .generics(vec![d.to_string()])
                    .sync_on(stream)
                    .map_err(|e| anyhow::anyhow!("prime fused lm_head_argmax failed: {e:?}"))?;
                } else {
                    // LM head GEMV
                    cublas::GemvInPlace {
                        matrix: &*self.lm_head,
                        vector: &bufs.normed,
                        out: &bufs.logits,
                        m: vocab_size as i32,
                        k: d as i32,
                    }
                    .sync_on(stream)
                    .map_err(|e| anyhow::anyhow!("prime lm_head gemv failed: {e:?}"))?;

                    // Prime two-stage argmax kernels so they're JIT'd before graph capture.
                    let logits_flat_prime = bufs
                        .logits
                        .view(&[vocab_size])
                        .map_err(|e| anyhow::anyhow!("view logits for prime failed: {e:?}"))?;
                    argmax_blocks_f16(
                        &logits_flat_prime,
                        (&mut bufs.argmax_block_max).partition([1]),
                        (&mut bufs.argmax_block_idx).partition([1]),
                        vocab_size as i32,
                    )
                    .generics(vec![argmax_block.to_string()])
                    .sync_on(stream)
                    .map_err(|e| anyhow::anyhow!("prime argmax_blocks failed: {e:?}"))?;
                }
                argmax_reduce_blocks_to_u32(
                    &bufs.argmax_block_max,
                    &bufs.argmax_block_idx,
                    (&mut token_ids).partition([1]),
                    num_argmax_blocks as i32,
                )
                .generics(vec![argmax_reduce_block.to_string()])
                .sync_on(stream)
                .map_err(|e| anyhow::anyhow!("prime argmax_reduce failed: {e:?}"))?;

                unsafe { stream.synchronize() }
                    .map_err(|e| anyhow::anyhow!("prime synchronize failed: {e:?}"))?;
            }

            // ── Graph capture via CudaGraph::scope ────────────────────────────
            let graph = CudaGraph::scope(stream, |s| {
                // Embedding: token_ids → hidden
                s.record(
                    embedding_batch_f16(
                        &token_ids,
                        &*self.embed_tokens,
                        (&mut bufs.hidden).partition([1, embed_block]),
                    )
                    .generics(vec![d.to_string(), embed_block.to_string()]),
                )?;

                for layer_idx in 0..num_layers {
                    let w = &self.layers[layer_idx].weights;
                    let (ref mut k_cache, ref mut v_cache) = kv_caches[layer_idx];

                    // Input norm (layer 0: plain RMS norm; layers 1+: fused add + RMS norm)
                    if layer_idx == 0 {
                        let hidden_2d = bufs
                            .hidden
                            .view(&[1, d])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                        s.record(
                            unsafe {
                                rms_norm_f16(
                                    &hidden_2d,
                                    &*w.input_layernorm,
                                    (&mut bufs.normed).partition([1, d]),
                                    eps,
                                )
                            }
                            .generics(vec![d.to_string(), RMS_BLOCK_HIDDEN.to_string()]),
                        )?;
                    } else {
                        s.record(
                            unsafe {
                                add_rms_norm_decode_raw_f16(
                                    bufs.hidden_after_attn.device_pointer().clone(),
                                    bufs.ff_down.device_pointer().clone(),
                                    w.input_layernorm.device_pointer().clone(),
                                    bufs.normed.device_pointer().clone(),
                                    bufs.hidden.device_pointer().clone(),
                                    eps,
                                )
                            }
                            .generics(vec![d.to_string(), rms_block.to_string()])
                            .grid((1u32, 1u32, 1u32)),
                        )?;
                    }

                    // QKV GEMV: normed → qkv
                    // cuBLAS uses raw device pointers + explicit m/k, so the
                    // base tensor works regardless of its metadata shape.
                    s.record(cublas::GemvInPlace {
                        matrix: &*w.qkv_proj,
                        vector: &bufs.normed,
                        out: &bufs.qkv,
                        m: qkv_width as i32,
                        k: d as i32,
                    })?;

                    // Zero-copy slice Q, K, V from qkv.
                    // Slice in 1D so views are contiguous (avoids stride mismatch).
                    let qkv_1d = bufs
                        .qkv
                        .view(&[qkv_width])
                        .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                    let q_1d = qkv_1d
                        .slice(&[0..attn_width])
                        .map_err(|e| anyhow::anyhow!("slice q: {e:?}"))?;
                    let k_1d = qkv_1d
                        .slice(&[attn_width..attn_width + kv_width])
                        .map_err(|e| anyhow::anyhow!("slice k: {e:?}"))?;
                    let v_1d = qkv_1d
                        .slice(&[attn_width + kv_width..qkv_width])
                        .map_err(|e| anyhow::anyhow!("slice v: {e:?}"))?;

                    if fuse_qk_rope_kv_decode {
                        s.record(
                            unsafe {
                                qk_norm_rope_kv_decode_raw_f16(
                                    bufs.qkv.device_pointer().clone(),
                                    w.q_norm.device_pointer().clone(),
                                    w.k_norm.device_pointer().clone(),
                                    self.inv_freq.device_pointer().clone(),
                                    bufs.qk_rope.device_pointer().clone(),
                                    k_cache.device_pointer().clone(),
                                    v_cache.device_pointer().clone(),
                                    &position,
                                    eps,
                                    attn_heads as i32,
                                    kv_heads as i32,
                                )
                            }
                            .generics(vec![
                                head_dim.to_string(),
                                (head_dim / 2).to_string(),
                                max_seq_len.to_string(),
                            ])
                            .grid((
                                (attn_heads + kv_heads) as u32,
                                2u32,
                                1u32,
                            )),
                        )?;
                    } else {
                        // Q flat [num_heads, head_dim] for per-head RMS norm
                        let q_flat = q_1d
                            .view(&[attn_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view q_flat: {e:?}"))?;
                        let k_flat = k_1d
                            .view(&[kv_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view k_flat: {e:?}"))?;

                        // Fused Q+K norm
                        s.record(
                            unsafe {
                                qk_norm_f16(
                                    &q_flat,
                                    &k_flat,
                                    &*w.q_norm,
                                    &*w.k_norm,
                                    (&mut bufs.qk_norm_flat).partition([1, head_dim]),
                                    eps,
                                    attn_heads as i32,
                                )
                            }
                            .generics(vec![head_dim.to_string(), RMS_BLOCK.to_string()]),
                        )?;

                        // Slice Q/K norm results and reshape to 3D for RoPE
                        let qk_norm_1d = bufs
                            .qk_norm_flat
                            .view(&[(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                        let q_norm_1d = qk_norm_1d
                            .slice(&[0..attn_heads * head_dim])
                            .map_err(|e| anyhow::anyhow!("slice: {e:?}"))?;
                        let k_norm_1d = qk_norm_1d
                            .slice(&[attn_heads * head_dim..(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("slice: {e:?}"))?;
                        let q_norm_3d = q_norm_1d
                            .view(&[1, attn_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                        let k_norm_3d = k_norm_1d
                            .view(&[1, kv_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;

                        // Fused Q+K RoPE
                        s.record(
                            unsafe {
                                qk_rope_dynpos_f16(
                                    &q_norm_3d,
                                    &k_norm_3d,
                                    &*self.inv_freq,
                                    &position,
                                    (&mut bufs.qk_rope).partition([1, 1, head_dim / 2]),
                                    attn_heads as i32,
                                )
                            }
                            .generics(vec![
                                head_dim.to_string(),
                                (head_dim / 2).to_string(),
                                qk_rope_latency.to_string(),
                            ])
                            .compile_options(qk_rope_compile_opts()),
                        )?;

                        // Slice RoPE K for KV cache update
                        let qk_rope_1d_kv = bufs
                            .qk_rope
                            .view(&[(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                        let rope_k_1d = qk_rope_1d_kv
                            .slice(&[attn_heads * head_dim..(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("slice: {e:?}"))?;
                        let rope_k_3d = rope_k_1d
                            .view(&[1, kv_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;

                        // KV cache update
                        let v_3d = v_1d
                            .view(&[1, kv_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view v_3d: {e:?}"))?;
                        s.record(
                            unsafe {
                                kv_cache_update_seq_dynpos_f16(
                                    &rope_k_3d,
                                    &v_3d,
                                    (k_cache).partition([1, max_seq_len, kv_cache_dyn_chunk_d]),
                                    (v_cache).partition([1, max_seq_len, kv_cache_dyn_chunk_d]),
                                    &position,
                                    1i32,
                                )
                            }
                            .generics(vec![
                                head_dim.to_string(),
                                kv_cache_dyn_chunk_d.to_string(),
                                max_seq_len.to_string(),
                            ]),
                        )?;
                    }

                    // Attention (Q from fused RoPE output)
                    if use_flash_decode {
                        // Flash decode via KernelGraphOp wrapper
                        let group_size = attn_heads / kv_heads;
                        let num_kv_splits = env_usize_or("GROUT_ATTN_NUM_KV_SPLITS", 2);
                        let kv_len_per_split = (max_seq_len + num_kv_splits - 1) / num_kv_splits;
                        let tile_n = attn_bn;

                        let q_ptr = bufs.qk_rope.device_pointer().clone();
                        let q_str2 = head_dim as i32;
                        let q_str1 = (group_size * head_dim) as i32;
                        let q_str0 = (kv_heads * group_size * head_dim) as i32;

                        let k_ptr = k_cache.device_pointer().clone();
                        let k_str2 = head_dim as i32;
                        let k_str1 = (max_seq_len * head_dim) as i32;
                        let k_str0 = (kv_heads * max_seq_len * head_dim) as i32;

                        let v_ptr = v_cache.device_pointer().clone();

                        let att_ptr = bufs.attn_out.device_pointer().clone();
                        let att_str3 = head_dim as i32;
                        let att_str2 = (num_kv_splits * head_dim) as i32;
                        let att_str1 = (group_size * num_kv_splits * head_dim) as i32;
                        let att_str0 = (kv_heads * group_size * num_kv_splits * head_dim) as i32;

                        let lse_ptr = bufs.lse_scratch.device_pointer().clone();
                        let lse_str2 = num_kv_splits as i32;
                        let lse_str1 = (group_size * num_kv_splits) as i32;
                        let lse_str0 = (kv_heads * group_size * num_kv_splits) as i32;

                        let s_kv_dptr = s_kv_device.device_pointer().clone();
                        let grid = (1u32, kv_heads as u32, num_kv_splits as u32);
                        let generics = vec![
                            "f16".to_string(),
                            head_dim.to_string(),
                            tile_n.to_string(),
                            kv_len_per_split.to_string(),
                            group_size.to_string(),
                            group_size.to_string(),
                            num_kv_splits.to_string(),
                        ];

                        s.record(KernelGraphOp(move |ctx: &ExecutionContext| unsafe {
                            attention_decode_kernel_grouped(
                                q_ptr.clone(),
                                1i32,
                                kv_heads as i32,
                                group_size as i32,
                                head_dim as i32,
                                q_str0,
                                q_str1,
                                q_str2,
                                k_ptr.clone(),
                                1i32,
                                kv_heads as i32,
                                max_seq_len as i32,
                                head_dim as i32,
                                k_str0,
                                k_str1,
                                k_str2,
                                v_ptr.clone(),
                                1i32,
                                kv_heads as i32,
                                max_seq_len as i32,
                                head_dim as i32,
                                k_str0,
                                k_str1,
                                k_str2,
                                att_ptr.clone(),
                                1i32,
                                kv_heads as i32,
                                group_size as i32,
                                num_kv_splits as i32,
                                head_dim as i32,
                                att_str0,
                                att_str1,
                                att_str2,
                                att_str3,
                                lse_ptr.clone(),
                                1i32,
                                kv_heads as i32,
                                group_size as i32,
                                num_kv_splits as i32,
                                lse_str0,
                                lse_str1,
                                lse_str2,
                                qk_scale,
                                s_kv_dptr.clone(),
                            )
                            .generics(generics.clone())
                            .grid(grid)
                            .execute(ctx)?;
                            Ok(())
                        }))?;
                    } else if use_fmha_split_kv {
                        // Split-K + GQA decode attention (tile-IR).
                        let qk_rope_1d_attn = bufs
                            .qk_rope
                            .view(&[(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                        let rope_q_1d = qk_rope_1d_attn
                            .slice(&[0..attn_heads * head_dim])
                            .map_err(|e| anyhow::anyhow!("slice: {e:?}"))?;
                        let rope_q_grouped = rope_q_1d
                            .view(&[kv_heads, fmha_group_size, head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                        let qk_scale_f16 = f16::from_f32(qk_scale);
                        s.record(
                            unsafe {
                                fmha_decode_gqa_split(
                                    &rope_q_grouped,
                                    &*k_cache,
                                    &*v_cache,
                                    (&mut bufs.fmha_att_partial).partition([
                                        1,
                                        fmha_group_size,
                                        head_dim,
                                    ]),
                                    (&mut bufs.fmha_lse_partial).partition([1, fmha_group_size]),
                                    qk_scale_f16,
                                    &position,
                                )
                            }
                            .generics(vec![
                                fmha_group_size.to_string(),
                                attn_bn.to_string(),
                                head_dim.to_string(),
                                fmha_num_kv_splits.to_string(),
                                fmha_decode_latency.to_string(),
                            ])
                            .compile_options(compile_options_with_occupancy(fmha_decode_occupancy)),
                        )?;
                        s.record(
                            unsafe {
                                splitk_reduce_merge(
                                    &bufs.fmha_att_partial,
                                    &bufs.fmha_lse_partial,
                                    (&mut bufs.attn_out).partition([
                                        1,
                                        fmha_group_size,
                                        fmha_merge_chunk_d,
                                    ]),
                                )
                            }
                            .generics(vec![
                                fmha_group_size.to_string(),
                                head_dim.to_string(),
                                fmha_merge_chunk_d.to_string(),
                                fmha_num_kv_splits.to_string(),
                                (fmha_num_kv_splits * fmha_group_size).to_string(),
                                fmha_merge_latency.to_string(),
                            ]),
                        )?;
                    } else {
                        let qk_rope_1d_attn = bufs
                            .qk_rope
                            .view(&[(attn_heads + kv_heads) * head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                        let rope_q_1d = qk_rope_1d_attn
                            .slice(&[0..attn_heads * head_dim])
                            .map_err(|e| anyhow::anyhow!("slice: {e:?}"))?;
                        let rope_q_view = rope_q_1d
                            .view(&[1, attn_heads, head_dim])
                            .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                        s.record(
                            unsafe {
                                flash_attn_causal_seq_dynpos_f16(
                                    &rope_q_view,
                                    &*k_cache,
                                    &*v_cache,
                                    (&mut bufs.attn_out).partition([ATTN_BM_DECODE, 1, head_dim]),
                                    qk_scale,
                                    query_group_size,
                                    &position,
                                )
                            }
                            .generics(vec![
                                ATTN_BM_DECODE.to_string(),
                                attn_bn.to_string(),
                                head_dim.to_string(),
                            ]),
                        )?;
                    }

                    // O projection: attn_out → attn_proj
                    s.record(cublas::GemvInPlace {
                        matrix: &*w.o_proj,
                        vector: &bufs.attn_out,
                        out: &bufs.attn_proj,
                        m: d as i32,
                        k: attn_width as i32,
                    })?;

                    // Fused add + RMS norm: (hidden + attn_proj) → (hidden_after_attn, ff_normed)
                    s.record(
                        unsafe {
                            add_rms_norm_decode_raw_f16(
                                bufs.hidden.device_pointer().clone(),
                                bufs.attn_proj.device_pointer().clone(),
                                w.post_attention_layernorm.device_pointer().clone(),
                                bufs.ff_normed.device_pointer().clone(),
                                bufs.hidden_after_attn.device_pointer().clone(),
                                eps,
                            )
                        }
                        .generics(vec![d.to_string(), rms_block.to_string()])
                        .grid((1u32, 1u32, 1u32)),
                    )?;

                    // Gate+Up GEMV: ff_normed → gate_up
                    s.record(cublas::GemvInPlace {
                        matrix: &*w.gate_up_proj,
                        vector: &bufs.ff_normed,
                        out: &bufs.gate_up,
                        m: (2 * inter_size) as i32,
                        k: d as i32,
                    })?;

                    // Zero-copy slice gate, up — 1D for contiguity, reshape to 2D for kernel
                    let gu_1d = bufs
                        .gate_up
                        .view(&[2 * inter_size])
                        .map_err(|e| anyhow::anyhow!("view: {e:?}"))?;
                    let gate_1d = gu_1d
                        .slice(&[0..inter_size])
                        .map_err(|e| anyhow::anyhow!("slice gate: {e:?}"))?;
                    let up_1d = gu_1d
                        .slice(&[inter_size..2 * inter_size])
                        .map_err(|e| anyhow::anyhow!("slice up: {e:?}"))?;
                    let gate_2d = gate_1d
                        .view(&[1, inter_size])
                        .map_err(|e| anyhow::anyhow!("view gate_2d: {e:?}"))?;
                    let up_2d = up_1d
                        .view(&[1, inter_size])
                        .map_err(|e| anyhow::anyhow!("view up_2d: {e:?}"))?;

                    // SiLU * Up
                    s.record(
                        silu_mul_2d_f16(
                            (&mut bufs.ff).partition([1, POINTWISE_BLOCK]),
                            &gate_2d,
                            &up_2d,
                        )
                        .generics(vec![POINTWISE_BLOCK.to_string()]),
                    )?;

                    // Down GEMV: ff → ff_down
                    s.record(cublas::GemvInPlace {
                        matrix: &*w.down_proj,
                        vector: &bufs.ff,
                        out: &bufs.ff_down,
                        m: d as i32,
                        k: inter_size as i32,
                    })?;

                    // Residual add is deferred to the next layer's input norm
                    // (or the final epilogue norm for the last layer).
                } // end layer loop

                // Final fused add + RMS norm: fold last layer's residual add into final norm
                s.record(
                    unsafe {
                        add_rms_norm_decode_raw_f16(
                            bufs.hidden_after_attn.device_pointer().clone(),
                            bufs.ff_down.device_pointer().clone(),
                            self.norm.device_pointer().clone(),
                            bufs.normed.device_pointer().clone(),
                            bufs.hidden.device_pointer().clone(),
                            eps,
                        )
                    }
                    .generics(vec![d.to_string(), rms_block.to_string()])
                    .grid((1u32, 1u32, 1u32)),
                )?;

                if use_fused_lm_head_argmax {
                    s.record(
                        unsafe {
                            lm_head_argmax_blocks_f16(
                                &*self.lm_head,
                                &bufs.normed,
                                (&mut bufs.argmax_block_max).partition([1]),
                                (&mut bufs.argmax_block_idx).partition([1]),
                                vocab_size as i32,
                            )
                        }
                        .generics(vec![d.to_string()]),
                    )?;
                } else {
                    // LM head GEMV: normed → logits
                    s.record(cublas::GemvInPlace {
                        matrix: &*self.lm_head,
                        vector: &bufs.normed,
                        out: &bufs.logits,
                        m: vocab_size as i32,
                        k: d as i32,
                    })?;

                    // In-graph greedy argmax: logits → token_ids[0]. Writing back to
                    // the same buffer the embedding reads means the next graph replay
                    // picks up this step's selection without any H2D copy.
                    let logits_flat = bufs
                        .logits
                        .view(&[vocab_size])
                        .map_err(|e| anyhow::anyhow!("view logits failed: {e:?}"))?;
                    s.record(
                        argmax_blocks_f16(
                            &logits_flat,
                            (&mut bufs.argmax_block_max).partition([1]),
                            (&mut bufs.argmax_block_idx).partition([1]),
                            vocab_size as i32,
                        )
                        .generics(vec![argmax_block.to_string()]),
                    )?;
                }
                s.record(
                    argmax_reduce_blocks_to_u32(
                        &bufs.argmax_block_max,
                        &bufs.argmax_block_idx,
                        (&mut token_ids).partition([1]),
                        num_argmax_blocks as i32,
                    )
                    .generics(vec![argmax_reduce_block.to_string()]),
                )?;

                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("CudaGraph::scope failed: {e:?}"))?;

            Ok(graph)
        })();

        match scope_result {
            Ok(graph) => {
                let logits_alias = unsafe { bufs.logits.into_shared_alias() };
                Ok(DecodeCudaGraphRunner {
                    graph,
                    token_host: [0u32; 1],
                    position_host: [position_start as u32; 1],
                    s_kv_host: [(position_start + 1) as i32; 1],
                    token_ids_device: token_ids,
                    position_device: position,
                    s_kv_device,
                    logits: logits_alias,
                    logits_valid: !use_fused_lm_head_argmax,
                    _bufs: bufs,
                    kv_caches,
                })
            }
            Err(err) => {
                // Restore KV caches so the non-graph fallback path works.
                for (layer_idx, (k, v)) in kv_caches.into_iter().enumerate() {
                    self.layers[layer_idx].state.k_cache = Some(Arc::new(k));
                    self.layers[layer_idx].state.v_cache = Some(Arc::new(v));
                }
                Err(err)
            }
        }
    }

    #[allow(dead_code)]
    fn execute_step_graph_decode_capture_ctx(
        &mut self,
        ctx: &ExecutionContext,
        graph: &StepGraph,
        pool: &mut TensorPool,
        token_ids_device: Arc<Tensor<u32>>,
        position_start: Arc<Tensor<u32>>,
        final_logits: &mut Option<Tensor<f16>>,
    ) -> Result<Arc<Tensor<f16>>> {
        let mut final_logits_policy = FinalLogitsPolicy::Preallocated(final_logits);
        self.execute_step_graph_common_ctx(
            ctx,
            graph,
            pool,
            TokenInput::Device(token_ids_device),
            PositionInput::Device(position_start),
            &mut final_logits_policy,
            false,
            false,
        )
    }

    fn build_step_graph(&self, seqlen: usize) -> Result<StepGraph> {
        ensure!(seqlen > 0, "step_seq expects at least one token");

        let mut specs = Vec::new();
        let mut ops = Vec::new();
        let v = TensorRef::Value;
        let lw = |layer_idx, slot| TensorRef::Weight(WeightRef::Layer { layer_idx, slot });

        let hidden_size = self.cfg.hidden_size;
        let attn_heads = self.cfg.num_attention_heads;
        let kv_heads = self.cfg.num_key_value_heads;
        let head_dim = self.cfg.head_dim;
        let inter_size = self.cfg.intermediate_size;
        let attn_width = attn_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let fuse_qk_rope_kv_prefill =
            seqlen > 1 && env_bool_or("GROUT_FUSED_QK_ROPE_KV_PREFILL", true);

        let mut hidden = push_value(&mut specs, vec![seqlen, hidden_size]);
        ops.push(GraphOp::EmbeddingBatch { out: hidden });

        let qkv_merged_width = attn_width + 2 * kv_width;

        for layer_idx in 0..self.cfg.num_hidden_layers {
            let o_proj = &self.layers[layer_idx].weights.o_proj;
            ensure!(
                o_proj.shape().len() == 2 && o_proj.shape()[1] as usize == attn_width,
                "o_proj expected input dim {}, got shape {:?}",
                attn_width,
                o_proj.shape()
            );

            // --- Input LayerNorm ---
            let normed = push_value(&mut specs, vec![seqlen, hidden_size]);
            ops.push(GraphOp::RmsNorm {
                x: v(hidden),
                weight: lw(layer_idx, LayerWeightSlot::InputLayerNorm),
                n: hidden_size,
                out: normed,
            });

            // --- QKV projection ---
            let q_2d = push_value(&mut specs, vec![seqlen, attn_width]);
            let k_2d = push_value(&mut specs, vec![seqlen, kv_width]);
            let v_2d = push_value(&mut specs, vec![seqlen, kv_width]);
            if seqlen > 1 {
                // Prefill: 3 row-sliced GEMMs against the merged weight — no copies.
                ops.push(GraphOp::MatMulSlice {
                    matrix: lw(layer_idx, LayerWeightSlot::QkvProj),
                    row_offset: 0,
                    out_features: attn_width,
                    rhs: v(normed),
                    out: q_2d,
                });
                ops.push(GraphOp::MatMulSlice {
                    matrix: lw(layer_idx, LayerWeightSlot::QkvProj),
                    row_offset: attn_width,
                    out_features: kv_width,
                    rhs: v(normed),
                    out: k_2d,
                });
                ops.push(GraphOp::MatMulSlice {
                    matrix: lw(layer_idx, LayerWeightSlot::QkvProj),
                    row_offset: attn_width + kv_width,
                    out_features: kv_width,
                    rhs: v(normed),
                    out: v_2d,
                });
            } else {
                // Decode: merged GEMM + contiguous memcpy slices (1 GEMV, 3 cheap copies).
                let qkv_merged = push_value(&mut specs, vec![seqlen, qkv_merged_width]);
                ops.push(GraphOp::MatMul {
                    matrix: lw(layer_idx, LayerWeightSlot::QkvProj),
                    rhs: v(normed),
                    out: qkv_merged,
                });
                ops.push(GraphOp::SliceCols {
                    input: v(qkv_merged),
                    col_offset: 0,
                    out_cols: attn_width,
                    out: q_2d,
                });
                ops.push(GraphOp::SliceCols {
                    input: v(qkv_merged),
                    col_offset: attn_width,
                    out_cols: kv_width,
                    out: k_2d,
                });
                ops.push(GraphOp::SliceCols {
                    input: v(qkv_merged),
                    col_offset: attn_width + kv_width,
                    out_cols: kv_width,
                    out: v_2d,
                });
            }

            let q_3d = push_value(&mut specs, vec![seqlen, attn_heads, head_dim]);
            let k_3d = push_value(&mut specs, vec![seqlen, kv_heads, head_dim]);
            let v_3d = push_value(&mut specs, vec![seqlen, kv_heads, head_dim]);
            ops.push(GraphOp::Reshape {
                input: v(q_2d),
                shape: vec![seqlen, attn_heads, head_dim],
                out: q_3d,
            });
            ops.push(GraphOp::Reshape {
                input: v(k_2d),
                shape: vec![seqlen, kv_heads, head_dim],
                out: k_3d,
            });
            ops.push(GraphOp::Reshape {
                input: v(v_2d),
                shape: vec![seqlen, kv_heads, head_dim],
                out: v_3d,
            });

            let q_rope = push_value(&mut specs, vec![seqlen, attn_heads, head_dim]);
            if fuse_qk_rope_kv_prefill {
                ops.push(GraphOp::QkNormRopeKvPrefill {
                    layer_idx,
                    q: v(q_3d),
                    k: v(k_3d),
                    v: v(v_3d),
                    out: q_rope,
                });
            } else {
                let q_flat = push_value(&mut specs, vec![seqlen * attn_heads, head_dim]);
                let k_flat = push_value(&mut specs, vec![seqlen * kv_heads, head_dim]);
                ops.push(GraphOp::Reshape {
                    input: v(q_3d),
                    shape: vec![seqlen * attn_heads, head_dim],
                    out: q_flat,
                });
                ops.push(GraphOp::Reshape {
                    input: v(k_3d),
                    shape: vec![seqlen * kv_heads, head_dim],
                    out: k_flat,
                });

                let q_norm_flat = push_value(&mut specs, vec![seqlen * attn_heads, head_dim]);
                let k_norm_flat = push_value(&mut specs, vec![seqlen * kv_heads, head_dim]);
                ops.push(GraphOp::RmsNorm {
                    x: v(q_flat),
                    weight: lw(layer_idx, LayerWeightSlot::QNorm),
                    n: head_dim,
                    out: q_norm_flat,
                });
                ops.push(GraphOp::RmsNorm {
                    x: v(k_flat),
                    weight: lw(layer_idx, LayerWeightSlot::KNorm),
                    n: head_dim,
                    out: k_norm_flat,
                });

                let q_rope_in = push_value(&mut specs, vec![seqlen, attn_heads, head_dim]);
                let k_rope_in = push_value(&mut specs, vec![seqlen, kv_heads, head_dim]);
                ops.push(GraphOp::Reshape {
                    input: v(q_norm_flat),
                    shape: vec![seqlen, attn_heads, head_dim],
                    out: q_rope_in,
                });
                ops.push(GraphOp::Reshape {
                    input: v(k_norm_flat),
                    shape: vec![seqlen, kv_heads, head_dim],
                    out: k_rope_in,
                });

                let k_rope = push_value(&mut specs, vec![seqlen, kv_heads, head_dim]);
                ops.push(GraphOp::Rope {
                    x: v(q_rope_in),
                    out: q_rope,
                });
                ops.push(GraphOp::Rope {
                    x: v(k_rope_in),
                    out: k_rope,
                });

                ops.push(GraphOp::KvCacheUpdate {
                    layer_idx,
                    new_k: v(k_rope),
                    new_v: v(v_3d),
                });
            }

            let attn_3d = push_value(&mut specs, vec![seqlen, attn_heads, head_dim]);
            ops.push(GraphOp::Attention {
                layer_idx,
                q: v(q_rope),
                out: attn_3d,
            });

            let attn_2d = push_value(&mut specs, vec![seqlen, attn_width]);
            ops.push(GraphOp::Reshape {
                input: v(attn_3d),
                shape: vec![seqlen, attn_width],
                out: attn_2d,
            });

            let attn_proj = push_value(&mut specs, vec![seqlen, hidden_size]);
            ops.push(GraphOp::MatMul {
                matrix: lw(layer_idx, LayerWeightSlot::OProj),
                rhs: v(attn_2d),
                out: attn_proj,
            });

            // --- Fused residual Add + RmsNorm (saves a kernel launch + memory pass) ---
            let hidden_after_attn = push_value(&mut specs, vec![seqlen, hidden_size]);
            let ff_normed = push_value(&mut specs, vec![seqlen, hidden_size]);
            ops.push(GraphOp::AddRmsNorm {
                residual: v(hidden),
                x: v(attn_proj),
                weight: lw(layer_idx, LayerWeightSlot::PostAttentionLayerNorm),
                n: hidden_size,
                out: ff_normed,
                residual_out: hidden_after_attn,
            });

            // --- Gate+up projection ---
            let gate = push_value(&mut specs, vec![seqlen, inter_size]);
            let up = push_value(&mut specs, vec![seqlen, inter_size]);
            if seqlen > 1 {
                // Prefill: 2 row-sliced GEMMs against the merged weight — no copies.
                ops.push(GraphOp::MatMulSlice {
                    matrix: lw(layer_idx, LayerWeightSlot::GateUpProj),
                    row_offset: 0,
                    out_features: inter_size,
                    rhs: v(ff_normed),
                    out: gate,
                });
                ops.push(GraphOp::MatMulSlice {
                    matrix: lw(layer_idx, LayerWeightSlot::GateUpProj),
                    row_offset: inter_size,
                    out_features: inter_size,
                    rhs: v(ff_normed),
                    out: up,
                });
            } else {
                // Decode: merged GEMM + contiguous memcpy slices.
                let gate_up_merged = push_value(&mut specs, vec![seqlen, 2 * inter_size]);
                ops.push(GraphOp::MatMul {
                    matrix: lw(layer_idx, LayerWeightSlot::GateUpProj),
                    rhs: v(ff_normed),
                    out: gate_up_merged,
                });
                ops.push(GraphOp::SliceCols {
                    input: v(gate_up_merged),
                    col_offset: 0,
                    out_cols: inter_size,
                    out: gate,
                });
                ops.push(GraphOp::SliceCols {
                    input: v(gate_up_merged),
                    col_offset: inter_size,
                    out_cols: inter_size,
                    out: up,
                });
            }

            let ff = push_value(&mut specs, vec![seqlen, inter_size]);
            ops.push(GraphOp::SiluMul {
                gate: v(gate),
                up: v(up),
                out: ff,
            });

            let ff_down = push_value(&mut specs, vec![seqlen, hidden_size]);
            ops.push(GraphOp::MatMul {
                matrix: lw(layer_idx, LayerWeightSlot::DownProj),
                rhs: v(ff),
                out: ff_down,
            });

            let next_hidden = push_value(&mut specs, vec![seqlen, hidden_size]);
            ops.push(GraphOp::Add {
                lhs: v(hidden_after_attn),
                rhs: v(ff_down),
                out: next_hidden,
            });
            hidden = next_hidden;
        }

        let hidden_norm = push_value(&mut specs, vec![seqlen, hidden_size]);
        ops.push(GraphOp::RmsNorm {
            x: v(hidden),
            weight: TensorRef::Weight(WeightRef::Norm),
            n: hidden_size,
            out: hidden_norm,
        });

        let last_hidden = push_value(&mut specs, vec![hidden_size]);
        ops.push(GraphOp::GatherRow {
            src: v(hidden_norm),
            row_idx: seqlen - 1,
            out: last_hidden,
        });

        let logits = push_value(&mut specs, vec![self.cfg.vocab_size]);
        ops.push(GraphOp::MatVec {
            matrix: TensorRef::Weight(WeightRef::LmHead),
            vector: v(last_hidden),
            out: logits,
        });

        StepGraph::new(ops, specs, logits)
    }

    fn execute_step_graph_ctx(
        &mut self,
        ctx: &ExecutionContext,
        graph: &StepGraph,
        pool: &mut TensorPool,
        token_ids: &[u32],
        position_start: usize,
    ) -> Result<Arc<Tensor<f16>>> {
        let profile_ops = self
            .active_profile
            .as_ref()
            .map(|p| p.op_profile_enabled)
            .unwrap_or(false);
        let profile_sync_ops = self
            .active_profile
            .as_ref()
            .map(|p| p.op_profile_sync_enabled)
            .unwrap_or(false);
        let mut final_logits_policy = FinalLogitsPolicy::Allocate;
        self.execute_step_graph_common_ctx(
            ctx,
            graph,
            pool,
            TokenInput::Host(token_ids),
            PositionInput::Host(position_start),
            &mut final_logits_policy,
            profile_ops,
            profile_sync_ops,
        )
    }

    fn execute_step_graph_common_ctx(
        &mut self,
        ctx: &ExecutionContext,
        graph: &StepGraph,
        pool: &mut TensorPool,
        token_input: TokenInput<'_>,
        position_input: PositionInput,
        final_logits_policy: &mut FinalLogitsPolicy<'_>,
        profile_ops: bool,
        profile_sync_ops: bool,
    ) -> Result<Arc<Tensor<f16>>> {
        let mut values: Vec<Option<Arc<Tensor<f16>>>> = vec![None; graph.specs.len()];
        let mut remaining_uses = graph.use_counts.clone();
        for op in &graph.ops {
            let op_start = if profile_ops {
                Some(Instant::now())
            } else {
                None
            };
            match op {
                GraphOp::EmbeddingBatch { out } => {
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.execute_embedding_op_ctx(ctx, &token_input, out_buf)?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::MatMul { matrix, rhs, out } => {
                    let matrix = self.resolve_tensor_ref(&values, *matrix)?;
                    let rhs = self.resolve_tensor_ref(&values, *rhs)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.gemm_into_ctx(ctx, matrix, rhs, out_buf)?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::MatVec {
                    matrix,
                    vector,
                    out,
                } => {
                    let matrix = self.resolve_tensor_ref(&values, *matrix)?;
                    let vector = self.resolve_tensor_ref(&values, *vector)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.gemv_into_ctx(ctx, matrix, vector, out_buf)?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::Add { lhs, rhs, out } => {
                    let lhs = self.resolve_tensor_ref(&values, *lhs)?;
                    let rhs = self.resolve_tensor_ref(&values, *rhs)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.add_2d_into_ctx(ctx, lhs, rhs, out_buf)?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::SiluMul { gate, up, out } => {
                    let gate = self.resolve_tensor_ref(&values, *gate)?;
                    let up = self.resolve_tensor_ref(&values, *up)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.silu_mul_2d_into_ctx(ctx, gate, up, out_buf)?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::RmsNorm { x, weight, n, out } => {
                    let x = self.resolve_tensor_ref(&values, *x)?;
                    let weight = self.resolve_tensor_ref(&values, *weight)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.rms_norm_arc_into_ctx(ctx, x, weight, *n, out_buf)?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::Reshape { input, shape, out } => {
                    let input_id = match input {
                        TensorRef::Value(v) => *v,
                        TensorRef::Weight(_) => bail!("reshape input must be a value"),
                    };
                    let src = if remaining_uses[input_id.idx()] == 1 {
                        values[input_id.idx()]
                            .take()
                            .context("missing reshape input value")?
                    } else {
                        values[input_id.idx()]
                            .as_ref()
                            .cloned()
                            .context("missing reshape input value")?
                    };
                    let reshaped = self
                        .take_or_copy_f16_ctx(ctx, src)?
                        .reshape(shape)
                        .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?;
                    values[out.idx()] = Some(Arc::new(reshaped));
                }
                GraphOp::Rope { x, out } => {
                    let x = self.resolve_tensor_ref(&values, *x)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.execute_rope_op_ctx(ctx, x, &position_input, out_buf)?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::KvCacheUpdate {
                    layer_idx,
                    new_k,
                    new_v,
                } => {
                    let new_k = self.resolve_tensor_ref(&values, *new_k)?;
                    let new_v = self.resolve_tensor_ref(&values, *new_v)?;
                    self.execute_kv_cache_update_op_ctx(
                        ctx,
                        *layer_idx,
                        new_k,
                        new_v,
                        &position_input,
                    )?;
                }
                GraphOp::QkNormRopeKvPrefill {
                    layer_idx,
                    q,
                    k,
                    v,
                    out,
                } => {
                    let q = self.resolve_tensor_ref(&values, *q)?;
                    let k = self.resolve_tensor_ref(&values, *k)?;
                    let v = self.resolve_tensor_ref(&values, *v)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.execute_qk_norm_rope_kv_prefill_op_ctx(
                        ctx,
                        *layer_idx,
                        q,
                        k,
                        v,
                        &position_input,
                        out_buf,
                    )?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::Attention { layer_idx, q, out } => {
                    let q = self.resolve_tensor_ref(&values, *q)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.execute_attention_op_ctx(
                        ctx,
                        *layer_idx,
                        q,
                        &position_input,
                        out_buf,
                    )?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::GatherRow { src, row_idx, out } => {
                    let src = self.resolve_tensor_ref(&values, *src)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.gather_row_into_ctx(ctx, src, *row_idx, out_buf)?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::SliceCols {
                    input,
                    col_offset,
                    out_cols,
                    out,
                } => {
                    let input_tensor = self.resolve_tensor_ref(&values, *input)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.slice_cols_into_ctx(
                        ctx,
                        input_tensor,
                        *col_offset,
                        *out_cols,
                        out_buf,
                    )?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::MatMulSlice {
                    matrix,
                    row_offset,
                    out_features,
                    rhs,
                    out,
                } => {
                    let matrix = self.resolve_tensor_ref(&values, *matrix)?;
                    let rhs = self.resolve_tensor_ref(&values, *rhs)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let out_tensor = self.gemm_row_slice_into_ctx(
                        ctx,
                        matrix,
                        *row_offset,
                        *out_features,
                        rhs,
                        out_buf,
                    )?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                }
                GraphOp::AddRmsNorm {
                    residual,
                    x,
                    weight,
                    n,
                    out,
                    residual_out,
                } => {
                    let residual_t = self.resolve_tensor_ref(&values, *residual)?;
                    let x_t = self.resolve_tensor_ref(&values, *x)?;
                    let weight_t = self.resolve_tensor_ref(&values, *weight)?;
                    let out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *out,
                        final_logits_policy,
                    )?;
                    let res_out_buf = self.checkout_graph_output_ctx(
                        ctx,
                        graph,
                        pool,
                        *residual_out,
                        final_logits_policy,
                    )?;
                    let (out_tensor, res_out_tensor) = self.add_rms_norm_into_ctx(
                        ctx,
                        residual_t,
                        x_t,
                        weight_t,
                        *n,
                        out_buf,
                        res_out_buf,
                    )?;
                    values[out.idx()] = Some(Arc::new(out_tensor));
                    values[residual_out.idx()] = Some(Arc::new(res_out_tensor));
                }
            }
            if let Some(op_start) = op_start {
                if profile_sync_ops {
                    unsafe {
                        ctx.get_cuda_stream()
                            .synchronize()
                            .map_err(|e| anyhow::anyhow!("profile op sync failed: {e:?}"))?;
                    }
                }
                self.profile_op(graph_op_name(op), op_start.elapsed());
            }

            self.consume_graph_inputs_ctx(ctx, graph, pool, &mut values, &mut remaining_uses, op)?;
        }

        values[graph.final_value.idx()]
            .as_ref()
            .cloned()
            .context("missing final logits value")
    }

    fn checkout_graph_output_ctx(
        &self,
        ctx: &ExecutionContext,
        graph: &StepGraph,
        pool: &mut TensorPool,
        out: ValueId,
        final_logits_policy: &mut FinalLogitsPolicy<'_>,
    ) -> Result<Tensor<f16>> {
        if out == graph.final_value {
            return match final_logits_policy {
                FinalLogitsPolicy::Allocate => alloc_f16_ctx(ctx, &graph.spec(out).shape),
                FinalLogitsPolicy::Preallocated(final_logits) => final_logits
                    .take()
                    .context("missing preallocated final logits buffer"),
            };
        }

        pool.checkout(ctx, graph.spec(out))
    }

    fn execute_embedding_op_ctx(
        &self,
        ctx: &ExecutionContext,
        token_input: &TokenInput<'_>,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        match token_input {
            TokenInput::Host(token_ids) => self.embedding_batch_into_ctx(ctx, token_ids, out),
            TokenInput::Device(token_ids_device) => {
                self.embedding_batch_from_device_ids_into_ctx(ctx, token_ids_device.clone(), out)
            }
        }
    }

    fn execute_rope_op_ctx(
        &self,
        ctx: &ExecutionContext,
        x: Arc<Tensor<f16>>,
        position_input: &PositionInput,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        match position_input {
            PositionInput::Host(position_start) => {
                self.rope_seq_arc_into_ctx(ctx, x, *position_start, out)
            }
            PositionInput::Device(position_start) => {
                self.rope_seq_arc_into_ctx_device_pos(ctx, x, position_start.clone(), out)
            }
        }
    }

    fn execute_kv_cache_update_op_ctx(
        &mut self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        new_k: Arc<Tensor<f16>>,
        new_v: Arc<Tensor<f16>>,
        position_input: &PositionInput,
    ) -> Result<()> {
        match position_input {
            PositionInput::Host(position_start) => {
                self.kv_cache_update_seq_arc_ctx(ctx, layer_idx, new_k, new_v, *position_start)
            }
            PositionInput::Device(position_start) => self.kv_cache_update_seq_arc_ctx_device_pos(
                ctx,
                layer_idx,
                new_k,
                new_v,
                position_start.clone(),
            ),
        }
    }

    fn execute_qk_norm_rope_kv_prefill_op_ctx(
        &mut self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        q: Arc<Tensor<f16>>,
        k: Arc<Tensor<f16>>,
        v: Arc<Tensor<f16>>,
        position_input: &PositionInput,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        let position_start = match position_input {
            PositionInput::Host(position_start) => *position_start,
            PositionInput::Device(_) => {
                bail!("QkNormRopeKvPrefill is prefill-only and requires host position")
            }
        };

        let seq_len = q.shape().first().copied().unwrap_or_default() as usize;
        ensure!(
            q.shape()
                == vec![
                    seq_len as i32,
                    self.cfg.num_attention_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "q shape mismatch in fused qk/rope/kv prefill: {:?}",
            q.shape()
        );
        ensure!(
            k.shape()
                == vec![
                    seq_len as i32,
                    self.cfg.num_key_value_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "k shape mismatch in fused qk/rope/kv prefill: {:?}",
            k.shape()
        );
        ensure!(
            v.shape()
                == vec![
                    seq_len as i32,
                    self.cfg.num_key_value_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "v shape mismatch in fused qk/rope/kv prefill: {:?}",
            v.shape()
        );
        ensure!(
            out.shape()
                == vec![
                    seq_len as i32,
                    self.cfg.num_attention_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "fused qk/rope/kv output shape mismatch: {:?}",
            out.shape()
        );
        ensure!(
            self.cfg.head_dim % 2 == 0,
            "fused qk/rope/kv requires even head_dim, got {}",
            self.cfg.head_dim
        );
        ensure!(
            position_start + seq_len <= self.max_seq_len,
            "fused qk/rope/kv range [{}..{}) exceeds max_seq_len {}",
            position_start,
            position_start + seq_len,
            self.max_seq_len
        );

        let layer = &self.layers[layer_idx];
        let k_cache = layer
            .state
            .k_cache
            .as_ref()
            .context("missing k_cache in layer state")?;
        let v_cache = layer
            .state
            .v_cache
            .as_ref()
            .context("missing v_cache in layer state")?;
        ensure!(
            k_cache.shape()
                == vec![
                    self.cfg.num_key_value_heads as i32,
                    self.max_seq_len as i32,
                    self.cfg.head_dim as i32
                ],
            "k_cache shape mismatch in fused qk/rope/kv: {:?}",
            k_cache.shape()
        );
        ensure!(
            v_cache.shape()
                == vec![
                    self.cfg.num_key_value_heads as i32,
                    self.max_seq_len as i32,
                    self.cfg.head_dim as i32
                ],
            "v_cache shape mismatch in fused qk/rope/kv: {:?}",
            v_cache.shape()
        );

        let weights = &layer.weights;
        unsafe {
            qk_norm_rope_kv_prefill_raw_f16(
                q.device_pointer().clone(),
                k.device_pointer().clone(),
                v.device_pointer().clone(),
                weights.q_norm.device_pointer().clone(),
                weights.k_norm.device_pointer().clone(),
                self.inv_freq.device_pointer().clone(),
                out.device_pointer().clone(),
                k_cache.device_pointer().clone(),
                v_cache.device_pointer().clone(),
                self.cfg.rms_norm_eps,
                position_start as i32,
                seq_len as i32,
                self.cfg.num_attention_heads as i32,
                self.cfg.num_key_value_heads as i32,
            )
            .generics(vec![
                self.cfg.head_dim.to_string(),
                (self.cfg.head_dim / 2).to_string(),
                self.max_seq_len.to_string(),
            ])
            .grid((
                seq_len as u32,
                (self.cfg.num_attention_heads + self.cfg.num_key_value_heads) as u32,
                1u32,
            ))
            .execute(ctx)?;
        }

        Ok(out)
    }

    fn execute_attention_op_ctx(
        &self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        q: Arc<Tensor<f16>>,
        position_input: &PositionInput,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        match position_input {
            PositionInput::Host(position_start) => {
                self.attend_seq_arc_into_ctx(ctx, layer_idx, q, *position_start, out)
            }
            PositionInput::Device(position_start) => self.attend_seq_arc_into_ctx_device_pos(
                ctx,
                layer_idx,
                q,
                position_start.clone(),
                out,
            ),
        }
    }

    fn consume_graph_inputs_ctx(
        &self,
        ctx: &ExecutionContext,
        graph: &StepGraph,
        pool: &mut TensorPool,
        values: &mut [Option<Arc<Tensor<f16>>>],
        remaining_uses: &mut [usize],
        op: &GraphOp,
    ) -> Result<()> {
        for input in op.value_inputs() {
            let idx = input.idx();
            ensure!(
                remaining_uses[idx] > 0,
                "invalid use-count state for value {idx}"
            );
            remaining_uses[idx] -= 1;
            if remaining_uses[idx] == 0
                && input != graph.final_value
                && let Some(tensor) = values[idx].take()
            {
                let tensor = self.take_or_copy_f16_ctx(ctx, tensor)?;
                pool.checkin(tensor, graph.spec(input))?;
            }
        }
        Ok(())
    }

    fn resolve_weight_ref(&self, weight: WeightRef) -> Arc<Tensor<f16>> {
        match weight {
            WeightRef::LmHead => self.lm_head.clone(),
            WeightRef::Norm => self.norm.clone(),
            WeightRef::Layer { layer_idx, slot } => {
                let layer = &self.layers[layer_idx].weights;
                match slot {
                    LayerWeightSlot::InputLayerNorm => layer.input_layernorm.clone(),
                    LayerWeightSlot::PostAttentionLayerNorm => {
                        layer.post_attention_layernorm.clone()
                    }
                    LayerWeightSlot::QNorm => layer.q_norm.clone(),
                    LayerWeightSlot::KNorm => layer.k_norm.clone(),
                    LayerWeightSlot::QkvProj => layer.qkv_proj.clone(),
                    LayerWeightSlot::OProj => layer.o_proj.clone(),
                    LayerWeightSlot::GateUpProj => layer.gate_up_proj.clone(),
                    LayerWeightSlot::DownProj => layer.down_proj.clone(),
                }
            }
        }
    }

    fn resolve_tensor_ref(
        &self,
        values: &[Option<Arc<Tensor<f16>>>],
        input: TensorRef,
    ) -> Result<Arc<Tensor<f16>>> {
        match input {
            TensorRef::Value(v) => values[v.idx()]
                .as_ref()
                .cloned()
                .with_context(|| format!("missing graph value {}", v.idx())),
            TensorRef::Weight(w) => Ok(self.resolve_weight_ref(w)),
        }
    }

    fn copy_f16_ctx(&self, ctx: &ExecutionContext, src: &Arc<Tensor<f16>>) -> Result<Tensor<f16>> {
        Ok(unsafe { api::dup(src).execute(ctx)? })
    }

    fn take_or_copy_f16_ctx(
        &self,
        ctx: &ExecutionContext,
        src: Arc<Tensor<f16>>,
    ) -> Result<Tensor<f16>> {
        match Arc::try_unwrap(src) {
            Ok(t) => Ok(t),
            Err(shared) => self.copy_f16_ctx(ctx, &shared),
        }
    }

    fn embedding_batch_ctx(
        &self,
        ctx: &ExecutionContext,
        token_ids: &[u32],
    ) -> Result<Tensor<f16>> {
        ensure!(
            !token_ids.is_empty(),
            "embedding_batch_ctx expects at least one token"
        );
        let out = alloc_f16_ctx(ctx, &[token_ids.len(), self.cfg.hidden_size])?;
        self.embedding_batch_into_ctx(ctx, token_ids, out)
    }

    fn embedding_batch_into_ctx(
        &self,
        ctx: &ExecutionContext,
        token_ids: &[u32],
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        ensure!(
            !token_ids.is_empty(),
            "embedding_batch_ctx expects at least one token"
        );
        let seqlen = token_ids.len();
        ensure!(
            out.shape() == vec![seqlen as i32, self.cfg.hidden_size as i32],
            "embedding output shape mismatch, got {:?}",
            out.shape()
        );

        let ids_host = Arc::new(token_ids.to_vec());
        let ids = unsafe { api::copy_host_vec_to_device(&ids_host).execute(ctx)? };
        let embed_block = env_usize_or("GROUT_EMBED_BLOCK", EMBED_BLOCK);
        let out = out.partition([1, embed_block]);
        let result = unsafe {
            embedding_batch_f16(
                value(Arc::new(ids)),
                value(self.embed_tokens.clone()),
                value(out),
            )
            .generics(vec![
                self.cfg.hidden_size.to_string(),
                embed_block.to_string(),
            ])
            .execute(ctx)?
        };
        let _ids: Arc<Tensor<u32>> = result.0;
        let out: Partition<Tensor<f16>> = result.2;
        Ok(out.unpartition())
    }

    fn embedding_batch_from_device_ids_into_ctx(
        &self,
        ctx: &ExecutionContext,
        token_ids: Arc<Tensor<u32>>,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        ensure!(
            token_ids.shape().len() == 1,
            "embedding token_ids must be rank-1, got {:?}",
            token_ids.shape()
        );
        let seqlen = token_ids.shape()[0] as usize;
        ensure!(
            seqlen > 0,
            "embedding token_ids must contain at least one token"
        );
        ensure!(
            out.shape() == vec![seqlen as i32, self.cfg.hidden_size as i32],
            "embedding output shape mismatch, got {:?}",
            out.shape()
        );

        let embed_block = env_usize_or("GROUT_EMBED_BLOCK", EMBED_BLOCK);
        let out = out.partition([1, embed_block]);
        let result = unsafe {
            embedding_batch_f16(
                value(token_ids),
                value(self.embed_tokens.clone()),
                value(out),
            )
            .generics(vec![
                self.cfg.hidden_size.to_string(),
                embed_block.to_string(),
            ])
            .execute(ctx)?
        };
        let out: Partition<Tensor<f16>> = result.2;
        Ok(out.unpartition())
    }

    fn gemv_ctx(
        &self,
        ctx: &ExecutionContext,
        matrix: Arc<Tensor<f16>>,
        vector: Arc<Tensor<f16>>,
    ) -> Result<Tensor<f16>> {
        ensure!(
            matrix.shape().len() == 2,
            "gemv matrix must be rank 2, got {:?}",
            matrix.shape()
        );
        let m = matrix.shape()[0] as usize;
        let out = alloc_f16_ctx(ctx, &[m])?;
        self.gemv_into_ctx(ctx, matrix, vector, out)
    }

    fn gemv_into_ctx(
        &self,
        ctx: &ExecutionContext,
        matrix: Arc<Tensor<f16>>,
        vector: Arc<Tensor<f16>>,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        ensure!(
            matrix.shape().len() == 2,
            "gemv matrix must be rank 2, got {:?}",
            matrix.shape()
        );
        ensure!(
            vector.shape().len() == 1,
            "gemv vector must be rank 1, got {:?}",
            vector.shape()
        );

        let m = matrix.shape()[0] as usize;
        let k = matrix.shape()[1] as usize;
        ensure!(k == vector.shape()[0] as usize, "gemv shape mismatch");
        ensure!(
            out.shape() == vec![m as i32],
            "gemv output shape mismatch, got {:?}",
            out.shape()
        );
        let op = cublas::gemv_f16_op(matrix, vector, out, m, k)?;
        unsafe { op.execute(ctx)? }
    }

    fn gemm_ctx(
        &self,
        ctx: &ExecutionContext,
        matrix: Arc<Tensor<f16>>,
        rhs: Arc<Tensor<f16>>,
    ) -> Result<Tensor<f16>> {
        ensure!(
            matrix.shape().len() == 2,
            "gemm matrix must be rank 2, got {:?}",
            matrix.shape()
        );
        ensure!(
            rhs.shape().len() == 2,
            "gemm rhs must be rank 2, got {:?}",
            rhs.shape()
        );
        let m = matrix.shape()[0] as usize;
        let n = rhs.shape()[0] as usize;
        let out = alloc_f16_ctx(ctx, &[n, m])?;
        self.gemm_into_ctx(ctx, matrix, rhs, out)
    }

    fn gemm_into_ctx(
        &self,
        ctx: &ExecutionContext,
        matrix: Arc<Tensor<f16>>,
        rhs: Arc<Tensor<f16>>,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        ensure!(
            matrix.shape().len() == 2,
            "gemm matrix must be rank 2, got {:?}",
            matrix.shape()
        );
        ensure!(
            rhs.shape().len() == 2,
            "gemm rhs must be rank 2, got {:?}",
            rhs.shape()
        );
        let m = matrix.shape()[0] as usize;
        let k = matrix.shape()[1] as usize;
        let n = rhs.shape()[0] as usize;
        ensure!(k == rhs.shape()[1] as usize, "gemm shape mismatch");
        ensure!(
            out.shape() == vec![n as i32, m as i32],
            "gemm output shape mismatch, got {:?}",
            out.shape()
        );
        let op = cublas::gemm_f16_op(matrix, rhs, out, m, n, k)?;
        unsafe { op.execute(ctx)? }
    }

    /// Row-sliced GEMM: multiplies input by a contiguous row-range of the weight matrix.
    /// Uses pointer arithmetic — no weight copy, no extra memory.
    fn gemm_row_slice_into_ctx(
        &self,
        ctx: &ExecutionContext,
        matrix: Arc<Tensor<f16>>,
        row_offset: usize,
        out_features: usize,
        rhs: Arc<Tensor<f16>>,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        let k = matrix.shape()[1] as usize;
        let n = rhs.shape()[0] as usize;
        ensure!(
            k == rhs.shape()[1] as usize,
            "gemm_row_slice shape mismatch"
        );
        ensure!(
            out.shape() == vec![n as i32, out_features as i32],
            "gemm_row_slice output shape mismatch: expected [{n}, {out_features}], got {:?}",
            out.shape()
        );
        let op = cublas::gemm_f16_row_slice_op(matrix, row_offset, rhs, out, out_features, n, k)?;
        unsafe { op.execute(ctx)? }
    }

    fn add_2d_ctx(
        &self,
        ctx: &ExecutionContext,
        lhs: Arc<Tensor<f16>>,
        rhs: Arc<Tensor<f16>>,
    ) -> Result<Tensor<f16>> {
        let rows = lhs.shape()[0] as usize;
        let cols = lhs.shape()[1] as usize;
        let out = alloc_f16_ctx(ctx, &[rows, cols])?;
        self.add_2d_into_ctx(ctx, lhs, rhs, out)
    }

    fn add_2d_into_ctx(
        &self,
        ctx: &ExecutionContext,
        lhs: Arc<Tensor<f16>>,
        rhs: Arc<Tensor<f16>>,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        ensure!(lhs.shape() == rhs.shape(), "add shape mismatch");
        ensure!(lhs.shape().len() == 2, "add_2d expects rank-2 tensors");
        let rows = lhs.shape()[0] as usize;
        let cols = lhs.shape()[1] as usize;
        ensure!(
            out.shape() == vec![rows as i32, cols as i32],
            "add output shape mismatch, got {:?}",
            out.shape()
        );
        let out = out.partition([1, POINTWISE_BLOCK]);
        let result = unsafe {
            add_2d_f16(value(out), value(lhs), value(rhs))
                .generics(vec![POINTWISE_BLOCK.to_string()])
                .execute(ctx)?
        };
        let out: Partition<Tensor<f16>> = result.0;
        Ok(out.unpartition())
    }

    fn silu_mul_2d_ctx(
        &self,
        ctx: &ExecutionContext,
        gate: Arc<Tensor<f16>>,
        up: Arc<Tensor<f16>>,
    ) -> Result<Tensor<f16>> {
        let rows = gate.shape()[0] as usize;
        let cols = gate.shape()[1] as usize;
        let out = alloc_f16_ctx(ctx, &[rows, cols])?;
        self.silu_mul_2d_into_ctx(ctx, gate, up, out)
    }

    fn silu_mul_2d_into_ctx(
        &self,
        ctx: &ExecutionContext,
        gate: Arc<Tensor<f16>>,
        up: Arc<Tensor<f16>>,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        ensure!(gate.shape() == up.shape(), "silu_mul shape mismatch");
        ensure!(gate.shape().len() == 2, "silu_mul expects rank-2 tensors");
        let rows = gate.shape()[0] as usize;
        let cols = gate.shape()[1] as usize;
        ensure!(
            out.shape() == vec![rows as i32, cols as i32],
            "silu_mul output shape mismatch, got {:?}",
            out.shape()
        );
        let out = out.partition([1, POINTWISE_BLOCK]);
        let result = unsafe {
            silu_mul_2d_f16(value(out), value(gate), value(up))
                .generics(vec![POINTWISE_BLOCK.to_string()])
                .execute(ctx)?
        };
        let out: Partition<Tensor<f16>> = result.0;
        Ok(out.unpartition())
    }

    /// Copies a column-slice from a rank-2 tensor.
    /// For input [rows, total_cols], copies columns [col_offset .. col_offset+out_cols)
    /// from each row into `out` [rows, out_cols].
    ///
    /// Uses `Tensor::slice()` to compute the source view, then copies:
    /// - seqlen=1 (decode): single contiguous memcpy
    /// - seqlen>1 (prefill): per-row strided memcpy
    fn slice_cols_into_ctx(
        &self,
        ctx: &ExecutionContext,
        input: Arc<Tensor<f16>>,
        col_offset: usize,
        out_cols: usize,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        let rows = input.shape()[0] as usize;

        ensure!(
            rows == 1,
            "SliceCols is only supported for seqlen=1 (decode); got rows={rows}"
        );
        let elem_size = size_of::<f16>();
        let src_ptr = input.device_pointer().cu_deviceptr() + (col_offset * elem_size) as u64;
        let dst_ptr = out.device_pointer().cu_deviceptr();
        unsafe {
            memcpy_dtod_async::<u8>(
                dst_ptr,
                src_ptr,
                out_cols * elem_size,
                ctx.get_cuda_stream(),
            );
        }
        Ok(out)
    }

    /// Fused add + RMS norm: combined = residual + x, out = rms_norm(combined, weight).
    /// Returns (normed_output, combined_residual).
    fn add_rms_norm_into_ctx(
        &self,
        ctx: &ExecutionContext,
        residual: Arc<Tensor<f16>>,
        x: Arc<Tensor<f16>>,
        weight: Arc<Tensor<f16>>,
        n: usize,
        out: Tensor<f16>,
        residual_out: Tensor<f16>,
    ) -> Result<(Tensor<f16>, Tensor<f16>)> {
        ensure!(
            residual.shape() == x.shape(),
            "add_rms_norm: residual shape {:?} != x shape {:?}",
            residual.shape(),
            x.shape()
        );
        let orig_shape: Vec<usize> = residual.shape().iter().map(|d| *d as usize).collect();
        let rows = match orig_shape.as_slice() {
            [d] => {
                ensure!(*d == n, "add_rms_norm expected dim {n}, got {d}");
                1
            }
            [r, d] => {
                ensure!(*d == n, "add_rms_norm expected inner dim {n}, got {d}");
                *r
            }
            _ => bail!(
                "add_rms_norm only supports rank 1 or 2 inputs, got {:?}",
                orig_shape
            ),
        };

        // Ensure inputs are rank-2 for the kernel.
        let residual = if orig_shape.len() == 1 {
            Arc::new(
                self.copy_f16_ctx(ctx, &residual)?
                    .reshape(&[1, n])
                    .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?,
            )
        } else {
            residual
        };
        let x = if orig_shape.len() == 1 {
            Arc::new(
                self.copy_f16_ctx(ctx, &x)?
                    .reshape(&[1, n])
                    .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?,
            )
        } else {
            x
        };

        ensure!(
            out.shape().iter().map(|d| *d as usize).product::<usize>() == rows * n,
            "add_rms_norm output numel mismatch, got {:?}",
            out.shape()
        );
        ensure!(
            residual_out
                .shape()
                .iter()
                .map(|d| *d as usize)
                .product::<usize>()
                == rows * n,
            "add_rms_norm residual_out numel mismatch, got {:?}",
            residual_out.shape()
        );
        let out = out
            .reshape(&[rows, n])
            .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?
            .partition([1, n]);
        let residual_out = residual_out
            .reshape(&[rows, n])
            .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?
            .partition([1, n]);
        let result = unsafe {
            add_rms_norm_f16(
                value(residual),
                value(x),
                value(weight),
                value(out),
                value(residual_out),
                value(self.cfg.rms_norm_eps),
            )
            .generics(vec![n.to_string(), self.add_rms_block.to_string()])
            .execute(ctx)?
        };
        let _residual: Arc<Tensor<f16>> = result.0;
        let _x: Arc<Tensor<f16>> = result.1;
        let out: Partition<Tensor<f16>> = result.3;
        let residual_out: Partition<Tensor<f16>> = result.4;
        Ok((
            out.unpartition()
                .reshape(&orig_shape)
                .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?,
            residual_out
                .unpartition()
                .reshape(&orig_shape)
                .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?,
        ))
    }

    fn rms_norm_ctx(
        &self,
        ctx: &ExecutionContext,
        x: Tensor<f16>,
        weight: Arc<Tensor<f16>>,
        n: usize,
    ) -> Result<Tensor<f16>> {
        let x_shape: Vec<usize> = x.shape().iter().map(|d| *d as usize).collect();
        let rows = match x_shape.as_slice() {
            [d] => {
                ensure!(*d == n, "rms_norm expected dim {n}, got {d}");
                1
            }
            [r, d] => {
                ensure!(*d == n, "rms_norm expected inner dim {n}, got {d}");
                *r
            }
            _ => bail!(
                "rms_norm only supports rank 1 or 2 inputs, got {:?}",
                x_shape
            ),
        };
        let out = alloc_f16_ctx(ctx, &[rows, n])?;
        self.rms_norm_arc_into_ctx(ctx, Arc::new(x), weight, n, out)
    }

    fn rms_norm_arc_into_ctx(
        &self,
        ctx: &ExecutionContext,
        x: Arc<Tensor<f16>>,
        weight: Arc<Tensor<f16>>,
        n: usize,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        let orig_shape: Vec<usize> = x.shape().iter().map(|d| *d as usize).collect();
        let (x, rows) = match orig_shape.as_slice() {
            [d] => {
                ensure!(*d == n, "rms_norm expected dim {n}, got {d}");
                (
                    Arc::new(
                        self.copy_f16_ctx(ctx, &x)?
                            .reshape(&[1, n])
                            .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?,
                    ),
                    1,
                )
            }
            [r, d] => {
                ensure!(*d == n, "rms_norm expected inner dim {n}, got {d}");
                (x, *r)
            }
            _ => bail!(
                "rms_norm only supports rank 1 or 2 inputs, got {:?}",
                orig_shape
            ),
        };

        ensure!(
            out.shape().iter().map(|d| *d as usize).product::<usize>() == rows * n,
            "rms_norm output numel mismatch, got {:?}",
            out.shape()
        );
        let out = out
            .reshape(&[rows, n])
            .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?
            .partition([1, n]);
        // Pick BLOCK_SIZE per n: small n (head_dim=128) gets RMS_BLOCK=128
        // because BS > N panics cutile. Hidden-size RMS can be retuned with
        // GROUT_RMS_HIDDEN_BLOCK; it must be a power of two and divide N
        // because cutile requires pow-2 tile lengths and this kernel uses
        // exact N / BLOCK_SIZE tiling.
        let bs = if n >= RMS_BLOCK_HIDDEN {
            if self.rms_hidden_block <= n && n % self.rms_hidden_block == 0 {
                self.rms_hidden_block
            } else {
                RMS_BLOCK_HIDDEN
            }
        } else {
            RMS_BLOCK
        };
        let result = unsafe {
            rms_norm_f16(
                value(x),
                value(weight),
                value(out),
                value(self.cfg.rms_norm_eps),
            )
            .generics(vec![n.to_string(), bs.to_string()])
            .execute(ctx)?
        };
        let _x: Arc<Tensor<f16>> = result.0;
        let out: Partition<Tensor<f16>> = result.2;
        Ok(out
            .unpartition()
            .reshape(&orig_shape)
            .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?)
    }

    fn rope_seq_ctx(
        &self,
        ctx: &ExecutionContext,
        x: Tensor<f16>,
        position_start: usize,
    ) -> Result<Tensor<f16>> {
        ensure!(
            x.shape().len() == 3
                && x.shape()[2] as usize == self.cfg.head_dim
                && x.shape()[2] as usize == ROPE_BLOCK,
            "rope expects [seqlen, heads, head_dim] where head_dim={ROPE_BLOCK}, got {:?}",
            x.shape()
        );
        let seq_len = x.shape()[0] as usize;
        let num_heads = x.shape()[1] as usize;
        let out = alloc_f16_ctx(ctx, &[seq_len, num_heads, self.cfg.head_dim])?;
        self.rope_seq_arc_into_ctx(ctx, Arc::new(x), position_start, out)
    }

    fn rope_seq_arc_into_ctx(
        &self,
        ctx: &ExecutionContext,
        x: Arc<Tensor<f16>>,
        position_start: usize,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        let position_input = PositionInput::Host(position_start);
        self.rope_seq_arc_into_ctx_with_position(ctx, x, &position_input, out)
    }

    fn rope_seq_arc_into_ctx_device_pos(
        &self,
        ctx: &ExecutionContext,
        x: Arc<Tensor<f16>>,
        position_start: Arc<Tensor<u32>>,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        let position_input = PositionInput::Device(position_start);
        self.rope_seq_arc_into_ctx_with_position(ctx, x, &position_input, out)
    }

    fn rope_seq_arc_into_ctx_with_position(
        &self,
        ctx: &ExecutionContext,
        x: Arc<Tensor<f16>>,
        position_input: &PositionInput,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        ensure!(
            x.shape().len() == 3
                && x.shape()[2] as usize == self.cfg.head_dim
                && x.shape()[2] as usize == ROPE_BLOCK,
            "rope expects [seqlen, heads, head_dim] where head_dim={ROPE_BLOCK}, got {:?}",
            x.shape()
        );
        if let PositionInput::Device(position_start) = position_input {
            ensure!(
                position_start.shape() == vec![1],
                "rope position tensor must be shape [1], got {:?}",
                position_start.shape()
            );
        }
        let seq_len = x.shape()[0] as usize;
        let num_heads = x.shape()[1] as usize;
        ensure!(
            out.shape() == vec![seq_len as i32, num_heads as i32, self.cfg.head_dim as i32],
            "rope output shape mismatch, got {:?}",
            out.shape()
        );
        let out = out.partition([1, 1, self.cfg.head_dim / 2]);
        let out: Partition<Tensor<f16>> = match position_input {
            PositionInput::Host(position_start) => {
                let result = unsafe {
                    rope_seq_f16(
                        value(x),
                        value(self.inv_freq.clone()),
                        value(out),
                        value(*position_start as i32),
                    )
                    .generics(vec![
                        self.cfg.head_dim.to_string(),
                        (self.cfg.head_dim / 2).to_string(),
                    ])
                    .execute(ctx)?
                };
                result.2
            }
            PositionInput::Device(position_start) => {
                let result = unsafe {
                    rope_seq_dynpos_f16(
                        value(x),
                        value(self.inv_freq.clone()),
                        value(position_start.clone()),
                        value(out),
                    )
                    .generics(vec![
                        self.cfg.head_dim.to_string(),
                        (self.cfg.head_dim / 2).to_string(),
                    ])
                    .execute(ctx)?
                };
                result.3
            }
        };
        Ok(out.unpartition())
    }

    fn kv_cache_update_seq_ctx(
        &mut self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        new_k: Tensor<f16>,
        new_v: Tensor<f16>,
        position_start: usize,
    ) -> Result<()> {
        self.kv_cache_update_seq_arc_ctx(
            ctx,
            layer_idx,
            Arc::new(new_k),
            Arc::new(new_v),
            position_start,
        )
    }

    fn kv_cache_update_seq_arc_ctx(
        &mut self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        new_k: Arc<Tensor<f16>>,
        new_v: Arc<Tensor<f16>>,
        position_start: usize,
    ) -> Result<()> {
        let position_input = PositionInput::Host(position_start);
        self.kv_cache_update_seq_arc_ctx_with_position(
            ctx,
            layer_idx,
            new_k,
            new_v,
            &position_input,
        )
    }

    fn kv_cache_update_seq_arc_ctx_device_pos(
        &mut self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        new_k: Arc<Tensor<f16>>,
        new_v: Arc<Tensor<f16>>,
        position_start: Arc<Tensor<u32>>,
    ) -> Result<()> {
        let position_input = PositionInput::Device(position_start);
        self.kv_cache_update_seq_arc_ctx_with_position(
            ctx,
            layer_idx,
            new_k,
            new_v,
            &position_input,
        )
    }

    fn kv_cache_update_seq_arc_ctx_with_position(
        &mut self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        new_k: Arc<Tensor<f16>>,
        new_v: Arc<Tensor<f16>>,
        position_input: &PositionInput,
    ) -> Result<()> {
        ensure!(
            new_k.shape().len() == 3,
            "new_k must be rank 3 [seqlen, kv_heads, head_dim], got {:?}",
            new_k.shape()
        );
        let seq_len = new_k.shape()[0] as usize;
        ensure!(
            new_k.shape()
                == vec![
                    seq_len as i32,
                    self.cfg.num_key_value_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "new_k shape mismatch: {:?}",
            new_k.shape()
        );
        ensure!(
            new_v.shape()
                == vec![
                    seq_len as i32,
                    self.cfg.num_key_value_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "new_v shape mismatch: {:?}",
            new_v.shape()
        );
        match position_input {
            PositionInput::Host(position_start) => ensure!(
                *position_start + seq_len <= self.max_seq_len,
                "kv_cache_update range [{}..{}) exceeds max_seq_len {}",
                position_start,
                position_start + seq_len,
                self.max_seq_len
            ),
            PositionInput::Device(position_start) => {
                ensure!(
                    seq_len == 1,
                    "decode graph path expects seq_len=1, got {seq_len}"
                );
                ensure!(
                    position_start.shape() == vec![1],
                    "kv_cache position tensor must be shape [1], got {:?}",
                    position_start.shape()
                );
            }
        }

        let layer = &mut self.layers[layer_idx];
        let k_cache = layer
            .state
            .k_cache
            .take()
            .context("missing k_cache in layer state")?;
        let v_cache = layer
            .state
            .v_cache
            .take()
            .context("missing v_cache in layer state")?;
        let bm_s = env_usize_or("GROUT_KV_CACHE_BM_S", KV_CACHE_BM_S_DEFAULT);
        let (k_cache, v_cache): (Partition<Tensor<f16>>, Partition<Tensor<f16>>) =
            match position_input {
                PositionInput::Host(position_start) => {
                    debug_assert_eq!(
                        *position_start, 0,
                        "kv_cache_update_seq_f16 assumes position_start==0 \
                         (prefill path); got {position_start}"
                    );
                    // Host (prefill) path uses the new BM_S-sharded
                    // kernel: partition tile is [1, BM_S, VEC_BLOCK] so
                    // the grid becomes (num_kv_heads, max_seq_len/BM_S, 1).
                    let k_cache_part = k_cache.partition([1, bm_s, VEC_BLOCK]);
                    let v_cache_part = v_cache.partition([1, bm_s, VEC_BLOCK]);
                    let result = unsafe {
                        kv_cache_update_seq_f16(
                            value(new_k),
                            value(new_v),
                            value(k_cache_part),
                            value(v_cache_part),
                            value(*position_start as i32),
                            value(seq_len as i32),
                        )
                        .generics(vec![
                            self.cfg.head_dim.to_string(),
                            VEC_BLOCK.to_string(),
                            bm_s.to_string(),
                        ])
                        .execute(ctx)?
                    };
                    (result.2, result.3)
                }
                PositionInput::Device(position_start) => {
                    // Device (decode) path. CHUNK_D sharding expands grid
                    // from (kv_heads, 1, 1) to (kv_heads, 1, head_dim/CHUNK_D).
                    let chunk_d =
                        env_usize_or("GROUT_KV_CACHE_DYN_CHUNK_D", KV_CACHE_DYN_CHUNK_D_DEFAULT);
                    let k_cache_part = k_cache.partition([1, self.max_seq_len, chunk_d]);
                    let v_cache_part = v_cache.partition([1, self.max_seq_len, chunk_d]);
                    let result = unsafe {
                        kv_cache_update_seq_dynpos_f16(
                            value(new_k),
                            value(new_v),
                            value(k_cache_part),
                            value(v_cache_part),
                            value(position_start.clone()),
                            value(seq_len as i32),
                        )
                        .generics(vec![
                            self.cfg.head_dim.to_string(),
                            chunk_d.to_string(),
                            self.max_seq_len.to_string(),
                        ])
                        .execute(ctx)?
                    };
                    (result.2, result.3)
                }
            };
        layer.state.k_cache = Some(Arc::new(k_cache.unpartition()));
        layer.state.v_cache = Some(Arc::new(v_cache.unpartition()));
        Ok(())
    }

    fn attend_seq_ctx(
        &self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        q: Tensor<f16>,
        position_start: usize,
    ) -> Result<Tensor<f16>> {
        ensure!(
            q.shape().len() == 3,
            "q must be rank 3 [seqlen, heads, head_dim], got {:?}",
            q.shape()
        );
        let q_len = q.shape()[0] as usize;
        let out = alloc_f16_ctx(
            ctx,
            &[q_len, self.cfg.num_attention_heads, self.cfg.head_dim],
        )?;
        self.attend_seq_arc_into_ctx(ctx, layer_idx, Arc::new(q), position_start, out)
    }

    fn attend_seq_arc_into_ctx(
        &self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        q: Arc<Tensor<f16>>,
        position_start: usize,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        let position_input = PositionInput::Host(position_start);
        self.attend_seq_arc_into_ctx_with_position(ctx, layer_idx, q, &position_input, out)
    }

    fn attend_seq_arc_into_ctx_device_pos(
        &self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        q: Arc<Tensor<f16>>,
        position_start: Arc<Tensor<u32>>,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        let position_input = PositionInput::Device(position_start);
        self.attend_seq_arc_into_ctx_with_position(ctx, layer_idx, q, &position_input, out)
    }

    fn attend_seq_arc_into_ctx_with_position(
        &self,
        ctx: &ExecutionContext,
        layer_idx: usize,
        q: Arc<Tensor<f16>>,
        position_input: &PositionInput,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        ensure!(
            q.shape().len() == 3,
            "q must be rank 3 [seqlen, heads, head_dim], got {:?}",
            q.shape()
        );
        let q_len = q.shape()[0] as usize;
        ensure!(
            q.shape()
                == vec![
                    q_len as i32,
                    self.cfg.num_attention_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "q shape mismatch in attend: {:?}",
            q.shape()
        );
        if let PositionInput::Device(position_start) = position_input {
            ensure!(q_len == 1, "decode graph path expects q_len=1, got {q_len}");
            ensure!(
                position_start.shape() == vec![1],
                "attention position tensor must be shape [1], got {:?}",
                position_start.shape()
            );
        }

        let layer = &self.layers[layer_idx];
        let k_cache = layer
            .state
            .k_cache
            .as_ref()
            .context("missing k_cache in layer state")?;
        let v_cache = layer
            .state
            .v_cache
            .as_ref()
            .context("missing v_cache in layer state")?;

        let qk_scale = 1.0f32 / (self.cfg.head_dim as f32).sqrt();
        let query_group_size = self.cfg.num_kv_groups() as i32;
        // Prefill tile: static defaults; tune via
        // GROUT_ATTN_BM_PREFILL / GROUT_ATTN_BN_PREFILL.
        let attn_bn = match position_input {
            PositionInput::Host(_) => {
                if q_len == 1 {
                    env_usize_or("GROUT_ATTN_BN_DECODE", ATTN_BN_DECODE)
                } else {
                    env_usize_or("GROUT_ATTN_BN_PREFILL", ATTN_BN_PREFILL)
                }
            }
            PositionInput::Device(_) => env_usize_or("GROUT_ATTN_BN_DECODE", ATTN_BN_DECODE),
        };
        // ATTN_BM split: prefill can have BM>1 to amortize MMA setup; decode
        // is structurally pinned to 1 (q_len=1). Prefill tunable via
        // GROUT_ATTN_BM_PREFILL.
        let attn_bm = match position_input {
            PositionInput::Host(_) => {
                if q_len == 1 {
                    ATTN_BM_DECODE
                } else {
                    env_usize_or("GROUT_ATTN_BM_PREFILL", ATTN_BM_PREFILL)
                }
            }
            PositionInput::Device(_) => ATTN_BM_DECODE,
        };
        ensure!(
            out.shape()
                == vec![
                    q_len as i32,
                    self.cfg.num_attention_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "attend output shape mismatch, got {:?}",
            out.shape()
        );
        let out_tensor: Tensor<f16> = match position_input {
            PositionInput::Host(position_start) => {
                let kv_len = (*position_start + q_len) as i32;
                // Long-prefill wrappers enable the TileGym-style GQA/LPT
                // path via GROUT_FMHA_PREFILL_GQA_LPT. The engine also
                // auto-enables that path for long sm_100 prefill cells.
                // Otherwise GROUT_FMHA_PREFILL (default ON) routes to the
                // regular Tile IR causal prefill kernel.
                let default_gqa_lpt =
                    q_len >= 2048 && query_group_size > 1 && device_is_sm100(ctx.get_device_id());
                let use_gqa_lpt = env_bool_or("GROUT_FMHA_PREFILL_GQA_LPT", default_gqa_lpt);
                let use_gqa = env_bool_or("GROUT_FMHA_PREFILL_GQA", false);
                let use_prefill_kernel = env_bool_or("GROUT_FMHA_PREFILL", true);
                if use_gqa_lpt {
                    let qgs = query_group_size as usize;
                    let group_env = env_usize_or("GROUT_FMHA_PREFILL_GQA_GROUP", 0);
                    let group = if group_env == 0 { qgs } else { group_env };
                    ensure!(
                        group >= 1 && qgs % group == 0,
                        "GROUT_FMHA_PREFILL_GQA_GROUP={group} must divide \
                         query_group_size={qgs}"
                    );
                    ensure!(
                        self.cfg.num_attention_heads % group == 0,
                        "GROUT_FMHA_PREFILL_GQA_GROUP={group} must divide num_attention_heads={}",
                        self.cfg.num_attention_heads
                    );
                    let m_eff = attn_bm * group;
                    let even_k: i32 = if kv_len % (attn_bn as i32) == 0 { 1 } else { 0 };
                    let prefill_latency =
                        env_usize_or("GROUT_FMHA_PREFILL_LATENCY", FMHA_PREFILL_LATENCY_DEFAULT);
                    let prefill_occupancy = env_usize_hint_or(
                        "GROUT_FMHA_PREFILL_OCCUPANCY",
                        FMHA_PREFILL_OCCUPANCY_DEFAULT,
                    );
                    let prefill_sched = env_usize_or("GROUT_FMHA_PREFILL_LPT_SCHED", 1);
                    ensure!(
                        prefill_sched <= 3,
                        "GROUT_FMHA_PREFILL_LPT_SCHED={prefill_sched} must be in 0..=3"
                    );
                    let prefill_mask_split =
                        if env_bool_or("GROUT_FMHA_PREFILL_LPT_MASK_SPLIT", false) {
                            1
                        } else {
                            0
                        };
                    let num_q_blocks = q_len.div_ceil(attn_bm);
                    let num_head_groups = self.cfg.num_attention_heads / group;
                    let swizzle_default =
                        prefill_lpt_swizzle(q_len, self.cfg.head_dim, num_head_groups);
                    let swizzle_env = env_usize_or("GROUT_FMHA_PREFILL_LPT_SWIZZLE", 0);
                    let swizzle = if swizzle_env == 0 {
                        swizzle_default
                    } else {
                        swizzle_env
                    }
                    .min(num_head_groups)
                    .max(1);
                    let num_hb_quotient = num_head_groups / swizzle;
                    let num_hb_remainder = (num_head_groups % swizzle).max(1);
                    let grid_x = (num_q_blocks * num_head_groups) as u32;
                    unsafe {
                        fmha_prefill_gqa_lpt(
                            q.device_pointer().clone(),
                            k_cache.device_pointer().clone(),
                            v_cache.device_pointer().clone(),
                            out.device_pointer().clone(),
                            value(qk_scale),
                            value(query_group_size),
                            value(q_len as i32),
                            value(kv_len),
                            value(*position_start as i32),
                            value(num_q_blocks as i32),
                            value(num_head_groups as i32),
                            value(swizzle as i32),
                            value(num_hb_quotient as i32),
                            value(num_hb_remainder as i32),
                        )
                        .generics(vec![
                            attn_bm.to_string(),
                            attn_bn.to_string(),
                            self.cfg.head_dim.to_string(),
                            group.to_string(),
                            m_eff.to_string(),
                            1.to_string(), // CAUSAL
                            even_k.to_string(),
                            prefill_latency.to_string(),
                            prefill_sched.to_string(),
                            prefill_mask_split.to_string(),
                        ])
                        .grid((grid_x, 1u32, 1u32))
                        .compile_options(compile_options_with_occupancy(prefill_occupancy))
                        .execute(ctx)?;
                    }
                    out
                } else if use_gqa {
                    let qgs = query_group_size as usize;
                    // GROUP = packing factor (how many q_heads per CTA).
                    // Must divide query_group_size. For Qwen3 qgs=4, valid
                    // values: {1, 2, 4}. Default = qgs (unchanged from old
                    // behavior, kv_head_idx = pid.1 directly).
                    let group_env = env_usize_or("GROUT_FMHA_PREFILL_GQA_GROUP", 0);
                    let group = if group_env == 0 { qgs } else { group_env };
                    ensure!(
                        group >= 1 && qgs % group == 0,
                        "GROUT_FMHA_PREFILL_GQA_GROUP={group} must divide \
                         query_group_size={qgs}"
                    );
                    let m_eff = attn_bm * group;
                    let out_part = out.partition([attn_bm, group, self.cfg.head_dim]);
                    let even_k: i32 = if kv_len % (attn_bn as i32) == 0 { 1 } else { 0 };
                    let prefill_latency =
                        env_usize_or("GROUT_FMHA_PREFILL_LATENCY", FMHA_PREFILL_LATENCY_DEFAULT);
                    let prefill_occupancy = env_usize_hint_or(
                        "GROUT_FMHA_PREFILL_OCCUPANCY",
                        FMHA_PREFILL_OCCUPANCY_DEFAULT,
                    );
                    let result = unsafe {
                        fmha_prefill_gqa(
                            value(q.clone()),
                            value(k_cache.clone()),
                            value(v_cache.clone()),
                            value(out_part),
                            value(qk_scale),
                            value(query_group_size),
                            value(kv_len),
                            value(*position_start as i32),
                        )
                    }
                    .generics(vec![
                        attn_bm.to_string(),
                        attn_bn.to_string(),
                        self.cfg.head_dim.to_string(),
                        group.to_string(),
                        m_eff.to_string(),
                        1.to_string(), // CAUSAL
                        even_k.to_string(),
                        prefill_latency.to_string(),
                    ])
                    .compile_options(compile_options_with_occupancy(prefill_occupancy));
                    let result = unsafe { result.execute(ctx)? };
                    result.3.unpartition()
                } else if use_prefill_kernel {
                    let out_part = out.partition([attn_bm, 1, self.cfg.head_dim]);
                    let even_k: i32 = if kv_len % (attn_bn as i32) == 0 { 1 } else { 0 };
                    let prefill_latency =
                        env_usize_or("GROUT_FMHA_PREFILL_LATENCY", FMHA_PREFILL_LATENCY_DEFAULT);
                    let prefill_occupancy = env_usize_hint_or(
                        "GROUT_FMHA_PREFILL_OCCUPANCY",
                        FMHA_PREFILL_OCCUPANCY_DEFAULT,
                    );
                    let result = unsafe {
                        fmha_prefill_causal(
                            value(q.clone()),
                            value(k_cache.clone()),
                            value(v_cache.clone()),
                            value(out_part),
                            value(qk_scale),
                            value(query_group_size),
                            value(kv_len),
                            value(*position_start as i32),
                        )
                    }
                    .generics(vec![
                        attn_bm.to_string(),
                        attn_bn.to_string(),
                        self.cfg.head_dim.to_string(),
                        1.to_string(), // CAUSAL
                        even_k.to_string(),
                        prefill_latency.to_string(),
                    ])
                    .compile_options(compile_options_with_occupancy(prefill_occupancy));
                    let result = unsafe { result.execute(ctx)? };
                    result.3.unpartition()
                } else {
                    let out_part = out.partition([attn_bm, 1, self.cfg.head_dim]);
                    let result = unsafe {
                        flash_attn_causal_seq_f16(
                            value(q.clone()),
                            value(k_cache.clone()),
                            value(v_cache.clone()),
                            value(out_part),
                            value(qk_scale),
                            value(query_group_size),
                            value(kv_len),
                            value(*position_start as i32),
                        )
                    }
                    .generics(vec![
                        attn_bm.to_string(),
                        attn_bn.to_string(),
                        self.cfg.head_dim.to_string(),
                    ]);
                    let result = unsafe { result.execute(ctx)? };
                    result.3.unpartition()
                }
            }
            PositionInput::Device(position_start) => {
                let out_part = out.partition([attn_bm, 1, self.cfg.head_dim]);
                let result = unsafe {
                    // flash_attn_causal_seq_dynpos_f16_async(
                    //     value(q.clone()),
                    //     value(k_cache.clone()),
                    //     value(v_cache.clone()),
                    //     value(out),
                    //     value(f16::from_f32(qk_scale)),
                    //     value(query_group_size),
                    //     value(position_start.clone()),
                    // )
                    // .generics(vec![
                    //     ATTN_BM_DECODE.to_string(),
                    //     attn_bn.to_string(),
                    //     self.cfg.head_dim.to_string(),
                    // ])

                    // Try this instead...
                    let m = q.shape()[0] as usize;
                    let d = self.cfg.head_dim;
                    fmha_causal(
                        value(q.clone()),
                        value(k_cache.clone()),
                        value(v_cache.clone()),
                        value(out_part),
                        value(f16::from_f32(qk_scale)),
                        value(query_group_size),
                        value(position_start.clone()),
                    )
                    .generics(vec![
                        attn_bm.to_string(),
                        attn_bn.to_string(),
                        d.to_string(),
                        1.to_string(),
                        ((m % attn_bn == 0) as i32).to_string(),
                    ])
                };
                let result = unsafe { result.execute(ctx)? };
                result.3.unpartition()
            }
        };
        Ok(out_tensor)
    }

    fn gather_row_ctx(
        &self,
        ctx: &ExecutionContext,
        src: Arc<Tensor<f16>>,
        row_idx: usize,
    ) -> Result<Tensor<f16>> {
        ensure!(src.shape().len() == 2, "gather_row expects rank-2 tensor");
        let cols = src.shape()[1] as usize;
        let out = alloc_f16_ctx(ctx, &[cols])?;
        self.gather_row_into_ctx(ctx, src, row_idx, out)
    }

    fn gather_row_into_ctx(
        &self,
        ctx: &ExecutionContext,
        src: Arc<Tensor<f16>>,
        row_idx: usize,
        out: Tensor<f16>,
    ) -> Result<Tensor<f16>> {
        ensure!(src.shape().len() == 2, "gather_row expects rank-2 tensor");
        let rows = src.shape()[0] as usize;
        let cols = src.shape()[1] as usize;
        ensure!(row_idx < rows, "row_idx {} out of bounds {}", row_idx, rows);
        ensure!(
            out.shape() == vec![cols as i32],
            "gather_row output shape mismatch, got {:?}",
            out.shape()
        );
        let out = out.partition([VEC_BLOCK]);
        let result = unsafe {
            gather_row_f16(value(src), value(out), value(row_idx as i32))
                .generics(vec![VEC_BLOCK.to_string()])
                .execute(ctx)?
        };
        let out: Partition<Tensor<f16>> = result.1;
        Ok(out.unpartition())
    }

    fn argmax_blocks_ctx(
        &self,
        ctx: &ExecutionContext,
        logits: Arc<Tensor<f16>>,
        len: usize,
    ) -> Result<(Tensor<f32>, Tensor<u32>)> {
        ensure!(len > 0, "argmax expects non-empty logits");
        let argmax_block = env_usize_or("GROUT_ARGMAX_BLOCK", ARGMAX_BLOCK);
        let num_blocks = (len + argmax_block - 1) / argmax_block;
        let block_max = unsafe { api::zeros::<f32>(&[num_blocks]).execute(ctx)? }.partition([1]);
        let block_idx = unsafe { api::zeros::<u32>(&[num_blocks]).execute(ctx)? }.partition([1]);
        let result = unsafe {
            argmax_blocks_f16(value(logits), block_max, block_idx, value(len as i32))
                .generics(vec![argmax_block.to_string()])
                .execute(ctx)?
        };
        let block_max: Partition<Tensor<f32>> = result.1;
        let block_idx: Partition<Tensor<u32>> = result.2;
        Ok((block_max.unpartition(), block_idx.unpartition()))
    }

    async fn argmax_device(&self, logits: Arc<Tensor<f16>>) -> Result<usize> {
        ensure!(
            logits.shape().len() == 1,
            "argmax expects rank-1 logits, got {:?}",
            logits.shape()
        );
        let len = logits.shape()[0] as usize;
        ensure!(len > 0, "argmax expects non-empty logits");

        let (block_max, block_idx) =
            with_context(|ctx| value(self.argmax_blocks_ctx(ctx, logits, len))).await??;
        let host_max = block_max.to_host_vec().await?;
        let host_idx = block_idx.to_host_vec().await?;
        let num_blocks = host_max.len();

        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0usize;
        for i in 0..num_blocks {
            let idx = host_idx[i] as usize;
            if idx >= len {
                continue;
            }
            let val = host_max[i];
            if val > best_val || (val == best_val && idx < best_idx) {
                best_val = val;
                best_idx = idx;
            }
        }
        Ok(best_idx)
    }
}

fn graph_op_name(op: &GraphOp) -> &'static str {
    match op {
        GraphOp::EmbeddingBatch { .. } => "EmbeddingBatch",
        GraphOp::MatMul { .. } => "MatMul",
        GraphOp::MatVec { .. } => "MatVec",
        GraphOp::Add { .. } => "Add",
        GraphOp::SiluMul { .. } => "SiluMul",
        GraphOp::RmsNorm { .. } => "RmsNorm",
        GraphOp::Reshape { .. } => "Reshape",
        GraphOp::Rope { .. } => "Rope",
        GraphOp::KvCacheUpdate { .. } => "KvCacheUpdate",
        GraphOp::QkNormRopeKvPrefill { .. } => "QkNormRopeKvPrefill",
        GraphOp::Attention { .. } => "Attention",
        GraphOp::GatherRow { .. } => "GatherRow",
        GraphOp::SliceCols { .. } => "SliceCols",
        GraphOp::MatMulSlice { .. } => "MatMulSlice",
        GraphOp::AddRmsNorm { .. } => "AddRmsNorm",
    }
}

fn build_inv_freq(stream: &Arc<cuda_core::Stream>, cfg: &Qwen3Config) -> Result<Arc<Tensor<f32>>> {
    let mut inv = Vec::with_capacity(cfg.head_dim / 2);
    for i in (0..cfg.head_dim).step_by(2) {
        let p = (i as f32) / (cfg.head_dim as f32);
        inv.push(1.0f32 / cfg.rope_theta.powf(p));
    }
    let inv = Arc::new(inv);
    let t = api::copy_host_vec_to_device(&inv)
        .sync_on(stream)
        .map_err(|e| anyhow::anyhow!("copy inv_freq to device failed: {e:?}"))?;
    Ok(Arc::new(t))
}

fn load_layer_weight(
    loader: &WeightLoader,
    stream: &Arc<cuda_core::Stream>,
    idx: usize,
    suffix: &str,
    human_name: &str,
) -> Result<Arc<Tensor<f16>>> {
    let name = format!("model.layers.{idx}.{suffix}");
    loader
        .load_device_f16(&name, stream)
        .with_context(|| format!("failed to load {human_name} ({name})"))
}

/// Concatenates multiple rank-2 weight tensors along dimension 0 (rows) on the GPU.
/// All inputs must have the same number of columns.
/// Returns an Arc<Tensor> of shape [sum_of_rows, cols].
fn concat_weight_rows_2d(
    stream: &Arc<cuda_core::Stream>,
    tensors: &[&Arc<Tensor<f16>>],
) -> Result<Arc<Tensor<f16>>> {
    ensure!(
        !tensors.is_empty(),
        "concat_weight_rows_2d requires at least one tensor"
    );
    let cols = tensors[0].shape()[1] as usize;
    let mut total_rows = 0usize;
    let mut src_parts: Vec<(u64, usize)> = Vec::with_capacity(tensors.len());
    for t in tensors {
        ensure!(
            t.shape().len() == 2,
            "concat_weight_rows_2d expects rank-2 tensors, got {:?}",
            t.shape()
        );
        ensure!(
            t.shape()[1] as usize == cols,
            "concat_weight_rows_2d: mismatched columns {} vs {}",
            t.shape()[1],
            cols
        );
        let t_rows = t.shape()[0] as usize;
        let t_bytes = t_rows * cols * size_of::<f16>();
        src_parts.push((t.device_pointer().cu_deviceptr(), t_bytes));
        total_rows += t_rows;
    }

    let total_elements = total_rows * cols;
    let total_bytes = total_elements * size_of::<f16>();

    // Allocate the merged tensor and copy each source into it.
    let ctx = cuda_async::device_operation::ExecutionContext::new(stream.clone());
    let dst_ptr = unsafe { cuda_core::malloc_async(total_bytes, stream) };
    let mut offset_bytes = 0u64;
    for (src_ptr, t_bytes) in &src_parts {
        unsafe {
            memcpy_dtod_async::<u8>(dst_ptr + offset_bytes, *src_ptr, *t_bytes, stream);
        }
        offset_bytes += *t_bytes as u64;
    }
    let merged = unsafe {
        Tensor::<f16>::from_raw_parts(
            dst_ptr,
            total_bytes,
            ctx.get_device_id(),
            vec![total_rows as i32, cols as i32],
            vec![cols as i32, 1],
        )
    };
    Ok(Arc::new(merged))
}

fn argmax_f16(values: &[f16]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in values.iter().enumerate() {
        let vf = v.to_f32();
        if vf > best_v {
            best_v = vf;
            best_i = i;
        }
    }
    best_i
}

fn summarize_logits(logits: &[f16], tokenizer: &Tokenizer) -> Result<String> {
    if logits.is_empty() {
        return Ok("empty".to_string());
    }
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut min_i = 0usize;
    let mut max_i = 0usize;
    let mut sum_abs = 0.0f64;
    for (i, v) in logits.iter().enumerate() {
        let x = v.to_f32();
        sum_abs += (x as f64).abs();
        if x < min_v {
            min_v = x;
            min_i = i;
        }
        if x > max_v {
            max_v = x;
            max_i = i;
        }
    }
    let max_tok = tokenizer
        .decode(&[max_i as u32], false)
        .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
    let min_tok = tokenizer
        .decode(&[min_i as u32], false)
        .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
    Ok(format!(
        "max[{max_i}]={max_v:.6} token={max_tok:?}, min[{min_i}]={min_v:.6} token={min_tok:?}, sum_abs={sum_abs:.6}"
    ))
}
