use crate::config::{GenerationConfig, Qwen3Config};
use crate::cublas;
use crate::cuda_graph::CudaGraphExec;
use crate::kernels::{
    KernelKind, TILE_KERNEL_KINDS, add_2d_f16, argmax_blocks_f16, embedding_batch_f16,
    flash_attn_causal_seq_dynpos_f16, flash_attn_causal_seq_f16, gather_row_f16,
    kv_cache_update_seq_dynpos_f16, kv_cache_update_seq_f16, rms_norm_f16, rope_seq_dynpos_f16,
    rope_seq_f16, silu_mul_2d_f16,
};
use crate::loader::WeightLoader;
use anyhow::{Context, Result, bail, ensure};
use cuda_async::device_operation::{DeviceOp, ExecutionContext, value, with_context};
use cuda_core::memcpy_htod_async;
use cutile::api;
use cutile::half::f16;
use cutile::tensor::{IntoPartition, IntoPartitionArc, Partition, Reshape, Tensor, ToHostVec};
use cutile::tile_kernel::{PartitionOp, TileKernel};
use rand::Rng;
use std::cmp::{Reverse, min};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const VEC_BLOCK: usize = 128;
const POINTWISE_BLOCK: usize = 256;
const RMS_BLOCK: usize = 128;
const ROPE_BLOCK: usize = 128;
const ARGMAX_BLOCK: usize = 128;
const ATTN_BM: usize = 1;
const ATTN_BN_PREFILL: usize = 32;
const ATTN_BN_DECODE: usize = 32;

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

struct DecodeCudaGraphRunner {
    graph: CudaGraphExec,
    _pool_keepalive: TensorPool,
    token_host: [u32; 1],
    position_host: [u32; 1],
    token_ids_device: Arc<Tensor<u32>>,
    position_device: Arc<Tensor<u32>>,
    logits: Arc<Tensor<f16>>,
}

impl DecodeCudaGraphRunner {
    fn launch_step(&mut self, token_id: u32, position_start: usize) -> Result<Arc<Tensor<f16>>> {
        ensure!(
            position_start <= u32::MAX as usize,
            "position_start {} exceeds u32 range",
            position_start
        );
        self.token_host[0] = token_id;
        self.position_host[0] = position_start as u32;
        unsafe {
            memcpy_htod_async(
                self.token_ids_device.device_pointer().cu_deviceptr(),
                self.token_host.as_ptr(),
                1,
                self.graph.stream(),
            );
            memcpy_htod_async(
                self.position_device.device_pointer().cu_deviceptr(),
                self.position_host.as_ptr(),
                1,
                self.graph.stream(),
            );
        }
        self.graph.launch()?;
        Ok(self.logits.clone())
    }

    fn synchronize(&self) -> Result<()> {
        let stream = self.graph.stream();
        stream
            .device()
            .bind_to_thread()
            .map_err(|e| anyhow::anyhow!("failed to bind CUDA context: {e:?}"))?;
        unsafe {
            stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("stream synchronize failed: {e:?}"))
        }
    }
}

enum TokenInput<'a> {
    Host(&'a [u32]),
    Device(Arc<Tensor<u32>>),
}

#[derive(Clone)]
enum PositionInput {
    Host(usize),
    Device(Arc<Tensor<u32>>),
}

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
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WeightRef {
    EmbedTokens,
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
            | Self::Attention { out, .. }
            | Self::GatherRow { out, .. } => Some(*out),
            Self::KvCacheUpdate { .. } => None,
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
            Self::Attention { q, .. } => {
                maybe_push(&mut values, *q);
            }
            Self::GatherRow { src, .. } => {
                maybe_push(&mut values, *src);
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
            return Ok(t.reshape(&spec.shape).unwrap());
        }
        if let Some(t) = self.take_compatible(spec) {
            return Ok(t.reshape(&spec.shape).unwrap());
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
            bin.push(tensor.reshape(&spec.shape).unwrap());
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
    Ok(unsafe { api::zeros::<f16>(shape).execute(ctx)? })
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
    q_proj: Arc<Tensor<f16>>,
    k_proj: Arc<Tensor<f16>>,
    v_proj: Arc<Tensor<f16>>,
    o_proj: Arc<Tensor<f16>>,
    gate_proj: Arc<Tensor<f16>>,
    up_proj: Arc<Tensor<f16>>,
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
    pub fn prompt_tps(&self) -> f64 {
        let secs = self.prompt_elapsed.as_secs_f64().max(1.0e-9);
        self.prompt_tokens as f64 / secs
    }

    pub fn decode_tps(&self) -> f64 {
        let secs = self.decode_elapsed.as_secs_f64().max(1.0e-9);
        self.generated_tokens as f64 / secs
    }

    pub fn total_tps(&self) -> f64 {
        let secs = self.total_elapsed.as_secs_f64().max(1.0e-9);
        self.generated_tokens as f64 / secs
    }
}

pub struct Qwen3Engine {
    cfg: Qwen3Config,
    tokenizer: Tokenizer,
    model_dir: std::path::PathBuf,
    embed_tokens: Arc<Tensor<f16>>,
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
    profile_enabled: bool,
    active_profile: Option<RunProfile>,
    kernel_warm_registry: KernelWarmRegistry,
    step_graph_cache: HashMap<usize, Arc<StepGraph>>,
    step_pool_cache: HashMap<usize, TensorPool>,
}

impl Qwen3Engine {
    pub async fn load(model_dir: &Path, max_seq_len: Option<usize>) -> Result<Self> {
        let cfg = Qwen3Config::from_model_dir(model_dir)?;
        let generation_cfg = GenerationConfig::from_model_dir(model_dir)?;
        ensure!(
            cfg.tie_word_embeddings,
            "this prototype currently requires tie_word_embeddings=true"
        );
        ensure!(
            !cfg.use_sliding_window,
            "sliding-window attention is not supported in this prototype"
        );
        ensure!(cfg.head_dim == ROPE_BLOCK, "expected head_dim={ROPE_BLOCK}");
        ensure!(
            cfg.hidden_size % VEC_BLOCK == 0,
            "hidden_size must be divisible by {VEC_BLOCK}"
        );
        ensure!(
            cfg.hidden_size % POINTWISE_BLOCK == 0,
            "hidden_size must be divisible by {POINTWISE_BLOCK}"
        );
        ensure!(
            cfg.intermediate_size % VEC_BLOCK == 0,
            "intermediate_size must be divisible by {VEC_BLOCK}"
        );
        ensure!(
            cfg.intermediate_size % POINTWISE_BLOCK == 0,
            "intermediate_size must be divisible by {POINTWISE_BLOCK}"
        );
        ensure!(
            cfg.vocab_size % VEC_BLOCK == 0,
            "vocab_size must be divisible by {VEC_BLOCK}"
        );
        ensure!(
            cfg.head_dim % RMS_BLOCK == 0,
            "head_dim must be divisible by {RMS_BLOCK}"
        );

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
        let use_device_argmax = std::env::var("GROUT_USE_DEVICE_ARGMAX")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(false);
        let profile_enabled = std::env::var("GROUT_PROFILE")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(false);

        let embed_tokens = loader
            .load_device_f16("model.embed_tokens.weight")
            .await
            .context("failed to load model.embed_tokens.weight")?;
        let norm = loader
            .load_device_f16("model.norm.weight")
            .await
            .context("failed to load model.norm.weight")?;

        let inv_freq = build_inv_freq(&cfg).await?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let weights = LayerWeights {
                input_layernorm: load_layer_weight(
                    &loader,
                    i,
                    "input_layernorm.weight",
                    "input layernorm",
                )
                .await?,
                post_attention_layernorm: load_layer_weight(
                    &loader,
                    i,
                    "post_attention_layernorm.weight",
                    "post-attention layernorm",
                )
                .await?,
                q_norm: load_layer_weight(&loader, i, "self_attn.q_norm.weight", "q_norm").await?,
                k_norm: load_layer_weight(&loader, i, "self_attn.k_norm.weight", "k_norm").await?,
                q_proj: load_layer_weight(&loader, i, "self_attn.q_proj.weight", "q_proj").await?,
                k_proj: load_layer_weight(&loader, i, "self_attn.k_proj.weight", "k_proj").await?,
                v_proj: load_layer_weight(&loader, i, "self_attn.v_proj.weight", "v_proj").await?,
                o_proj: load_layer_weight(&loader, i, "self_attn.o_proj.weight", "o_proj").await?,
                gate_proj: load_layer_weight(&loader, i, "mlp.gate_proj.weight", "gate_proj")
                    .await?,
                up_proj: load_layer_weight(&loader, i, "mlp.up_proj.weight", "up_proj").await?,
                down_proj: load_layer_weight(&loader, i, "mlp.down_proj.weight", "down_proj")
                    .await?,
            };

            let k_cache =
                api::zeros::<f16>(&[cfg.num_key_value_heads, max_seq_len, cfg.head_dim]).await?;
            let v_cache =
                api::zeros::<f16>(&[cfg.num_key_value_heads, max_seq_len, cfg.head_dim]).await?;
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
            profile_enabled,
            active_profile: None,
            kernel_warm_registry: KernelWarmRegistry::default(),
            step_graph_cache: HashMap::new(),
            step_pool_cache: HashMap::new(),
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
        let _ = with_context(|ctx| value(self.warm_tile_kernels_ctx(ctx))).await?;

        self.reset_cache().await?;
        Ok(())
    }

    fn warm_tile_kernels_ctx(&mut self, ctx: &ExecutionContext) -> Result<()> {
        for kind in TILE_KERNEL_KINDS {
            self.warm_tile_kernel_ctx(ctx, kind)?;
        }
        if env_bool_or("GROUT_CUDA_GRAPH_DECODE", false) {
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
                let q_proj = self.layers[0].weights.q_proj.clone();
                let _ = self.gemm_ctx(ctx, q_proj, x)?;
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
        }

        self.kernel_warm_registry.mark_warmed(kind);
        Ok(())
    }

    pub async fn reset_cache(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.state.k_cache = Some(Arc::new(
                api::zeros::<f16>(&[
                    self.cfg.num_key_value_heads,
                    self.max_seq_len,
                    self.cfg.head_dim,
                ])
                .await?,
            ));
            layer.state.v_cache = Some(Arc::new(
                api::zeros::<f16>(&[
                    self.cfg.num_key_value_heads,
                    self.max_seq_len,
                    self.cfg.head_dim,
                ])
                .await?,
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
        self.reset_cache().await?;
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
        let use_cuda_graph_decode =
            env_bool_or("GROUT_CUDA_GRAPH_DECODE", false) && max_new_tokens > 0;
        let mut decode_graph_runner = if use_cuda_graph_decode {
            let stream = with_context(|ctx| value(ctx.get_cuda_stream().clone())).await?;
            let capture_ctx = ExecutionContext::new(stream);
            match self.build_decode_graph_runner_ctx(&capture_ctx, cur_pos) {
                Ok(runner) => Some(runner),
                Err(err) => {
                    eprintln!(
                        "warning: failed to initialize CUDA decode graph ({err:#}); falling back"
                    );
                    None
                }
            }
        } else {
            None
        };

        let decode_start = Instant::now();
        let mut generated_ids: Vec<u32> = Vec::new();
        let mut rng = rand::thread_rng();
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
            if let Some(runner) = decode_graph_runner.as_mut() {
                let graph_res = runner.launch_step(next, cur_pos).and_then(|new_logits| {
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
                        decode_graph_runner = None;
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

    fn build_decode_graph_runner_ctx(
        &mut self,
        ctx: &ExecutionContext,
        position_start: usize,
    ) -> Result<DecodeCudaGraphRunner> {
        let _ = cuda_async::device_context::with_global_device_context(ctx.get_device_id(), |_| ());
        ensure!(
            position_start <= u32::MAX as usize,
            "position_start {} exceeds u32 range",
            position_start
        );

        let seqlen = 1usize;
        let graph = self.get_or_build_step_graph(seqlen)?;

        let mut pool = TensorPool::from_plan_ctx(ctx, &graph.pool_plan)?;
        let token_host = [0u32; 1];
        let position_host = [position_start as u32; 1];
        let token_init = Arc::new(vec![token_host[0]]);
        let position_init = Arc::new(vec![position_host[0]]);
        let token_ids_device =
            Arc::new(unsafe { api::copy_host_vec_to_device(&token_init).execute(ctx)? });
        let position_device =
            Arc::new(unsafe { api::copy_host_vec_to_device(&position_init).execute(ctx)? });

        // Prime stream-local allocations and pool bins before capture so replay does not
        // rely on first-launch allocation side effects.
        let mut prime_logits = Some(alloc_f16_ctx(ctx, &graph.spec(graph.final_value).shape)?);
        let _ = self.execute_step_graph_decode_capture_ctx(
            ctx,
            graph.as_ref(),
            &mut pool,
            token_ids_device.clone(),
            position_device.clone(),
            &mut prime_logits,
        )?;
        ctx.device()
            .bind_to_thread()
            .map_err(|e| anyhow::anyhow!("failed to bind CUDA context: {e:?}"))?;
        unsafe {
            ctx.get_cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("stream synchronize failed: {e:?}"))?;
        }

        let mut final_logits = Some(alloc_f16_ctx(ctx, &graph.spec(graph.final_value).shape)?);
        let mut captured_logits: Option<Arc<Tensor<f16>>> = None;
        let stream = ctx.get_cuda_stream().clone();
        let graph_exec = CudaGraphExec::capture(stream, || {
            let logits = self.execute_step_graph_decode_capture_ctx(
                ctx,
                graph.as_ref(),
                &mut pool,
                token_ids_device.clone(),
                position_device.clone(),
                &mut final_logits,
            )?;
            captured_logits = Some(logits);
            Ok(())
        })?;
        let logits = captured_logits.context("decode graph capture produced no logits output")?;

        Ok(DecodeCudaGraphRunner {
            graph: graph_exec,
            _pool_keepalive: pool,
            token_host,
            position_host,
            token_ids_device,
            position_device,
            logits,
        })
    }

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

        let mut hidden = push_value(&mut specs, vec![seqlen, hidden_size]);
        ops.push(GraphOp::EmbeddingBatch { out: hidden });

        for layer_idx in 0..self.cfg.num_hidden_layers {
            let o_proj = &self.layers[layer_idx].weights.o_proj;
            ensure!(
                o_proj.shape().len() == 2 && o_proj.shape()[1] as usize == attn_width,
                "o_proj expected input dim {}, got shape {:?}",
                attn_width,
                o_proj.shape()
            );

            let normed = push_value(&mut specs, vec![seqlen, hidden_size]);
            ops.push(GraphOp::RmsNorm {
                x: v(hidden),
                weight: lw(layer_idx, LayerWeightSlot::InputLayerNorm),
                n: hidden_size,
                out: normed,
            });

            let q_2d = push_value(&mut specs, vec![seqlen, attn_width]);
            let k_2d = push_value(&mut specs, vec![seqlen, kv_width]);
            let v_2d = push_value(&mut specs, vec![seqlen, kv_width]);
            ops.push(GraphOp::MatMul {
                matrix: lw(layer_idx, LayerWeightSlot::QProj),
                rhs: v(normed),
                out: q_2d,
            });
            ops.push(GraphOp::MatMul {
                matrix: lw(layer_idx, LayerWeightSlot::KProj),
                rhs: v(normed),
                out: k_2d,
            });
            ops.push(GraphOp::MatMul {
                matrix: lw(layer_idx, LayerWeightSlot::VProj),
                rhs: v(normed),
                out: v_2d,
            });

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

            let q_rope = push_value(&mut specs, vec![seqlen, attn_heads, head_dim]);
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

            let hidden_after_attn = push_value(&mut specs, vec![seqlen, hidden_size]);
            ops.push(GraphOp::Add {
                lhs: v(hidden),
                rhs: v(attn_proj),
                out: hidden_after_attn,
            });

            let ff_normed = push_value(&mut specs, vec![seqlen, hidden_size]);
            ops.push(GraphOp::RmsNorm {
                x: v(hidden_after_attn),
                weight: lw(layer_idx, LayerWeightSlot::PostAttentionLayerNorm),
                n: hidden_size,
                out: ff_normed,
            });

            let gate = push_value(&mut specs, vec![seqlen, inter_size]);
            let up = push_value(&mut specs, vec![seqlen, inter_size]);
            ops.push(GraphOp::MatMul {
                matrix: lw(layer_idx, LayerWeightSlot::GateProj),
                rhs: v(ff_normed),
                out: gate,
            });
            ops.push(GraphOp::MatMul {
                matrix: lw(layer_idx, LayerWeightSlot::UpProj),
                rhs: v(ff_normed),
                out: up,
            });

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
            matrix: TensorRef::Weight(WeightRef::EmbedTokens),
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
        let mut final_logits_policy = FinalLogitsPolicy::Allocate;
        self.execute_step_graph_common_ctx(
            ctx,
            graph,
            pool,
            TokenInput::Host(token_ids),
            PositionInput::Host(position_start),
            &mut final_logits_policy,
            profile_ops,
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
                    let reshaped = self.take_or_copy_f16_ctx(ctx, src)?.reshape(shape).unwrap();
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
            }
            if let Some(op_start) = op_start {
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
            WeightRef::EmbedTokens => self.embed_tokens.clone(),
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
                    LayerWeightSlot::QProj => layer.q_proj.clone(),
                    LayerWeightSlot::KProj => layer.k_proj.clone(),
                    LayerWeightSlot::VProj => layer.v_proj.clone(),
                    LayerWeightSlot::OProj => layer.o_proj.clone(),
                    LayerWeightSlot::GateProj => layer.gate_proj.clone(),
                    LayerWeightSlot::UpProj => layer.up_proj.clone(),
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
            out.shape() == [seqlen as i32, self.cfg.hidden_size as i32],
            "embedding output shape mismatch, got {:?}",
            out.shape()
        );

        let ids_host = Arc::new(token_ids.to_vec());
        let ids = unsafe { api::copy_host_vec_to_device(&ids_host).execute(ctx)? };
        let out = out.partition([1, VEC_BLOCK]);
        let result = unsafe {
            embedding_batch_f16(
                value(Arc::new(ids)),
                value(self.embed_tokens.clone()),
                value(out),
            )
            .generics(vec![
                self.cfg.hidden_size.to_string(),
                VEC_BLOCK.to_string(),
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
            out.shape() == [seqlen as i32, self.cfg.hidden_size as i32],
            "embedding output shape mismatch, got {:?}",
            out.shape()
        );

        let out = out.partition([1, VEC_BLOCK]);
        let result = unsafe {
            embedding_batch_f16(
                value(token_ids),
                value(self.embed_tokens.clone()),
                value(out),
            )
            .generics(vec![
                self.cfg.hidden_size.to_string(),
                VEC_BLOCK.to_string(),
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
            out.shape() == [m as i32],
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
            out.shape() == [n as i32, m as i32],
            "gemm output shape mismatch, got {:?}",
            out.shape()
        );
        let op = cublas::gemm_f16_op(matrix, rhs, out, m, n, k)?;
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
            cols.is_multiple_of(POINTWISE_BLOCK),
            "add cols {cols} not divisible by {POINTWISE_BLOCK}"
        );
        ensure!(
            out.shape() == [rows as i32, cols as i32],
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
            cols.is_multiple_of(POINTWISE_BLOCK),
            "silu_mul cols {cols} not divisible by {POINTWISE_BLOCK}"
        );
        ensure!(
            out.shape() == [rows as i32, cols as i32],
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
        ensure!(
            n.is_multiple_of(RMS_BLOCK),
            "rms_norm n={n} must be divisible by {RMS_BLOCK}"
        );
        let orig_shape: Vec<usize> = x.shape().iter().map(|d| *d as usize).collect();
        let (x, rows) = match orig_shape.as_slice() {
            [d] => {
                ensure!(*d == n, "rms_norm expected dim {n}, got {d}");
                (
                    Arc::new(self.copy_f16_ctx(ctx, &x)?.reshape(&[1, n]).unwrap()),
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
        let out = out.reshape(&[rows, n]).unwrap().partition([1, n]);
        let result = unsafe {
            rms_norm_f16(
                value(x),
                value(weight),
                value(out),
                value(self.cfg.rms_norm_eps),
            )
            .generics(vec![n.to_string(), RMS_BLOCK.to_string()])
            .execute(ctx)?
        };
        let _x: Arc<Tensor<f16>> = result.0;
        let out: Partition<Tensor<f16>> = result.2;
        Ok(out.unpartition().reshape(&orig_shape).unwrap())
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
                position_start.shape() == [1],
                "rope position tensor must be shape [1], got {:?}",
                position_start.shape()
            );
        }
        let seq_len = x.shape()[0] as usize;
        let num_heads = x.shape()[1] as usize;
        ensure!(
            out.shape() == [seq_len as i32, num_heads as i32, self.cfg.head_dim as i32],
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
                == [
                    seq_len as i32,
                    self.cfg.num_key_value_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "new_k shape mismatch: {:?}",
            new_k.shape()
        );
        ensure!(
            new_v.shape()
                == [
                    seq_len as i32,
                    self.cfg.num_key_value_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "new_v shape mismatch: {:?}",
            new_v.shape()
        );
        ensure!(
            self.cfg.head_dim.is_multiple_of(VEC_BLOCK),
            "head_dim must be divisible by {VEC_BLOCK}"
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
                    position_start.shape() == [1],
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
        let k_cache_part = k_cache.partition([1, self.max_seq_len, VEC_BLOCK]);
        let v_cache_part = v_cache.partition([1, self.max_seq_len, VEC_BLOCK]);
        let (k_cache, v_cache): (Partition<Tensor<f16>>, Partition<Tensor<f16>>) =
            match position_input {
                PositionInput::Host(position_start) => {
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
                            self.max_seq_len.to_string(),
                        ])
                        .execute(ctx)?
                    };
                    (result.2, result.3)
                }
                PositionInput::Device(position_start) => {
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
                            VEC_BLOCK.to_string(),
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
                == [
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
                position_start.shape() == [1],
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
        ensure!(
            out.shape()
                == [
                    q_len as i32,
                    self.cfg.num_attention_heads as i32,
                    self.cfg.head_dim as i32
                ],
            "attend output shape mismatch, got {:?}",
            out.shape()
        );
        let out = out.partition([1, 1, self.cfg.head_dim]);
        let out: Partition<Tensor<f16>> = match position_input {
            PositionInput::Host(position_start) => {
                let kv_len = (*position_start + q_len) as i32;
                let result = unsafe {
                    flash_attn_causal_seq_f16(
                        value(q.clone()),
                        value(k_cache.clone()),
                        value(v_cache.clone()),
                        value(out),
                        value(qk_scale),
                        value(query_group_size),
                        value(kv_len),
                        value(*position_start as i32),
                    )
                }
                .generics(vec![
                    ATTN_BM.to_string(),
                    attn_bn.to_string(),
                    self.cfg.head_dim.to_string(),
                ]);
                let result = unsafe { result.execute(ctx)? };
                result.3
            }
            PositionInput::Device(position_start) => {
                let result = unsafe {
                    flash_attn_causal_seq_dynpos_f16(
                        value(q.clone()),
                        value(k_cache.clone()),
                        value(v_cache.clone()),
                        value(out),
                        value(qk_scale),
                        value(query_group_size),
                        value(position_start.clone()),
                    )
                }
                .generics(vec![
                    ATTN_BM.to_string(),
                    attn_bn.to_string(),
                    self.cfg.head_dim.to_string(),
                ]);
                let result = unsafe { result.execute(ctx)? };
                result.3
            }
        };
        Ok(out.unpartition())
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
            cols.is_multiple_of(VEC_BLOCK),
            "gather_row cols {cols} not divisible by {VEC_BLOCK}"
        );
        ensure!(
            out.shape() == [cols as i32],
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
        ensure!(
            len.is_multiple_of(ARGMAX_BLOCK),
            "argmax kernel path expects len divisible by {ARGMAX_BLOCK}, got {len}"
        );

        let num_blocks = len / ARGMAX_BLOCK;
        let block_max = api::zeros::<f32>(&[num_blocks]).partition([1]);
        let block_idx = api::zeros::<u32>(&[num_blocks]).partition([1]);
        let result = unsafe {
            argmax_blocks_f16(value(logits), block_max, block_idx, value(len as i32))
                .generics(vec![ARGMAX_BLOCK.to_string()])
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
        if !len.is_multiple_of(ARGMAX_BLOCK) {
            let host = logits.to_host_vec().await?;
            return Ok(argmax_f16(&host));
        }

        let (block_max, block_idx) =
            with_context(|ctx| value(self.argmax_blocks_ctx(ctx, logits, len))).await??;
        let host_max: Vec<f32> = block_max.to_host_vec().await?;
        let host_idx: Vec<u32> = block_idx.to_host_vec().await?;
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
        GraphOp::Attention { .. } => "Attention",
        GraphOp::GatherRow { .. } => "GatherRow",
    }
}

async fn build_inv_freq(cfg: &Qwen3Config) -> Result<Arc<Tensor<f32>>> {
    let mut inv = Vec::with_capacity(cfg.head_dim / 2);
    for i in (0..cfg.head_dim).step_by(2) {
        let p = (i as f32) / (cfg.head_dim as f32);
        inv.push(1.0f32 / cfg.rope_theta.powf(p));
    }
    let inv = Arc::new(inv);
    Ok(Arc::new(api::copy_host_vec_to_device(&inv).await?))
}

async fn load_layer_weight(
    loader: &WeightLoader,
    idx: usize,
    suffix: &str,
    human_name: &str,
) -> Result<Arc<Tensor<f16>>> {
    let name = format!("model.layers.{idx}.{suffix}");
    loader
        .load_device_f16(&name)
        .await
        .with_context(|| format!("failed to load {human_name} ({name})"))
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
