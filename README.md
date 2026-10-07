# Grout

Qwen3 inference engine built on
[cuTile Rust](https://github.com/NVlabs/cutile-rs) 0.4.0.

## Requirements

- **CUDA**: 13.2 is the minimum the cuTile compiler accepts for Blackwell;
  13.3 is recommended (and required for FP4 packing and block-scaled MMA in
  cuTile Rust); the DGX Spark tutorial was tested on 13.4. Tuning records
  carry the `tileiras` fingerprint they were produced with and warn (not
  refuse) when the host's toolkit differs.

## Setup

Place the model directories alongside this repo:

```
parent/
  grout/        # this repo
  hf_models/
    qwen3_4b/   # RTX 5090 / sm_120 benchmark model
    qwen3_32b/  # B200 / sm_100 benchmark model
```

Environment is configured via `.cargo/config.toml`:

| Variable | Default | Purpose |
|---|---|---|
| `CUDA_TOOLKIT_PATH` | `/usr/local/cuda-13` | CUDA toolkit location |
| `GROUT_CUBLAS_COMPUTE16` | auto | cuBLAS accumulation mode for decode GEMVs |
| `GROUT_ATTN_BN_DECODE` | `32` | Attention tile size for decode |

## Build and run

```bash
cargo run --release -- \
  --model "../hf_models/qwen3_4b" \
  --prompt "Hello, how are you?" \
  --max-new-tokens 50
```

The first run compiles all cuTile kernels (MLIR -> PTX -> CUBIN). Subsequent runs use the kernel cache.

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--model <PATH>` | (required) | Path to model directory |
| `--prompt <TEXT>` | (required) | Input prompt |
| `--max-new-tokens <N>` | `128` | Maximum tokens to generate |
| `--max-seq-len <N>` | model default | Override max sequence length |
| `--sample` | `false` | Enable sampling (temperature/top-k) |
| `--raw-prompt` | `false` | Skip chat template wrapping |
| `--device-argmax` | `false` | Run greedy argmax on the GPU |
| `--profile` | `false` | Print prefill/decode step averages; with `GROUT_PROFILE_OPS=1` also a per-op table (StepGraph ops only: prefill and warm-up, since decode replays a CUDA graph) |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GROUT_CUBLAS_COMPUTE16` | auto | `1` = fp16 accumulate, `0` = fp32. Auto uses fp16 accumulation |
| `GROUT_CUBLAS_COMPUTE16_MAX_M` | unset | Max M dimension for fp16 accumulation |
| `GROUT_CUBLAS_FAST_ALGO` | `default_tensor_op` | cuBLAS algorithm selection |
| `GROUT_FUSED_LM_HEAD_ARGMAX` | `0` | Experimental greedy decode path that fuses LM-head scoring with block argmax and skips materializing logits |
| `GROUT_ATTN_BN_DECODE` | `32` | KV tile size for decode attention |
| `GROUT_PROFILE_OPS` | `0` | `1` = collect the per-op table printed by `--profile` (StepGraph ops only; graph-replayed decode steps are not itemized) |
| `GROUT_PROFILE_SYNC_OPS` | `0` | `1` = synchronize the stream after each profiled op so per-op times are kernel times rather than launch overhead |
| `GROUT_TUNING_RECORD_DIR` | `benchmarks/tuning` | Root of the per-arch tuning records (`<root>/sm_<xy>/*.json`); a missing arch directory warns and falls back to built-in defaults |
| `GROUT_DEBUG_POOL_ALLOC` | `0` | `1` = log tensor pool fallback allocations |

## Architecture

```
main.rs         CLI (clap + tokio)
config.rs       Qwen3Config deserialization from config.json
loader.rs       SafeTensors weight loading (mmap -> fp16 -> GPU)
kernels.rs      cuTile Rust GPU kernels (#[cutile::module])
cublas.rs       cuBLAS GEMM/GEMV wrapper (via cudarc)
model.rs        Qwen3Engine: StepGraph IR, forward pass, generation loop
```

### Execution model

The engine uses a **StepGraph IR** — a sequence of ops (GEMM, RmsNorm, RoPE, Attention, etc.) compiled once per sequence length class (prefill vs decode). For decode, the graph is captured as a **CUDA graph** for replay without CPU overhead.

Key execution path:
1. **Prefill**: Encode the full prompt in one pass (batched GEMM + flash attention)
2. **CUDA graph capture**: Run one decode step to capture the graph
3. **Decode loop**: Replay the captured graph, updating only token ID and position via `memcpy_htod_async`

### Kernels

All GPU kernels are written in cuTile Rust's DSL (`#[cutile::module]`), which compiles Rust to MLIR to PTX:

- `embedding_batch_f16` — batched token embedding lookup
- `rms_norm_f16` — RMS normalization
- `add_rms_norm_f16` — fused residual add + RMS norm
- `add_rms_norm_decode_raw_f16` — decode-specialized fused residual add + RMS norm
- `rope_seq_f16` / `rope_seq_dynpos_f16` — rotary position embeddings
- `kv_cache_update_seq_f16` / `kv_cache_update_seq_dynpos_f16` — KV cache write
- `qk_norm_rope_kv_prefill_raw_f16` / `qk_norm_rope_kv_decode_raw_f16` — fused Q/K norm + RoPE + KV-cache write paths
- `flash_attn_causal_seq_f16`, `fmha_prefill_*`, `fmha_decode_gqa_split` — prefill/decode attention kernels
- `splitk_reduce_merge` — decode split-K attention merge
- `silu_mul_2d_f16` — SiLU activation * up projection
- `add_2d_f16` — element-wise add
- `argmax_blocks_f16` — block-parallel argmax for greedy decoding
- `gather_row_f16` — single-row extraction

cuBLAS handles the linear projections (GEMM/GEMV) via `cublas.rs`.

## Benchmarking

The paper-facing benchmark harness lives in [`benchmarks/`](benchmarks/README.md).
It compares Grout against SGLang and vLLM by default, with llama.cpp and
TRT-LLM available as opt-in baselines.

Run the current RTX 5090 / sm_120 sweeps from the repository root:

```bash
./benchmarks/sweep_tg_sm120.sh
./benchmarks/sweep_pp_sm120.sh
```

Run the B200 / sm_100 profile by overriding the model path as needed:

```bash
MODEL_HF=../hf_models/qwen3_32b ./benchmarks/sweep_tg_sm100.sh
MODEL_HF=../hf_models/qwen3_32b ./benchmarks/sweep_pp_sm100.sh
```

The sweep scripts use driver-controlled clocks by default. They still accept
an optional MHz argument for debugging clock-locked runs, but paper runs should
either leave clocks unlocked or explicitly disclose the lock policy.

### Current result bundles

The paper-facing numbers live in the sweep aggregates, not in manually updated
tables. Use `aggregate.csv` for plots and `aggregate.md` for quick inspection.

- RTX 5090 / Qwen3-4B TG sweep:
  `benchmarks/results/sweep/20260508_111703_plus_115728_tg8192/`
- RTX 5090 / Qwen3-4B PP sweep:
  `benchmarks/results/sweep/20260508_114340/`
- B200 / Qwen3-32B final results:
  `benchmarks/results/final/b200_qwen3_32b/`

The B200 results are also indexed in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

For a direct Grout-only run:

```bash
cargo build --release --features benchmarks --bin grout_bench
target/release/grout_bench \
  --model "../hf_models/qwen3_4b" \
  --prompt "Hello, how are you?" \
  --max-new-tokens 512 \
  --max-seq-len 4096 \
  --reps 10 \
  --warmup-reps 3 \
  --ignore-eos \
  --quiet
```

See [`benchmarks/README.md`](benchmarks/README.md) for benchmark
policy, engine versions, and canonical run commands.
