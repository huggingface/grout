#!/usr/bin/env bash
# Paper-grade benchmark sweep: tg scan at fixed pp=18 (chat-templated).
#
# Usage: sweep_tg.sh [GPU_CLOCK_MHZ]
#
# Without an arg, clocks are driver-controlled (report "unlocked" in paper).
# With a number (e.g. 2400), GPU core clock is locked for the duration and
# reset on exit. Warm steady-state ≈ 2400 MHz on RTX 5090 under load.
#
# Output layout:
#   benchmarks/results/sweep/<timestamp>/
#     run.jsonl              -- concatenation of all per-run JSON lines
#     aggregate.csv          -- per-cell median + IQR
#     aggregate.md           -- same, markdown table
#     summary_<ts>.txt       -- captured stdout of the whole sweep
#
# Each engine emits its own records to run.jsonl:
#   {engine, variant, pp, tg, rep, prompt_tokens, gen_tokens, e2e_ms}
# Engines may also emit prefill_ms/decode_ms; aggregation keeps those direct
# phase timers separate from request_gen_tps = gen_tokens / e2e_ms.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$GROUT_DIR/../llama.cpp}"
GGUF_PATH="${GGUF_PATH:-$GROUT_DIR/../hf_models/qwen3_4b_f16.gguf}"
BENCH_ENVS_DIR="${BENCH_ENVS_DIR:-$GROUT_DIR/../bench_envs}"
BENCH_CACHE_DIR="${BENCH_CACHE_DIR:-$BENCH_ENVS_DIR/.cache}"
BENCH_HOME="${BENCH_HOME:-${SCRATCH:-$GROUT_DIR/..}}"
export PATH="$BENCH_ENVS_DIR/sglang_env/bin:$BENCH_ENVS_DIR/vllm_env/bin:$PATH"

# Paper config: pp=18 (chat-templated), tg scan.
PP_LABEL="${SWEEP_PP_LABEL:-18}"
read -r -a TG_VALUES <<< "${SWEEP_TG_VALUES:-36 128 512}"
if [[ "${#TG_VALUES[@]}" -eq 0 ]]; then
    echo "ERROR: SWEEP_TG_VALUES produced an empty tg list" >&2
    exit 2
fi
# 10 measured reps, 3 warmup. Median + IQR from 10 reps is robust without
# being absurdly long (~25s per cell × 15 cells ≈ 6 min wall).
BENCH_REPS="${BENCH_REPS:-10}"
BENCH_REPS_LONG="${BENCH_REPS_LONG:-}"
WARMUP_REPS="${WARMUP_REPS:-3}"
MODEL_MAX_LEN="${MODEL_MAX_LEN:-16384}"
GROUT_MAX_SEQ_LEN="${GROUT_MAX_SEQ_LEN:-}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-$MODEL_MAX_LEN}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MODEL_MAX_LEN}"
# vLLM budgets gpu_memory_utilization x total VRAM; on a desktop GPU the
# display compositor's share can push the 0.9 default into startup OOM
# ("warming up sampler with 256 dummy requests"). Set 0.8 on such boxes.
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"

# Request-level upper-bound lines for the tg report. The aggregator reads
# MODEL_HF/config.json and model.safetensors.index.json so the weight bytes
# and KV-cache shape match the model under test. It reports nominal peak as the
# paper-facing roof and an optional effective-bandwidth roof; BW_eff defaults
# to a simple 0.85 × 1792 GB/s multiplier.
ROOFLINE_GPU_BANDWIDTH_GBPS="${ROOFLINE_GPU_BANDWIDTH_GBPS:-1792}"
ROOFLINE_BANDWIDTH_FRACTION="${ROOFLINE_BANDWIDTH_FRACTION:-0.85}"
ROOFLINE_EFFECTIVE_BANDWIDTH_GBPS="${ROOFLINE_EFFECTIVE_BANDWIDTH_GBPS:-}"
ROOFLINE_MODEL_PARAMS="${ROOFLINE_MODEL_PARAMS:-}"
ROOFLINE_PREFILL_TFLOPS="${ROOFLINE_PREFILL_TFLOPS:-417.792}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$SCRIPT_DIR/results/sweep/$TS"
mkdir -p "$OUT_DIR"
JSONL="$OUT_DIR/run.jsonl"
SUMMARY="$OUT_DIR/summary_${TS}.txt"

LOCK_MHZ="${1:-}"
if [[ -n "$LOCK_MHZ" ]]; then
    if ! [[ "$LOCK_MHZ" =~ ^[0-9]+$ ]]; then
        echo "ERROR: expected integer MHz, got $LOCK_MHZ" >&2
        exit 2
    fi
    echo "Locking GPU clock to ${LOCK_MHZ} MHz for this sweep…"
    sudo nvidia-smi --lock-gpu-clocks="${LOCK_MHZ},${LOCK_MHZ}" >/dev/null
    cleanup_clocks() {
        echo "Resetting GPU clocks…"
        sudo nvidia-smi --reset-gpu-clocks >/dev/null || true
    }
    trap cleanup_clocks EXIT
fi

# LD_PRELOAD workaround for FlashInfer/sgl-kernel loading system libstdc++.
SYSTEM_LIBSTDCXX="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if [[ -f "$SYSTEM_LIBSTDCXX" ]]; then
    export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$SYSTEM_LIBSTDCXX"
fi

# Keep Python engine compile/model caches off the small container home mount.
mkdir -p "$BENCH_CACHE_DIR"/{flashinfer_base,flashinfer_cubins,huggingface,torch,torchinductor,triton,tvm-ffi,vllm,vllm_config,xdg}
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$BENCH_CACHE_DIR/flashinfer_base}"
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-$BENCH_CACHE_DIR/flashinfer_cubins}"
export HF_HOME="${HF_HOME:-$BENCH_CACHE_DIR/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$BENCH_CACHE_DIR/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$BENCH_CACHE_DIR/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BENCH_CACHE_DIR/triton}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$BENCH_CACHE_DIR/tvm-ffi}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$BENCH_CACHE_DIR/vllm}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$BENCH_CACHE_DIR/vllm_config}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BENCH_CACHE_DIR/xdg}"

venv_python() {
    local env_name="$1"
    local venv_dir="$BENCH_ENVS_DIR/$env_name"
    if [[ -x "$venv_dir/bin/python3" ]]; then
        echo "$venv_dir/bin/python3"
    else
        echo ""
    fi
}
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
err()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; }
ts_now() { date "+%Y-%m-%dT%H:%M:%S%z"; }
env_or_tg_default() {
    local base="$1"
    local tg="$2"
    local default="$3"
    local per_tg="${base}_TG_${tg}"
    if [[ -n "${!per_tg+x}" ]]; then
        echo "${!per_tg}"
    elif [[ -n "${!base+x}" ]]; then
        echo "${!base}"
    else
        echo "$default"
    fi
}
bench_reps_for_tg() {
    local tg="$1"
    local per_tg="BENCH_REPS_TG_${tg}"
    if [[ -n "${!per_tg+x}" ]]; then
        echo "${!per_tg}"
    elif (( tg > 512 )) && [[ -n "$BENCH_REPS_LONG" ]]; then
        echo "$BENCH_REPS_LONG"
    else
        echo "$BENCH_REPS"
    fi
}
grout_max_seq_len_for() {
    local pp="$1"
    local tg="$2"
    if [[ -n "$GROUT_MAX_SEQ_LEN" ]]; then
        echo "$GROUT_MAX_SEQ_LEN"
        return
    fi
    local total=$((pp + tg))
    if (( total < 4096 )); then
        echo 4096
    else
        echo "$total"
    fi
}

{
echo "═══════════════════════════════════════════════════════════════"
echo " Paper benchmark sweep  (pp=${PP_LABEL}, tg ∈ {${TG_VALUES[*]}})"
if [[ -n "$BENCH_REPS_LONG" ]]; then
    echo " reps=${BENCH_REPS}, reps_long_gt512=${BENCH_REPS_LONG}, warmup=${WARMUP_REPS}, clock_lock=${LOCK_MHZ:-none}"
else
    echo " reps=${BENCH_REPS}, warmup=${WARMUP_REPS}, clock_lock=${LOCK_MHZ:-none}"
fi
echo " grout_tuning_profile=${GROUT_TUNING_PROFILE:-default}"
if [[ -n "$ROOFLINE_EFFECTIVE_BANDWIDTH_GBPS" ]]; then
    echo " request_roofline: BW_nominal=${ROOFLINE_GPU_BANDWIDTH_GBPS} GB/s, BW_eff=${ROOFLINE_EFFECTIVE_BANDWIDTH_GBPS} GB/s, model=${MODEL_HF}, alpha=0"
else
    echo " request_roofline: BW_nominal=${ROOFLINE_GPU_BANDWIDTH_GBPS} GB/s, BW_eff=${ROOFLINE_BANDWIDTH_FRACTION}×nominal, model=${MODEL_HF}, alpha=0"
fi
echo " $(ts_now)"
echo "═══════════════════════════════════════════════════════════════"

log "GPU info"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
           --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
nvidia-smi -q -d CLOCK 2>/dev/null | grep -A3 "Max Clocks" | head -8 || true

# --------------------------------------------------------------------------
# 1. GROUT
# --------------------------------------------------------------------------
log "Benchmark: grout"
if [[ -f "$GROUT_DIR/Cargo.toml" ]]; then
    (cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench 2>&1 | tail -3)

    # Prefill tile for the tg scan is always the small-pp shape because
    # pp is fixed at 18. These match the baked defaults but we pin them
    # explicitly so the sweep output is reproducible if defaults later
    # drift.
    PREFILL_BM=16
    PREFILL_BN=32

    # Decode tile shape varies with max kv_len = pp + tg. See the
    # `tg_to_decode_tile` lookup below — populated from sweep_tg_tile.sh.
    tg_to_decode_tile() {
        # echoes "BN_DECODE NUM_KV_SPLITS"
        # From 2026-04-21 sweep_tg_tile.sh at pp=18. Decoded kv_len grows from pp+1
        # to pp+tg over the generation, so the right NKS depends on
        # the *average* kv_len, not the final one:
        #   tg=36  (avg kv≈36):  (BN=16, NKS=4)   = 212 ms  (grid-flat)
        #   tg=128 (avg kv≈82):  (BN=32, NKS=4)   = 767 ms  winner
        #   tg=512 (avg kv≈274): (BN=32, NKS=8)   = 3114 ms winner
        local tg="$1"
        case "$tg" in
            36)   echo "16 4" ;;
            128)  echo "32 4" ;;
            512)  echo "32 8" ;;
            *)    echo "32 8" ;;
        esac
    }

    for TG in "${TG_VALUES[@]}"; do
        read DEC_BN DEC_NKS <<< "$(tg_to_decode_tile "$TG")"
        RUN_PREFILL_BM="$(env_or_tg_default GROUT_ATTN_BM_PREFILL "$TG" "$PREFILL_BM")"
        RUN_PREFILL_BN="$(env_or_tg_default GROUT_ATTN_BN_PREFILL "$TG" "$PREFILL_BN")"
        RUN_DEC_BN="$(env_or_tg_default GROUT_ATTN_BN_DECODE "$TG" "$DEC_BN")"
        RUN_DEC_NKS="$(env_or_tg_default GROUT_FMHA_NUM_KV_SPLITS "$TG" "$DEC_NKS")"
        RUN_MAX_SEQ_LEN="$(grout_max_seq_len_for "$PP_LABEL" "$TG")"
        RUN_BENCH_REPS="$(bench_reps_for_tg "$TG")"
        echo ""
        echo "--- grout pp=${PP_LABEL} tg=${TG}  (reps=$RUN_BENCH_REPS BM=$RUN_PREFILL_BM BN_PRE=$RUN_PREFILL_BN BN_DEC=$RUN_DEC_BN NKS=$RUN_DEC_NKS max_seq=$RUN_MAX_SEQ_LEN) ---"
        (cd "$GROUT_DIR" && \
            GROUT_ATTN_BM_PREFILL=$RUN_PREFILL_BM \
            GROUT_ATTN_BN_PREFILL=$RUN_PREFILL_BN \
            GROUT_ATTN_BN_DECODE=$RUN_DEC_BN \
            GROUT_FMHA_NUM_KV_SPLITS=$RUN_DEC_NKS \
            ./target/release/grout_bench \
            --model "$MODEL_HF" \
            --prompt "Hello, how are you?" \
            --max-new-tokens "$TG" \
            --max-seq-len "$RUN_MAX_SEQ_LEN" \
            --reps "$RUN_BENCH_REPS" \
            --warmup-reps "$WARMUP_REPS" \
            --json "$JSONL" \
            --variant default \
            --pp-label "$PP_LABEL" \
            --quiet \
            --ignore-eos 2>&1) | tail -25
    done
else
    err "grout not found; skipping"
fi

# --------------------------------------------------------------------------
# 2. SGLANG (no-radix = radix/prefix cache disabled)
# --mode no-radix disables the RadixAttention prefix cache, matching
# vLLM's --no-prefix-cache. Must match sweep_pp.sh config so reviewers
# see consistent sglang settings across the two sweeps.
# --mem-fraction 0.85 matches sweep_pp.sh (default 0.9 OOMs at pp=2048).
# --------------------------------------------------------------------------
log "Benchmark: SGLang (no-radix)"
SGLANG_PYTHON="$(venv_python sglang_env)"
if [[ "${SWEEP_ENABLE_BASELINES:-1}" == 1 ]] && [[ -n "$SGLANG_PYTHON" ]] && "$SGLANG_PYTHON" -c "import sglang" 2>/dev/null; then
    for TG in "${TG_VALUES[@]}"; do
        RUN_BENCH_REPS="$(bench_reps_for_tg "$TG")"
        echo ""
        echo "--- sglang pp=${PP_LABEL} tg=${TG}  (reps=$RUN_BENCH_REPS) ---"
        PATH="$(dirname "$SGLANG_PYTHON"):$PATH" \
            "$SGLANG_PYTHON" "$SCRIPT_DIR/bench_sglang.py" \
            --model "$MODEL_HF" \
            --max-new-tokens "$TG" \
            --reps "$RUN_BENCH_REPS" \
            --warmup-reps "$WARMUP_REPS" \
            --json "$JSONL" \
            --pp-label "$PP_LABEL" \
            --context-length "$SGLANG_CONTEXT_LENGTH" \
            --mem-fraction 0.85 \
            --mode no-radix 2>&1 | tail -25
    done
else
    err "sglang env missing; skipping"
fi

# --------------------------------------------------------------------------
# 3. VLLM (cuda-graph; eager was consistently behind)
# --------------------------------------------------------------------------
log "Benchmark: vLLM (cuda-graph, prefix-cache OFF)"
VLLM_PYTHON="$(venv_python vllm_env)"
if [[ "${SWEEP_ENABLE_BASELINES:-1}" == 1 ]] && [[ -n "$VLLM_PYTHON" ]] && "$VLLM_PYTHON" -c "import vllm" 2>/dev/null; then
    for TG in "${TG_VALUES[@]}"; do
        RUN_BENCH_REPS="$(bench_reps_for_tg "$TG")"
        echo ""
        echo "--- vllm pp=${PP_LABEL} tg=${TG}  (reps=$RUN_BENCH_REPS) ---"
        # --no-prefix-cache: repeated-prompt reps would otherwise hit
        # cache starting from rep 2 and report near-zero prefill; must
        # match sweep_pp.sh for consistent cross-sweep comparison.
        PATH="$(dirname "$VLLM_PYTHON"):$PATH" \
            "$VLLM_PYTHON" "$SCRIPT_DIR/bench_vllm.py" \
            --model "$MODEL_HF" \
            --max-new-tokens "$TG" \
            --reps "$RUN_BENCH_REPS" \
            --warmup-reps "$WARMUP_REPS" \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            --gpu-mem-util "$VLLM_GPU_MEM_UTIL" \
            --json "$JSONL" \
            --pp-label "$PP_LABEL" \
            --mode cuda-graph \
            --no-prefix-cache 2>&1 | tail -25
    done
else
    err "vllm env missing at $VLLM_PYTHON; skipping"
fi

# --------------------------------------------------------------------------
# 4. LLAMA.CPP (flash-attn ON; no-fa was consistently behind) — DISABLED
# --------------------------------------------------------------------------
if [[ "${SWEEP_ENABLE_LLAMA:-0}" == "1" ]]; then
    log "Benchmark: llama.cpp (flash-attn)"
    if [[ -x "$LLAMA_CPP_DIR/build/bin/llama-server" && -f "$GGUF_PATH" ]]; then
        for TG in "${TG_VALUES[@]}"; do
            RUN_BENCH_REPS="$(bench_reps_for_tg "$TG")"
            echo ""
            echo "--- llama.cpp pp=${PP_LABEL} tg=${TG}  (reps=$RUN_BENCH_REPS) ---"
            python3 "$SCRIPT_DIR/bench_llama_cpp.py" \
                --llama-server "$LLAMA_CPP_DIR/build/bin/llama-server" \
                --gguf "$GGUF_PATH" \
                --max-new-tokens "$TG" \
                --reps "$RUN_BENCH_REPS" \
                --warmup-reps "$WARMUP_REPS" \
                --json "$JSONL" \
                --pp-label "$PP_LABEL" \
                --mode fa 2>&1 | tail -25
        done
    else
        err "llama.cpp artefacts missing; skipping"
    fi
fi

# --------------------------------------------------------------------------
# 5. TRT-LLM (TRT-engine backend only) — DISABLED
# --------------------------------------------------------------------------
# Kept as-is for future re-enable. Set SWEEP_ENABLE_TRTLLM=1 to run it.
if [[ "${SWEEP_ENABLE_TRTLLM:-0}" == "1" ]]; then
    log "Benchmark: TRT-LLM (trt backend)"
    TRTLLM_PYTHON="$(venv_python trtllm_env)"
    TRTLLM_ENV_BIN="$BENCH_ENVS_DIR/trtllm_env/bin"
    if [[ -n "$TRTLLM_PYTHON" ]] \
        && "$TRTLLM_PYTHON" -c "import tensorrt_llm" 2>/dev/null \
        && [[ -x "$TRTLLM_ENV_BIN/trtllm-llmapi-launch" ]]; then
        # Amortize the engine build: compile ONCE with max_seq_len covering the
        # largest tg, save to --engine-dir, then reload from disk for tg=128, 512.
        # First call is ~20 s build; subsequent two are ~5 s load.
        # NOTE: do NOT pre-create the dir — bench_trtllm.py treats an existing
        # empty dir as "cached" and would try to load. Python creates it on save.
        TRT_ENGINE_DIR="$OUT_DIR/trt_engine_amortized"
        for TG in "${TG_VALUES[@]}"; do
            RUN_BENCH_REPS="$(bench_reps_for_tg "$TG")"
            echo ""
            echo "--- trt-llm pp=${PP_LABEL} tg=${TG}  (reps=$RUN_BENCH_REPS) ---"
            PATH="$TRTLLM_ENV_BIN:$PATH" \
                "$TRTLLM_ENV_BIN/trtllm-llmapi-launch" \
                "$TRTLLM_PYTHON" "$SCRIPT_DIR/bench_trtllm.py" \
                --model "$MODEL_HF" \
                --max-new-tokens "$TG" \
                --reps "$RUN_BENCH_REPS" \
                --warmup-reps "$WARMUP_REPS" \
                --json "$JSONL" \
                --pp-label "$PP_LABEL" \
                --max-input-len 64 \
                --max-seq-len 2048 \
                --engine-dir "$TRT_ENGINE_DIR" \
                --backend trt 2>&1 | tail -25
        done
    else
        err "trt-llm env or trtllm-llmapi-launch missing; skipping"
    fi
fi

# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
log "Aggregating sweep results"
ROOFLINE_ARGS=(
    --request-roofline
    --roofline-model-dir "$MODEL_HF"
    --roofline-gpu-bandwidth-gbps "$ROOFLINE_GPU_BANDWIDTH_GBPS"
    --roofline-bandwidth-fraction "$ROOFLINE_BANDWIDTH_FRACTION"
    --roofline-prefill-tflops "$ROOFLINE_PREFILL_TFLOPS"
)
if [[ -n "$ROOFLINE_MODEL_PARAMS" ]]; then
    ROOFLINE_ARGS+=(--roofline-param-count "$ROOFLINE_MODEL_PARAMS")
fi
if [[ -n "$ROOFLINE_EFFECTIVE_BANDWIDTH_GBPS" ]]; then
    ROOFLINE_ARGS+=(--roofline-effective-bandwidth-gbps "$ROOFLINE_EFFECTIVE_BANDWIDTH_GBPS")
fi
python3 "$SCRIPT_DIR/aggregate_sweep.py" \
    --jsonl "$JSONL" \
    --csv "$OUT_DIR/aggregate.csv" \
    --markdown "$OUT_DIR/aggregate.md" \
    "${ROOFLINE_ARGS[@]}"

echo ""
echo "Done. Results in: $OUT_DIR"
echo "  run.jsonl      : all raw per-rep records"
echo "  aggregate.csv  : per-cell medians/IQR + request_gen/decode_direct/decode_fit/roofline metrics"
echo "  aggregate.md   : same, markdown table"

} 2>&1 | tee "$SUMMARY"
