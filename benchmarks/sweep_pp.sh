#!/usr/bin/env bash
# Prefill scan: fix tg=36, vary pp ∈ {18, 128, 512, 2048}.
# Companion to sweep_tg.sh (which scans tg at fixed pp=18). Same JSONL +
# aggregator pipeline; the cells just have different pp values. Aggregation
# reports request_gen_tps separately from optional direct decode phase timers.
#
# Usage: sweep_pp.sh [GPU_CLOCK_MHZ]
#   ./sweep_pp.sh           # unlocked clocks
#   SWEEP_PP_VALUES="18 128 512 2048 8192" ./sweep_pp.sh
#   ./sweep_pp.sh 2400      # lock clocks for the run
#
# Requires: prompt files at $SWEEP_DIR/prompts/pp_<N>.txt — generated
# automatically via make_prompts.py if missing.
#
# Resilience: single-engine failures (OOM, missing env, etc.) are logged
# and skipped so the rest of the sweep still produces aggregated results.

# NOTE: intentionally *not* using `set -e`. We want to continue past
# engine failures. `set -u` + `set -o pipefail` still catch undefined
# vars and silent pipe failures.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$GROUT_DIR/../llama.cpp}"
GGUF_PATH="${GGUF_PATH:-$GROUT_DIR/../hf_models/qwen3_4b_f16.gguf}"
BENCH_ENVS_DIR="${BENCH_ENVS_DIR:-$GROUT_DIR/../bench_envs}"
BENCH_CACHE_DIR="${BENCH_CACHE_DIR:-$BENCH_ENVS_DIR/.cache}"
BENCH_HOME="${BENCH_HOME:-${SCRATCH:-$GROUT_DIR/..}}"
export PATH="$BENCH_ENVS_DIR/sglang_env/bin:$BENCH_ENVS_DIR/vllm_env/bin:$PATH"

# Prefill scan config: vary pp at fixed tg.
read -r -a PP_VALUES <<< "${SWEEP_PP_VALUES:-18 128 512 2048}"
TG_FIXED="${SWEEP_TG_FIXED:-36}"
if [[ "${#PP_VALUES[@]}" -eq 0 ]]; then
    echo "ERROR: SWEEP_PP_VALUES produced an empty pp list" >&2
    exit 2
fi
BENCH_REPS="${BENCH_REPS:-10}"
BENCH_REPS_LONG="${BENCH_REPS_LONG:-}"
WARMUP_REPS="${WARMUP_REPS:-3}"
# Engine context length must cover the largest pp in the sweep (+ tg + slack),
# else vLLM/SGLang reject long prompts. Auto-grow with the sweep, 16384 floor.
_MAX_PP=0; for _pp in "${PP_VALUES[@]}"; do (( _pp > _MAX_PP )) && _MAX_PP=$_pp; done
_DEFAULT_MAX_LEN=$(( _MAX_PP + TG_FIXED + 256 )); (( _DEFAULT_MAX_LEN < 16384 )) && _DEFAULT_MAX_LEN=16384
MODEL_MAX_LEN="${MODEL_MAX_LEN:-$_DEFAULT_MAX_LEN}"
GROUT_MAX_SEQ_LEN="${GROUT_MAX_SEQ_LEN:-}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-$MODEL_MAX_LEN}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MODEL_MAX_LEN}"
# vLLM budgets gpu_memory_utilization x total VRAM; on a desktop GPU the
# display compositor's share can push the 0.9 default into startup OOM
# ("warming up sampler with 256 dummy requests"). Set 0.8 on such boxes.
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$SCRIPT_DIR/results/sweep/$TS"
PROMPTS_DIR="$OUT_DIR/prompts"
mkdir -p "$PROMPTS_DIR"
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
env_or_pp_default() {
    local base="$1"
    local pp="$2"
    local default="$3"
    local per_pp="${base}_PP_${pp}"
    if [[ -n "${!per_pp+x}" ]]; then
        echo "${!per_pp}"
    elif [[ -n "${!base+x}" ]]; then
        echo "${!base}"
    else
        echo "$default"
    fi
}
bench_reps_for_pp() {
    local pp="$1"
    local per_pp="BENCH_REPS_PP_${pp}"
    if [[ -n "${!per_pp+x}" ]]; then
        echo "${!per_pp}"
    elif (( pp > 512 )) && [[ -n "$BENCH_REPS_LONG" ]]; then
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
echo " Prefill scan  (pp ∈ {${PP_VALUES[*]}}, tg=${TG_FIXED})"
if [[ -n "$BENCH_REPS_LONG" ]]; then
    echo " reps=${BENCH_REPS}, reps_long_gt512=${BENCH_REPS_LONG}, warmup=${WARMUP_REPS}, clock_lock=${LOCK_MHZ:-none}"
else
    echo " reps=${BENCH_REPS}, warmup=${WARMUP_REPS}, clock_lock=${LOCK_MHZ:-none}"
fi
echo " grout_tuning_profile=${GROUT_TUNING_PROFILE:-default}"
echo " $(ts_now)"
echo "═══════════════════════════════════════════════════════════════"

# --------------------------------------------------------------------------
# Generate prompt files. Use the vllm env's python (has transformers).
# --------------------------------------------------------------------------
log "Generating prompt files at exact token counts"
PROMPT_GEN_PY="$(venv_python vllm_env)"
if [[ -z "$PROMPT_GEN_PY" ]]; then
    PROMPT_GEN_PY="python3"
fi
"$PROMPT_GEN_PY" "$SCRIPT_DIR/make_prompts.py" \
    --model "$MODEL_HF" \
    --out-dir "$PROMPTS_DIR" \
    --pp "${PP_VALUES[@]}"

# --------------------------------------------------------------------------
# 1. GROUT
# --------------------------------------------------------------------------
log "Benchmark: grout"
if [[ -f "$GROUT_DIR/Cargo.toml" ]]; then
    (cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench 2>&1 | tail -3)

    # Prefill tile dispatch — must match prefill_tile_by_pp in src/model.rs.
    # Tuned from the 2026-04-20 sweep over pp ∈ {18, 128, 512, 2048}. Clean crossover
    # at pp=512: short-pp (<512) is SM-undersat so small BM wins; long-pp
    # (≥512) is SM-saturated so wider BM amortizes MMA setup.
    pp_to_prefill_tile() {
        # echoes "BM BN"
        # 2026-07-05 per-arm tile sweep (both legacy and safe mapped kernels
        # agree): BM=64 rows win at pp>=512 for both arms (legacy best 64/32
        # = 14.36 ms, safe best 64/32 = 14.51 at pp=512; 64/64 within noise
        # for both). The old 32/16 was stale for both arms. pp>=2048 cells
        # are overridden by the per-pp LPT profile in the sm120 wrapper.
        local pp="$1"
        if (( pp < 512 )); then
            echo "16 32"
        else
            echo "64 32"
        fi
    }

    # Decode tile shape dispatched by pp (really by kv_len = pp + tg).
    # 2026-04-21 sweep_tg_tile.sh data at tg=36 (unlocked clocks):
    #   pp=18   (kv≈54):   (BN=16, NKS=4)  = 212  (tied across the whole grid)
    #   pp=128  (kv≈164):  (BN=32, NKS=16) = 211  vs (16,4) = 227    (-7%)
    #   pp=512  (kv≈548):  (BN=32, NKS=16) = 211  vs (16,4) = 226    (-7%)
    #   pp=2048 (kv≈2084): (BN=32, NKS=16) = 232  vs (16,4) = 279   (-17%)
    # NKS=16 is the universal winner at kv_len ≥ 164; BN=32 is within
    # noise of (16, …) and (64, …) at every pp. Threshold at pp=128.
    pp_to_decode_tile() {
        # echoes "BN NKS"
        local pp="$1"
        if (( pp >= 128 )); then
            echo "32 16"
        else
            echo "16 4"
        fi
    }

    for PP in "${PP_VALUES[@]}"; do
        read PRE_BM PRE_BN <<< "$(pp_to_prefill_tile "$PP")"
        read DEC_BN DEC_NKS <<< "$(pp_to_decode_tile "$PP")"
        RUN_PREFILL_BM="$(env_or_pp_default GROUT_ATTN_BM_PREFILL "$PP" "$PRE_BM")"
        RUN_PREFILL_BN="$(env_or_pp_default GROUT_ATTN_BN_PREFILL "$PP" "$PRE_BN")"
        RUN_DEC_BN="$(env_or_pp_default GROUT_ATTN_BN_DECODE "$PP" "$DEC_BN")"
        RUN_DEC_NKS="$(env_or_pp_default GROUT_FMHA_NUM_KV_SPLITS "$PP" "$DEC_NKS")"
        RUN_FMHA_PREFILL="$(env_or_pp_default GROUT_FMHA_PREFILL "$PP" "1")"
        RUN_FMHA_PREFILL_GQA="$(env_or_pp_default GROUT_FMHA_PREFILL_GQA "$PP" "0")"
        RUN_FMHA_PREFILL_GQA_LPT="$(env_or_pp_default GROUT_FMHA_PREFILL_GQA_LPT "$PP" "0")"
        RUN_FMHA_PREFILL_GQA_GROUP="$(env_or_pp_default GROUT_FMHA_PREFILL_GQA_GROUP "$PP" "0")"
        RUN_FMHA_PREFILL_LPT_SWIZZLE="$(env_or_pp_default GROUT_FMHA_PREFILL_LPT_SWIZZLE "$PP" "0")"
        RUN_FMHA_PREFILL_LPT_SCHED="$(env_or_pp_default GROUT_FMHA_PREFILL_LPT_SCHED "$PP" "1")"
        RUN_FMHA_PREFILL_LPT_MASK_SPLIT="$(env_or_pp_default GROUT_FMHA_PREFILL_LPT_MASK_SPLIT "$PP" "0")"
        RUN_FMHA_PREFILL_LATENCY="$(env_or_pp_default GROUT_FMHA_PREFILL_LATENCY "$PP" "2")"
        RUN_FMHA_PREFILL_OCCUPANCY="$(env_or_pp_default GROUT_FMHA_PREFILL_OCCUPANCY "$PP" "2")"
        RUN_FUSED_QK_ROPE_KV_PREFILL="$(env_or_pp_default GROUT_FUSED_QK_ROPE_KV_PREFILL "$PP" "1")"
        RUN_RMS_BLOCK="$(env_or_pp_default GROUT_RMS_BLOCK "$PP" "0")"
        RUN_ADD_RMS_BLOCK="$(env_or_pp_default GROUT_ADD_RMS_BLOCK "$PP" "0")"
        RUN_RMS_HIDDEN_BLOCK="$(env_or_pp_default GROUT_RMS_HIDDEN_BLOCK "$PP" "0")"
        RUN_MAX_SEQ_LEN="$(grout_max_seq_len_for "$PP" "$TG_FIXED")"
        RUN_BENCH_REPS="$(bench_reps_for_pp "$PP")"
        echo ""
        if [[ "${GROUT_TUNING_PROFILE:-default}" == "records" ]]; then
            echo "--- grout pp=${PP} tg=${TG_FIXED}  (reps=$RUN_BENCH_REPS max_seq=$RUN_MAX_SEQ_LEN BN_DEC=$RUN_DEC_BN NKS=$RUN_DEC_NKS; profile=records: prefill knobs from benchmarks/tuning/<arch> records, no env exported for them) ---"
        else
            echo "--- grout pp=${PP} tg=${TG_FIXED}  (reps=$RUN_BENCH_REPS BM=$RUN_PREFILL_BM BN_PRE=$RUN_PREFILL_BN BN_DEC=$RUN_DEC_BN NKS=$RUN_DEC_NKS max_seq=$RUN_MAX_SEQ_LEN GQA=$RUN_FMHA_PREFILL_GQA LPT=$RUN_FMHA_PREFILL_GQA_LPT GROUP=$RUN_FMHA_PREFILL_GQA_GROUP SW=$RUN_FMHA_PREFILL_LPT_SWIZZLE SCHED=$RUN_FMHA_PREFILL_LPT_SCHED MASK_SPLIT=$RUN_FMHA_PREFILL_LPT_MASK_SPLIT) ---"
        fi
        # GROUT_TUNING_PROFILE=records: every prefill-attention / hint knob
        # is left to the engine, which resolves env > tuning record
        # (benchmarks/tuning/<arch>) > built-in default. The legacy per-pp
        # table above is NOT exported for those knobs in this mode — env
        # beats records, so exporting it silently benches the hand profile
        # instead of the tuned one (this is exactly what happened to every
        # sweep before 2026-09-04). Decode tile knobs are still exported:
        # no decode records ship yet, and the wrappers carry the per-arch
        # decode profile through the same per-pp variables.
        GROUT_RUN_ENV=(
            GROUT_ATTN_BN_DECODE=$RUN_DEC_BN
            GROUT_FMHA_NUM_KV_SPLITS=$RUN_DEC_NKS
        )
        if [[ "${GROUT_TUNING_PROFILE:-default}" != "records" ]]; then
            GROUT_RUN_ENV+=(
                GROUT_ATTN_BM_PREFILL=$RUN_PREFILL_BM
                GROUT_ATTN_BN_PREFILL=$RUN_PREFILL_BN
                GROUT_FMHA_PREFILL=$RUN_FMHA_PREFILL
                GROUT_FMHA_PREFILL_GQA=$RUN_FMHA_PREFILL_GQA
                GROUT_FMHA_PREFILL_GQA_LPT=$RUN_FMHA_PREFILL_GQA_LPT
                GROUT_FMHA_PREFILL_GQA_GROUP=$RUN_FMHA_PREFILL_GQA_GROUP
                GROUT_FMHA_PREFILL_LPT_SWIZZLE=$RUN_FMHA_PREFILL_LPT_SWIZZLE
                GROUT_FMHA_PREFILL_LPT_SCHED=$RUN_FMHA_PREFILL_LPT_SCHED
                GROUT_FMHA_PREFILL_LPT_MASK_SPLIT=$RUN_FMHA_PREFILL_LPT_MASK_SPLIT
                GROUT_FMHA_PREFILL_LATENCY=$RUN_FMHA_PREFILL_LATENCY
                GROUT_FMHA_PREFILL_OCCUPANCY=$RUN_FMHA_PREFILL_OCCUPANCY
                GROUT_FUSED_QK_ROPE_KV_PREFILL=$RUN_FUSED_QK_ROPE_KV_PREFILL
                GROUT_RMS_BLOCK=$RUN_RMS_BLOCK
                GROUT_ADD_RMS_BLOCK=$RUN_ADD_RMS_BLOCK
                GROUT_RMS_HIDDEN_BLOCK=$RUN_RMS_HIDDEN_BLOCK
            )
        fi
        (cd "$GROUT_DIR" && \
            env "${GROUT_RUN_ENV[@]}" \
            ./target/release/grout_bench \
            --model "$MODEL_HF" \
            --prompt-file "$PROMPTS_DIR/pp_${PP}.txt" \
            --raw-prompt \
            --max-new-tokens "$TG_FIXED" \
            --max-seq-len "$RUN_MAX_SEQ_LEN" \
            --reps "$RUN_BENCH_REPS" \
            --warmup-reps "$WARMUP_REPS" \
            --json "$JSONL" \
            --variant default \
            --pp-label "$PP" \
            --quiet \
            --ignore-eos 2>&1) | tail -25
    done
fi

# --------------------------------------------------------------------------
# 2. SGLANG (no-radix = radix/prefix cache disabled)
# --mode no-radix disables the RadixAttention prefix cache, matching the
# prefix-cache-OFF vLLM and llama.cpp (cache_prompt: False) configs for
# honest prefill measurement.
# --------------------------------------------------------------------------
log "Benchmark: SGLang (no-radix)"
SGLANG_PYTHON="$(venv_python sglang_env)"
if [[ "${SWEEP_ENABLE_BASELINES:-1}" == 1 ]] && [[ -n "$SGLANG_PYTHON" ]] && "$SGLANG_PYTHON" -c "import sglang" 2>/dev/null; then
    echo " sglang_version=$("$SGLANG_PYTHON" -c 'import sglang, torch; print(sglang.__version__, "torch", torch.__version__)' 2>/dev/null)"
    for PP in "${PP_VALUES[@]}"; do
        RUN_BENCH_REPS="$(bench_reps_for_pp "$PP")"
        echo ""
        echo "--- sglang pp=${PP} tg=${TG_FIXED}  (reps=$RUN_BENCH_REPS) ---"
        # --mem-fraction 0.85 gives sglang enough scratch for the forward
        # pass at pp=2048 (default 0.9 OOMs with Qwen3-4B on RTX 5090 32 GiB).
        "$SGLANG_PYTHON" "$SCRIPT_DIR/bench_sglang.py" \
            --model "$MODEL_HF" \
            --prompt-file "$PROMPTS_DIR/pp_${PP}.txt" \
            --max-new-tokens "$TG_FIXED" \
            --reps "$RUN_BENCH_REPS" \
            --warmup-reps "$WARMUP_REPS" \
            --json "$JSONL" \
            --pp-label "$PP" \
            --context-length "$SGLANG_CONTEXT_LENGTH" \
            --mem-fraction 0.85 \
            --mode no-radix 2>&1 | tail -25
    done
fi

# --------------------------------------------------------------------------
# 3. VLLM (cuda-graph, prefix cache OFF)
# Apples-to-apples prefill requires disabling prefix cache — otherwise
# repeated-prompt reps hit cache and report near-zero prefill at long pp.
# --------------------------------------------------------------------------
log "Benchmark: vLLM (cuda-graph, prefix-cache OFF)"
VLLM_PYTHON="$(venv_python vllm_env)"
if [[ "${SWEEP_ENABLE_BASELINES:-1}" == 1 ]] && [[ -n "$VLLM_PYTHON" ]] && "$VLLM_PYTHON" -c "import vllm" 2>/dev/null; then
    # Baseline provenance in the summary: the bench scripts' tail output
    # does not carry the engine version, and cross-session comparisons
    # need it.
    echo " vllm_version=$("$VLLM_PYTHON" -c 'import vllm, torch; print(vllm.__version__, "torch", torch.__version__)' 2>/dev/null)"
    for PP in "${PP_VALUES[@]}"; do
        RUN_BENCH_REPS="$(bench_reps_for_pp "$PP")"
        echo ""
        echo "--- vllm pp=${PP} tg=${TG_FIXED}  (reps=$RUN_BENCH_REPS) ---"
        "$VLLM_PYTHON" "$SCRIPT_DIR/bench_vllm.py" \
            --model "$MODEL_HF" \
            --prompt-file "$PROMPTS_DIR/pp_${PP}.txt" \
            --max-new-tokens "$TG_FIXED" \
            --reps "$RUN_BENCH_REPS" \
            --warmup-reps "$WARMUP_REPS" \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            --gpu-mem-util "$VLLM_GPU_MEM_UTIL" \
            --json "$JSONL" \
            --pp-label "$PP" \
            --mode cuda-graph \
            --no-prefix-cache 2>&1 | tail -25
    done
fi

# --------------------------------------------------------------------------
# 4. LLAMA.CPP (flash-attn ON) — DISABLED
# --------------------------------------------------------------------------
if [[ "${SWEEP_ENABLE_LLAMA:-0}" == "1" ]]; then
    log "Benchmark: llama.cpp (flash-attn)"
    if [[ -x "$LLAMA_CPP_DIR/build/bin/llama-server" && -f "$GGUF_PATH" ]]; then
        for PP in "${PP_VALUES[@]}"; do
            RUN_BENCH_REPS="$(bench_reps_for_pp "$PP")"
            echo ""
            echo "--- llama.cpp pp=${PP} tg=${TG_FIXED}  (reps=$RUN_BENCH_REPS) ---"
            python3 "$SCRIPT_DIR/bench_llama_cpp.py" \
                --llama-server "$LLAMA_CPP_DIR/build/bin/llama-server" \
                --gguf "$GGUF_PATH" \
                --prompt-file "$PROMPTS_DIR/pp_${PP}.txt" \
                --max-new-tokens "$TG_FIXED" \
                --reps "$RUN_BENCH_REPS" \
                --warmup-reps "$WARMUP_REPS" \
                --json "$JSONL" \
                --pp-label "$PP" \
                --mode fa 2>&1 | tail -25
        done
    fi
fi

# TRT-LLM skipped in pp scan: requires per-engine tuning at this prompt
# length that we don't have time for. Re-add later by cloning the block
# structure above with trtllm-llmapi-launch + --engine-dir.

# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
log "Aggregating prefill scan"
python3 "$SCRIPT_DIR/aggregate_sweep.py" \
    --jsonl "$JSONL" \
    --csv "$OUT_DIR/aggregate.csv" \
    --markdown "$OUT_DIR/aggregate.md"

echo ""
echo "Done. Results in: $OUT_DIR"
echo "  run.jsonl, aggregate.csv, aggregate.md"

} 2>&1 | tee "$SUMMARY"
