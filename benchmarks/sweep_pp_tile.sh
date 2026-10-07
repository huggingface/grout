#!/usr/bin/env bash
# Cross-kernel BM × BN tile sweep at multiple pp values.
# For each pp, sweeps both fmha_prefill_causal and fmha_prefill_gqa over a
# (BM, BN) grid and reports prefill_ms median per cell. Output informs the
# prefill_tile_by_pp dispatch table in src/model.rs.
#
# Usage:
#   ./sweep_pp_tile.sh                 # default pp=(18 128 512)
#   ./sweep_pp_tile.sh 18 128 512 2048 # custom pp list
#
# Cells where BM > pp (kernel would read past Q's allocated rows) are
# skipped — the grout binary would crash on those, so we mark them '-'.
#
# Runtime estimate at default pp set: ~8–10 min unlocked, similar locked.
# Lock clocks first for paper-quality numbers:
#   sudo nvidia-smi --lock-gpu-clocks=2400,2400

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"
BENCH_ENVS_DIR="${BENCH_ENVS_DIR:-$GROUT_DIR/../bench_envs}"
BENCH_CACHE_DIR="${BENCH_CACHE_DIR:-$BENCH_ENVS_DIR/.cache}"
BENCH_HOME="${BENCH_HOME:-${SCRATCH:-$GROUT_DIR/..}}"
export HOME="$BENCH_HOME"
# Repointing HOME breaks rustup's toolchain lookup ($HOME/.rustup), which
# silently fails the cargo build and lets the sweep run a STALE binary —
# fatal with the ../cutile-rs path dependency. Keep rustup/cargo anchored
# to the real home.
REAL_HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
export RUSTUP_HOME="${RUSTUP_HOME:-$REAL_HOME/.rustup}"
export CARGO_HOME="${CARGO_HOME:-$REAL_HOME/.cargo}"
export PATH="$BENCH_ENVS_DIR/sglang_env/bin:$BENCH_ENVS_DIR/vllm_env/bin:$PATH"
BENCH_REPS="${BENCH_REPS:-10}"
WARMUP_REPS="${WARMUP_REPS:-3}"
PREFILL_BN_VALUES_RAW="${PREFILL_BN_VALUES:-16 32 64 128}"
PREFILL_MODES_RAW="${PREFILL_MODES:-causal gqa}"
CAUSAL_BM_VALUES_RAW="${CAUSAL_BM_VALUES:-16 32 64 128}"
GQA_BM_VALUES_RAW="${GQA_BM_VALUES:-4 8 16 32}"
read -r -a PREFILL_BN_VALUES <<< "$PREFILL_BN_VALUES_RAW"
read -r -a PREFILL_MODES <<< "$PREFILL_MODES_RAW"
read -r -a CAUSAL_BM_VALUES <<< "$CAUSAL_BM_VALUES_RAW"
read -r -a GQA_BM_VALUES <<< "$GQA_BM_VALUES_RAW"

# Keep tokenizer/model/JIT caches off the small container home mount.
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

if [[ $# -eq 0 ]]; then
    PP_VALUES=(18 128 512)
else
    PP_VALUES=("$@")
fi

echo "Building grout…"
(cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench 2>&1 | tail -2) \
    || { echo "FATAL: grout_bench build failed — refusing to sweep a stale binary." >&2; exit 1; }
echo

# Generate prompt files at exact token counts via make_prompts.py.
PROMPTS_DIR="$(mktemp -d)"
VLLM_PY="${VLLM_PY:-$BENCH_ENVS_DIR/vllm_env/bin/python3}"
[[ -x "$VLLM_PY" ]] || VLLM_PY="python3"
echo "Generating prompts in $PROMPTS_DIR …"
"$VLLM_PY" "$SCRIPT_DIR/make_prompts.py" \
    --model "$MODEL_HF" \
    --out-dir "$PROMPTS_DIR" \
    --pp "${PP_VALUES[@]}" >/dev/null
echo

# Per-cell runner: returns prefill_ms median from the 10 timed reps.
run_one() {
    local pp="$1" gqa="$2" bm="$3" bn="$4"
    local max_seq_len=$((pp + 36))
    (( max_seq_len < 4096 )) && max_seq_len=4096
    GROUT_FMHA_PREFILL_GQA_LPT=0 \
        GROUT_FMHA_PREFILL_GQA="$gqa" \
        GROUT_ATTN_BM_PREFILL="$bm" \
        GROUT_ATTN_BN_PREFILL="$bn" \
        "$GROUT_DIR/target/release/grout_bench" \
        --model "$MODEL_HF" \
        --prompt-file "$PROMPTS_DIR/pp_${pp}.txt" \
        --raw-prompt \
        --max-new-tokens 36 \
        --max-seq-len "$max_seq_len" \
        --reps "$BENCH_REPS" --warmup-reps "$WARMUP_REPS" --ignore-eos --quiet 2>&1 \
    | grep -E '^\s+\[timed\]' \
    | awk -F'prefill_ms=' '{print $2}' | awk -F',' '{print $1}' \
    | sort -n \
    | awk -v reps="$BENCH_REPS" 'BEGIN{mid=int((reps+1)/2)} NR==mid{med=$1} END{printf "%.2f", med}'
}

for PP in "${PP_VALUES[@]}"; do
    for MODE in "${PREFILL_MODES[@]}"; do
        if [[ "$MODE" == "gqa" ]]; then
            GQA=1
            LABEL="fmha_prefill_gqa   (head-grouped, GROUP=4, M_EFF=BM*4)"
            # GQA uses smaller BM because the effective tile is BM*GROUP rows.
            BM_VALUES=("${GQA_BM_VALUES[@]}")
        else
            GQA=0
            LABEL="fmha_prefill_causal"
            BM_VALUES=("${CAUSAL_BM_VALUES[@]}")
        fi

        echo "=== $LABEL   pp=$PP   (prefill_ms median; '-' = BM > pp, skipped) ==="
        printf 'BM\\BN'
        for bn in "${PREFILL_BN_VALUES[@]}"; do printf '\t%d' "$bn"; done
        echo
        for bm in "${BM_VALUES[@]}"; do
            printf '%d' "$bm"
            for bn in "${PREFILL_BN_VALUES[@]}"; do
                if (( bm > PP )); then
                    printf '\t-'
                    continue
                fi
                v=$(run_one "$PP" "$GQA" "$bm" "$bn" 2>/dev/null || echo "FAIL")
                printf '\t%s' "$v"
            done
            echo
        done
        echo
    done
done
