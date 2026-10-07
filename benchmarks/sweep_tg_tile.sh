#!/usr/bin/env bash
# BN_DECODE × NUM_KV_SPLITS tile sweep for fmha_decode_gqa_split, per tg.
# Decode tile performance depends on kv_len = pp + tg; at fixed pp=18 the
# tables are parametrized by tg but the real independent variable is kv_len
# (printed alongside each table). Output informs the decode tile dispatch in
# sweep_tg.sh, sweep_pp.sh, and src/model.rs.
#
# Usage:
#   ./sweep_tg_tile.sh                    # default tg=(36 128 512 2048), pp=18
#   ./sweep_tg_tile.sh 36 128 512         # custom tg list
#   PROMPT_FILE=/path/to/p.txt PP_LEN=2048 ./sweep_tg_tile.sh 36
#     # sweep BN × NKS at pp=2048 tg=36 (for tuning sweep_pp.sh decode tile)
#
# Caveat: grout reports decode_ms as TOTAL over the generation, so each
# cell is really the integral of per-step decode times across kv_len ∈
# [pp+1, pp+tg]. A tile that wins uniformly across that range also wins
# here; a tile that's only optimal at one end may tie with a different
# tile that's optimal at the other end. Interpret accordingly.
#
# Runtime estimate at default tg set: ~10–15 min unlocked. Lock clocks
# first for paper-quality numbers:
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
BN_VALUES_RAW="${BN_VALUES:-16 32 64 128}"
NKS_VALUES_RAW="${NKS_VALUES:-2 4 8 16}"
read -r -a BN_VALUES <<< "$BN_VALUES_RAW"
read -r -a NKS_VALUES <<< "$NKS_VALUES_RAW"

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
    TG_VALUES=(36 128 512 2048)
else
    TG_VALUES=("$@")
fi

# Prompt: default to the canonical chat-templated pp=18 prompt used in the
# paper's tg sweep. Override with PROMPT_FILE=... for non-default pp (and
# set PP_LEN=... so the kv_len label is correct).
PROMPT_FILE="${PROMPT_FILE:-}"
PP_LEN="${PP_LEN:-18}"
if [[ -z "$PROMPT_FILE" ]]; then
    PROMPT_DIR="$(mktemp -d)"
    PROMPT_FILE="$PROMPT_DIR/pp_18.txt"
    echo "Generating pp=18 chat-templated prompt at $PROMPT_FILE …"
    VLLM_PY="${VLLM_PY:-$BENCH_ENVS_DIR/vllm_env/bin/python3}"
    [[ -x "$VLLM_PY" ]] || VLLM_PY="python3"
    "$VLLM_PY" "$SCRIPT_DIR/make_prompts.py" \
        --model "$MODEL_HF" \
        --out-dir "$PROMPT_DIR" \
        --pp 18 >/dev/null
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: prompt file not found at $PROMPT_FILE" >&2
    exit 2
fi
echo "Using prompt file: $PROMPT_FILE  (pp≈${PP_LEN})"
echo

echo "Building grout…"
(cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench 2>&1 | tail -2) \
    || { echo "FATAL: grout_bench build failed — refusing to sweep a stale binary." >&2; exit 1; }
echo

# Per-cell runner: returns decode_ms median from the 10 timed reps.
run_one() {
    local tg="$1" bn="$2" nks="$3"
    local max_seq_len=$((PP_LEN + tg))
    (( max_seq_len < 4096 )) && max_seq_len=4096
    GROUT_ATTN_BN_DECODE="$bn" \
        GROUT_FMHA_NUM_KV_SPLITS="$nks" \
        "$GROUT_DIR/target/release/grout_bench" \
        --model "$MODEL_HF" \
        --prompt-file "$PROMPT_FILE" \
        --raw-prompt \
        --max-new-tokens "$tg" \
        --max-seq-len "$max_seq_len" \
        --reps "$BENCH_REPS" --warmup-reps "$WARMUP_REPS" --ignore-eos --quiet 2>&1 \
    | grep -E '^\s+\[timed\]' \
    | awk -F'decode_ms=' '{print $2}' | awk -F',' '{print $1}' \
    | sort -n \
    | awk -v reps="$BENCH_REPS" 'BEGIN{mid=int((reps+1)/2)} NR==mid{med=$1} END{printf "%.2f", med}'
}

for TG in "${TG_VALUES[@]}"; do
    KV_LEN=$((PP_LEN + TG))
    echo "=== fmha_decode_gqa_split   tg=${TG}  kv_len≈${KV_LEN}   (decode_ms median over ${TG} tokens; lower=better) ==="
    printf 'BN\\NKS'
    for nks in "${NKS_VALUES[@]}"; do printf '\t%d' "$nks"; done
    echo
    for bn in "${BN_VALUES[@]}"; do
        printf '%d' "$bn"
        for nks in "${NKS_VALUES[@]}"; do
            v=$(run_one "$TG" "$bn" "$nks" 2>/dev/null || echo "FAIL")
            printf '\t%s' "$v"
        done
        echo
    done
    echo
done
