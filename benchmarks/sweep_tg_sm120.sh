#!/usr/bin/env bash
# RTX 5090 / sm_120 profile for the canonical tg sweep.
#
# This wrapper intentionally keeps benchmark flow, Python baselines, and
# aggregation in sweep_tg.sh. Only the sm_120-specific roofline and Grout
# tuning knobs live here so they are easy to inspect and revise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${GROUT_TUNING_PROFILE:=sm120_5090}"

# RTX 5090 nominal GDDR7 bandwidth. The aggregate report also includes the
# 85% effective-bandwidth line unless explicitly overridden.
: "${ROOFLINE_GPU_BANDWIDTH_GBPS:=1792}"
: "${ROOFLINE_BANDWIDTH_FRACTION:=0.85}"

# Canonical 5090 tg sweep includes the long-token points used for capability
# checks. Override SWEEP_TG_VALUES externally for smoke runs.
: "${SWEEP_TG_VALUES:=36 128 512 2048 8192}"

# Keep optional baselines out of local profile sweeps unless explicitly
# requested. sweep_tg.sh owns the actual opt-in behavior.
: "${SWEEP_ENABLE_LLAMA:=0}"
: "${SWEEP_ENABLE_TRTLLM:=0}"

# Long-context tg cells dominate runtime; keep short cells at BENCH_REPS and
# run tg > 512 with fewer measured reps unless explicitly overridden.
: "${BENCH_REPS_LONG:=3}"

# Current sm_120 Grout tile profile. Short-token cells are from the 5090
# decode-tile sweep; long-token cells are the measured 5090 candidates that
# kept tg=8192 near SGLang/vLLM in the May 2026 long-context run.
: "${GROUT_ATTN_BM_PREFILL:=16}"
: "${GROUT_ATTN_BN_PREFILL:=32}"
: "${GROUT_ATTN_BN_DECODE_TG_36:=16}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_36:=4}"
: "${GROUT_ATTN_BN_DECODE_TG_128:=32}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_128:=4}"
: "${GROUT_ATTN_BN_DECODE_TG_512:=32}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_512:=8}"
# BN=32 is the mapped decode kernel's optimum at long kv too — the old BN=64
# values here were tuned on the legacy kernels and cost ~4-7% at tg>=2048
# (paired shape check, 2026-07-07: tg=2048 12.09s vs 12.65s; tg=8192 50.61s
# vs 54.73s). Per-kernel-form lesson applies to this wrapper as well.
: "${GROUT_ATTN_BN_DECODE_TG_2048:=32}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_2048:=16}"
: "${GROUT_ATTN_BN_DECODE_TG_8192:=32}"
: "${GROUT_FMHA_NUM_KV_SPLITS_TG_8192:=16}"

# Current CUDA/cuBLAS stack prefers f16 accumulation for the Qwen3-4B decode
# GEMVs on RTX 5090. Override externally for diagnostics only.
: "${GROUT_CUBLAS_COMPUTE16:=1}"
: "${GROUT_CUDA_GRAPH_DECODE:=1}"
: "${GROUT_FLASH_DECODE:=0}"
: "${GROUT_FMHA_SPLIT_KV:=1}"
: "${GROUT_FMHA_DECODE_LATENCY:=4}"
: "${GROUT_FMHA_DECODE_OCCUPANCY:=2}"
: "${GROUT_FMHA_MERGE_CHUNK_D:=16}"
: "${GROUT_FMHA_MERGE_LATENCY:=2}"
: "${GROUT_FUSED_QK_ROPE_KV_DECODE:=1}"
: "${GROUT_QK_ROPE_LATENCY:=2}"
: "${GROUT_QK_ROPE_OCCUPANCY:=1}"
: "${GROUT_QK_ROPE_CGA:=0}"
: "${GROUT_KV_CACHE_DYN_CHUNK_D:=32}"
: "${GROUT_KV_CACHE_BM_S:=16}"
: "${GROUT_EMBED_BLOCK:=1024}"
: "${GROUT_RMS_BLOCK:=4096}"
: "${GROUT_ARGMAX_BLOCK:=128}"
: "${GROUT_FMHA_PREFILL:=1}"
: "${GROUT_FMHA_PREFILL_GQA:=0}"
: "${GROUT_FMHA_PREFILL_LATENCY:=2}"
: "${GROUT_FMHA_PREFILL_OCCUPANCY:=2}"
: "${GROUT_FUSED_QK_ROPE_KV_PREFILL:=1}"

export \
    GROUT_TUNING_PROFILE \
    ROOFLINE_GPU_BANDWIDTH_GBPS \
    ROOFLINE_BANDWIDTH_FRACTION \
    SWEEP_TG_VALUES \
    SWEEP_ENABLE_LLAMA \
    SWEEP_ENABLE_TRTLLM \
    BENCH_REPS_LONG \
    GROUT_ATTN_BM_PREFILL \
    GROUT_ATTN_BN_PREFILL \
    GROUT_ATTN_BN_DECODE_TG_36 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_36 \
    GROUT_ATTN_BN_DECODE_TG_128 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_128 \
    GROUT_ATTN_BN_DECODE_TG_512 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_512 \
    GROUT_ATTN_BN_DECODE_TG_2048 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_2048 \
    GROUT_ATTN_BN_DECODE_TG_8192 \
    GROUT_FMHA_NUM_KV_SPLITS_TG_8192 \
    GROUT_CUBLAS_COMPUTE16 \
    GROUT_CUDA_GRAPH_DECODE \
    GROUT_FLASH_DECODE \
    GROUT_FMHA_SPLIT_KV \
    GROUT_FMHA_DECODE_LATENCY \
    GROUT_FMHA_DECODE_OCCUPANCY \
    GROUT_FMHA_MERGE_CHUNK_D \
    GROUT_FMHA_MERGE_LATENCY \
    GROUT_FUSED_QK_ROPE_KV_DECODE \
    GROUT_QK_ROPE_LATENCY \
    GROUT_QK_ROPE_OCCUPANCY \
    GROUT_QK_ROPE_CGA \
    GROUT_KV_CACHE_DYN_CHUNK_D \
    GROUT_KV_CACHE_BM_S \
    GROUT_EMBED_BLOCK \
    GROUT_RMS_BLOCK \
    GROUT_ARGMAX_BLOCK \
    GROUT_FMHA_PREFILL \
    GROUT_FMHA_PREFILL_GQA \
    GROUT_FMHA_PREFILL_LATENCY \
    GROUT_FMHA_PREFILL_OCCUPANCY \
    GROUT_FUSED_QK_ROPE_KV_PREFILL

exec bash "$SCRIPT_DIR/sweep_tg.sh" "$@"
