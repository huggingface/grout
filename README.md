# grout

A Qwen 3 inference engine written in Rust, build using [cutile-rs](https://github.com/NVlabs/cutile-rs).

## Usage

1) Follow the [installation guide of `cutile-rs`](https://github.com/NVlabs/cutile-rs?tab=readme-ov-file#install).
2) Configure your [environment variables for `cutile-rs`](https://github.com/NVlabs/cutile-rs?tab=readme-ov-file#configure-environment).

- Set `CUDA_TOOLKIT_PATH` to your CUDA 13.2 install directory.

```
CUDA_TOOLKIT_PATH=/usr/local/cuda-13 cargo run --release -- --model <path-to-qwen3-model> --prompt "Hello, how are you?" --max-new-tokens 128
```

### Options

| Flag | Description |
|------|-------------|
| `--model <path>` | Path to model directory (safetensors + config.json) |
| `--prompt <text>` | Input prompt |
| `--max-new-tokens <n>` | Number of tokens to generate (default: 128) |
| `--max-seq-len <n>` | Override max sequence length |
| `--sample` | Enable sampling (default: greedy) |
| `--raw-prompt` | Skip chat template, use prompt as-is |
| `--device-argmax` | Run argmax on device |
| `--profile` | Print per-step profiling report |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GROUT_CUDA_GRAPH_DECODE` | Set to `1` to enable CUDA graph capture for decode |
| `GROUT_CUBLAS_COMPUTE16` | Set to `1` to use FP16 accumulation in cuBLAS |
| `GROUT_CUBLAS_COMPUTE16_MAX_M` | Max M dimension for FP16 compute |
| `GROUT_CUBLAS_FAST_ALGO` | cuBLAS algorithm selection |
| `GROUT_ATTN_BN_DECODE` | Attention block size for decode |

## License

Apache-2.0
