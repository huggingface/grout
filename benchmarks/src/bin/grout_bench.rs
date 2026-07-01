use anyhow::{Result, bail};
use clap::Parser;
use grout::kernels;
use grout::model::{GenerationOutput, Qwen3Engine};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3 inference engine on cutile-rs")]
struct Args {
    #[arg(long, default_value = "../hf_models/qwen3_4b")]
    model: PathBuf,

    #[arg(long)]
    prompt: Option<String>,

    /// Path to a prompt file (UTF-8 text). Used by the pp sweep to load
    /// synthetic prompts of specific token counts. Mutually exclusive with
    /// `--prompt`; if both are given, `--prompt-file` wins.
    #[arg(long)]
    prompt_file: Option<PathBuf>,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long)]
    max_seq_len: Option<usize>,

    #[arg(long, default_value_t = false)]
    sample: bool,

    #[arg(long, default_value_t = false)]
    raw_prompt: bool,

    /// Run argmax on the host (CPU). Default: device-side argmax, which
    /// avoids copying the full [vocab, fp16] logits tensor (~300 KB) to
    /// host every decode token. Enabling this flag reverts to the old
    /// host-argmax path for diagnostic / correctness comparison.
    #[arg(long, default_value_t = false)]
    host_argmax: bool,

    #[arg(long, default_value_t = false)]
    profile: bool,

    #[arg(long, default_value_t = 5)]
    reps: usize,

    /// Warmup reps run after JIT compile + graph capture (discarded).
    /// JIT itself acts as the first warmup; 2 extra smooths out the
    /// cold-run dip we see on the first measured rep (runs 2+ hit peak).
    #[arg(long, default_value_t = 2)]
    warmup_reps: usize,

    /// Measure only the cold-start JIT compile phase. Loads the engine,
    /// times `warm_all_kernels()`, prints machine-readable keys, and exits
    /// without running prefill, decode, or graph capture.
    #[arg(long, default_value_t = false)]
    jit_breakdown: bool,

    /// Append one JSON object per measured run to this file (JSONL).
    /// Format: {"engine": "grout", "variant": ..., "pp": ..., "tg": ...,
    ///          "rep": i, "prompt_tokens": N, "gen_tokens": M,
    ///          "prefill_ms": X, "decode_ms": Y, "e2e_ms": Z}
    #[arg(long)]
    json: Option<PathBuf>,

    /// Label for the `variant` field in JSON output (e.g. "default",
    /// "flash-decode"). Has no effect on execution.
    #[arg(long, default_value = "default")]
    variant: String,

    /// Label for the `pp` (prompt-length target) field in JSON output.
    /// Informational only — actual `prompt_tokens` comes from the tokenizer.
    #[arg(long, default_value_t = 0)]
    pp_label: usize,

    /// Skip EOS-based early termination so decode runs exactly
    /// `max_new_tokens` steps. Matches `ignore_eos=true` / `min_tokens`
    /// semantics on the Python bench scripts; required for a paper-grade
    /// tg sweep so every engine generates the same fixed decode window.
    #[arg(long, default_value_t = false)]
    ignore_eos: bool,

    /// Suppress printing the generated text and profile report at the
    /// end of a run. Keeps only per-rep and summary timing lines —
    /// matches the output style of bench_{sglang,vllm,llama_cpp}.py.
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

fn log_output(label: &str, i: usize, reps: usize, output: &GenerationOutput) {
    println!(
        "  [{label}] run {}/{reps}: \
         prompt_tokens={}, gen_tokens={}, \
         prefill_ms={:.2}, decode_ms={:.2}, e2e_ms={:.2}, \
         decode_phase_tps={:.1}, request_gen_tps={:.1}, e2e_tps={:.1}",
        i + 1,
        output.prompt_tokens,
        output.generated_tokens,
        output.prompt_elapsed.as_secs_f64() * 1000.0,
        output.decode_elapsed.as_secs_f64() * 1000.0,
        output.total_elapsed.as_secs_f64() * 1000.0,
        output.decode_phase_tps(),
        output.request_gen_tps(),
        output.total_tps(),
    );
}

fn append_json_line(
    path: &PathBuf,
    variant: &str,
    pp_label: usize,
    tg_label: usize,
    rep: usize,
    output: &GenerationOutput,
) -> Result<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(
        f,
        "{{\"engine\":\"grout\",\"variant\":\"{}\",\"pp\":{},\"tg\":{},\"rep\":{},\
         \"prompt_tokens\":{},\"gen_tokens\":{},\
         \"prefill_ms\":{:.3},\"decode_ms\":{:.3},\"e2e_ms\":{:.3}}}",
        variant,
        pp_label,
        tg_label,
        rep,
        output.prompt_tokens,
        output.generated_tokens,
        output.prompt_elapsed.as_secs_f64() * 1000.0,
        output.decode_elapsed.as_secs_f64() * 1000.0,
        output.total_elapsed.as_secs_f64() * 1000.0,
    )?;
    Ok(())
}

fn log_summary(label: &str, results: &[GenerationOutput], quiet: bool) {
    if results.is_empty() {
        return;
    }
    let n = results.len() as f64;
    let mean_decode_phase = results.iter().map(|r| r.decode_phase_tps()).sum::<f64>() / n;
    let mean_request_gen = results.iter().map(|r| r.request_gen_tps()).sum::<f64>() / n;
    let mean_e2e = results.iter().map(|r| r.total_tps()).sum::<f64>() / n;
    let mean_elapsed = results
        .iter()
        .map(|r| r.total_elapsed.as_secs_f64())
        .sum::<f64>()
        / n;
    println!(
        "  [{label}] mean over {} runs: decode_phase_tps={mean_decode_phase:.1}, request_gen_tps={mean_request_gen:.1}, e2e_tps={mean_e2e:.1}, elapsed={mean_elapsed:.3}s",
        results.len()
    );
    if quiet {
        return;
    }
    // Print the generated text from the last run.
    println!();
    println!("{}", results.last().unwrap().text);
    if let Some(ref report) = results.last().unwrap().profile_report {
        println!();
        println!("{report}");
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut engine = Qwen3Engine::load(&args.model, args.max_seq_len).await?;
    engine.set_sampling_enabled(args.sample);
    engine.set_chat_template_enabled(!args.raw_prompt);
    engine.set_device_argmax_enabled(!args.host_argmax);
    engine.set_profile_enabled(args.profile);
    engine.set_ignore_eos(args.ignore_eos);

    println!("Loaded model from {}", engine.model_dir().display());

    if args.jit_breakdown {
        println!("JIT breakdown: timing warm_all_kernels() only (no prefill/decode).");
        let t0 = Instant::now();
        engine.warm_all_kernels().await?;
        let elapsed = t0.elapsed();
        println!("JIT_WARM_WALL_MS={:.3}", elapsed.as_secs_f64() * 1000.0);
        println!("JIT_KERNELS_WARMED={}", kernels::TILE_KERNEL_KINDS.len());
        return Ok(());
    }

    // --prompt-file takes precedence over --prompt. The pp-sweep wires
    // synthetic prompts this way since quoting long strings through shell
    // is fragile.
    let prompt_from_file: Option<String> = match args.prompt_file.as_ref() {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("failed to read --prompt-file {:?}: {e}", path))?,
        ),
        None => None,
    };
    let prompt: &str = if let Some(text) = prompt_from_file.as_deref() {
        text
    } else if let Some(p) = args.prompt.as_deref() {
        p
    } else {
        bail!("--prompt or --prompt-file is required unless --jit-breakdown is set");
    };

    // Truncate display of very long prompts so stdout stays readable.
    let preview = if prompt.len() > 160 {
        format!("{}… [{} chars]", &prompt[..160], prompt.len())
    } else {
        prompt.to_string()
    };
    println!("Prompt: {}", preview);
    println!(
        "Generating {} tokens x {} measured reps (+ {} warmup reps after JIT, all discarded)...",
        args.max_new_tokens, args.reps, args.warmup_reps,
    );

    // JIT compilation run (not counted at all).
    println!("\nJIT compile + graph capture...");
    let jit = engine.generate(prompt, args.max_new_tokens).await?;
    println!(
        "  done: decode_phase_tps={:.1}, request_gen_tps={:.1}, prompt_s={:.3}",
        jit.decode_phase_tps(),
        jit.request_gen_tps(),
        jit.prompt_elapsed.as_secs_f64(),
    );

    // Additional warmup reps (discarded).
    for i in 0..args.warmup_reps {
        let output = engine.generate(prompt, args.max_new_tokens).await?;
        log_output("warmup", i, args.warmup_reps, &output);
    }

    // Measured reps.
    println!();
    let mut results = Vec::with_capacity(args.reps);
    for i in 0..args.reps {
        let output = engine.generate(prompt, args.max_new_tokens).await?;
        log_output("timed", i, args.reps, &output);
        if let Some(ref json_path) = args.json {
            append_json_line(
                json_path,
                &args.variant,
                args.pp_label,
                args.max_new_tokens,
                i,
                &output,
            )?;
        }
        results.push(output);
    }
    log_summary("grout", &results, args.quiet);

    Ok(())
}
