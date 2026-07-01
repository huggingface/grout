use anyhow::Result;
use clap::Parser;
use grout::model::Qwen3Engine;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3 inference engine on cutile-rs")]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long)]
    max_seq_len: Option<usize>,

    #[arg(long, default_value_t = false)]
    sample: bool,

    #[arg(long, default_value_t = false)]
    raw_prompt: bool,

    #[arg(long, default_value_t = false)]
    device_argmax: bool,

    #[arg(long, default_value_t = false)]
    profile: bool,

    /// Discarded warmup generations before the measured run. The first
    /// generate() pays JIT compile + decode-graph capture (~0.85s of cold
    /// prefill), which otherwise lands in the reported prompt t/s. The
    /// default 1 warmup makes the reported t/s reflect steady state; set 0
    /// to see cold-start numbers.
    #[arg(long, default_value_t = 1)]
    warmup_reps: usize,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut engine = Qwen3Engine::load(&args.model, args.max_seq_len).await?;
    engine.set_sampling_enabled(args.sample);
    engine.set_chat_template_enabled(!args.raw_prompt);
    engine.set_device_argmax_enabled(args.device_argmax);
    engine.set_profile_enabled(args.profile);

    println!("Loaded model from {}", engine.model_dir().display());
    println!("Prompt: {}", args.prompt);
    println!("Generating {} tokens...", args.max_new_tokens);

    // Warmup (discarded): the first generate() pays JIT compile + decode-graph
    // capture, which would otherwise pollute the reported prompt t/s. A few
    // decode tokens are enough to JIT the prefill path and capture/replay the
    // decode graph, so cap warmup length to keep it cheap. --warmup-reps 0 opts out.
    let warmup_tokens = args.max_new_tokens.min(8);
    for _ in 0..args.warmup_reps {
        let _ = engine.generate(&args.prompt, warmup_tokens).await?;
    }

    let output = engine.generate(&args.prompt, args.max_new_tokens).await?;
    println!();
    println!("{}", output.text);
    println!(
        "t/s: {:.2} prompt, {:.2} decode phase, {:.2} end-to-end (prompt_tokens={}, generated_tokens={}, prompt_s={:.3}, decode_s={:.3}, total_s={:.3})",
        output.prompt_tps(),
        output.decode_phase_tps(),
        output.total_tps(),
        output.prompt_tokens,
        output.generated_tokens,
        output.prompt_elapsed.as_secs_f64(),
        output.decode_elapsed.as_secs_f64(),
        output.total_elapsed.as_secs_f64(),
    );
    if let Some(report) = output.profile_report {
        println!();
        println!("{report}");
    }
    Ok(())
}
