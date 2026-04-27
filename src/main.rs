use anyhow::Result;
use clap::Parser;
use grout::model::Qwen3Engine;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3-4B inference prototype on cuda-tile")]
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

    let output = engine.generate(&args.prompt, args.max_new_tokens).await?;
    println!();
    println!("{}", output.text);
    println!(
        "t/s: {:.2} prompt, {:.2} decode, {:.2} end-to-end (prompt_tokens={}, generated_tokens={}, prompt_s={:.3}, decode_s={:.3}, total_s={:.3})",
        output.prompt_tps(),
        output.decode_tps(),
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
