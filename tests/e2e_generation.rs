use std::process::Command;

/// End-to-end smoke test: run the `grout` CLI on a fixed prompt and assert the
/// generated text looks like a coherent Qwen3 chat response.
///
/// Requires a working CUDA/cuTile stack and a local model, so it is `#[ignore]`d
/// by default — a plain `cargo test` reports it as *ignored*, not passed. Run it
/// explicitly with `cargo test -- --ignored` (or `--include-ignored`). Point it
/// at a model directory with `GROUT_E2E_MODEL` (defaults to `../hf_models/qwen3_4b`).
#[test]
#[ignore = "GPU e2e: needs CUDA/cuTile + a local model; run with `cargo test -- --ignored`"]
fn qwen3_hello_prompt_generates_coherent_text() {
    let bin = env!("CARGO_BIN_EXE_grout");
    let model = std::env::var("GROUT_E2E_MODEL").unwrap_or_else(|_| "../hf_models/qwen3_4b".into());
    // The plain `grout` CLI does a single greedy generation with host-side
    // argmax (device_argmax defaults to false) and the chat template enabled,
    // which matches the deterministic single-rep intent of this test.
    let output = Command::new(bin)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "--model",
            &model,
            "--prompt",
            "Hello, how are you?",
            "--max-new-tokens",
            "32",
        ])
        .output()
        .expect("failed to run grout binary");

    assert!(
        output.status.success(),
        "grout failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The plain CLI prints the generated text directly (no per-rep timing
    // header), so inspect the whole stdout for coherence markers.
    let has_greeting = stdout.contains("Hello!") || stdout.contains("I'm");
    let has_assistant_marker =
        stdout.contains("Qwen") || stdout.contains("assist") || stdout.contains("language model");
    assert!(
        has_greeting && has_assistant_marker,
        "generated text did not look coherent\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr),
    );
}
