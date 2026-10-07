//! Engine-objective autotuning on the cutile-rs 0.3.0 `cutile::tune` stack.
//!
//! The poster-example shape for multi-architecture tuning:
//!
//! - **Declared spaces** (`Config`) per tunable site, searched by the
//!   library's resumable `GridSearch` through the public `Objective` trait.
//! - **Engine objective**: each trial times real `Qwen3Engine` steps (whole
//!   prefill or decode window), not an isolated kernel — grout's tile optima
//!   are only meaningful end-to-end (per-kernel-form tuning lesson).
//! - **Correctness gate**: a candidate whose generated text differs from the
//!   default configuration's is recorded `Invalid`, never timed as a winner.
//! - **Per-arch persistence**: winners land in a provenance-checked
//!   `tune::Record` at `benchmarks/tuning/<arch>/<site>.json`. Records are
//!   refused at load when kernel source, toolchain, or candidate space
//!   changed. Run this same binary on each target arch (sm_120 locally,
//!   sm_100 on a B200) to produce that arch's records; nothing is shared or
//!   approximated across arches.
//!
//! Usage:
//!   grout_autotune --model ../hf_models/qwen3_4b \
//!       --prompt-dir benchmarks/results/sweep/<ts>/prompts \
//!       [--site prefill_attention|prefill_hints|decode_attention|wide_prefill|all] \
//!       [--out-dir benchmarks/tuning] [--reps 3]
//!
//! Trials append to `<out-dir>/<arch>/<site>.<bucket>.trials.jsonl`; an
//! interrupted sweep resumes where it stopped.

use anyhow::{Context, Result};
use clap::Parser;
use cutile::tune::{
    best_config, space_hash, Config, GridSearch, Objective, ParamValue, Record, RecordEntry,
    Searcher, Trial, TrialState, Workspace,
};

/// `Trial`/`TrialState` are #[non_exhaustive]; the public constructors
/// (`Trial::measured` / `Trial::invalid`, landed with cutile-rs #239) are the
/// out-of-crate `Objective` implementor's way to build a return value.
/// `Trial::measured` records a non-finite timing as `Invalid` so it can
/// round-trip through the JSONL log.
fn invalid_trial(config_id: &str, reason: String) -> Trial {
    Trial::invalid(config_id, reason)
}

fn measured_trial(config_id: &str, median_ms: f32, min_ms: f32, reps: usize) -> Trial {
    Trial::measured(config_id, median_ms, min_ms, reps)
}
use grout::model::Qwen3Engine;
use std::fs;
use std::io::Write as _;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: String,
    /// Directory containing pp_<n>.txt prompt files (sweep layout).
    #[arg(long)]
    prompt_dir: String,
    #[arg(long, default_value = "all")]
    site: String,
    #[arg(long, default_value = "benchmarks/tuning")]
    out_dir: String,
    /// Timed engine steps per trial (after one untimed warmup step).
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Max sequence length for the engine (bounds which pp buckets run).
    #[arg(long, default_value_t = 16384)]
    max_seq_len: usize,
    /// Wall-clock budget per BUCKET, minutes (0 = none). Shipping incumbents
    /// are visited first, so a truncated search still yields a record no
    /// worse than what ships; rerun without a budget to finish the grid
    /// (trial logs resume).
    #[arg(long, default_value_t = 0)]
    budget_min: u64,
    /// Debug: run the default config N times on the first bucket and print
    /// each generated text hash (in-process determinism probe), then exit.
    #[arg(long, default_value_t = 0)]
    gate_probe: usize,
    /// Proactive device-state hygiene: every N completed trials, drop the
    /// engine, quiesce, evict every cached kernel specialization and reload
    /// — before an allocation fails rather than after (0 = reactive only).
    /// Each eviction costs one engine reload (seconds to a minute on a
    /// 32B model), so pick N from how many specializations fit: ~40 on a
    /// 180 GB B200 with Qwen3-32B, more on smaller models.
    #[arg(long, default_value_t = 0)]
    evict_every: usize,
    /// Prefill buckets (prompt lengths) for the prefill sites, e.g.
    /// "18,512,2048,8192". Records apply from a bucket's length upward, and
    /// a prompt shorter than the smallest bucket receives that bucket's
    /// winner — include a short bucket when short prompts matter.
    #[arg(long, default_value = "512,2048,8192")]
    pp_buckets: String,
    /// Extra explicit candidate(s) for the selected site, e.g. a hand
    /// profile to wire in as-is: `--require GROUT_ATTN_BM_PREFILL=128,GROUT_ATTN_BN_PREFILL=128,GROUT_FMHA_PREFILL_OCCUPANCY=2`.
    /// Repeatable. Measured before the grid (like the shipping incumbents),
    /// logged, and eligible to win — the value 0 means "unset". Keys need
    /// not be axes of the site (the engine honors any knob), but then use
    /// `--site` to target one site.
    #[arg(long)]
    require: Vec<String>,
}

/// One tunable engine site: env-var axes, shape buckets, and an objective.
struct Site {
    name: &'static str,
    /// (env var, values). Value 0 means "unset" (compiler/engine default).
    axes: Vec<(&'static str, Vec<i64>)>,
    /// (bucket label, pp tokens file stem, max_new_tokens, decode_objective)
    buckets: Vec<Bucket>,
}

struct Bucket {
    label: String,
    prompt_pp: usize,
    max_new_tokens: usize,
    /// false: objective = prompt_elapsed; true: objective = decode_elapsed.
    decode_objective: bool,
}

fn sites(max_seq_len: usize, pp_buckets: &[usize]) -> Vec<Site> {
    // Axes mirror the original tile-sweep scripts (sweep_pp_tile.sh /
    // sweep_tg_tile.sh) plus the knobs the shipping sm_100/sm_120 profiles
    // set — critically including the KERNEL DISPATCH flag
    // (GROUT_FMHA_PREFILL_GQA_LPT) and occupancy. The first B200 tuning
    // attempt failed its parity gate because the space omitted the
    // dispatch axis: on sm_100 auto-LPT engaged for every candidate while
    // the shipping config is the mapped kernel with LPT off. Rule learned:
    // the incumbent shipping config must be expressible within the space
    // (it is, for both arches: sm_100 mapped 128/128/occ2/LPT0 and the
    // sm_120 profile corners are all covered).
    let pp_bucket = |pp: usize| Bucket {
        label: format!("pp={pp}"),
        prompt_pp: pp,
        max_new_tokens: 16,
        decode_objective: false,
    };
    // Thread-block-cluster axis (num_cta_in_cga), opt-in via
    // GROUT_TUNE_CGA_AXIS=1. On sm_120 (RTX 5090) clusters on the attention
    // kernels are a cliff, not a knob: paired screen 2026-09-04, CGA=2 ran
    // 4.7x slower at pp=2048 and 11.5x at pp=8192, CGA=4 8x/22x — every
    // cluster candidate would cost ~10x the tuning time to confirm what the
    // screen already showed. Untested on sm_100, where clusters are native
    // and the K/V multicast argument actually applies: screen one config
    // there first (`--require ...,GROUT_FMHA_PREFILL_CGA=2`) before turning
    // the axis on.
    let cga_axis: Vec<i64> = if std::env::var("GROUT_TUNE_CGA_AXIS").is_ok_and(|v| v == "1") {
        vec![0, 2, 4]
    } else {
        vec![0]
    };
    vec![
        Site {
            name: "prefill_attention",
            axes: vec![
                // 0 = mapped kernel, 1 = LPT kernel; LPT-specific knobs
                // (swizzle/sched/mask-split) ride engine defaults.
                ("GROUT_FMHA_PREFILL_GQA_LPT", vec![0, 1]),
                // 1 = head-grouped GQA-mapped kernel (fmha_prefill_gqa_mapped),
                // the third dispatch the legacy sweep_pp_tile.sh scanned and
                // this space initially omitted. On the 5090 a paired check
                // (15 vs 15, alternating) put its best cell at parity with
                // the record (1.010 @2048, 0.992 @8192) — included for
                // completeness of the space, not because it is expected to
                // win. Zero is ELIDED from config ids (see ELIDE_ZERO_AXES)
                // so trial logs recorded before the axis existed stay valid.
                // Only meaningful with LPT=0 and small BM (the effective tile
                // is BM x group); see prune().
                ("GROUT_FMHA_PREFILL_GQA", vec![0, 1]),
                ("GROUT_ATTN_BM_PREFILL", vec![4, 8, 16, 32, 64, 128]),
                ("GROUT_ATTN_BN_PREFILL", vec![16, 32, 64, 128]),
                ("GROUT_FMHA_PREFILL_OCCUPANCY", vec![1, 2]),
                // num_worker_warps_per_cta (0 = compiler default). Full
                // range, not just 4: the sm_100 08-23 run already moved
                // two prefill buckets to warps=4 and wide prefill to 2.
                ("GROUT_FMHA_PREFILL_WARPS", vec![0, 2, 4, 8]),
            ],
            buckets: pp_buckets
                .iter()
                .copied()
                .filter(|pp| *pp < max_seq_len)
                .map(pp_bucket)
                .collect(),
        },
        // Architecture-specific optimization hints for the prefill attention
        // kernels, tuned as a second layer ON TOP of the tile/dispatch winners
        // above: the engine resolves every knob env > record > default, and
        // the driver reloads the engine after each site's record is saved, so
        // candidates here run with the prefill_attention record already
        // applied. Kept as its own site because the joint space (128 x 192
        // per bucket) is not searchable in an evening; the layering is the
        // classic coordinate-descent compromise. Per bucket, knobs the base
        // dispatch cannot see (LPT schedule/swizzle/mask-split under a
        // mapped base; GQA group under a causal-mapped base) are pruned to
        // one representative, so a causal-mapped base searches 4 live cells
        // instead of 192 duplicates (see hints_inert_duplicate).
        Site {
            name: "prefill_hints",
            axes: vec![
                // load_pipelined depth for the K/V loads (engine default 2).
                ("GROUT_FMHA_PREFILL_LATENCY", vec![1, 2, 3, 4]),
                // GQA heads per CTA (0 = query_group_size, i.e. all of them).
                ("GROUT_FMHA_PREFILL_GQA_GROUP", vec![0, 2, 4, 8]),
                // LPT schedule: 1 = linear, 2/3 = swizzled (reverse/forward).
                ("GROUT_FMHA_PREFILL_LPT_SCHED", vec![1, 2, 3]),
                // LPT swizzle width (0 = derived from L2 budget).
                ("GROUT_FMHA_PREFILL_LPT_SWIZZLE", vec![0, 8]),
                // Boolean: -1 = explicit off (see apply_config), 1 = on.
                ("GROUT_FMHA_PREFILL_LPT_MASK_SPLIT", vec![-1, 1]),
                // Thread-block cluster size (num_cta_in_cga): adjacent CTAs
                // share K/V tiles, so cluster multicast can cut L2 traffic.
                // Live for every dispatch; 0 = compiler default (elided).
                ("GROUT_FMHA_PREFILL_CGA", cga_axis.clone()),
            ],
            buckets: pp_buckets
                .iter()
                .copied()
                .filter(|pp| *pp < max_seq_len)
                .map(pp_bucket)
                .collect(),
        },
        Site {
            name: "decode_attention",
            axes: vec![
                ("GROUT_ATTN_BN_DECODE", vec![16, 32, 64, 128]),
                ("GROUT_FMHA_NUM_KV_SPLITS", vec![4, 8, 16, 32]),
                ("GROUT_FMHA_DECODE_WARPS", vec![0, 2, 4, 8]),
                ("GROUT_FMHA_DECODE_CGA", cga_axis.clone()),
            ],
            // Canonical decode cells use a short prompt; tuning in a long
            // kv context (the first attempt used pp=512) skews winners.
            //
            // The bucket is labeled by max_seq_len, not tg: split-kv
            // geometry partitions the ALLOCATED cache (kv_len_per_split =
            // ceil(max_seq_len / splits)), so the NKS optimum is a
            // function of the engine's max_seq_len. The sm_100 gate
            // proved a winner tuned at msl=16384 does not transfer to
            // the canonical msl=4096 engine. Tune once per max_seq_len
            // the deployment uses; records coexist as separate buckets.
            buckets: vec![Bucket {
                label: format!("msl={max_seq_len}"),
                prompt_pp: 18,
                max_new_tokens: 128,
                decode_objective: true,
            }],
        },
        Site {
            name: "wide_prefill",
            axes: vec![
                ("GROUT_QK_PREFILL_BM", vec![16, 32, 64]),
                ("GROUT_QK_PREFILL_WARPS", vec![0, 1, 2, 4]),
                ("GROUT_QK_PREFILL_CGA", cga_axis.clone()),
            ],
            // Wide fused-Q/KV prefill only pays off at long prompts.
            buckets: pp_buckets
                .iter()
                .copied()
                .filter(|pp| *pp >= 2048 && *pp < max_seq_len)
                .map(pp_bucket)
                .collect(),
        },
    ]
}

/// Shipping incumbent configs per arch: coverage is sufficient only if
/// every incumbent is expressible inside the declared space — verified at
/// startup, so a candidate space can never again omit the config it must
/// beat (the failure mode of the first B200 tuning attempt).
fn incumbents(arch: &str, site: &str) -> Vec<Vec<(&'static str, i64)>> {
    match (arch, site) {
        (_, "prefill_attention") if arch.starts_with("sm_100") => vec![
            // sweep_pp_sm100.sh pp<=8192: mapped kernel, 128x128, occ 2.
            vec![
                ("GROUT_FMHA_PREFILL_GQA_LPT", 0),
                ("GROUT_ATTN_BM_PREFILL", 128),
                ("GROUT_ATTN_BN_PREFILL", 128),
                ("GROUT_FMHA_PREFILL_OCCUPANCY", 2),
                ("GROUT_FMHA_PREFILL_WARPS", 0),
            ],
        ],
        (_, "prefill_attention") => vec![
            // sweep_pp_sm120.sh: mapped 64x32 short, LPT on at >=2048.
            vec![
                ("GROUT_FMHA_PREFILL_GQA_LPT", 0),
                ("GROUT_ATTN_BM_PREFILL", 64),
                ("GROUT_ATTN_BN_PREFILL", 32),
                ("GROUT_FMHA_PREFILL_OCCUPANCY", 1),
                ("GROUT_FMHA_PREFILL_WARPS", 0),
            ],
            vec![
                ("GROUT_FMHA_PREFILL_GQA_LPT", 1),
                ("GROUT_ATTN_BM_PREFILL", 16),
                ("GROUT_ATTN_BN_PREFILL", 64),
                ("GROUT_FMHA_PREFILL_OCCUPANCY", 1),
                ("GROUT_FMHA_PREFILL_WARPS", 0),
            ],
        ],
        (_, "prefill_hints") if arch.starts_with("sm_100") => vec![
            // sweep_pp_sm100.sh pp=2048/8192: latency 2, group auto, LPT
            // knobs (swizzle 8 / sched 1 / mask-split off) — inert while the
            // incumbent runs the mapped kernel, but the profile sets them.
            vec![
                ("GROUT_FMHA_PREFILL_LATENCY", 2),
                ("GROUT_FMHA_PREFILL_GQA_GROUP", 0),
                ("GROUT_FMHA_PREFILL_LPT_SCHED", 1),
                ("GROUT_FMHA_PREFILL_LPT_SWIZZLE", 8),
                ("GROUT_FMHA_PREFILL_LPT_MASK_SPLIT", -1),
                ("GROUT_FMHA_PREFILL_CGA", 0),
            ],
        ],
        (_, "prefill_hints") => vec![
            // sm_120 records run LPT with engine defaults for every hint.
            vec![
                ("GROUT_FMHA_PREFILL_LATENCY", 2),
                ("GROUT_FMHA_PREFILL_GQA_GROUP", 0),
                ("GROUT_FMHA_PREFILL_LPT_SCHED", 1),
                ("GROUT_FMHA_PREFILL_LPT_SWIZZLE", 0),
                ("GROUT_FMHA_PREFILL_LPT_MASK_SPLIT", 1),
                ("GROUT_FMHA_PREFILL_CGA", 0),
            ],
        ],
        (_, "decode_attention") if arch.starts_with("sm_100") => vec![
            // sweep_tg_sm100.sh TG_128 cell.
            vec![
                ("GROUT_ATTN_BN_DECODE", 32),
                ("GROUT_FMHA_NUM_KV_SPLITS", 4),
                ("GROUT_FMHA_DECODE_WARPS", 0),
                ("GROUT_FMHA_DECODE_CGA", 0),
            ],
        ],
        (_, "decode_attention") => vec![vec![
            ("GROUT_ATTN_BN_DECODE", 32),
            ("GROUT_FMHA_NUM_KV_SPLITS", 16),
            ("GROUT_FMHA_DECODE_WARPS", 0),
            ("GROUT_FMHA_DECODE_CGA", 0),
        ]],
        (_, "wide_prefill") => vec![vec![
            ("GROUT_QK_PREFILL_BM", 32),
            ("GROUT_QK_PREFILL_WARPS", 0),
            ("GROUT_QK_PREFILL_CGA", 0),
        ]],
        _ => vec![],
    }
}

/// Every incumbent parameter value must be a member of its axis.
fn verify_coverage(arch: &str, site: &Site) -> Result<()> {
    for incumbent in incumbents(arch, site.name) {
        for (key, value) in &incumbent {
            let axis = site
                .axes
                .iter()
                .find(|(name, _)| name == key)
                .with_context(|| format!("{}: incumbent key {key} has no axis", site.name))?;
            anyhow::ensure!(
                axis.1.contains(value),
                "{}: incumbent {key}={value} is NOT in the declared axis {:?} —                  the space cannot beat a config it does not contain",
                site.name,
                axis.1
            );
        }
    }
    println!(
        "  coverage ok: {} shipping incumbent(s) inside the {} space",
        incumbents(arch, site.name).len(),
        site.name
    );
    Ok(())
}

/// LPT-only hint axes are meaningless when the prefill_attention record for
/// this arch runs the mapped kernel in every bucket; collapse them to the
/// incumbent value so the search spends its budget on knobs that fire.
/// Short stable tag of the prefill_attention record's winning config for a
/// bucket ("none" when no record exists yet).
fn base_config_tag(out_dir: &Path, bucket: &str) -> String {
    let Ok(text) = fs::read_to_string(out_dir.join("prefill_attention.json")) else {
        return "none".into();
    };
    let Ok(record) = serde_json::from_str::<serde_json::Value>(&text) else {
        return "none".into();
    };
    let id = record["entries"]
        .as_array()
        .and_then(|es| es.iter().find(|e| e["bucket"].as_str() == Some(bucket)))
        .and_then(|e| e["config"]["id"].as_str())
        .unwrap_or("");
    if id.is_empty() {
        return "none".into();
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(id, &mut h);
    format!("{:08x}", std::hash::Hasher::finish(&h) as u32)
}

/// The winning parameters recorded for `bucket` in `<out_dir>/<site>.json`.
fn record_winner(out_dir: &Path, site: &str, bucket: &str) -> Option<BTreeMap<String, i64>> {
    let text = fs::read_to_string(out_dir.join(format!("{site}.json"))).ok()?;
    let record: serde_json::Value = serde_json::from_str(&text).ok()?;
    let entry = record["entries"]
        .as_array()?
        .iter()
        .find(|e| e["bucket"].as_str() == Some(bucket))?;
    let params = entry["config"]["params"].as_object()?;
    Some(
        params
            .iter()
            .filter_map(|(k, v)| v.as_i64().map(|v| (k.clone(), v)))
            .collect(),
    )
}

/// The prefill_attention winner a prefill_hints bucket runs on top of.
fn base_config(out_dir: &Path, bucket: &str) -> Option<BTreeMap<String, i64>> {
    record_winner(out_dir, "prefill_attention", bucket)
}

fn base_dispatch_name(base: &BTreeMap<String, i64>) -> &'static str {
    let get = |k: &str| base.get(k).copied().unwrap_or(0);
    if get("GROUT_FMHA_PREFILL_GQA_LPT") == 1 {
        "LPT"
    } else if get("GROUT_FMHA_PREFILL_GQA") == 1 {
        "GQA-mapped"
    } else {
        "causal-mapped"
    }
}

/// Hint knobs the engine does not read under a given base dispatch. The
/// LPT schedule/swizzle/mask-split knobs exist only in the LPT kernel; the
/// GQA group is read by the LPT and GQA-mapped kernels but not by the
/// causal-mapped one. Every other combination of the inert knobs is a
/// re-measurement of the same kernel — 188 of the 192 hint candidates for a
/// causal-mapped base (the 2026-09-04 B200 pp=8192 bucket).
fn inert_hint_knobs(base: &BTreeMap<String, i64>) -> &'static [&'static str] {
    const LPT_ONLY: &[&str] = &[
        "GROUT_FMHA_PREFILL_LPT_SCHED",
        "GROUT_FMHA_PREFILL_LPT_SWIZZLE",
        "GROUT_FMHA_PREFILL_LPT_MASK_SPLIT",
    ];
    const LPT_ONLY_AND_GROUP: &[&str] = &[
        "GROUT_FMHA_PREFILL_LPT_SCHED",
        "GROUT_FMHA_PREFILL_LPT_SWIZZLE",
        "GROUT_FMHA_PREFILL_LPT_MASK_SPLIT",
        "GROUT_FMHA_PREFILL_GQA_GROUP",
    ];
    match base_dispatch_name(base) {
        "LPT" => &[],
        "GQA-mapped" => LPT_ONLY,
        _ => LPT_ONLY_AND_GROUP,
    }
}

/// A hint candidate is an inert duplicate when any knob the base cannot
/// see is off its canonical (incumbent) value — exactly one representative
/// per live combination survives. Without a base record nothing is pruned.
fn hints_inert_duplicate(
    c: &Config,
    base: Option<&BTreeMap<String, i64>>,
    canonical: &[(&'static str, i64)],
) -> bool {
    let Some(base) = base else {
        return false;
    };
    inert_hint_knobs(base).iter().any(|k| {
        let canon = canonical
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| *v)
            .unwrap_or(0);
        let actual = match c.params.get(*k) {
            Some(ParamValue::Int(v)) => *v,
            _ => 0,
        };
        actual != canon
    })
}

/// Number of knobs on which `c` differs from a reference configuration
/// (absent = 0 on either side, matching the tuner's unset convention).
fn config_distance(c: &Config, reference: &BTreeMap<String, i64>) -> usize {
    let mut keys: std::collections::BTreeSet<&str> =
        c.params.keys().map(String::as_str).collect();
    keys.extend(reference.keys().map(String::as_str));
    keys.into_iter()
        .filter(|k| {
            let a = match c.params.get(*k) {
                Some(ParamValue::Int(v)) => *v,
                _ => 0,
            };
            let b = reference.get(*k).copied().unwrap_or(0);
            a != b
        })
        .count()
}

/// Axes added after trial logs already existed: a zero (= unset) value is
/// omitted from the config instead of recorded as `KEY=0`, so config ids of
/// the pre-existing grid are unchanged and resume keeps every prior trial.
const ELIDE_ZERO_AXES: &[&str] = &[
    "GROUT_FMHA_PREFILL_GQA",
    "GROUT_FMHA_PREFILL_CGA",
    "GROUT_FMHA_DECODE_CGA",
    "GROUT_QK_PREFILL_CGA",
];

/// Candidates that cannot express anything the rest of the grid does not.
fn prune(site: &str, params: &[(&'static str, i64)]) -> bool {
    let get = |k: &str| params.iter().find(|(n, _)| *n == k).map(|(_, v)| *v).unwrap_or(0);
    if site == "prefill_attention" {
        let gqa = get("GROUT_FMHA_PREFILL_GQA");
        let lpt = get("GROUT_FMHA_PREFILL_GQA_LPT");
        let bm = get("GROUT_ATTN_BM_PREFILL");
        // LPT dispatch takes precedence in the engine: GQA=1 is inert there.
        if gqa == 1 && lpt == 1 {
            return true;
        }
        // The GQA-mapped tile is BM x group rows: 64/128 explode it, while
        // 4/8 only make sense there (the causal/LPT kernels take BM >= 16).
        if gqa == 1 && bm >= 64 {
            return true;
        }
        if gqa == 0 && bm < 16 {
            return true;
        }
    }
    false
}

fn cartesian(site: &str, axes: &[(&'static str, Vec<i64>)]) -> Vec<Config> {
    let mut configs: Vec<Vec<(&'static str, i64)>> = vec![vec![]];
    for (name, values) in axes {
        configs = configs
            .into_iter()
            .flat_map(|base| {
                values.iter().map(move |v| {
                    let mut c = base.clone();
                    c.push((name, *v));
                    c
                })
            })
            .collect();
    }
    configs
        .into_iter()
        .filter(|params| !prune(site, params))
        .map(|params| {
            Config::new(
                params
                    .into_iter()
                    .filter(|(k, v)| !(ELIDE_ZERO_AXES.contains(k) && *v == 0))
                    .map(|(k, v)| (k, ParamValue::Int(v))),
            )
        })
        .collect()
}

/// `KEY=V,KEY=V` from --require into a Config (elision rule applied so a
/// spec naming a grid point gets the grid point's id).
fn parse_required_config(spec: &str) -> Result<Config> {
    let mut params: Vec<(String, ParamValue)> = Vec::new();
    for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (k, v) = item
            .split_once('=')
            .with_context(|| format!("--require item {item:?} is not KEY=VALUE"))?;
        let v: i64 = v
            .trim()
            .parse()
            .with_context(|| format!("--require {k}: value {v:?} is not an integer"))?;
        if ELIDE_ZERO_AXES.contains(&k.trim()) && v == 0 {
            continue;
        }
        params.push((k.trim().to_string(), ParamValue::Int(v)));
    }
    anyhow::ensure!(!params.is_empty(), "--require {spec:?} names no parameters");
    Ok(Config::new(params))
}

/// Incumbent membership with the elision rule: an absent elided key equals 0.
fn config_matches(c: &Config, incumbent: &[(&'static str, i64)]) -> bool {
    incumbent.iter().all(|(k, v)| match c.params.get(*k) {
        Some(p) => *p == ParamValue::Int(*v),
        None => *v == 0 && ELIDE_ZERO_AXES.contains(k),
    })
}

fn apply_config(config: &Config) {
    for (key, value) in &config.params {
        // SAFETY: the tuner is single-threaded; env mutation happens strictly
        // between engine steps, and the engine reads these vars on this same
        // thread inside block_on.
        unsafe {
            match value {
                ParamValue::Int(0) => std::env::remove_var(key),
                // Negative = an explicit "0" for boolean knobs whose unset
                // default is true (0 itself means "unset" in this space).
                ParamValue::Int(v) if *v < 0 => std::env::set_var(key, "0"),
                ParamValue::Int(v) => std::env::set_var(key, v.to_string()),
                ParamValue::Str(s) => std::env::set_var(key, s),
            }
        }
    }
}

fn clear_config(config: &Config) {
    for key in config.params.keys() {
        // SAFETY: see apply_config.
        unsafe { std::env::remove_var(key) };
    }
}

/// Engine-objective oracle: measures whole engine steps per candidate.
///
/// This composes with the library's `GridSearch`/`Searcher` through the
/// public `Objective` trait (named `Oracle` before cutile-rs #239); the
/// closure-based `Autotuner` front-end assumes a CUDA-event-timed launch on
/// one stream, which does not fit an engine step that spans streams and host
/// logic (reported upstream).
struct EngineOracle<'a> {
    configs: Vec<Config>,
    engine: Option<Qwen3Engine>,
    model_dir: PathBuf,
    max_seq_len: usize,
    /// `--evict-every`: 0 = reactive recovery only.
    evict_every: usize,
    /// Trials completed since the last eviction/reload.
    trials_since_evict: usize,
    rt: &'a tokio::runtime::Runtime,
    prompt: String,
    max_new_tokens: usize,
    decode_objective: bool,
    reps: usize,
    reference_text: Option<String>,
    deadline: Option<Instant>,
    log: fs::File,
}

/// All tuner engines must run a fixed decode window: with EOS active, a
/// "128-token" candidate trial measures however many tokens that sample
/// happened to emit, and the winner is not comparable to a fixed-length
/// shipping run (found the hard way in the sm_100 parity gate).
fn load_engine(
    rt: &tokio::runtime::Runtime,
    model_dir: &Path,
    max_seq_len: usize,
) -> Result<Qwen3Engine> {
    let mut engine = rt.block_on(Qwen3Engine::load(model_dir, Some(max_seq_len)))?;
    // Benchmark semantics, identical to grout_bench's defaults and to the
    // sweeps the records are gated against:
    // - greedy: the engine inherits do_sample from the model's
    //   generation_config.json (true for Qwen3, temperature 0.7 / top-k 20 /
    //   top-p 0.8), and sampling also forces the host-side token-selection
    //   path instead of the in-graph argmax — a different decode program
    //   from the one that ships (found by the cutile-rs review, 2026-09-21;
    //   every earlier tuner run measured under sampling);
    // - raw prompts: the pp_<n>.txt files are exact token counts, and the
    //   chat template would add its wrapper tokens (kv_len no longer the
    //   bucket length, EVEN_K flips);
    // - fixed decode window (EOS ignored).
    engine.set_sampling_enabled(false);
    engine.set_chat_template_enabled(false);
    engine.set_ignore_eos(true);
    Ok(engine)
}

fn is_alloc_reason(msg: &str) -> bool {
    msg.contains("ALLOC_FAILED")
        || msg.contains("OUT_OF_MEMORY")
        || msg.contains("OutOfMemory")
        || msg.contains("out of memory")
}

fn is_alloc_failure(e: &anyhow::Error) -> bool {
    is_alloc_reason(&format!("{e:#}"))
}

/// Last resort when in-process recovery (`recover_device_state`: drop the
/// engine, quiesce, evict the kernel cache, reload) itself fails because the
/// reload cannot allocate: exit non-zero WITHOUT logging the candidate so
/// `autotune_loop.sh` restarts a fresh process, whose resumed search
/// measures that candidate first. With `--evict-every` set this path
/// should be rare; it exists because eviction cannot guarantee that every
/// later allocation succeeds.
fn exit_for_fresh_process(context: &str) -> ! {
    eprintln!(
        "  device allocation failure {context} — exiting with code 3 so autotune_loop.sh \
         restarts a fresh process (nothing logged for the interrupted candidate; the resumed \
         search measures it first)"
    );
    std::process::exit(3);
}

impl EngineOracle<'_> {
    /// A sweep churns kernel specializations by design, and cutile's
    /// in-memory (L1) kernel cache is intentionally unbounded — every
    /// candidate's modules stay resident until evicted. After enough
    /// candidates, allocations fail for reasons that have nothing to do
    /// with the candidate. Dropping the engine alone does not help (the
    /// 2026-09-04 B200 run: after one reload every remaining pp=8192
    /// candidate came back OOM) — the modules live in the process-global
    /// cache, not in the engine. Recovery: drop the engine (its CUDA graphs
    /// and warm registry reference cached modules), quiesce the device so no
    /// cached kernel can still be executing (the eviction API's safety
    /// contract), evict every cached specialization, reload. The reload
    /// recompiles what it needs; the disk cache serves stage 2.
    fn recover_device_state(&mut self) -> Result<usize> {
        self.engine = None;
        grout::model::device_synchronize();
        // SAFETY: the engine — the only launcher of cached kernels in this
        // process — has been dropped and the device synchronized above, so
        // no cached module can still be executing on any stream.
        let evicted = unsafe { cutile::tile_kernel::clear_kernel_cache() };
        self.engine = Some(load_engine(self.rt, &self.model_dir, self.max_seq_len)?);
        self.trials_since_evict = 0;
        Ok(evicted)
    }

    fn step_ms(&mut self) -> Result<(f32, String)> {
        let engine = self.engine.as_mut().expect("engine");
        let out = self
            .rt
            .block_on(engine.generate(&self.prompt, self.max_new_tokens))?;
        let ms = if self.decode_objective {
            out.decode_elapsed.as_secs_f32() * 1e3
        } else {
            out.prompt_elapsed.as_secs_f32() * 1e3
        };
        Ok((ms, out.text))
    }

    fn measure_inner(&mut self, index: usize) -> Result<Trial> {
        let config = self.configs[index].clone();
        apply_config(&config);
        // Warmup step doubles as the compile/launch/correctness gate.
        let (_, text) = match self.step_ms() {
            Ok(v) => v,
            // Allocation failures are decided by `measure` (leak vs genuine).
            Err(e) if is_alloc_failure(&e) => {
                clear_config(&config);
                return Err(e);
            }
            Err(e) => {
                clear_config(&config);
                return Ok(invalid_trial(
                    &config.id,
                    format!("engine step failed: {e:#}"),
                ));
            }
        };
        // Gate v1 = the step succeeded (compile/launch/shape errors above
        // invalidate the candidate).
        //
        // History: an earlier version of this comment blamed the lack of a
        // text-match gate on "allocation-address-dependent reduction order"
        // making greedy output nondeterministic. That was wrong. The engine
        // inherited do_sample=true from the model's generation_config.json,
        // so the --gate-probe that "verified" it was sampling (temperature
        // 0.7, top-k 20, top-p 0.8). With sampling off (load_engine), the
        // probe is deterministic: five runs of the default config on the
        // 5090 produced one text hash (2026-09-21). Caught by the cutile-rs
        // review.
        //
        // Why text-match is still not the gate: a *different* candidate
        // legitimately changes floating-point summation order (tile shape,
        // LPT vs mapped, split-K), so greedy text can diverge after a
        // near-tie token without any kernel being wrong — a strict match
        // would reject valid configurations. Numeric correctness of every
        // kernel form is covered by the GPU test suite and the bitwise
        // cross-arm gates in the exp4 harnesses; a tolerance-based gate
        // (logit-level agreement) is the tracked follow-up.
        let _ = (&text, &self.reference_text);
        let mut samples = Vec::with_capacity(self.reps);
        for _ in 0..self.reps {
            samples.push(self.step_ms()?.0);
        }
        clear_config(&config);
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[samples.len() / 2];
        Ok(measured_trial(&config.id, median, samples[0], samples.len()))
    }
}

impl Objective for EngineOracle<'_> {
    fn configs(&self) -> &[Config] {
        &self.configs
    }

    fn measure(&mut self, index: usize) -> Trial {
        let id = self.configs[index].id.clone();
        // Proactive hygiene: same drop -> quiesce -> evict -> reload as the
        // reactive path, run before the trial so an allocation failure is
        // never the trigger. A failed reload here is the same last resort.
        if self.evict_every > 0 && self.trials_since_evict >= self.evict_every {
            let since = self.trials_since_evict;
            match self.recover_device_state() {
                Ok(evicted) => {
                    eprintln!(
                        "  proactive eviction after {since} trials: {evicted} cached kernel \
                         specialization(s) evicted, engine reloaded"
                    );
                }
                Err(err) => {
                    let _ = self.log.flush();
                    exit_for_fresh_process(&format!(
                        "before {id} (proactive eviction reload failed: {err:#})"
                    ));
                }
            }
        }
        let mut recovered = false;
        let trial = loop {
            match self.measure_inner(index) {
                Ok(t) => break t,
                Err(e) if is_alloc_failure(&e) => {
                    clear_config(&self.configs[index]);
                    if !recovered {
                        recovered = true;
                        match self.recover_device_state() {
                            Ok(evicted) => {
                                eprintln!(
                                    "  device allocation failure on {id}: engine dropped, \
                                     {evicted} cached kernel specialization(s) evicted, engine \
                                     reloaded — retrying once"
                                );
                                continue;
                            }
                            Err(err) => {
                                let _ = self.log.flush();
                                exit_for_fresh_process(&format!(
                                    "on {id} (in-process recovery failed: {err:#})"
                                ));
                            }
                        }
                    }
                    // Failed again on a fresh engine with an empty kernel
                    // cache: this one is the candidate's own.
                    break invalid_trial(
                        &id,
                        format!("device allocation failure after in-process recovery: {e:#}"),
                    );
                }
                Err(e) => break invalid_trial(&id, format!("harness error: {e:#}")),
            }
        };
        if let Ok(line) = serde_json::to_string(&trial) {
            let _ = writeln!(self.log, "{line}");
        }
        self.trials_since_evict += 1;
        match &trial.state {
            TrialState::Measured { median_ms, .. } => {
                println!("  {} -> {median_ms:.2} ms", trial.config_id)
            }
            TrialState::Invalid { reason } => {
                println!("  {} -> invalid: {reason}", trial.config_id)
            }
            other => println!("  {} -> {other:?}", trial.config_id),
        }
        trial
    }

    fn budget_remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
}

fn load_existing_trials(path: &Path) -> Vec<Trial> {
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| serde_json::from_str::<Trial>(l).ok())
        // An OOM-flavored Invalid was recorded against the candidate by
        // older tuner builds when the engine had leaked (2026-09-04 B200
        // pp=8192 logs). Treat those as unvisited so resume re-measures them.
        .filter(|t| match &t.state {
            TrialState::Invalid { reason } => !is_alloc_reason(reason),
            _ => true,
        })
        .collect()
}

fn detect_arch() -> String {
    std::env::var("GROUT_TUNE_ARCH").unwrap_or_else(|_| grout::model::device_arch(0))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let arch = detect_arch();
    let pp_buckets: Vec<usize> = args
        .pp_buckets
        .split(',')
        .map(|v| v.trim().parse::<usize>().context("--pp-buckets"))
        .collect::<Result<_>>()?;
    anyhow::ensure!(!pp_buckets.is_empty(), "--pp-buckets is empty");
    let out_dir = PathBuf::from(&args.out_dir).join(&arch);
    fs::create_dir_all(&out_dir)?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let model_dir = PathBuf::from(&args.model);
    let mut engine_slot: Option<Qwen3Engine> =
        Some(load_engine(&rt, &model_dir, args.max_seq_len)?);

    let tileiras = cutile::cutile_compiler::cuda_tile_runtime_utils::tileiras_fingerprint().to_string();

    if args.gate_probe > 0 {
        let site_list = sites(args.max_seq_len, &pp_buckets);
        let bucket = &site_list[0].buckets[0];
        let prompt_path =
            PathBuf::from(&args.prompt_dir).join(format!("pp_{}.txt", bucket.prompt_pp));
        let prompt = fs::read_to_string(&prompt_path)?;
        let mut engine = engine_slot.take().expect("engine");
        for i in 0..args.gate_probe {
            let out = rt.block_on(engine.generate(&prompt, bucket.max_new_tokens))?;
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&out.text, &mut hash);
            println!(
                "probe {i}: prefill={:.2} ms hash={:016x} text[..48]={:?}",
                out.prompt_elapsed.as_secs_f32() * 1e3,
                std::hash::Hasher::finish(&hash),
                out.text.chars().take(48).collect::<String>()
            );
        }
        return Ok(());
    }

    // Engines constructed below load records from this directory, so a
    // site tuned later in the run (prefill_hints) sees the winners saved by
    // an earlier one (prefill_attention).
    if std::env::var("GROUT_TUNING_RECORD_DIR").is_err() {
        // SAFETY: single-threaded, before any engine exists.
        unsafe { std::env::set_var("GROUT_TUNING_RECORD_DIR", &args.out_dir) };
    }
    for site in sites(args.max_seq_len, &pp_buckets) {
        if args.site != "all" && args.site != site.name {
            continue;
        }
        verify_coverage(&arch, &site)?;
        // The declared space (space_hash is taken over this list); each
        // bucket below prunes and orders its own copy.
        let mut site_configs = cartesian(site.name, &site.axes);
        let incumbent_set = incumbents(&arch, site.name);
        // Explicit --require configs; duplicates of a grid point are
        // dropped by id so the trial log stays one entry per id.
        for spec in args.require.iter().rev() {
            let config = parse_required_config(spec)?;
            site_configs.retain(|c| c.id != config.id);
            site_configs.insert(0, config);
        }
        println!(
            "site {} — {} declared candidates x {} buckets on {arch}",
            site.name,
            site_configs.len(),
            site.buckets.len()
        );
        let mut entries: Vec<RecordEntry> = Vec::new();
        for bucket in &site.buckets {
            let prompt_path =
                PathBuf::from(&args.prompt_dir).join(format!("pp_{}.txt", bucket.prompt_pp));
            let prompt = fs::read_to_string(&prompt_path)
                .with_context(|| format!("prompt file {}", prompt_path.display()))?;
            // prefill_hints is coordinate descent on top of the
            // prefill_attention winner, so its trials are only comparable
            // while that base is unchanged: tag the log with the base
            // config's id so a new attention winner starts a fresh log
            // instead of resuming stale measurements.
            let base_tag = if site.name == "prefill_hints" {
                format!(".base-{}", base_config_tag(&out_dir, &bucket.label))
            } else {
                String::new()
            };
            let trials_path = out_dir.join(format!(
                "{}.{}{}.trials.jsonl",
                site.name, bucket.label, base_tag
            ));

            // Per-bucket candidate list: prune knobs the base dispatch
            // makes inert, then order by distance from what is known to be
            // good, so a budget cut (or an impatient operator) leaves the
            // informative neighborhood measured and the far corners unmeasured
            // — never the other way round.
            let mut configs = site_configs.clone();
            if site.name == "prefill_hints" {
                let base = base_config(&out_dir, &bucket.label);
                let canonical = incumbent_set.first().cloned().unwrap_or_default();
                let before = configs.len();
                configs.retain(|c| !hints_inert_duplicate(c, base.as_ref(), &canonical));
                match &base {
                    Some(b) => println!(
                        "  [{}] base {}: {} of {} hint candidates are live for this dispatch \
                         ({} inert duplicates pruned)",
                        bucket.label,
                        base_dispatch_name(b),
                        configs.len(),
                        before,
                        before - configs.len()
                    ),
                    None => println!(
                        "  [{}] no prefill_attention record yet: tuning all {} hint candidates",
                        bucket.label, before
                    ),
                }
            }
            // Reference points: shipping incumbents plus this bucket's
            // current record winner, if any. Required configs stay first,
            // then incumbents, then everything else by Hamming distance to
            // the nearest reference (stable, so grid order breaks ties).
            let mut references: Vec<BTreeMap<String, i64>> = incumbent_set
                .iter()
                .map(|inc| inc.iter().map(|(k, v)| (k.to_string(), *v)).collect())
                .collect();
            if let Some(winner) = record_winner(&out_dir, site.name, &bucket.label) {
                references.push(winner);
            }
            let required: Vec<String> = args
                .require
                .iter()
                .filter_map(|s| parse_required_config(s).ok().map(|c| c.id))
                .collect();
            configs.sort_by_key(|c| {
                if required.contains(&c.id) {
                    return (0usize, 0usize);
                }
                if incumbent_set.iter().any(|inc| config_matches(c, inc)) {
                    return (1, 0);
                }
                let d = references
                    .iter()
                    .map(|r| config_distance(c, r))
                    .min()
                    .unwrap_or(0);
                (2, d)
            });
            let known = load_existing_trials(&trials_path);
            if !known.is_empty() {
                println!("  [{}] resuming: {} prior trials", bucket.label, known.len());
            }
            let log = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&trials_path)?;

            // Reference output for the correctness gate: default config.
            let mut oracle = EngineOracle {
                configs: configs.clone(),
                engine: Some(match engine_slot.take() {
                    Some(e) => e,
                    None => load_engine(&rt, &model_dir, args.max_seq_len)?,
                }),
                model_dir: model_dir.clone(),
                max_seq_len: args.max_seq_len,
                evict_every: args.evict_every,
                trials_since_evict: 0,
                rt: &rt,
                prompt,
                max_new_tokens: bucket.max_new_tokens,
                decode_objective: bucket.decode_objective,
                reps: args.reps,
                reference_text: None,
                deadline: (args.budget_min > 0)
                    .then(|| Instant::now() + Duration::from_secs(args.budget_min * 60)),
                log,
            };
            let reference = match oracle.step_ms() {
                Err(e) if is_alloc_failure(&e) => match oracle.recover_device_state() {
                    Ok(evicted) => {
                        eprintln!(
                            "  [{}] device allocation failure at the reference step: engine \
                             dropped, {evicted} cached kernel specialization(s) evicted, engine \
                             reloaded — retrying once",
                            bucket.label
                        );
                        oracle.step_ms()?
                    }
                    Err(err) => exit_for_fresh_process(&format!(
                        "at the [{}] reference step (in-process recovery failed: {err:#})",
                        bucket.label
                    )),
                },
                other => other?,
            };
            oracle.reference_text = Some(reference.1);

            println!("  [{}] searching...", bucket.label);
            let trials = GridSearch::new().resume(known).search(&mut oracle);
            engine_slot = oracle.engine;
            let Some(best) = best_config(&configs, &trials) else {
                println!("  [{}] no valid winner", bucket.label);
                continue;
            };
            let best_trial = trials
                .iter()
                .filter(|t| t.config_id == best.id)
                .find_map(|t| match &t.state {
                    TrialState::Measured { median_ms, reps, .. } => Some((*median_ms, *reps)),
                    _ => None,
                })
                .unwrap_or((f32::NAN, 0));
            println!(
                "  [{}] winner: {} ({:.2} ms)",
                bucket.label, best.id, best_trial.0
            );
            entries.push(RecordEntry {
                bucket: bucket.label.clone(),
                config: best.clone(),
                median_ms: best_trial.0,
                samples: best_trial.1,
                l2_key: None,
            });
        }
        if entries.is_empty() {
            continue;
        }
        let ws = Workspace {
            kernel: site.name.to_string(),
            source_hash: grout::kernels::_SOURCE_HASH.to_string(),
            arch: arch.clone(),
            tileiras_fingerprint: tileiras.clone(),
            space_hash: Some(space_hash(&site_configs)),
        };
        let mut record = Record::new(&ws);
        record.entries = entries;
        record.gate = Some("step-success-v1".into());
        let record_path = out_dir.join(format!("{}.json", site.name));
        record.save(&record_path)?;
        println!("saved {}", record_path.display());
        // Records are read at engine construction; drop the engine so the
        // next site runs on top of the winners just saved.
        engine_slot = None;
        grout::model::device_synchronize();
    }
    Ok(())
}
