#!/usr/bin/env bash
# Restart-loop driver for grout_autotune — the documented last resort.
#
# The tuner recovers from device-state exhaustion in-process: on an
# allocation failure it drops the engine, synchronizes the device, evicts
# every cached kernel specialization (cutile::tile_kernel::clear_kernel_cache,
# cutile-rs >= 0.3.1) and reloads; `--evict-every N` does the same
# proactively every N trials. This wrapper only matters when that reload
# itself fails (the tuner exits 3 without logging the candidate): trials are
# journaled and the search resumes, so the driver is simply rerun until a
# clean exit. Each restart loses at most the in-flight trial.
#
# The tuner is behind the `benchmarks` cargo feature; a plain
# `cargo build --release` does not produce it (and leaves any stale copy in
# place), so build it here.
set -u
MAX_RESTARTS="${MAX_RESTARTS:-25}"
cargo build --release --features benchmarks --bin grout_autotune || exit 1
for i in $(seq 1 "$MAX_RESTARTS"); do
  target/release/grout_autotune "$@"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "autotune_loop: clean exit after $i run(s)"
    exit 0
  fi
  echo "autotune_loop: run $i exited rc=$rc — restarting (resumes from trial logs)"
done
echo "autotune_loop: exceeded MAX_RESTARTS=$MAX_RESTARTS" >&2
exit 1
