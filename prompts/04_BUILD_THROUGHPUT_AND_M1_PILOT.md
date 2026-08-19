# Codex prompt: build the throughput benchmark and direct-serialization M1 pilot

Work in the current `temporal-ehr-ml4h` repository. Read the locked protocol/configuration, all
Phase 1 and Phase 3 aggregate reports, and the preceding prompts before editing.

## Hard preconditions

Stop and report the missing item if:

- the experiment configuration is not `LOCKED`;
- the Phase 1 protocol-review gate is not passing;
- the Phase 3 baseline job lacks a verified `COMPLETE.json` sentinel;
- any baseline has incomplete or mismatched hashes;
- the Qwen3-Embedding-0.6B local artifact is absent; or
- any code path would inspect test performance.

## Goal

Build two DGX-ready notebooks:

1. `notebooks/05_throughput_benchmark.ipynb`
2. `notebooks/06_m1_correct_time_pilot.ipynb`

Also create reusable modules, tests, and separate Slurm runners:

- `scripts/slurm/13_run_throughput_benchmark.sbatch`
- `scripts/slurm/14_run_m1_correct_time_pilot.sbatch`

Do not run the full C0-C3 matrix, summarization model, sequence model, 8B confirmatory experiment,
or test evaluation in this task.

## Notebook 05: throughput benchmark

Benchmark feasibility without using outcome performance to choose the scientific design.

Required benchmark cells:

- Qwen3-Embedding-0.6B at 4,096 tokens;
- Qwen3-Embedding-0.6B at 8,192 tokens;
- Qwen3-Embedding-8B at 8,192 tokens only if an already-downloaded, pinned local artifact exists;
- per-window Qwen3-Embedding-0.6B at the prespecified per-window budget used by the future sequence
  model.

Use privacy-safe short, median, and near-limit inputs derived through governed aggregate sampling or
fully synthetic fixtures. Never print real text. Record warm-up policy, batch sizes, precision,
device, model/tokenizer revision, tokens/second, documents/second, peak GPU memory, failures, and
estimated full-run wall time with a 2x contingency. Do not select tasks, windows, or temporal
conditions based on outcome metrics.

Write a deterministic scope recommendation with explicit rules:

- retain the 0.6B 8K primary only if it fits the deadline with contingency;
- retain 8B only as a limited confirmation if its estimated total time fits the prespecified budget;
- otherwise preserve the scientific matrix and reduce model scale, not task/test integrity.

## Notebook 06: M1 correct-time pilot

Run only the direct flat-serialization model M1 with `C1_correct_compact_time`, at the token budget
selected by the throughput gate, for train and validation patients in the locked panel.

- Use `content_matched_primary` packing even though only C1 is embedded now; generate and persist the
  retained-content manifest needed to compare C0-C3 later.
- Freeze and normalize the encoder and use the exact Phase 3 logistic-readout procedure.
- Cache embeddings resumably under immutable hashes.
- Compare validation results descriptively and with paired patient bootstrap intervals against the
  latest-window, latest-code, count, and prevalence references.
- Treat this as a pipeline and signal sanity check, not the final hypothesis test.
- Do not modify the task panel, window sizes, label definitions, token budget, instructions, or
  regularization grid in response to isolated task results without recording a protocol amendment.

Report aggregate validation metrics, token/retention summaries, throughput, calibration, and failure
counts. Store all patient-level artifacts privately. Never display real histories.

## Engineering requirements

- Both Slurm jobs must use job-specific `RUNNING.json`, `COMPLETE.json`, and `FAILED.json` sentinels.
- Include writable private cache/config directories and no external downloads.
- Fail closed on partial caches, configuration drift, uncommitted-code policy if required by locked
  config, and missing model revisions.
- Record Git commit, `pip check`, environment audit, CUDA/model revisions, seeds, and all artifact
  hashes in private manifests.
- Make it possible to resume embedding batches without duplicating or misaligning rows.

## Testing and validation

Add tests for throughput accounting, scope-decision rules, resumable caches, content-manifest reuse,
row alignment, no-test access, and privacy-safe reports. Run Ruff, the full test suite, shell syntax
checks for both Slurm scripts, and the pre-push safety gate.

Do not commit or push. Finish with exact DGX commands, expected run time based only on the benchmark,
files to inspect, and the next prompt that would be required to run the complete M1 C0-C3 matrix.
