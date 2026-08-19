# Codex prompt: build the locked-panel Phase 3 baselines

Work in the current `temporal-ehr-ml4h` repository. Read the repository, the Phase 1 reports, and
`prompts/02_AUDIT_PHASE1_AND_ALIGN_PROTOCOL.md` before editing.

## Hard preconditions

Stop without implementing or running models if any condition is false:

- `configs/experiments/locked_panel.yaml` exists and says `status: LOCKED`.
- The locked task panel, task-family window sizes, censoring rule, selection seed, temporal seeds,
  tokenizer/model revisions, and train/validation/test governance are explicit.
- The governed cohort exists and its private manifest matches the locked panel. Never silently
  reuse a cohort generated for a different panel.
- Phase 1 alignment tests pass.
- The aggregate Phase 1 review says content-matched packing and required temporal invariants pass.
- The test split remains locked for model selection.

If a precondition fails, report exactly what is missing and stop. Do not create a fake `LOCKED`
configuration and do not decide the task panel yourself.

## Goal

Build and validate the simple references that must precede the main temporal experiment. Use
Qwen3-Embedding-0.6B for embedding baselines unless the locked configuration explicitly names a
different pinned checkpoint. Build code and tests locally; run model forward passes only on the DGX.

Create:

- `notebooks/04_phase3_baselines.ipynb`
- reusable baseline and embedding modules under `src/temporal_ehr/`
- `scripts/slurm/12_run_phase3_baselines.sbatch`
- focused synthetic tests
- a privacy-safe aggregate report under `reports/generated/phase3_baselines/`

## Required baselines

Run the following separately for each locked task:

1. **B4 prevalence-only reference.** Fit the training prevalence and apply the constant probability
   to validation.
2. **B3 count-based reference.** Reuse or rerun the governed count baseline with the locked panel.
3. **B1 latest-window Qwen reference.** Serialize the complete most recent task-family observation
   window, embed it, and fit the frozen-embedding logistic readout.
4. **B2 latest-code Qwen reference.** Retain only the latest occurrence of each code, serialize,
   embed, and use the same readout procedure.
5. **Optional laboratory sanity reference.** For `lab_anemia` only, include the most recent eligible
   hemoglobin value/reference information if and only if it can be identified without leakage and
   the rule is prespecified and tested. Otherwise document why it was not implemented.

The latest-window and latest-code baselines answer different questions. Never merge or relabel them.

## Embedding and model controls

- Use the exact checkpoint and local revision from locked configuration; no external downloads in
  a Slurm job.
- Use the exact task-specific embedding instruction from configuration and hash it into the
  manifest.
- Use the official Qwen tokenizer from the same pinned artifact.
- Start with a 4,096-token pilot.
- Use full embedding dimensionality and L2 normalization unless the locked protocol says otherwise.
- Freeze the encoder. Do not fine-tune it.
- Fit one logistic-regression readout per task using the same prespecified regularization grid and
  deterministic tie rule as the count reference.
- Tune on validation only. Do not query test performance.
- Use canonical patient splits and exactly one governed prediction instance per patient/task where
  the locked cohort requires it.
- Assert patient/task/instance/label alignment before joining embeddings to outcomes.
- Use immutable cache keys containing data, cohort, renderer, tokenizer, model, instruction, token
  budget, and Git revision hashes.
- Make embedding generation resumable. A partial cache must never be treated as complete.

## Outcomes and reporting

For train and validation only, report per task:

- sample size and prevalence;
- AUROC and AUPRC;
- Brier score and log loss;
- calibration intercept/slope or a documented fallback if estimation is unstable;
- the selected regularization value and all validation candidates;
- token-length, truncation, and retained-event summaries; and
- inference throughput and peak memory.

Use patient-clustered bootstrap confidence intervals where repeated instances could exist; otherwise
state that the governed cohort contains one instance per patient and use a paired patient-level
bootstrap for model contrasts. Generate paired validation contrasts between B1, B2, B3, and B4.
Do not perform large unplanned families of hypothesis tests.

Store all instance-level texts, hashes, embeddings, labels, predictions, and models under
`outputs/private/`. Aggregate report cells smaller than 20 must be suppressed. The notebook must
show only aggregate results and fully synthetic text examples.

## Tests and validation

Test:

- hard failure on a non-`LOCKED` configuration;
- hard failure on cohort/config/hash mismatch;
- no test split in fit, selection, or reports;
- deterministic renderer/tokenization/cache keys;
- latest-window versus latest-code semantics;
- row and label alignment;
- interrupted/resumed embedding generation;
- train-only fitting of vocabulary/model parameters;
- validation metric correctness on synthetic arrays; and
- clean source notebook/output-path/privacy rules.

Run:

```bash
python -m ruff check .
python -m pytest
bash -n scripts/slurm/12_run_phase3_baselines.sbatch
bash scripts/git/pre_push_safety.sh
```

Do not commit or push. Do not implement M1 C0-C3, summarization, a GRU, an 8B confirmation, or test
evaluation in this task.

Finish with the exact DGX submission command, expected private and aggregate outputs, verification
commands, and an explicit list of work intentionally deferred.
