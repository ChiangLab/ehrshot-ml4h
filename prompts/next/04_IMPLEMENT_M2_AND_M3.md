Prompt 4 — Implement Jeff’s M2 and M3 modeling approaches
Work in temporal-ehr-ml4h. Read AGENTS.md, the locked protocol, decision log, final M1 validation report, and all immutable run manifests.

Do not change the final panel, cohort, serializers, windows, temporal conditions, prediction boundaries, test lock, or M1 results.

Goal: implement the two additional modeling approaches Jeff requested.

M2: single-pass summary embedding
- not recursive RAPTOR;
- take the ordered serialized observation windows;
- use the same content-matched 8,192-token source history retained for M1 unless an explicit
  pre-execution protocol amendment authorizes additional source coverage;
- use one authorized on-premise instruction model;
- deterministic decoding where supported;
- preserve diagnoses, procedures, medications, measurements, values, units, recurrence, trends, and supplied time labels;
- do not repair falsified timestamps;
- do not infer diagnoses, causality, or outcomes;
- cache by source hash, condition, prompt, model revision and decoding configuration;
- embed the resulting summary with frozen Qwen3;
- use the same logistic-regression procedure as M1.

Required M2 fidelity validation:
- validation-only stratified sample of at least 40 histories;
- balanced across four task families and correct/shuffled time;
- concept, numeric-value, unit, recurrence, direction-of-change, and time-label retention;
- unsupported concepts and contradictions;
- whether the summarizer repaired falsified timing;
- compression ratio and generation failure rate;
- no held-out prompt refinement.
- combine the automated audit with manual review of the same frozen 40-history sample; keep the
  packet and decisions private and export only aggregate pass/fail counts and hashes.

M3: per-window embeddings plus sequence model
- frozen Qwen3-Embedding-0.6B per window;
- maximum 512 tokens per window;
- maximum 64 most recent nonempty windows;
- mean-pooled window embeddings plus logistic regression as required baseline;
- one-layer unidirectional GRU;
- optional fixed projection to 256;
- hidden size 128;
- input dropout 0.1;
- AdamW;
- validation-AUROC early stopping, with log loss only as a deterministic tie-breaker;
- three frozen training seeds;
- average probabilities across seeds for final contrasts.
- prepare and verify seed-independent window sources/embeddings once per task and stage; seed jobs
  may reuse only that immutable cache and must not reload or re-encode it.

First run M2 and M3 under complete correctly timed history. Apply the prespecified approach-selection rule:
- select an approach only if macro-AUROC exceeds alternatives by at least 0.005;
- otherwise prefer M1, then M2, then M3;
- use log loss only as the second tie-breaker.

After the reference cells pass, implement the temporal conditions needed for within-approach
sensitivity. The selected approach must have latest-window and all three C3 seeds for the two
co-primary held-out contrasts. C0 and C2 remain prespecified secondary sensitivities. Preserve M1
as the required full temporal matrix even if scope limits the M2/M3 expansion.

Create:
- notebooks/08_m2_summary_validation.ipynb
- notebooks/09_m3_sequence_validation.ipynb
- scripts/slurm/24_m2_summary_task_array.sbatch
- scripts/slurm/25_m2_fit_validation.sbatch
- scripts/slurm/26_m3_task_seed_array.sbatch
- scripts/slurm/27_m3_fit_validation.sbatch
- scripts/slurm/28_freeze_protocol.sbatch

All governed summarization occurs on-premise. No record text may be sent to OpenAI, Anthropic, or another external API.

Add comprehensive tests for summary caching/fidelity, falsified-time preservation, sequence masking/padding, seed handling, patient alignment, no-test access, resume behavior, privacy, and batch sentinels.

Do not evaluate the test set. Finish by producing a cryptographically hashed frozen experiment manifest and a clear statement of which approach and contrasts are authorized for the one-time held-out run.
