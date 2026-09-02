# Current scientific protocol

Status date: 2026-09-01. This is a concise operational index; the final August 14 protocol and
`PROTOCOL_DEVIATIONS.md` remain authoritative for detail and unresolved differences.

## Cohort and panel

- The existing governed Phase 1 cohort is a **PILOT** with `lab_anemia`, `new_acutemi`, and
  `guo_los`.
- The final four-family panel is prospectively **LOCKED** as `lab_anemia`, `guo_readmission`,
  `new_hypertension`, and `chexpert`. It was selected before pilot embedding results and informed
  by Jeff's recommendation for clinical-family breadth; this does not claim his approval of the
  exact lock wording.
- The complete three-task engineering pilot passed for Qwen3-Embedding-0.6B, 4,096 tokens, M1
  direct serialization, B1-B4, and train/validation only. Pilot artifacts cannot satisfy
  final-panel gates and pilot performance cannot alter the prespecified final panel.
- Prompt 3's versioned `locked_four_task_m1_v1` foundation, runtime-only feasibility gate, four
  task-level embedding caches, final validation readout, and aggregate notebook are complete.
  Fit-only recovery job 451 validated exact upstream provenance while preserving failed job 446.
  Prompt 4 M2/M3 source and local synthetic validation are complete; governed reference execution
  is the next gate. Held-out evaluation remains locked.

## Time, windows, and evidence identity

- `T` is each task instance's official prediction time. The current conservative project rule is
  strict censoring: retain only events with `event_time < T`. Whether any task should instead use a
  benchmark-specific exact-boundary convention remains DEV-002; future events are never allowed.
- Windows are prediction-relative and half-open: `[T-(i+1)w, T-iw)`. They are 30 days for
  laboratory, imaging, and operational tasks and 90 days for incident-diagnosis tasks.
- Remove only seven-field exact duplicates across `subject_id`, `time`, `code`, `numeric_value`,
  `text_value`, `unit`, and `omop_table`. Preserve recurrence at different times or with any
  differing structured value.
- Select at most one official label instance per patient and task by sorting eligible official
  prediction times and mapping a SHA-256 digest of dataset/release version, task, official split,
  patient, frozen seed, and the sorted eligible prediction-time-index domain to one index. The
  rule is independent of outcome, history, record length, and model results.
- Before ranking, exclude every patient/task/prediction-time group represented by multiple official
  label rows, regardless of whether their label values agree. This prevents outcome- or row-order-
  dependent adjudication; patients with no remaining singleton time are excluded for that task.
- The primary C0-C3 comparison uses identical retained clinical events and order:
  - C0 order only: no elapsed-time labels;
  - C1 correct compact prediction-relative labels;
  - C2 reversed earlier inter-window gaps, monotonic reconstruction, preserved gap multiset/span;
  - C3 shuffled earlier time labels under seeds 20260815, 20260816, and 20260817.
- The most recent/final window keeps its correct label in C1, C2, and C3. An impossible or
  textually inert perturbation is ineligible, never fabricated.
- `latest_window_only` and `latest_code_occurrence` are separate information-reduction baselines,
  not rows of the C0-C3 temporal stress test.

## Models and evaluation boundary

- M1: one frozen embedding of the flat complete chronological document, then logistic regression.
- Frozen Qwen inference uses BF16 where configured, but finite raw vectors are promoted and
  explicitly L2-normalized in FP32 before caching; see DEC-011 and DEV-016.
- Final M1 uses Qwen3-Embedding-0.6B at 8,192 tokens. A separate runtime/memory-only throughput
  gate benchmarks the complete pinned 8B model and admits only a feasible prefix of C1, C3,
  latest-window, and latest-code priority, using a 2x time contingency and no outcome metric.
- The DGX site limit is 12 hours. Verified job 444 retained the full 2x measured contingency and a
  five-minute execution reserve; this engineering allocation rule changes no scientific input,
  model, condition, or evaluation definition.
- M2: one authorized local, non-recursive longitudinal summary, frozen embedding, then logistic
  regression. It is RAPTOR-inspired compression, not a RAPTOR hierarchy. Summary generation must
  be deterministic where supported, preserve supplied falsified time labels rather than repair
  them, and remain entirely on the authorized on-premise endpoint. Its primary source is the same
  content-matched 8,192-token retained history as M1. Eligibility requires both automated fidelity
  checks and manual review of the frozen 40-history validation sample.
- M3: frozen per-window embeddings followed by a small unidirectional GRU; mean pooling plus
  logistic regression is the required capacity comparator. The locked architecture uses at most 64
  recent nonempty windows, 512 tokens per window, hidden size 128, dropout 0.1, AdamW, and three
  frozen seeds. Early stopping maximizes validation AUROC, with validation log loss only resolving
  an AUROC tie.
- M3's immutable source and embedding caches are prepared once per task/stage before seed training.
  Projection is applied to retained windows before right-padding, which is mathematically
  equivalent to projecting the zero-padded tensor while avoiding unnecessary padded 1,024-D
  storage. These are execution-only optimizations; the three seeds and every condition remain.
- Whichever approach is selected must have complete-correct versus latest-window and
  complete-correct versus all three fixed C3 seeds available before the held-out run. C0 and C2 are
  secondary temporal sensitivities; M1 retains the complete temporal matrix regardless of selection.
- Development and selection use training/validation only. The held-out test split is prohibited
  until panel, serializers, prompts, hyperparameters, models, seeds, statistics, and an immutable
  experiment manifest are frozen. Test results may be evaluated once and may not change design.

## Governance boundary

Raw data, cohort rows, subject identifiers, prediction timestamps, serialized histories, private
manifests, labels, predictions, summaries, embeddings, and model artifacts stay under governed,
Git-ignored DGX storage. Only source code, synthetic fixtures, hashes, environment summaries, job
sentinels, and reviewed privacy-safe aggregate reports may be used in an off-VPN Codex handoff.
