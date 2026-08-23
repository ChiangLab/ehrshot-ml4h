# Current scientific protocol

Status date: 2026-08-22. This is a concise operational index; the final August 14 protocol and
`PROTOCOL_DEVIATIONS.md` remain authoritative for detail and unresolved differences.

## Cohort and panel

- The existing governed Phase 1 cohort is a **PILOT** with `lab_anemia`, `new_acutemi`, and
  `guo_los`.
- The final four-family panel is prospectively **LOCKED** as `lab_anemia`, `guo_readmission`,
  `new_hypertension`, and `chexpert`. It was selected before pilot embedding results and informed
  by Jeff's recommendation for clinical-family breadth; this does not claim his approval of the
  exact lock wording.
- The complete three-task engineering pilot is separately authorized for Qwen3-Embedding-0.6B,
  4,096 tokens, M1 direct serialization, B1-B4, and train/validation only. Pilot artifacts cannot
  satisfy final-panel gates.

## Time, windows, and evidence identity

- `T` is each task instance's official prediction time. The current conservative project rule is
  strict censoring: retain only events with `event_time < T`. Whether any task should instead use a
  benchmark-specific exact-boundary convention remains DEV-002; future events are never allowed.
- Windows are prediction-relative and half-open: `[T-(i+1)w, T-iw)`. They are 30 days for
  laboratory, imaging, and operational tasks and 90 days for incident-diagnosis tasks.
- Remove only seven-field exact duplicates across `subject_id`, `time`, `code`, `numeric_value`,
  `text_value`, `unit`, and `omop_table`. Preserve recurrence at different times or with any
  differing structured value.
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
- M2: one authorized local, non-recursive longitudinal summary, frozen embedding, then logistic
  regression. It is RAPTOR-inspired compression, not a RAPTOR hierarchy.
- M3: frozen per-window embeddings followed by a small unidirectional GRU; mean pooling plus
  logistic regression is the required capacity comparator.
- Development and selection use training/validation only. The held-out test split is prohibited
  until panel, serializers, prompts, hyperparameters, models, seeds, statistics, and an immutable
  experiment manifest are frozen. Test results may be evaluated once and may not change design.

## Governance boundary

Raw data, cohort rows, subject identifiers, prediction timestamps, serialized histories, private
manifests, labels, predictions, summaries, embeddings, and model artifacts stay under governed,
Git-ignored DGX storage. Only source code, synthetic fixtures, hashes, environment summaries, job
sentinels, and reviewed privacy-safe aggregate reports may be used in an off-VPN Codex handoff.
