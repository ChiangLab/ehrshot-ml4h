# Current scientific protocol

M2 amendment (2026-09-22; DEC-035/DEV-043): after job 980 failed, William authorized
`m2_ordered_structured_v1`, a new single-pass, LLM-generated structured-summary interface.
Clinical evidence is still exactly M1's retained source; task definitions, strict pre-T censoring,
30/90-day windows, final-window correctness, and C1/C3 label-only perturbations do not change.
The source groups each window by code/source/unit, preserving the ordered value/text sequence
and exact run counts, including returns to earlier values. Grouping does NOT retain global
cross-code event interleaving; do not call it globally lossless. No individual retained series
may be changed, reordered, partially shortened, or moved to another window. Whole-series omission
is allowed only under the retention floors below. The model emits facts, not IDs; rendering uses
only validated returned facts and never substitutes a deterministic source summary.

The unchanged numeric fidelity floors (90% concepts, 95% numbers/text/units, 80% recurrence/trends)
now use window/code/source/unit-specific denominators, not pooled lexical overlap. Recurrence is
occurrence-weighted; directional-series retention derives endpoint change and monotonicity from
the full ordered numeric runs. It is not a clinical-trend interpretation. Missing numeric values
make that trajectory unknown. Exact time-label sequence is mandatory. This clarified accounting,
JSON schema, prompt, ordered source and rendering are a versioned protocol amendment.

Keep `nvidia/Qwen3.6-35B-A3B-NVFP4`, temperature 0, top-p 1, fixed seed, thinking off, and the
Qwen3-Embedding-0.6B encoder. A separate 16,384-token JSON generation ceiling does not expand the
8,192-token downstream embedding cap or the clinical input. No truncation/fallback is permitted.
Two frozen 32-input training-only engineering samples precede production: development, then
patient-disjoint confirmation also excluding job-980 people. All inputs must pass. The fixed
40-history validation manual review remains separate and mandatory; it must not tune this prompt.
No automatic scientific resumption is implemented until the new engineering evidence is reviewed.
This post-M1 exploratory extension cannot reopen held-out evaluation or establish Markovianity.

Status date: 2026-09-08. This is a concise operational index; the final August 14 protocol and
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
  Prompt 4 M3 reference execution and aggregation are complete. M2 was not completed after repeated
  fail-closed fidelity/runtime attempts. Under the authorized 2026-09-07 scope amendment, M1 became
  the sole primary held-out approach, M3 remained validation-only secondary evidence, and M2 became
  future work. The M1-only manifest was frozen at commit `7315451`; recovery commit `e9f8f59`
  retained that scientific source and completed preparation jobs `583_0`-`583_3`, embedding jobs
  `585_0`-`585_3`, fit/report job 586, and publication job 587. The one-time held-out evaluation is
  complete, and its protocol may not be changed in response to the observed result.

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
- M2 summaries must use the versioned compact structured format: exact supplied codes only, no
  code-name expansion or headings, grouped recurrence counts, distinct supplied numeric values,
  exact units, and unaltered time-label order. Only responses with API `finish_reason=stop` are
  cache-eligible. The structured audit recognizes the plain `LOINC/...`, `SNOMED/...`, and similar
  code form actually emitted by the locked serializer and does not count digits inside codes or
  time labels as clinical numeric values.
- Before M2 inference, Python deterministically groups rendered facts within each unchanged
  prediction-relative window by clinical code, OMOP source, and unit. It preserves each distinct
  numeric and text value in first-occurrence order and records the exact total occurrence count.
  This preserves the represented structured evidence and window order while preventing the
  instruction model from doing unreliable aggregation or exhausting its output on repetitions.
  Prompt, transform, and cache schemas are versioned together. A summary is eligible only with
  `finish_reason=stop` and an offline tokenizer check showing that it fits the frozen 8,192-token
  downstream embedding context.
- The post-529 recovery preserves the primary 4,096-token decoding/cache identity and admits one
  prior row only when its assistant text exactly equals the reconstructed grouped source and fits
  the downstream tokenizer. Missing or rejected rows use a distinct 8,192-token completion-budget
  identity. Eight deterministic shards per task and a two-shard cap change scheduling only; they do
  not change source evidence, prompt, model, condition, split, or evaluation. If validation selects
  M2, the already-authorized M2 sensitivity conditions use the same bounded shard policy.
- Fidelity-v5 job 545 demonstrated that exact-copy output can exceed the frozen downstream context
  and therefore cannot implement bounded summarization. Fidelity-v6 restores a genuine structured
  compression step under one new 6,144-token generation/cache identity. It requires exact supplied
  codes, values, units, recurrence, and time-label order; rejects unsupported or contradictory
  facts or movement of facts between time blocks; applies the prespecified retention floors per row;
  and retains the unchanged 8,192-token
  embedding gate. Sixteen shards per task with four active and two requests each keep the same
  maximum eight endpoint requests while reducing timeout exposure. Legacy caches remain preserved
  but cannot satisfy the new prompt/cache identity.
- Fidelity-v6 job 559 demonstrated that eight total endpoint requests were too slow for its
  16-shard-per-task execution envelope. Fidelity-v7 keeps the exact prompt, decoding, model,
  source, row validator, and cache identity; it uses an outcome-free synthetic 8/16/24/32 request
  ladder to select operational concurrency and 192 shards per task to bound each resumable unit.
  These are scheduling controls only and cannot be selected from validation performance.
- M3: frozen per-window embeddings followed by a small unidirectional GRU; mean pooling plus
  logistic regression is the required capacity comparator. The locked architecture uses at most 64
  recent nonempty windows, 512 tokens per window, hidden size 128, dropout 0.1, AdamW, and three
  frozen seeds. Early stopping maximizes validation AUROC, with validation log loss only resolving
  an AUROC tie.
- M3's immutable source and embedding caches are prepared once per task/stage before seed training.
  Projection is applied to retained windows before right-padding, which is mathematically
  equivalent to projecting the zero-padded tensor while avoiding unnecessary padded 1,024-D
  storage. These are execution-only optimizations; the three seeds and every condition remain.
- M1 is the prospectively selected held-out approach. Its held-out matrix includes C0, C1, C2,
  all three fixed C3 seeds, latest-window, latest-code, count, and prevalence baselines. The two
  co-primary contrasts are C1 minus latest-window and C1 minus the mean prediction across the three
  C3 seeds. M3 remains validation-only; M2 remains incomplete future work.
- Development and selection used training/validation only. The held-out manifest was frozen before
  access, its audit records deterministic cohort construction, and the held-out evaluation ran
  exactly once. The aggregate report records no post-access protocol modification and no held-out
  performance in job logs. No rerun or design change is permitted.

## Governance boundary

Raw data, cohort rows, subject identifiers, prediction timestamps, serialized histories, private
manifests, labels, predictions, summaries, embeddings, and model artifacts stay under governed,
Git-ignored DGX storage. Only source code, synthetic fixtures, hashes, environment summaries, job
sentinels, and reviewed privacy-safe aggregate reports may be used in an off-VPN Codex handoff.
