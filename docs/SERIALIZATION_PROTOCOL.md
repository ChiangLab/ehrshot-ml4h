# Protocol-aligned longitudinal serialization

`T` is the task instance's prediction timestamp. Every source history is censored with
`event_time < T` before serialization. Primary blocks are prediction-relative intervals
`[T-(i+1)w, T-iw)`, where `i=0` is the most recent window. In elapsed-time form these are
`(i*w, (i+1)*w]`, so an event exactly 30 days before `T` belongs only to the 1–30-day window,
while an event exactly at `T` is excluded.

The configured window is 30 days for laboratory-result, imaging-finding, and operational-outcome
tasks and 90 days for incident-diagnosis tasks. Task family—not the task name—selects the width.
Within a window, full timestamp and the remaining structured fields provide a deterministic tie
break. This ordering does not claim clinical precedence among truly concurrent events. Calendar-day
blocks remain available only as a QC view.

Before block construction, an event is considered an exact duplicate only when all seven available
EHRSHOT fields match: `subject_id`, `time`, `code`, `numeric_value`, `text_value`, `unit`, and
`omop_table`. This is the narrowest reproducible row-identity rule supported by the event schema.
It removes repeated physical rows while retaining recurrence whenever a code reappears at another
time or with any different structured field.

Compact labels encode ranges without calendar dates or patient timestamps, for example
`[TIME=1_30_DAYS_BEFORE_T]`. C0 omits elapsed-time labels; C1 uses correct labels; C2 reverses the
earlier inter-window gap sequence while preserving chronology, the final label, gap multiset, and
history span; C3 permutes earlier labels under each of three configured seeds while keeping the
final label correct. An impossible or no-op perturbation is explicitly ineligible.

The primary packing policy finds one most-recent event suffix that fits the token budget in every
eligible condition, then uses exactly that content across C0–C3. Independent per-condition packing
is an operational sensitivity analysis and reports evidence displaced by temporal tokens. The
separate `latest_window_only` and `latest_code_occurrence` baselines reduce information and are not
rows of the C0–C3 causal contrast.

Final M1 uses an 8,192-token budget. Empty pre-`T` histories remain valid for instruction-only C0
and C1 inputs but are explicitly unavailable for information-reduction or perturbation conditions.
C2 and C3 alter displayed label indices only: clinical event arrays and their order never move.
Each immutable model/condition cache has a patient-window metadata table with exactly one row per
embedding row and records input, model, tokenizer, configuration, and vector hashes. Pilot cache
keys and paths are categorically ineligible for the final namespace.

Patient-level manifests contain hashes only and remain under `outputs/private/`. Aggregate review
reports remain under `reports/generated/`. Neither location is tracked by Git.

Prompt 4 M2 and M3 consume this locked serialization; they do not create a new censoring,
windowing, deduplication, or rendering rule. M2 receives one ordered serialized representation and
must preserve supplied compact labels exactly, including falsified C3 order. M3 reconstructs the
complete blocks from the immutable final cohort and event source using these same functions,
cross-checks every block against the Prompt 3 patient-window index hashes, retains chronological
order, keeps at most the 64 most recent nonempty blocks, and applies its independent 512-token
encoder limit per block. Neither approach may rearrange clinical event arrays or repair labels.
