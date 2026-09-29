# Decision log

- 2026-09-29: Use a private DuckDB index plus localhost browser explorer for presentation review.
  The original event parquet and full embedding arrays remain immutable sources; the index covers
  only train/validation selected patients and strictly pre-T events. The held-out submission is
  excluded, arbitrary SQL/download is unavailable, and no M4 is inferred because none is defined.

2026-09-25 — DEC-038: after verified structured-v2 engineering 988/989/990 and William's request
to proceed, implement a separate reference production path. Keep source/prompt/model/decoding/
renderer/fidelity fixed; use at most 16 unique inputs per shard and four total requests. Preserve
raw responses and a lifetime two-attempt budget; reuse admitted rows after semantic/hash checks;
halt on rejected inputs rather than dropping patients. The frozen manual fidelity review gates
validation logistic regression. Reference completion does not authorize the old approach-selection
or one-time test chain; subsequent sensitivity must be explicitly validation-only and prospectively
documented. No M1/M3/test rerun or encoder replacement. Status: local implementation, DGX pending.

2026-09-25 — DEC-037: William authorized the versioned repair after 986's diagnosed fidelity
failures. Preserve strict validation; do not deduplicate, reorder, fill, or repair LLM answers.
Strengthen copying instructions and canonicalize JSON object-key serialization (not arrays),
version the complete request identity, and isolate outputs. Freeze 32 new training confirmation
inputs before generation, excluding all prior engineering people; regress all 64 inspected inputs.
Automate regression -> confirmation -> readiness with afterok plus recursive semantic/hash checks.
Do not equate readiness with production, manual fidelity review, model fitting or scientific M2
completion. Production integration remains a separate required step. No old caches/sentinels are
overwritten, and no held-out/M1/M3 computation is authorized. Real v2 effectiveness is pending.

2026-09-24 — DEC-036: recover job 982 with a versioned reversible downstream renderer. Its
saved latest responses passed every factual gate in a user-run hash-verified diagnostic; two
rendered outputs exceeded 8,192 by 56/73 tokens. Reuse the fixed last saved response per input,
retain generation source/prompt/schema/decoding/model identity, preserve 982 FAILED, and record
original generation and recovery commits separately. Short field names with an explicit legend
and canonical escaping preserve every returned fact; decoding must recover the identical JSON.
Token-check the entire new text, and version embedding keys. Replay is followed by the originally
frozen independent confirmation, never a newly selected easier sample. Real fit remains pending;
no guarantee is inferred from character savings or local synthetic tests. No full M2, M1/M3 or
test computation is submitted by this implementation.

2026-09-22 — DEC-035: William authorized versioned single-pass structured M2 summaries that
preserve ordered measurements, retaining the clinical source, Qwen instruction model, 0.6B encoder
and validation-only boundary. Preserve the failed job-980 evidence and all prior versions. Use a
new ordered run representation, strict JSON facts, exact retained-series matching, keyed fidelity,
and a 16,384-token JSON generation ceiling with unchanged 8,192-token encoder cap. This explicitly
amends representation/prompt/validation accounting; it is not a runtime-only fix or a fact-ID
extractive substitute. Development and independently held-out engineering samples are training-
only; confirmation excludes all development and 980 people. Neither performance nor outcomes
select these samples. Two gated engineering jobs are implemented; production integration follows
only reviewed real evidence, with the original 40-history manual review still required. No new
M1/M3/test computation, local real-model execution, commit, push or remote submission is performed.
Status: local implementation; real DGX development/confirmation pending. DEC-034's job-980 canary
ran and failed, so its full-resumption gate was not met.

2026-09-16 — DEC-034: repair the demonstrable source/output singleton grammar mismatch before
changing the M2 scientific interface. Accept `occurrences=1` only under all existing source
matching and retention gates; retain prompt, decoding, encoder and cache identity. Require a
32-input train-only C1/C3 engineering canary with private rejected responses and hashed original
job evidence before any full job-596 recovery. The 333 recurrence-classified failed logs are not
proof of model counting errors. No fact-ID/extractive redesign or held-out reopening is approved.
Status update 2026-09-22: repair ran in job 980, which FAILED (24/32 final PASS). The full
resumption gate was not met; DEC-035 supersedes the next engineering action.

Status date: 2026-09-16. Open scientific deviations are tracked in greater detail in
`PROTOCOL_DEVIATIONS.md`.

| ID | Decision | Status | Evidence or next gate |
|---|---|---|---|
| DEC-001 | Keep the existing three-task panel labeled PILOT and prospectively lock the separate four-task breadth panel (`lab_anemia`, `guo_readmission`, `new_hypertension`, `chexpert`). | LOCKED 2026-08-21 | Selected before pilot embeddings, informed by Jeff's clinical-family breadth recommendation; no claim of his approval of this wording. |
| DEC-002 | Use each task's official prediction time `T` and strict `event_time < T` as the current conservative project censoring rule. | ACTIVE; EXCEPTION DECISION OPEN | DEV-002 records the possible task-specific benchmark-boundary exception. Future events are always prohibited. |
| DEC-003 | Use exact prediction-relative half-open 30-day windows for lab/imaging/operational tasks and 90-day windows for incident diagnoses. | VERIFIED | Job 408 recovered and validated job-407 aggregate evidence. |
| DEC-004 | Remove only exact seven-field duplicate rows and retain clinical recurrence. | VERIFIED | Job 408 recovered and validated job-407 aggregate evidence. |
| DEC-005 | C0-C3 share retained content; C1-C3 keep the final window correct; C2 reverses earlier gaps and C3 shuffles earlier labels under three fixed seeds. | VERIFIED | Job 408 recovered and validated content matching and Notebook 03. |
| DEC-006 | Keep latest-window and latest-code as separate baselines rather than crossing either with C0-C3. | ACTIVE | Required before temporal model comparisons. |
| DEC-007 | Compare M1 flat embedding, M2 one-pass summary embedding, and M3 per-window embedding plus GRU as operational systems. | ACTIVE; M1 COMPLETE | M2 must use only an authorized local model; no governed history may reach an external API. |
| DEC-008 | Do not access held-out test performance until the panel and full experiment manifest are frozen. | ACTIVE CONTROL | Train/validation-only development continues. |
| DEC-009 | Recover job 407 instead of rerunning Notebook 03. Preserve its FAILED sentinel permanently. | COMPLETE | Job 408 wrote COMPLETE with exit status 0 and preserved the source failure hash. |
| DEC-010 | Authorize the complete three-task engineering pilot without allowing it to satisfy the final-panel gate. | COMPLETE | Jobs 415-418 and the semantic report/notebook validators passed; the aggregate records `PILOT_COMPLETE_FINAL_PANEL_UNCHANGED` and no test use. |
| DEC-011 | Retain BF16 Qwen inference but normalize the finite raw embedding vectors explicitly in FP32. | VERIFIED | The corrected synthetic smoke gate produced finite 1,024-dimensional embeddings with norms within approximately 2e-9 of one before the completed pilot chain ran. |
| DEC-012 | Build the final cohort in a new immutable namespace with the exact sorted eligible prediction-time-index selection hash, independent of outcome, history, record length, and model results. | VERIFIED | Foundation job 437 and its semantic/notebook validators passed. |
| DEC-013 | Admit 8B confirmatory cells only as the longest feasible prefix of C1, C3, latest-window, and latest-code under runtime/memory evidence and a 2x time contingency. | VERIFIED | Runtime-only throughput job 444 passed; no outcome performance informed feasibility. |
| DEC-014 | Exclude every official patient/task/prediction-time group containing multiple label rows before deterministic final-cohort ranking, regardless of label agreement. | VERIFIED | Foundation job 437 passed the governed outcome-independent duplicate-time accounting and semantic checks. |
| DEC-015 | Retain the required 2x throughput contingency and use a five-minute additional reserve within the 12-hour site limit. | VERIFIED | Job 444 passed with the full 2x contingency and 300-second reserve; no scientific input changed. |
| DEC-016 | Recover fit job 446 by reusing only exact COMPLETE job-437/444/445 evidence and submitting job 23 alone. | COMPLETE | Job 451 at `ef1b0de` wrote COMPLETE/exit 0, preserved 446 FAILED, passed final aggregate/notebook validation, and used no held-out data. |
| DEC-017 | Implement Prompt 4 M2/M3 first under C1, select only by the prespecified validation rule, then authorize fixed within-approach sensitivities before freezing. | SOURCE ALIGNMENT UPDATE; LOCAL REVALIDATION PENDING | M2 is one-pass on-premise summarization plus frozen Qwen/logistic regression; M3 is frozen per-window Qwen plus mean-pool/logistic and a one-layer GRU under three seeds. Reference compute now pauses for the required manual M2 fidelity review before selection. |
| DEC-018 | Preserve the two co-primary held-out contrasts within the validation-selected approach: complete-correct minus latest-window and complete-correct minus three-seed shuffled earlier time. | SOURCE IMPLEMENTED; DGX PENDING | M1 already has these cells. If M2 or M3 is selected, its conditional sensitivity stage must produce B1 and all three C3 seeds before freeze; C0/C2 remain secondary. |
| DEC-019 | Use validation AUROC for M3 early stopping and use log loss only to break an AUROC tie. | SOURCE IMPLEMENTED; DGX PENDING | Restores the metric named in the August 14 protocol before any governed M3 fit. |
| DEC-020 | Define primary M2 as compression of the same content-matched 8,192-token clinical source retained for M1. | PROPOSED FOR JEFF CONFIRMATION; SOURCE IMPLEMENTED | This controls clinical evidence across M1/M2 and protects the deadline; it does not claim that M2 expands history coverage. |
| DEC-021 | Separate seed-independent M3 window preparation from seed training and tune only Slurm concurrency from synthetic capacity evidence. | SOURCE IMPLEMENTED; DGX CAPACITY PENDING | One preparation pass per task/stage feeds all three immutable seeds; projection-before-padding reduces memory, and concurrent GRU jobs do not alter inputs, architecture, seeds, or metrics. |
| DEC-022 | Require request-level non-thinking M2 generation, reject null-like assistant content, and recover by rerunning M2 while reusing only cryptographically validated M3 jobs 470/471. | SOURCE IMPLEMENTED; DGX RECOVERY PENDING | Exact synthetic prompting returned all required fields with thinking disabled; the changed decoding hash prevents job-502 rows from satisfying recovery keys. |
| DEC-023 | Version M2 as compact structured summarization after validation-only inspection exposed repetition-driven output exhaustion and unsupported code expansion. | SOURCE IMPLEMENTED; DGX PREFLIGHT PENDING | Exact codes may not be translated; recurrence is grouped; only `finish_reason=stop` is eligible; token usage is recorded; plain-code fidelity parsing and a dense synthetic live gate precede the rerun. Jobs 507/513 remain superseded evidence, and M3 470/471 is reused unchanged. |
| DEC-024 | Compute exact-duplicate M2 recurrence deterministically before the instruction-model call. | SOURCE IMPLEMENTED; DGX PREFLIGHT PENDING | The `f25080d` synthetic gate submitted no jobs because Qwen intermittently counted 64 identical rows as 65. Collapsing exact duplicates within their unchanged window preserves all represented evidence, reduces tokens, removes unreliable arithmetic from Qwen, and is bound into a new cache schema; three consecutive synthetic passes are required. |
| DEC-025 | Replace instruction-model aggregation with deterministic within-window grouping before the fidelity-v3 M2 call. | SOURCE IMPLEMENTED; LOCAL VALIDATION PENDING | Job 519 showed that exact-repeat-only compaction still caused 12-hour overruns and 549/650 CheXpert length rejections. Group by code/source/unit in Python, preserve ordered distinct numeric/text values and exact counts, increase only the output ceiling to 4,096, and require clean completion plus an 8,192-token downstream fit. M1/M3, clinical evidence, conditions, splits, and test lock do not change. |
| DEC-026 | Recover fidelity-v3 job 529 with exact cache validation, a separate 8,192-token overflow identity, and deterministic bounded shards. | SOURCE IMPLEMENTED; LOCAL/DGX VALIDATION PENDING | The primary 4,096-token identity remains unchanged, so verified exact-copy rows can be reused. Missing, non-exact, truncated, or over-context rows cannot satisfy that gate and are regenerated under the overflow identity. Eight shards per task with at most two active keep each job below the 12-hour site limit without changing the experiment. |
| DEC-027 | Separate the instruction-model generation cap from the frozen downstream embedding context after job 536. | SOURCE IMPLEMENTED; LOCAL VALIDATION PENDING | The first two fidelity-v4 shards rejected 49 and 34 responses at `finish_reason=length` despite successful caching of 400 and 375 rows. Preserve both old decoding identities, use a separately hashed 16,384-token cap only for unresolved generation, and continue to reject any exact output that exceeds the unchanged 8,192-token embedding tokenizer. |
| DEC-028 | Restore genuine bounded M2 summarization after job 545 proved exact copying incompatible with the frozen downstream context. | SOURCE IMPLEMENTED; LOCAL VALIDATION COMPLETE; DGX PENDING | Preserve job 545 and all legacy caches; use one new 6,144-token structured-summary identity, row-level unsupported-fact, text/numeric value, within-time-block association, retention, clean-finish, and tokenizer checks, three fixed contract attempts, and the unchanged 8,192-token embedding gate. Use 64 smaller shards with eight total endpoint requests; reuse canonical M3 470/471 and keep test locked. |
| DEC-029 | Recover slow fidelity-v6 array 559 by changing only sharding and capacity-governed request concurrency. | SOURCE IMPLEMENTED; LOCAL VALIDATION COMPLETE; DGX PENDING | Preserve/revalidate every accepted fidelity-v6 row; benchmark only synthetic records at 8/16/24/32 requests; choose concurrency without outcomes; use 192 shards per task and eight requests per shard so interruption cannot strand hundreds of rows. The 768-element array also remains below common Slurm array-size ceilings. The model, prompt, decoding/cache identity, clinical evidence, conditions, fidelity gates, M3 evidence, and test lock do not change. |
| DEC-030 | Use M1 direct serialization as the sole primary approach in the one-time held-out evaluation; retain M3 as validation-only evidence and M2 as incomplete future work. | COMPLETE; ONE-TIME TEST CLOSED | William authorized this scope on 2026-09-07 before held-out access. The frozen manifest retained every M1 task, cohort, condition, baseline, hyperparameter, seed, and statistical test. Jobs 583/585/586 completed the one-time M1 evaluation; M2 and M3 did not access held-out outcomes. |
| DEC-033 | Resume M2 after the completed one-time M1 evaluation as a validation-only extension, using the existing fidelity-v7 summary identity and unchanged Qwen3-Embedding-0.6B encoder. | ACTIVE | Jeff requested continued M2 work on 2026-09-09. This extension may reuse only rows that pass the existing immutable cache and fidelity gates, must not access held-out outcomes, and cannot modify or select against the completed M1 test result. A fresh synthetic endpoint-capacity ladder must precede scheduling because serving capacity may have changed. |
| DEC-031 | Recover Prompt 5 preparation after jobs `574_0`-`574_3` without creating a second held-out experiment. | VERIFIED COMPLETE | Recovery commit `e9f8f59` retained frozen source commit `7315451`, jobs 572/573, the access audit, cohort, and all 574 failures. Jobs `583_0`-`583_3`, `585_0`-`585_3`, 586, and 587 wrote COMPLETE and the aggregate/publication validators passed. |
| DEC-032 | Report the held-out null and negative temporal contrasts exactly as observed and make no post-access design change. | LOCKED 2026-09-08 | C1 macro-AUROC was 0.646. C1 minus latest-window was -0.014 (95% CI -0.032 to 0.003; Holm p=0.224), and C1 minus the three-seed shuffled-time mean was 0.002 (95% CI -0.001 to 0.006; Holm p=0.224). The count baseline exceeded C1 by 0.066 macro-AUROC. |
