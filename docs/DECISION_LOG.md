# Decision log

Status date: 2026-09-01. Open scientific deviations are tracked in greater detail in
`PROTOCOL_DEVIATIONS.md`.

| ID | Decision | Status | Evidence or next gate |
|---|---|---|---|
| DEC-001 | Keep the existing three-task panel labeled PILOT and prospectively lock the separate four-task breadth panel (`lab_anemia`, `guo_readmission`, `new_hypertension`, `chexpert`). | LOCKED 2026-08-21 | Selected before pilot embeddings, informed by Jeff's clinical-family breadth recommendation; no claim of his approval of this wording. |
| DEC-002 | Use each task's official prediction time `T` and strict `event_time < T` as the current conservative project censoring rule. | ACTIVE; EXCEPTION DECISION OPEN | DEV-002 records the possible task-specific benchmark-boundary exception. Future events are always prohibited. |
| DEC-003 | Use exact prediction-relative half-open 30-day windows for lab/imaging/operational tasks and 90-day windows for incident diagnoses. | VERIFIED | Job 408 recovered and validated job-407 aggregate evidence. |
| DEC-004 | Remove only exact seven-field duplicate rows and retain clinical recurrence. | VERIFIED | Job 408 recovered and validated job-407 aggregate evidence. |
| DEC-005 | C0-C3 share retained content; C1-C3 keep the final window correct; C2 reverses earlier gaps and C3 shuffles earlier labels under three fixed seeds. | VERIFIED | Job 408 recovered and validated content matching and Notebook 03. |
| DEC-006 | Keep latest-window and latest-code as separate baselines rather than crossing either with C0-C3. | ACTIVE | Required before temporal model comparisons. |
| DEC-007 | Compare M1 flat embedding, M2 one-pass summary embedding, and M3 per-window embedding plus GRU as operational systems. | ACTIVE; NOT YET RUN | M2 must use only an authorized local model; no governed history may reach an external API. |
| DEC-008 | Do not access held-out test performance until the panel and full experiment manifest are frozen. | ACTIVE CONTROL | Train/validation-only development continues. |
| DEC-009 | Recover job 407 instead of rerunning Notebook 03. Preserve its FAILED sentinel permanently. | COMPLETE | Job 408 wrote COMPLETE with exit status 0 and preserved the source failure hash. |
| DEC-010 | Authorize the complete three-task engineering pilot without allowing it to satisfy the final-panel gate. | COMPLETE | Jobs 415-418 and the semantic report/notebook validators passed; the aggregate records `PILOT_COMPLETE_FINAL_PANEL_UNCHANGED` and no test use. |
| DEC-011 | Retain BF16 Qwen inference but normalize the finite raw embedding vectors explicitly in FP32. | VERIFIED | The corrected synthetic smoke gate produced finite 1,024-dimensional embeddings with norms within approximately 2e-9 of one before the completed pilot chain ran. |
| DEC-012 | Build the final cohort in a new immutable namespace with the exact sorted eligible prediction-time-index selection hash, independent of outcome, history, record length, and model results. | VERIFIED | Foundation job 437 and its semantic/notebook validators passed. |
| DEC-013 | Admit 8B confirmatory cells only as the longest feasible prefix of C1, C3, latest-window, and latest-code under runtime/memory evidence and a 2x time contingency. | VERIFIED | Runtime-only throughput job 444 passed; no outcome performance informed feasibility. |
| DEC-014 | Exclude every official patient/task/prediction-time group containing multiple label rows before deterministic final-cohort ranking, regardless of label agreement. | VERIFIED | Foundation job 437 passed the governed outcome-independent duplicate-time accounting and semantic checks. |
| DEC-015 | Retain the required 2x throughput contingency and use a five-minute additional reserve within the 12-hour site limit. | VERIFIED | Job 444 passed with the full 2x contingency and 300-second reserve; no scientific input changed. |
| DEC-016 | Recover fit job 446 by reusing only exact COMPLETE job-437/444/445 evidence and submitting job 23 alone. | SOURCE IMPLEMENTED; RECOVERY RUN PENDING | The gate records source/recovery commits and hashes, preserves 446 FAILED, rejects partial outputs, and prohibits rerunning foundation, throughput, embeddings, models, M2/M3, or test evaluation. |
