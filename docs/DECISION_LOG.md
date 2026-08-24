# Decision log

Status date: 2026-08-24. Open scientific deviations are tracked in greater detail in
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
| DEC-010 | Authorize the complete three-task engineering pilot without allowing it to satisfy the final-panel gate. | ACTIVE PILOT | Qwen3-Embedding-0.6B, 4,096 tokens, M1 and B1-B4, train/validation only. |
| DEC-011 | Retain BF16 Qwen inference but normalize the finite raw embedding vectors explicitly in FP32. | IMPLEMENTED; DGX VERIFICATION PENDING | Synthetic job 411 failed built-in BF16 normalization; a synthetic DGX diagnostic produced unit vectors under explicit FP32 L2 normalization. A new smoke job must pass before the pilot chain. |
