# Decision log

Status date: 2026-08-21. Open scientific deviations are tracked in greater detail in
`PROTOCOL_DEVIATIONS.md`.

| ID | Decision | Status | Evidence or next gate |
|---|---|---|---|
| DEC-001 | Keep the existing three-task panel labeled PILOT and the written four-task breadth panel labeled PROPOSED. | ACTIVE | Exactly one reviewed panel may later be copied into `locked_panel.yaml` and marked LOCKED with Jeff. |
| DEC-002 | Use each task's official prediction time `T` and strict `event_time < T` as the current conservative project censoring rule. | ACTIVE; EXCEPTION DECISION OPEN | DEV-002 records the possible task-specific benchmark-boundary exception. Future events are always prohibited. |
| DEC-003 | Use exact prediction-relative half-open 30-day windows for lab/imaging/operational tasks and 90-day windows for incident diagnoses. | IMPLEMENTED; REAL-RUN RECOVERY PENDING | Job-407 aggregate report and executed Notebook 03 await recovery validation. |
| DEC-004 | Remove only exact seven-field duplicate rows and retain clinical recurrence. | IMPLEMENTED; REAL-RUN RECOVERY PENDING | Job-407 recovery must validate report and notebook. |
| DEC-005 | C0-C3 share retained content; C1-C3 keep the final window correct; C2 reverses earlier gaps and C3 shuffles earlier labels under three fixed seeds. | IMPLEMENTED; REAL-RUN RECOVERY PENDING | Synthetic tests pass; real aggregate validation requires recovery COMPLETE. |
| DEC-006 | Keep latest-window and latest-code as separate baselines rather than crossing either with C0-C3. | ACTIVE | Required before temporal model comparisons. |
| DEC-007 | Compare M1 flat embedding, M2 one-pass summary embedding, and M3 per-window embedding plus GRU as operational systems. | ACTIVE; NOT YET RUN | M2 must use only an authorized local model; no governed history may reach an external API. |
| DEC-008 | Do not access held-out test performance until the panel and full experiment manifest are frozen. | ACTIVE CONTROL | Train/validation-only development continues. |
| DEC-009 | Recover job 407 instead of rerunning Notebook 03. Preserve its FAILED sentinel permanently. | IMPLEMENTED LOCALLY; DGX PENDING | DEV-013 closes only after the separate recovery job produces verified COMPLETE. |
