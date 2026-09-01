# Results registry

Status date: 2026-09-01

This registry records only privacy-safe phase state, source provenance, job identifiers, aggregate
paths, and gates. It never records patient-level values or held-out performance.

| Phase | Source commit | DGX jobs | Aggregate evidence | State | Next gate |
|---|---|---|---|---|---|
| Phase 1 notebook pack | `bc5a746` | Historical execution not used as the final alignment gate | Aggregate foundation/count/serialization reports remain governed by their later review | COMPLETE AS ENGINEERING FOUNDATION | Phase 2 alignment |
| Phase 2 alignment source | `f9a370c7fe1cb13edd126128cfddc79100852913` | 407 | `reports/generated/phase1_protocol_review/protocol_alignment.json` | ARTIFACTS COMPLETE; JOB FAILED AFTERWARD | Recovery validation |
| Phase 2 recovery | `e12d475d5ec2bc1f6c02aeda28e1a252ac84d6ea` | 408 | Recovery sentinel and validated alignment aggregate | COMPLETE | Prompt 2 pilot |
| Prompt 2 implementation | `25aaed1` | None at source-completion time | None | SOURCE COMPLETE | Pinned 0.6B smoke |
| Prompt 2 numerical repair | `cb07cb5` | 411 is retained failed smoke evidence; corrected successor smoke passed | Synthetic smoke report records finite unit-normalized 1,024-dimensional output | VERIFIED | Complete dependency-locked pilot |
| Prompt 2 complete pilot | `ae35c50982ed56cee9a90875cdc98a05f63ffd54` | 415, 416, 417_0, 417_1, 417_2, 418 | `reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json`, SHA-256 `6b2d812cf430ed186fae00615e3b302dd8da58294b23902a74ef514c4047494d`; handoff SHA-256 `e95300137aaa317a65138f9fa55a4f121ab466fb5ed3d869c5f6660e21930042` | COMPLETE; FINAL PANEL UNCHANGED; NO TEST USE | Prompt 3 final M1 implementation |
| Prompt 3 final M1 validation | Foundation `1eb0d3535b0afaff0b82b4d430144ca7df372a8c`; throughput/embeddings `77c47f322b98fc74196cc56a0a16edc43604c887`; fit-only validator pending commit | 437 foundation COMPLETE; 438 preserved throughput failure; 444 throughput COMPLETE; 445_0–445_3 embeddings COMPLETE; 446 fit gate FAILED before fitting | Foundation, throughput, and four embedding aggregates/caches passed their gates; fit-only recovery passed 98 local tests and all static/privacy checks; no final predictions/report, M2/M3, or test use | FIT-ONLY RECOVERY SOURCE/LOCAL VALIDATION PASS | Preserve 437/444/445/446; submit job 23 only through the fit-recovery helper |
| Prompt 4 M2/M3 validation and freeze | Pending | Pending | Pending | BLOCKED | Complete Prompt 3 M1 validation |
| Prompt 5 one-time held-out evaluation | Pending | Pending | Pending | LOCKED | Frozen hashed manifest and every Prompt 5 authorization gate |
