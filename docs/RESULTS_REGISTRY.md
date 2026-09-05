# Results registry

Status date: 2026-09-05

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
| Prompt 3 final M1 validation | Foundation `1eb0d3535b0afaff0b82b4d430144ca7df372a8c`; throughput/embeddings `77c47f322b98fc74196cc56a0a16edc43604c887`; fit recovery `ef1b0debfa963ee365645ec2f754d8d129e071ce` | 437 foundation COMPLETE; 438 preserved throughput failure; 444 throughput COMPLETE; 445_0–445_3 embeddings COMPLETE; 446 preserved FAILED; 451 fit recovery COMPLETE | Final report SHA-256 `6d16d6b2494042fd2c84d25ef790f030355738cf8512fb87cf5477f73a3e0c7a`; handoff SHA-256 `2bd2223b13b9395ee7a725da9c260f1560c387db6cf5177f28f35999e0c56814`; report decision `FINAL_M1_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED` | COMPLETE; NO M2/M3 OR TEST USE | Prompt 4 M2/M3 validation and freeze |
| Prompt 4 M2/M3 validation and freeze | Canonical M3 `1a97d8f`; M2 fidelity-v3 `818103e`; fidelity-v4 pending | M3 470_0-470_3 and 471_0-471_11 COMPLETE; M2 502/507/513/519 superseded; 529_0-529_3 FAILED; 530/531 canceled | Job 529 preserved candidate rows but exposed 12-hour task overruns and 333 CheXpert length completions. Fidelity-v4 keeps the primary cache identity, revalidates exact copies, uses a separate 8,192-token overflow identity, and runs eight deterministic shards per task with two active. | M3 REFERENCE COMPUTE COMPLETE; M2 FIDELITY-V4 SOURCE PENDING VALIDATION; TEST LOCKED | Validate/commit/push, run `submit_prompt4_m2_fidelity_v4.sh`, complete manual fidelity review, then reference finalization |
| Prompt 5 one-time held-out evaluation | Pending | Pending | Pending | LOCKED | Frozen hashed manifest and every Prompt 5 authorization gate |
