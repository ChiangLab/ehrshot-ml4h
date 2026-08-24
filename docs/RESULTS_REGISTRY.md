# Results registry

Status date: 2026-08-24

This registry records only privacy-safe phase state, source provenance, job identifiers, aggregate
paths, and gates. It never records patient-level values or held-out performance.

| Phase | Source commit | DGX jobs | Aggregate evidence | State | Next gate |
|---|---|---|---|---|---|
| Phase 1 notebook pack | `bc5a746` | Historical execution not used as the final alignment gate | Aggregate foundation/count/serialization reports remain governed by their later review | COMPLETE AS ENGINEERING FOUNDATION | Phase 2 alignment |
| Phase 2 alignment source | `f9a370c7fe1cb13edd126128cfddc79100852913` | 407 | `reports/generated/phase1_protocol_review/protocol_alignment.json` | ARTIFACTS COMPLETE; JOB FAILED AFTERWARD | Recovery validation |
| Phase 2 recovery | `e12d475d5ec2bc1f6c02aeda28e1a252ac84d6ea` | 408 | Recovery sentinel and validated alignment aggregate | COMPLETE | Prompt 2 pilot |
| Prompt 2 implementation | `25aaed1` | None at source-completion time | None | SOURCE COMPLETE | Pinned 0.6B smoke |
| Prompt 2 numerical repair | `cb07cb5` | 411 is retained failed smoke evidence | Synthetic diagnostic only; no aggregate pilot result | SOURCE COMPLETE; DGX RECHECK REQUIRED | New smoke plus dependency-locked pilot chain |
| Prompt 2 complete pilot | Pending | Pending | `reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json` | NOT COMPLETE | All job sentinels, semantic validators, and privacy-safe handoff must pass |
| Prompt 3 final M1 validation | Pending | Pending | Pending | BLOCKED | Complete Prompt 2 pilot |
| Prompt 4 M2/M3 validation and freeze | Pending | Pending | Pending | BLOCKED | Complete Prompt 3 M1 validation |
| Prompt 5 one-time held-out evaluation | Pending | Pending | Pending | LOCKED | Frozen hashed manifest and every Prompt 5 authorization gate |
