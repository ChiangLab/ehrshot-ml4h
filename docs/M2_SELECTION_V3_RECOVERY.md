# M2 selection-v3 recovery after failed production array 993

Status: source implemented locally on 2026-09-30; no new DGX job has run yet.

## Why this is a new version

Array 993 did not time out. Its root shard generated three accepted inputs and rejected one input;
the global circuit breaker then failed the full 553-task graph. Across the preserved cache there
are 58 sealed passes and two assessment failures. The failures include duplicated or changed/
reordered series, and one rejected response repeated identically at temperature zero. Increasing
attempts or rerunning the same prompt would not address that deterministic failure.

`m2_selection_v3_reference_v1` changes only the instruction-model output contract. Qwen receives
the same ordered, strictly pre-prediction M1-retained source and returns ordered integer positions
of source fact series. Python validates those positions and materializes every code, value, unit,
count, time label, and within-series order directly from the immutable source. The model cannot
emit or alter clinical facts. Existing retention thresholds and the 8,192-token encoder limit are
still enforced. The on-premise Qwen instruction model, temperature zero, Qwen3-Embedding-0.6B,
four tasks, train/validation boundary, and 40-history manual review remain unchanged. Old caches
cannot satisfy the new request keys. Held-out test, M1, and M3 stay closed.

## Fail-closed graph

1. The planner recursively validates the job-993 plan, HALT evidence, all 553 FAILED sentinels,
   and both rejected inputs, then freezes a new worklist and manual-review sample.
2. A 32-input canary includes both job-993 rejected inputs plus outcome-independent training C1
   inputs across all four tasks.
3. A disjoint 32-input training confirmation follows only after canary success.
4. A read-only readiness job replays every source/request/response/materialization/hash check.
5. Production uses deterministic 16-input shards, at most two active shards and two requests per
   shard. A rejection writes a new immutable HALT and blocks all successors.
6. Coverage sealing recursively verifies every input, then creates the governed 40-history review
   packet. Four task embeddings run only after full coverage.
7. The validation fit is not queued automatically. It requires William's completed manual review.
8. The final read-only verifier recomputes provenance, coverage, prediction-derived metrics and
   macro metrics. It never generates, embeds, fits, or repairs.

## Deploy and launch

Publish only the named recovery files from the Mac:

```bash
cd /Users/williamdelery/code/temporal-ehr-ml4h
bash scripts/git/publish_m2_selection_v3.sh
git --no-pager log -1 --oneline
```

Connect to VPN, SSH to the DGX, require an empty queue and clean checkout, then pull and launch:

```bash
ssh wdelery@10.45.62.90
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
squeue -u "$USER"
git status --short --branch
eval "$(ssh-agent -s)"
ssh-add /home/wdelery/.ssh/id_ed25519_github_dgx
git pull --ff-only origin main
git --no-pager log -1 --oneline
bash scripts/slurm/submit_m2_selection_recovery.sh start
bash scripts/slurm/m2_selection_v3_status.sh
```

The launcher sets the project/model roots. It first freezes the plan in the NVIDIA container, then
submits canary -> confirmation -> readiness -> production -> coverage -> embeddings. Leave the
checkout unchanged while jobs are active. The work continues after SSH/VPN disconnect.

## Monitor and recover infrastructure interruptions

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/m2_selection_v3_status.sh
source outputs/private/m2_selection_v3_job_ids.env
grep -h '"event": "m2_selection_' slurm-tehr-m2-selection-v3-*.out | tail -n 30
```

An empty queue is not proof of success. Require COMPLETE sentinels from the status helper and later
the semantic verifier. `resume` is allowed only for a reviewed infrastructure interruption, after
the queue is empty, on the unchanged commit, and only when the v3 HALT file is absent:

```bash
bash scripts/slurm/submit_m2_selection_recovery.sh resume
```

If `outputs/private/final/m2_selection_v3_reference_v1/HALT.json` exists, do not delete, rename,
or bypass it. Preserve the evidence and diagnose before any further generation.

## Manual review and validation fit

After the coverage job is COMPLETE, review all 40 governed histories on the DGX:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
python3 scripts/review_m2_selection_v3_packet.py
```

Use Space/`b`/`q` in `less`; answer each criterion `pass`, `fail`, or `quit`. Rerunning resumes.
Only type `COMPLETE` after personally reviewing all items. A failed criterion blocks fitting.
Do not copy the private packet off the DGX.

After all four embeddings and the manual review pass, require an empty queue and submit the fit:

```bash
bash scripts/slurm/submit_m2_selection_recovery.sh finalize
bash scripts/slurm/m2_selection_v3_status.sh
```

When fit is COMPLETE, run the independent semantic verifier:

```bash
bash scripts/slurm/verify_m2_selection_v3.sh
```

Only then copy the public aggregate namespace and privacy-safe handoff to the Mac. Do not copy
private generations, review packets, labels, predictions, identifiers, or embeddings.
