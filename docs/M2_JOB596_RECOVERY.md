# M2 recovery after array 596

Status: source repair implemented; real DGX canary pending. Do not declare M2 recovered yet.

## Diagnosis and unchanged controls

The user reported 12 COMPLETE, 369 FAILED, and 387 unfinished shards after stopping 596/597/598.
An earlier first-error classification assigned 333 of 365 failed logs to recurrence. Those
categories are not counts of bad summaries and do not distinguish singleton syntax from changed
counts. The source formatter writes `occurrences=1`; the old grammar rejected it. This repair
accepts that explicit singleton as equivalent to omission. Source matching and all remaining
fidelity gates are unchanged. Genuine wrong counts still fail.

Do not change the prompt, source grouping, Qwen3.6 instruction model, decoding, 8,192-token encoder
context, Qwen3-Embedding-0.6B encoder, clinical tasks, windows, or outcomes. No fact-ID/extractive
replacement is authorized by this repair. Do not rerun M1/M3 or held-out evaluation.

Preserve all old logs, status sentinels, summary rows, and task reports. Accepted caches retain
their immutable generation identities and are revalidated before reuse. Shard execution reports
now live under `m2/tasks/summary_runs/<commit>/` so a new commit cannot overwrite an old report.
The new canary uses a separate private diagnostic directory, not a scientific embedding cache.

## 1. Deploy the scoped local repair

Run on the **Mac**, not in SSH. Do not stage the unrelated manuscript or deliverable edits.

```bash
cd /Users/williamdelery/code/temporal-ehr-ml4h
git --no-pager diff --check
git add -- AGENTS.md docs/CURRENT_SCIENTIFIC_PROTOCOL.md docs/PROJECT_STATE.md docs/DECISION_LOG.md docs/NEXT_ACTION.md docs/DGX_BATCH_RUNBOOK.md docs/PROTOCOL_DEVIATIONS.md docs/RESULTS_REGISTRY.md docs/M2_JOB596_RECOVERY.md src/temporal_ehr/final_m23/m2.py src/temporal_ehr/final_m23/recurrence_recovery.py scripts/run_m2_after596_canary.py scripts/slurm/24d_m2_after596_canary.sbatch scripts/slurm/submit_m2_after596_canary.sh tests/test_m2_recurrence_recovery.py
git --no-pager diff --cached --stat
git diff --cached --check
bash scripts/git/pre_push_safety.sh
```

Review the staged list. If it contains only the scoped repair, commit and push:

```bash
git commit -m "Repair M2 singleton validation and gate recovery after job 596"
git push origin main
git rev-parse HEAD
```

## 2. Submit only the small DGX engineering check

Connect/stay on MedNet VPN, then SSH. Enter passwords/passphrases only at the terminal prompt.

```bash
ssh wdelery@10.45.62.90
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
squeue -u "$USER" -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
git status --short --branch
```

Stop if any old job remains active or the DGX checkout is dirty; do not overwrite it or cancel
unrelated work. When the queue is empty and checkout clean:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github_dgx
git pull --ff-only origin main
git log -1 --oneline
bash scripts/slurm/submit_m2_after596_canary.sh
```

This helper restores the two required paths itself, requires the unchanged scientific config,
and queues one four-hour batch job with four total endpoint requests. It reserves no GPU and loads
only the local tokenizer, not embedding weights. The check covers four source-length strata
(shortest, two intermediate ranks, longest) in each of C1 and seed-20260815 C3 for all four tasks:
32 distinct-within-stratum training inputs selected without labels or results. It never uses the
frozen validation review sample for prompt development. Three unchanged generation attempts are
allowed, with a 300-second request timeout; there is no unbounded retry loop.

It hashes original 596/597/598 evidence and verifies canonical M3 470/471, records API responses
and source text privately, and writes an aggregate report even when fidelity fails. A PASS
requires every input to meet all row-level gates. A small engineering sample cannot establish
population-wide fidelity, capacity for 32 requests, or scientific M2 completion.

## 3. Monitor from the DGX repository

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/m2_after596_canary.env
squeue -j "$M2_CANARY_JOB_ID" -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
tail -n 80 "slurm-tehr-m2-recurrence-canary-${M2_CANARY_JOB_ID}.out"
find "outputs/private/run_status/${M2_CANARY_JOB_ID}" -maxdepth 1 -type f -printf '%f\n'
```

The log prints a progress event for each finished input, including failed inputs. It prints no
clinical content. Queue disappearance alone is not success. Once it finishes:

```bash
cat "reports/generated/engineering/m2_after596/canary-${M2_CANARY_JOB_ID}.json"
```

If FAILED or no report, inspect the last 100 lines of the log. Do not rerun 768 shards, erase
sentinels, increase limits, or loosen fidelity thresholds. Share only that aggregate and sanitized
error messages. Raw `source.json` and `response-*.json` live under
`outputs/private/recovery/m2_after596/<job>/item-00` through `item-31` and stay on DGX.
The first eight items are anemia, then eight readmission, hypertension, and CheXpert; each task
has four C1 then four C3 inputs. Full source-to-output review is done there, not in cloud tools.

If COMPLETE, validate the sentinel, original evidence, source/config commit and all private file
hashes inside the container:

```bash
srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data --container-workdir="$PWD" --export=ALL \
  env PYTHONPATH=src .venv/bin/python scripts/run_m2_after596_canary.py verify --job-id "$M2_CANARY_JOB_ID"
```

## 4. Next gate, not another blind full run

Return the small aggregate report and final status for review. It determines whether singleton
syntax was the main issue or whether actual count, retention, timing, or trend mismatches remain.
No new full-array launcher is armed by these commands. A passing canary must precede a separately
reviewed full resume/capacity check with current-source provenance; the old v7 launcher is not the
job-596 recovery authorization.

After that recovery is authorized, the unchanged scientific sequence is: finish/revalidate all M2
summary caches; embed with frozen 0.6B; prepare the 40-history review packet; complete automated
and human fidelity review; fit/evaluate train/validation-only logistic classifiers. Report this as
the post-M1 exploratory M2 extension, not a new held-out analysis.

Slurm continues after SSH logout and VPN disconnection. Stay on VPN while submitting, inspecting,
or copying only aggregate reports. No laptop sleep timer is required. Mail delivery is site-dependent;
the status files and log remain the authoritative completion evidence.
