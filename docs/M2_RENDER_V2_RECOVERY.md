# M2: recover job 982's rendering and run frozen confirmation

2026-09-24. These instructions supersede the structured-v1 submission guide.

## Evidence and correction

Job 982 at `f9cce3d09196177ab123e5467854f0435773c051` checked 32 training inputs in about
39 minutes 34 seconds. Thirty passed. A user-run hash-verified diagnostic showed that every latest
saved response passed the factual checks with measured retention 1.0. Two rendered strings were
8,248 and 8,265 tokens, exceeding the frozen 8,192-token encoder cap by 56 and 73. Confirmation
983 did not start. This was not the 12-hour limit and does not require another development
generation pass. The aggregate does not establish that summarization improves prediction.

The generation prompt, JSON schema, clinical source, model, decoding and sample plan stay pinned.
The new renderer uses a short field legend and compact, escaped fields. Ordered multi-value runs
retain all value/text/count tuples. Null, empty strings, literal `null`, units, source, recurrence,
and time-label ordering remain distinguishable. Decoding must recover the exact returned JSON.
The whole new text, including the legend, is counted with the same 0.6B tokenizer and 8,192 cap.
This changes the textual representation seen by an eventual encoder, so embedding-input identity
is versioned; it does not claim identical embeddings. No facts are added from the source or dropped
to make an oversized response fit. We cannot know actual token savings until DGX replay runs.

The new launcher queues exactly two jobs:

1. **Replay** (30-minute allocation): verify original hashes and FAILED sentinel; read the fixed
   last saved response for each of 32 development inputs; revalidate fidelity; render and check
   exact round trip plus token size; write new recovery evidence. No endpoint calls or embeddings.
2. **Confirmation** (four-hour allocation, dependent on replay success): use the original frozen
   independent 32-input training sample, not a new easier sample; same model request contract,
   four concurrent requests, two bounded attempts and deadline/circuit-breaker controls. Apply
   the new renderer uniformly. No embeddings or classifiers. No full-cohort graph is submitted.

Original job 982 remains FAILED permanently. Its private manifest is pinned to
`480effa80c708f4e04e29fdb863cf1f80cc3f49048da232c820f90c27ce120d1`.
An old confirmation RUNNING/COMPLETE sentinel or private directory stops automatic reuse of that
sample. Private evidence never leaves DGX. A new failure blocks downstream work and is preserved.

## Mac: publish only this repair

Run outside SSH. Unrelated manuscript edits remain unstaged. The block stops on pre-existing
staged work, validation failure, failed commit, or failed push. The assistant leaves it uncommitted.

```bash
(
  set -e
  cd /Users/williamdelery/code/temporal-ehr-ml4h
  git diff --cached --quiet || { echo 'STOP: inspect existing staged changes'; exit 1; }
  .venv/bin/python -m pytest
  .venv/bin/python -m ruff check .
  .venv/bin/python scripts/check_project_context.py
  bash -n scripts/slurm/24f_m2_render_recovery.sbatch
  bash -n scripts/slurm/submit_m2_render_recovery.sh
  bash -n scripts/slurm/m2_render_status.sh
  command -v rg >/dev/null
  bash scripts/git/pre_push_safety.sh
  git add AGENTS.md docs/CURRENT_SCIENTIFIC_PROTOCOL.md docs/PROJECT_STATE.md \
    docs/DECISION_LOG.md docs/NEXT_ACTION.md docs/DGX_BATCH_RUNBOOK.md \
    docs/PROTOCOL_DEVIATIONS.md docs/RESULTS_REGISTRY.md \
    docs/M2_STRUCTURED_V1_RUNBOOK.md docs/M2_RENDER_V2_RECOVERY.md \
    configs/recovery/m2_render_v2_after_982.yaml \
    src/temporal_ehr/final_m23/structured_recovery.py \
    src/temporal_ehr/final_m23/structured_render_v2.py \
    src/temporal_ehr/final_m23/render_recovery.py \
    scripts/run_m2_render_recovery.py scripts/slurm/24f_m2_render_recovery.sbatch \
    scripts/slurm/submit_m2_render_recovery.sh scripts/slurm/m2_render_status.sh \
    tests/test_m2_render_recovery.py
  git diff --cached --check
  git --no-pager diff --cached --stat
  bash scripts/git/pre_push_safety.sh
  git commit -m "Recover M2 job 982 with reversible compact rendering"
  git push origin main
  git rev-parse HEAD
)
```

Save the final commit hash. Connect to MedNet VPN, then SSH from the Mac:

```bash
ssh wdelery@10.45.62.90
```

## DGX: deploy and submit once

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
squeue -u "$USER" -o '%.18i %.12P %.30j %.2t %.10M %.10l %R'
git status --short --branch
```

Only with an empty queue and clean checkout, update. A pull/authentication failure means stop.

```bash
eval "$(ssh-agent -s)"
ssh-add /home/wdelery/.ssh/id_ed25519_github_dgx
git pull --ff-only origin main
git rev-parse HEAD
```

Compare the commit to the Mac commit. Then:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/submit_m2_render_recovery.sh
bash scripts/slurm/m2_render_status.sh
```

The script supplies project/model roots, reserves a commit-specific submission record, and records
each job ID immediately. If submission partially fails, inspect that record instead of submitting
another pair. No GPU or model download is needed for replay. The tokenizer and all dependencies
run in the NVIDIA container; the confirmation model remains on the inference server.

After confirming submission, `exit` SSH and disconnect VPN if desired; batch jobs continue.
Do not update the checkout while these jobs are queued/running. Allocation limits are not ETAs.

## Monitor and verify

After reconnecting to VPN and SSH:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/m2_render_status.sh
```

If needed, inspect the recorded logs (no need to guess job IDs):

```bash
source outputs/private/m2_render_job_ids.env
tail -n 100 "slurm-tehr-m2-render-${REPLAY_JOB_ID}.out"
tail -n 100 "slurm-tehr-m2-render-${CONFIRMATION_JOB_ID}.out"
```

After both are COMPLETE with report PASS and no FAILED sentinel, verify existing evidence:

```bash
(
  set -e
  cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
  source outputs/private/m2_render_job_ids.env
  test "$(git rev-parse HEAD)" = "$M2_RENDER_COMMIT"
  export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
  for phase in replay confirmation; do
    if test "$phase" = replay; then job="$REPLAY_JOB_ID"; else job="$CONFIRMATION_JOB_ID"; fi
    srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G \
      --time=00:30:00 \
      --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
      --container-mounts=/data:/data --container-workdir="$PWD" --export=ALL \
      env PYTHONPATH=src .venv/bin/python scripts/run_m2_render_recovery.py verify \
      --phase "$phase" --job-id "$job"
  done
  echo 'PASS: replay and confirmation semantically verified'
)
```

Verification makes no model requests. It checks original evidence/plan, current generation contract,
completed parent, response binding, factual fidelity, reversible rendering, tokenizer identity,
embedding-input key and 8,192-token cap. A missing report is not success.

## Aggregate handoff and remaining M2 work

On the Mac, connected to VPN, copy only safe aggregate reports:

```bash
mkdir -p /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/engineering/m2_render_v2
rsync -avP --checksum --include='*/' --include='replay-*.json' \
  --include='confirmation-*.json' --exclude='*' \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/engineering/m2_render_v2/ \
  /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/engineering/m2_render_v2/
```

Return those reports and verification output. Review live token fit and fidelity before production
worker/cache integration and capacity sizing. Full train/validation generation, the frozen
40-history manual review, 0.6B embedding and validation fitting remain to be completed. M1 and M3
are preserved, and no further held-out evaluation is authorized. Full retention in a small sample
does not establish useful summarization or clinical benefit; that remains an empirical question.
