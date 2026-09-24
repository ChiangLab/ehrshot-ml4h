# Ordered structured M2: engineering release gates

Status: local implementation, real DGX verification pending (2026-09-22).
This supersedes the job-596 singleton recovery instructions. Do not re-arm the old M2 graph.

## What changes, and what does not

Job 980 checked 32 training inputs and failed despite 24 final PASS. More hours cannot repair a
changed count, unsupported fact, truncated completion, or a validator that misreads its source.
Synthetic regressions additionally demonstrated lost return-to-baseline information in distinct-
value grouping and problems in historical text/trend matching. Preserve all old evidence; these
findings do not prove every historical failure was spurious or every accepted summary was sound.

William authorized `m2_ordered_structured_v1`. It consumes exactly the prepared clinical evidence
already retained for M1, without new history, labels, or prediction targets. Within each unchanged
window, code/source/unit series retain ordered value/text runs and exact counts. For example,
`1,1,2,1` becomes `[(1,2),(2,1),(1,1)]`, not distinct values `1,2`. This preserves each series,
NOT global cross-code interleaving. The model returns JSON clinical facts, not selected fact IDs.
The renderer formats only those returned and validated facts; it never supplies missing source
facts or a deterministic fallback. A retained series must match its source in full. Omission of
whole series is permitted only under fixed keyed retention gates.

- Same four tasks, strict pre-T source, 30/90-day windows and supplied C1/C3 time-label order.
- Same on-premise `nvidia/Qwen3.6-35B-A3B-NVFP4`; no external proxy or redirect.
- Temperature 0, top-p 1, seed 20260815, thinking disabled; one pass, at most two identical-contract
  attempts. No recursive/adaptive re-prompting. Temperature zero does not guarantee byte identity.
- New strict JSON schema and 16,384 generation-token ceiling; downstream summary still must fit
  the unchanged Qwen3-Embedding-0.6B tokenizer at 8,192 tokens. Generation and embedding token
  budgets are different. Oversized summaries fail rather than truncate or silently drop patients.
- Same numeric fidelity floors, now window/code/source/unit-specific: concepts 90%, numeric and
  text-value associations 95%, units 95%, occurrence-weighted recurrence 80%, directional series
  80%; exact time-label sequence. This change in denominator/accounting is explicitly versioned.
  Direction means numeric endpoint change and monotonicity, not a clinical interpretation.
- Old caches never satisfy the new identity. Prompt/schema/source/condition/decoding/served metadata
  are hashed; changing execution concurrency alone does not change a generation identity.
- Endpoint identity is a hash of available model metadata, not a verified server weight digest.
  Do not claim full weight-revision reproducibility without obtaining that from the server owner.
- No local clinical data/model execution, M1/M3 rerun, held-out access, or automatic full M2 launch.

## What runs now

The launcher submits only development -> independent confirmation, each 32 training inputs:
four tasks x C1/C3 reference conditions x four length strata. Both samples are frozen before
generation, independent of outcomes. Confirmation excludes every development and identifiable
job-980 patient across all tasks. Job-980 evidence and upstream M1 prepared files are hash-checked.
Each job first checks exact schema/fidelity support on synthetic input; unsupported endpoint schema
behavior stops before any clinical generation. Preparing the training sample still reads DGX source.

Each job requests four CPUs, 16 GB RAM and four hours on `dgit00038673`, no local GPU allocation.
The model runs at the separate inference endpoint. At most four requests are outstanding, each
with a 600-second timeout and two attempts. A failure circuit breaker and generation deadline
of 3.5 hours (including a 1,300-second no-new-wave reserve) prevent open-ended retrying. A wave
can be quiet for up to about 20 minutes; progress is flushed as that wave's responses are checked.
Development failure cancels the dependent confirmation through Slurm's invalid-dependency policy;
an unstarted canceled job may have no sentinel. Mail notifications are requested, not guaranteed.

These are finite engineering samples, not a throughput guarantee for the whole cohort. Either
job may fail. A four-hour allocation is a ceiling, not an ETA; queue time is additional. Existing
no-clobber evidence is never overwritten, and a source commit cannot be blindly submitted twice.

## 1. Mac: validate and publish the scoped code

Run in a **Mac** terminal, not inside SSH. These commands deliberately exclude manuscript edits
and deliverables. If other files are already staged, the block stops so they are not committed
accidentally. Git commands use no pager; there should be no `(END)` screen.

```bash
(
  set -e
  cd /Users/williamdelery/code/temporal-ehr-ml4h
  git diff --cached --quiet || { echo 'STOP: inspect existing staged changes first'; exit 1; }
  .venv/bin/python -m pytest
  .venv/bin/python -m ruff check .
  .venv/bin/python scripts/check_project_context.py
  bash -n scripts/slurm/24e_m2_structured_gate.sbatch
  bash -n scripts/slurm/submit_m2_structured_gates.sh
  bash -n scripts/slurm/m2_structured_status.sh
  command -v rg >/dev/null
  bash scripts/git/pre_push_safety.sh
  git add AGENTS.md docs/CURRENT_SCIENTIFIC_PROTOCOL.md docs/PROJECT_STATE.md \
    docs/DECISION_LOG.md docs/NEXT_ACTION.md docs/DGX_BATCH_RUNBOOK.md \
    docs/PROTOCOL_DEVIATIONS.md docs/RESULTS_REGISTRY.md docs/M2_STRUCTURED_V1_RUNBOOK.md \
    configs/experiments/m2_structured_v1.yaml scripts/run_m2_structured.py \
    scripts/slurm/24e_m2_structured_gate.sbatch scripts/slurm/submit_m2_structured_gates.sh \
    scripts/slurm/m2_structured_status.sh src/temporal_ehr/final_m23/structured_summary.py \
    src/temporal_ehr/final_m23/structured_recovery.py tests/test_m2_structured.py
  git diff --cached --check
  git --no-pager diff --cached --stat
  bash scripts/git/pre_push_safety.sh
  git commit -m "Add versioned ordered M2 summaries and gated engineering checks"
  git push origin main
  git rev-parse HEAD
)
```

The implementation assistant has not committed, pushed or run this block. Record the commit
printed at the end; it must match the DGX after pull. A failed check or push means stop, not proceed.

## 2. Connect to VPN and SSH

Keep MedNet VPN connected for SSH and copying reports. Run on the Mac:

```bash
ssh wdelery@10.45.62.90
```

Now, on **DGX**, inspect before changing source:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
squeue -u "$USER" -o '%.18i %.12P %.30j %.2t %.10M %.10l %R'
git status --short --branch
```

Only if the queue is empty and checkout clean, update. Do not cancel other work automatically.

```bash
eval "$(ssh-agent -s)"
ssh-add /home/wdelery/.ssh/id_ed25519_github_dgx
git pull --ff-only origin main
git --no-pager log -1 --oneline
git status --short --branch
```

Compare that commit to the Mac commit. If pull failed, do not continue. No package installation
or model download is required. Python dependencies run in the existing NVIDIA container.

## 3. Submit the two bounded engineering jobs

Still on DGX:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/submit_m2_structured_gates.sh
bash scripts/slurm/m2_structured_status.sh
```

The launcher sets project/model roots itself and records both IDs immediately in
`outputs/private/m2_structured_job_ids.env`, pointing to an immutable commit-specific submission
record. If submission partially fails, inspect that record and do not blindly rerun the launcher.
Do not modify/pull the checkout while either job is queued/running. After confirming submission
and initial logs, you can `exit` SSH and disconnect VPN; `sbatch` jobs continue independently.

## 4. Monitor after reconnecting

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/m2_structured_status.sh
```

No environment-variable reconstruction or manual job-number guessing is needed. Reports contain
only aggregate status, suppressed error categories, runtime, feasibility statistics and hashes.
Sources, subject hashes, model responses and individual assessments stay under the private root.
To see more of a failed job's nonclinical log, use the recorded ID:

```bash
source outputs/private/m2_structured_job_ids.env
tail -n 100 "slurm-tehr-m2-structured-${DEVELOPMENT_JOB_ID}.out"
tail -n 100 "slurm-tehr-m2-structured-${CONFIRMATION_JOB_ID}.out"
```

Do not cat private JSON into a chat. If failure occurs before an aggregate is written, share the
exception name/message and job sentinel, after checking that no clinical content is present.

## 5. Require semantic verification, not only COMPLETE files

Run after both jobs have left the queue with COMPLETE, no FAILED, and report PASS. This reads
existing evidence and the offline tokenizer only; it makes no new model request.

```bash
(
  set -e
  cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
  source outputs/private/m2_structured_job_ids.env
  test "$(git rev-parse HEAD)" = "$M2_STRUCTURED_COMMIT"
  export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
  for phase in development confirmation; do
    if test "$phase" = development; then
      job="$DEVELOPMENT_JOB_ID"
    else
      job="$CONFIRMATION_JOB_ID"
    fi
    srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G \
      --time=00:30:00 \
      --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
      --container-mounts=/data:/data --container-workdir="$PWD" --export=ALL \
      env PYTHONPATH=src .venv/bin/python scripts/run_m2_structured.py verify \
      --phase "$phase" --job-id "$job"
  done
  echo 'PASS: both engineering gates semantically verified; production still requires review'
)
```

Verification checks current source/configuration, upstream/history hashes, no-clobber private
manifest, exact model-response-to-assessment binding, tokenizer identity, sample balance, train-only
boundary, subject separation, retention/time/count/order/token constraints, and the confirmation's
completed development parent. Do not delete or alter any old artifact to satisfy these gates.

## 6. Return only safe engineering reports; next decision

On the **Mac**, while connected to VPN:

```bash
mkdir -p /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/engineering/m2_structured_v1
rsync -avP --checksum --include='*/' --include='development-*.json' \
  --include='confirmation-*.json' --exclude='*' \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/engineering/m2_structured_v1/ \
  /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/engineering/m2_structured_v1/
```

Share those reports plus semantic verification output. Disconnect VPN whenever you no longer
need SSH/transfer access. Do not run any old summary, embedding or fit launcher on these artifacts.

After both real gates pass, review representative source/output pairs on DGX and measured budget
feasibility. The next implementation is version-specific resumable production workers with
per-row admission, capacity-sized bounded shards, complete four-task train/validation coverage,
then 0.6B embeddings and the original frozen 40-history manual review before validation fitting.
It must not silently import old summaries, drop hard cases, relax fidelity, alter source coverage,
switch encoders, or reopen test. A failure here is useful engineering evidence, not an instruction
to force a PASS. No code can guarantee that the live instruction model will meet every contract.
