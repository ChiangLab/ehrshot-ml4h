# DGX batch runbook

## Operating rules

- Connect to MedNet VPN before SSH, Git operations against the DGX, `scp`, or `rsync` to
  `10.45.62.90`. Disconnect only after the job has been submitted and the initial scheduler/log
  checks succeed; `sbatch` jobs continue after SSH and VPN disconnect.
- Code lives in `/data/shared73/projects/wdelery/temporal-ehr-ml4h`; governed data, private outputs,
  environments, and model assets stay on DGX storage and outside Git.
- Use `sbatch --partition=batch` for formal jobs. Do not run heavy workloads on the login node.
- A valid run needs `sacct` state COMPLETED, exit `0:0`, a clean log, a job-specific COMPLETE
  sentinel, and semantic postconditions. File existence alone is never success.

## Evidence-bound recovery of job 407

Do not submit the recovery until the recovery-validator commit has been pushed and pulled onto the
DGX. From the DGX repository:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
git status --short --branch
git pull --ff-only origin main

RECOVERY_JOB_ID=$(
  sbatch --parsable \
    --partition=batch \
    --export=ALL,SOURCE_ALIGNMENT_JOB_ID=407 \
    scripts/slurm/11b_recover_phase1_alignment.sbatch
)
echo "Recovery job: ${RECOVERY_JOB_ID}"
```

This recovery does not execute Notebook 03 and needs no EHRSHOT or tokenizer environment-variable
exports. It uses the existing report, executed notebook, source environment audit, and failed
sentinel. Save the printed recovery job ID.

Initial check:

```bash
squeue -j "${RECOVERY_JOB_ID}" -o "%.18i %.12P %.26j %.2t %.10M %R"
```

If the job is pending, inspect the reason in the final column. A log may not exist until the job
starts. Once it is RUNNING:

```bash
tail -f "slurm-tehr-phase1-recovery-${RECOVERY_JOB_ID}.out"
```

After it leaves `squeue`, verify all postconditions:

```bash
sacct -j "${RECOVERY_JOB_ID}" \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,AllocCPUS

tail -n 200 "slurm-tehr-phase1-recovery-${RECOVERY_JOB_ID}.out"

test -s "outputs/private/run_status/${RECOVERY_JOB_ID}/COMPLETE.json"
test ! -e "outputs/private/run_status/${RECOVERY_JOB_ID}/FAILED.json"
test -s "outputs/private/run_status/${RECOVERY_JOB_ID}/recovery_validation.json"
test -s "outputs/private/run_status/407/FAILED.json"

.venv/bin/python - <<PY
import json
from pathlib import Path

job_id = "${RECOVERY_JOB_ID}"
root = Path("outputs/private/run_status")
complete = json.loads((root / job_id / "COMPLETE.json").read_text())
validation = json.loads((root / job_id / "recovery_validation.json").read_text())
assert complete["state"] == "COMPLETE"
assert complete["exit_status"] == 0
assert complete["source_job_id"] == "407"
assert complete["source_git_commit"] == "f9a370c7fe1cb13edd126128cfddc79100852913"
assert validation["status"] == "PASS"
assert all(validation["postconditions"].values())
for artifact in validation["artifacts"].values():
    assert artifact["size_bytes"] > 0
    assert len(artifact["sha256"]) == 64
print("PASS: recovery status, provenance, artifact hashes, and postconditions verified")
PY
```

Then build the aggregate-only handoff:

```bash
.venv/bin/python scripts/build_handoff_bundle.py \
  --job-id 407 \
  --job-id "${RECOVERY_JOB_ID}" \
  --output reports/generated/handoff/project_handoff.json
```

Copy only reviewed aggregate files to the Mac; never copy private executed notebooks, manifests,
labels, predictions, embeddings, or cohort rows into a Codex conversation.
