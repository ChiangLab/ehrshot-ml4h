# DGX batch runbook

## Operating rules

- Connect to MedNet VPN before SSH, Git operations against the DGX, `scp`, or `rsync` to
  `10.45.62.90`. Disconnect only after the job has been submitted and the initial scheduler/log
  checks succeed; `sbatch` jobs continue after SSH and VPN disconnect.
- Code lives in `/data/shared73/projects/wdelery/temporal-ehr-ml4h`; governed data, private outputs,
  environments, and model assets stay on DGX storage and outside Git.
- Use `sbatch --partition=batch` for formal jobs. Do not run heavy workloads on the login node.
- Prompt 3 container jobs are pinned to `dgit00038673`: synthetic 8B smoke job 423 passed there,
  while foundation job 424 failed before scientific execution when `dgit00038674` could not create
  the Pyxis runtime directory `/run/user/2051`. Do not remove the node constraint until the second
  node's container runtime is repaired and independently verified.
- Foundation job 430 subsequently reached governed cohort construction on the verified node and
  failed closed on multiple official label rows sharing a patient/task/prediction time. Jobs
  431–433 were canceled. The reviewed repair excludes all such groups before selection regardless
  of label values; do not resubmit until its commit is pulled onto the DGX.
- At the repaired commit, foundation job 437 completed with every validator. Throughput job 438
  measured anemia's 2x contingent primary runtime at 42,700.11 seconds and failed only because the
  extra 900-second reserve exceeded the 43,200-second site limit. Preserve both sentinels. The
  reviewed recovery changes only the reserve to 300 seconds and reuses job 437 through exact
  provenance/config/hash validation; it does not rerun the foundation.
- That recovery completed throughput 444 and embedding tasks 445_0 through 445_3 at commit
  `77c47f322b98fc74196cc56a0a16edc43604c887`. Fit 446 failed before fitting on an overly strict
  foundation source-equality check. Preserve all of these artifacts. The next run must be fit-only;
  do not submit the full chain or either earlier recovery helper again.
- A valid run needs scheduler completion when accounting is available, a clean log, a job-specific
  COMPLETE sentinel with exit status 0, and semantic postconditions. This site has disabled
  historical `sacct`; in that case require disappearance from `squeue` plus the durable evidence.
  File existence alone is never success.

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

## Complete three-task engineering pilot

The pilot is intentionally separate from the locked final panel. It uses only `lab_anemia`,
`new_acutemi`, and `guo_los`; Qwen3-Embedding-0.6B; 4,096 tokens; M1; and train/validation. Never
substitute a pilot cohort, prepared table, or cache for a final-panel artifact.

Before connecting to the DGX, commit and push the reviewed source change. Then connect to MedNet VPN,
SSH to the compute host, and fast-forward the clean DGX checkout:

```bash
ssh wdelery@10.45.62.90
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
git status --short --branch
git pull --ff-only origin main

export TEMPORAL_EHR_PROJECT_ROOT="$PWD"
export TEMPORAL_EHR_MODEL_ROOT="/data/shared73/projects/wdelery/model_cache/qwen3-embedding"
export EHRSHOT_ASSET_ROOT="/data/shared73/projects/wdelery/ehrshot-assets"
export EHRSHOT_DATA_PATH="${EHRSHOT_ASSET_ROOT}/data.parquet"
export EHRSHOT_ASSET_ROOT="/data/shared73/projects/wdelery/ehrshot-assets"
export EHRSHOT_DATA_PATH="${EHRSHOT_ASSET_ROOT}/data.parquet"
```

Verify the exact governed inputs without printing rows:

```bash
test -s "${EHRSHOT_DATA_PATH}"
test -d "${EHRSHOT_ASSET_ROOT}"
test -s "outputs/private/cohorts/primary_panel.parquet"
test -s "reports/generated/count_baseline/validation_results.json"
for task in lab_anemia new_acutemi guo_los; do
  test -s "outputs/private/count_baseline/${task}/validation_predictions.parquet"
done
```

If either pinned-model control file is absent, download the 0.6B snapshot outside the scientific
chain. The download node must have Hugging Face access; no patient data are read:

```bash
MODEL_DIR="${TEMPORAL_EHR_MODEL_ROOT}/Qwen3-Embedding-0.6B"
DOWNLOAD_JOB_ID=""
if [[ ! -s "${MODEL_DIR}/PINNED_REVISION.txt" || \
      ! -s "${MODEL_DIR}/MODEL_ARTIFACT_MANIFEST.json" ]]; then
  DOWNLOAD_JOB_ID="$(
    sbatch --parsable --partition=batch --export=ALL \
      scripts/slurm/02_download_qwen_06b.sbatch
  )"
  DOWNLOAD_JOB_ID="${DOWNLOAD_JOB_ID%%;*}"
  echo "DOWNLOAD_JOB_ID=${DOWNLOAD_JOB_ID}"
fi
```

Monitor an actual download with `squeue` and its log. After it succeeds, or immediately if the
model was already complete, submit the synthetic-only GPU smoke test:

```bash
squeue -j "${DOWNLOAD_JOB_ID}" -o "%.18i %.12P %.26j %.2t %.10M %R" 2>/dev/null || true
tail -n 100 "slurm-qwen3e-06b-download-${DOWNLOAD_JOB_ID}.out" 2>/dev/null || true

SMOKE_JOB_ID="$(
  sbatch --parsable --partition=batch --export=ALL \
    ${DOWNLOAD_JOB_ID:+--dependency=afterok:${DOWNLOAD_JOB_ID}} \
    scripts/slurm/03_smoke_qwen_06b.sbatch
)"
SMOKE_JOB_ID="${SMOKE_JOB_ID%%;*}"
echo "SMOKE_JOB_ID=${SMOKE_JOB_ID}"
```

If the download job fails specifically because the DGX has no Hugging Face egress, disconnect from
VPN and run the same pinned downloader in the Mac checkout, then reconnect to VPN and transfer the
complete directory. `rsync` is resumable and the DGX smoke test re-verifies every manifest hash:

```bash
cd "/Users/williamdelery/Library/Mobile Documents/com~apple~CloudDocs/raptor_biodesign_fellowship/ehrshot/temporal-ehr-ml4h"
source .venv/bin/activate
PYTHONPATH=src .venv/bin/python scripts/dgx/download_qwen_embeddings.py \
  --models 0.6B \
  --output-root "/Users/williamdelery/Downloads/qwen3-embedding"

# Reconnect to MedNet VPN before this transfer.
caffeinate -i rsync -avP --checksum \
  "/Users/williamdelery/Downloads/qwen3-embedding/Qwen3-Embedding-0.6B/" \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/model_cache/qwen3-embedding/Qwen3-Embedding-0.6B/
```

After this fallback, set `DOWNLOAD_JOB_ID=""` and submit the smoke job without a dependency.

Do not submit the scientific chain until the smoke job has left `squeue`, its log ends in PASS, the
private smoke JSON says PASS, and both pinned-model control files are nonempty:

Smoke job 411 is retained failed evidence: the pinned weights loaded and returned finite BF16
vectors, but SentenceTransformers' built-in BF16 normalization exceeded the unit-norm tolerance.
The shared encoder now requests raw BF16 vectors and performs explicit FP32 L2 normalization. A
new smoke job from the repaired source must pass; job 411 must never be treated as the gate.

After commit `cb07cb5` or a descendant is cleanly checked out, the preferred Prompt 2 path is the
single fail-closed submission helper below. It verifies the model controls, governed inputs, count
baseline inputs, and clean Git state; submits the smoke job; makes pilot preparation depend on
smoke success; submits the remaining chain; and writes all IDs under
`outputs/private/submissions/` without overwriting an earlier record:

```bash
bash scripts/slurm/submit_prompt2_execution.sh
source outputs/private/prompt2_job_ids.env
squeue -j "${SMOKE_JOB_ID},${PREPARE_JOB_ID},${THROUGHPUT_JOB_ID},${EMBED_ARRAY_JOB_ID},${REPORT_JOB_ID}" \
  -o "%.18i %.12P %.26j %.2t %.10M %R"
```

The explicit commands below remain useful for diagnosis, but do not manually resubmit part of the
chain after the helper succeeds.

```bash
squeue -j "${SMOKE_JOB_ID}" -o "%.18i %.12P %.26j %.2t %.10M %R"
tail -n 100 "slurm-qwen3e-06b-smoke-${SMOKE_JOB_ID}.out"
SMOKE_REPORT="outputs/private/smoke/qwen_embedding_smoke-${SMOKE_JOB_ID}.json"
test -s "${SMOKE_REPORT}"
SMOKE_REPORT="${SMOKE_REPORT}" PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

report = json.loads(Path(os.environ["SMOKE_REPORT"]).read_text(encoding="utf-8"))
assert report["status"] == "PASS"
assert report["patient_data_used"] is False
assert report["local_files_only"] is True
assert report["model_repository"] == "Qwen/Qwen3-Embedding-0.6B"
for field in ("model_revision_hash", "model_artifact_hash", "tokenizer_hash"):
    assert len(report[field]) == 64
print("PASS: pinned offline 0.6B smoke test")
PY
test -s "${MODEL_DIR}/PINNED_REVISION.txt"
test -s "${MODEL_DIR}/MODEL_ARTIFACT_MANIFEST.json"
```

Submit the full dependency chain and capture every returned job ID:

```bash
CHAIN_OUTPUT="$(bash scripts/slurm/submit_pilot_chain.sh)"
printf '%s\n' "${CHAIN_OUTPUT}"
eval "${CHAIN_OUTPUT}"
printf '%s\n' "${CHAIN_OUTPUT}" > outputs/private/pilot_chain_job_ids.env
```

The helper submits this exact graph:

```text
12 prepare -> 13 throughput -> 14 embedding array (0-2%1) -> 15 fit/report
               afterok             afterok                    afterok
```

All four jobs use `partition=batch` and the NVIDIA container. Only throughput and embedding request
the site-supported `mps:1` shared-GPU resource. The three array indices map, in order, to
`lab_anemia`, `new_acutemi`, and `guo_los`; `%1` prevents simultaneous use of the single GB10 GPU.
At this point the jobs are scheduler-owned, so SSH can be exited and the VPN disconnected.

To monitor later, reconnect to VPN, SSH back in, restore the IDs, and inspect scheduler state and
logs. A log may not exist while a job is pending:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
export TEMPORAL_EHR_PROJECT_ROOT="$PWD"
export TEMPORAL_EHR_MODEL_ROOT="/data/shared73/projects/wdelery/model_cache/qwen3-embedding"
export EHRSHOT_ASSET_ROOT="/data/shared73/projects/wdelery/ehrshot-assets"
export EHRSHOT_DATA_PATH="${EHRSHOT_ASSET_ROOT}/data.parquet"
source outputs/private/pilot_chain_job_ids.env
squeue -j "${PREPARE_JOB_ID},${THROUGHPUT_JOB_ID},${EMBED_ARRAY_JOB_ID},${REPORT_JOB_ID}" \
  -o "%.18i %.12P %.26j %.2t %.10M %R"
for log in \
  "slurm-tehr-pilot-prepare-${PREPARE_JOB_ID}.out" \
  "slurm-tehr-pilot-throughput-${THROUGHPUT_JOB_ID}.out" \
  slurm-tehr-pilot-embed-${EMBED_ARRAY_JOB_ID}_*.out \
  "slurm-tehr-pilot-report-${REPORT_JOB_ID}.out"; do
  [[ -e "${log}" ]] && tail -n 80 "${log}"
done
```

If `sacct` is enabled, use it as additional evidence. This site has previously disabled Slurm
accounting, so an unavailable `sacct` is not itself failure; job-specific atomic sentinels, clean
logs, cache hashes, and semantic validators are the durable requirements.

After every job leaves `squeue`, verify every postcondition:

```bash
source outputs/private/pilot_chain_job_ids.env

for job_id in "${PREPARE_JOB_ID}" "${THROUGHPUT_JOB_ID}" "${REPORT_JOB_ID}"; do
  test -s "outputs/private/run_status/${job_id}/COMPLETE.json"
  test ! -e "outputs/private/run_status/${job_id}/FAILED.json"
done
for index in 0 1 2; do
  test -s "outputs/private/run_status/${EMBED_ARRAY_JOB_ID}_${index}/COMPLETE.json"
  test ! -e "outputs/private/run_status/${EMBED_ARRAY_JOB_ID}_${index}/FAILED.json"
done
for task in lab_anemia new_acutemi guo_los; do
  test -s "outputs/private/pilot/three_task_m1_engineering_pilot_v1/prepared/${task}/COMPLETE.json"
  test -s "outputs/private/pilot/three_task_m1_engineering_pilot_v1/embeddings/${task}/COMPLETE.json"
done

REPORT="reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json"
NOTEBOOK="outputs/private/executed_notebooks/04_pilot_results.ipynb"
test -s "${REPORT}"
test -s "${NOTEBOOK}"
PYTHONPATH=src .venv/bin/python scripts/review_pilot_results.py --report "${REPORT}"
PYTHONPATH=src .venv/bin/python scripts/validate_executed_notebook.py "${NOTEBOOK}"
```

Build a concise privacy-safe handoff containing only allowlisted aggregate metadata and hashes:

```bash
HANDOFF_ARGS=(
  --job-id "${PREPARE_JOB_ID}"
  --job-id "${THROUGHPUT_JOB_ID}"
  --job-id "${EMBED_ARRAY_JOB_ID}_0"
  --job-id "${EMBED_ARRAY_JOB_ID}_1"
  --job-id "${EMBED_ARRAY_JOB_ID}_2"
  --job-id "${REPORT_JOB_ID}"
  --output reports/generated/handoff/pilot_handoff.json
)
PYTHONPATH=src .venv/bin/python scripts/build_handoff_bundle.py "${HANDOFF_ARGS[@]}"
```

Copy only these aggregate files to the Mac:

```bash
exit
rsync -avP \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/handoff/pilot_handoff.json \
  "/Users/williamdelery/Downloads/"
```

Remain on VPN through the `rsync`, then disconnect. Do not copy the patient-window index, prepared
texts, predictions, embeddings, cache files, executed notebook, or private sentinels to the Mac.

## Prompt 3 locked four-task foundation and M1 validation

The source must first be reviewed, committed, and pushed from the Mac; the implementation task
itself intentionally leaves it uncommitted. Connect to MedNet VPN, then use a normal Mac terminal:

```bash
ssh wdelery@10.45.62.90
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
git status --short --branch
git pull --ff-only origin main

export TEMPORAL_EHR_PROJECT_ROOT="$PWD"
export TEMPORAL_EHR_MODEL_ROOT="/data/shared73/projects/wdelery/model_cache/qwen3-embedding"
export EHRSHOT_ASSET_ROOT="/data/shared73/projects/wdelery/ehrshot-assets"
export EHRSHOT_DATA_PATH="${EHRSHOT_ASSET_ROOT}/data.parquet"
```

The `git status` output must be clean and the pulled commit must contain Prompt 3. Verify governed
prerequisites without printing rows:

```bash
test -s "${EHRSHOT_DATA_PATH}"
test -d "${EHRSHOT_ASSET_ROOT}"
test -s reports/generated/handoff/project_handoff.json
test -s reports/generated/handoff/pilot_handoff.json
test -s reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json
for size in 0.6B 8B; do
  test -s "${TEMPORAL_EHR_MODEL_ROOT}/Qwen3-Embedding-${size}/PINNED_REVISION.txt"
  test -s "${TEMPORAL_EHR_MODEL_ROOT}/Qwen3-Embedding-${size}/MODEL_ARTIFACT_MANIFEST.json"
done
srun \
  --partition=batch \
  --job-name=tehr-prompt3-preflight \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:10:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data \
  --container-workdir="$PWD" \
  --export=ALL \
  env PYTHONPATH=src .venv/bin/python scripts/run_final_m1.py hard-gates
```

An old 8B directory containing only tokenizer files is not a model snapshot. If the 8B manifest is
missing, or the prior directory is only about 16 MiB, submit the resumable full snapshot download:

```bash
DOWNLOAD_8B_JOB_ID="$(sbatch --parsable --partition=batch --export=ALL scripts/slurm/04_download_qwen_8b.sbatch)"
DOWNLOAD_8B_JOB_ID="${DOWNLOAD_8B_JOB_ID%%;*}"
echo "DOWNLOAD_8B_JOB_ID=${DOWNLOAD_8B_JOB_ID}"
squeue -j "${DOWNLOAD_8B_JOB_ID}" -o "%.18i %.12P %.26j %.2t %.10M %R"
```

Once it runs, inspect `slurm-qwen3e-8b-download-${DOWNLOAD_8B_JOB_ID}.out`. If the DGX cannot reach
Hugging Face, exit SSH, disconnect VPN so the Mac has ordinary internet access, and run:

```bash
cd /Users/williamdelery/code/temporal-ehr-ml4h
source .venv/bin/activate
PYTHONPATH=src .venv/bin/python scripts/dgx/download_qwen_embeddings.py \
  --models 8B \
  --output-root /Users/williamdelery/Downloads/qwen3-embedding
```

Reconnect to MedNet VPN and transfer the complete directory. This is a large model transfer; keep
the Mac awake and let `rsync` resume if the VPN drops:

```bash
caffeinate -i rsync -avP --checksum \
  /Users/williamdelery/Downloads/qwen3-embedding/Qwen3-Embedding-8B/ \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/model_cache/qwen3-embedding/Qwen3-Embedding-8B/
```

Do not hand-edit or remove partial snapshot files. The downloader resumes into the directory and
rewrites a complete artifact manifest; the smoke job verifies every listed file and hash.

After the full model is ready, stay on VPN, SSH back in, restore the four exports above, and submit
the entire Prompt 3 graph through the one fail-closed helper:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/submit_prompt3_execution.sh
source outputs/private/prompt3_job_ids.env
squeue -j "${SMOKE_8B_JOB_ID},${FOUNDATION_JOB_ID},${THROUGHPUT_JOB_ID},${EMBED_ARRAY_JOB_ID},${FIT_JOB_ID}" \
  -o "%.18i %.12P %.28j %.2t %.10M %R"
```

The exact dependency graph is:

```text
06 8B smoke ----\
                 +--afterok--> 21 throughput --afterok--> 22 array 0-3%1 --afterok--> 23 fit/report
20 foundation --/
```

Jobs 20, 22, and 23 allow up to 12 hours; job 21 allows four hours. The array mapping is fixed as
0=`lab_anemia`, 1=`guo_readmission`, 2=`new_hypertension`, 3=`chexpert`; `%1` serializes GPU use.
Once all five IDs appear in `squeue` as pending or running with the expected dependencies, it is
safe to exit SSH and disconnect the VPN. Slurm continues independently.

Expected governed outputs are:

- `outputs/private/final/locked_four_task_m1_v1/cohort/final_panel_cohort_v1.parquet`;
- per-task `prepared/{task}/{patient_window_index,patient_window_metadata,texts}.parquet` and
  `COMPLETE.json`;
- final-only B3 artifacts under `outputs/private/final/locked_four_task_m1_v1/count_baseline/`;
- `foundation.json`, `throughput.json`, four `embedding/{task}.json` aggregates, and
  `final_m1_validation.json` under `reports/generated/final/locked_four_task_m1_v1/`;
- condition-level 0.6B and runtime-authorized 8B caches plus row-mirrored metadata under the final
  private namespace; and
- executed aggregate notebooks 05 and 06 under `outputs/private/executed_notebooks/`.

### Post-foundation throughput recovery

If job 437 is the validated foundation and job 438 is the preserved reserve-gate failure, do not
rerun the full graph. After the reviewed recovery commit is on the clean DGX checkout, submit only
the remaining stages:

```bash
export TEMPORAL_EHR_PROJECT_ROOT="$PWD"
export TEMPORAL_EHR_MODEL_ROOT="/data/shared73/projects/wdelery/model_cache/qwen3-embedding"
export SOURCE_FOUNDATION_JOB_ID=437
export SOURCE_SMOKE_8B_JOB_ID=436
export SOURCE_FAILED_THROUGHPUT_JOB_ID=438

bash scripts/slurm/submit_prompt3_after_foundation.sh
source outputs/private/prompt3_job_ids.env
squeue -j "${THROUGHPUT_JOB_ID},${EMBED_ARRAY_JOB_ID},${FIT_JOB_ID}" \
  -o "%.18i %.12P %.28j %.2t %.10M %.10l %R"
```

The helper first validates job 437's COMPLETE/exit-0 source identity, the exact prior configuration
hash, the sole reserve change, the aggregate semantics, and all private foundation hashes. It also
hash-records job 438's FAILED sentinel and refuses an existing throughput aggregate. Only then does
it submit a new throughput job, task array, and fit/report job.

### Fit-only recovery after job 446

This recovery is now complete. Job 451 ran at commit
`ef1b0debfa963ee365645ec2f754d8d129e071ce`, wrote COMPLETE/exit 0, preserved job 446's FAILED
sentinel, and passed the aggregate and notebook validators. The copied final M1 report SHA-256 is
`6d16d6b2494042fd2c84d25ef790f030355738cf8512fb87cf5477f73a3e0c7a`; the copied Prompt 3 handoff
SHA-256 is `2bd2223b13b9395ee7a725da9c260f1560c387db6cf5177f28f35999e0c56814`. The commands below remain
as the historical recovery and verification procedure; do not resubmit it.

Jobs 444 and 445_0 through 445_3 are COMPLETE. Job 446 failed before fitting, so no final prediction
table, final M1 report, or Notebook 06 output exists. After the fit-recovery source is committed,
pushed, and pulled into a clean DGX checkout, run exactly:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
git status --short --branch
git pull --ff-only origin main

export TEMPORAL_EHR_PROJECT_ROOT="$PWD"
export TEMPORAL_EHR_MODEL_ROOT="/data/shared73/projects/wdelery/model_cache/qwen3-embedding"

bash scripts/slurm/submit_prompt3_fit_recovery.sh
source outputs/private/prompt3_job_ids.env
squeue -j "${FIT_JOB_ID}" -o "%.18i %.12P %.28j %.2t %.10M %.10l %R"
```

The helper validates the exact experiment hash; job 437, 444, and 445 COMPLETE sentinels; all
upstream aggregate hashes; the unchanged job-446 FAILED hash; and the absence of partial fit
outputs. It runs package-dependent preflight inside the pinned container and submits only
`23_final_m1_fit_validation.sbatch`. It never schedules foundation, throughput, embedding, or
model inference. Once the new fit ID is pending or running, SSH may be exited and the VPN may be
disconnected.

After the fit leaves `squeue`, reconnect to VPN and verify it independently:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/prompt3_job_ids.env

tail -n 300 "slurm-tehr-final-m1-fit-${FIT_JOB_ID}.out"
test -s "outputs/private/run_status/${FIT_JOB_ID}/COMPLETE.json"
test ! -e "outputs/private/run_status/${FIT_JOB_ID}/FAILED.json"
test -s outputs/private/run_status/446/FAILED.json
test -s reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json
test -s outputs/private/executed_notebooks/06_final_m1_results.ipynb

SOURCE_FAILED_FIT_SENTINEL_SHA256="${SOURCE_FAILED_FIT_SENTINEL_SHA256}" \
SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}" \
FIT_JOB_ID="${FIT_JOB_ID}" \
PYTHONPATH=src .venv/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

complete = json.loads(
    Path(f"outputs/private/run_status/{os.environ['FIT_JOB_ID']}/COMPLETE.json").read_text()
)
assert complete["state"] == "COMPLETE"
assert complete["exit_status"] == 0
assert complete["git_commit"] == os.environ["SOURCE_GIT_COMMIT"]
assert complete["source_job_id"] == "446"
assert complete["source_failed_sentinel_sha256"] == os.environ[
    "SOURCE_FAILED_FIT_SENTINEL_SHA256"
]
assert digest(Path("outputs/private/run_status/446/FAILED.json")) == os.environ[
    "SOURCE_FAILED_FIT_SENTINEL_SHA256"
]
print("PASS: fit recovery COMPLETE and job 446 FAILED sentinel preserved")
PY

srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data --container-workdir="$PWD" \
  env PYTHONPATH=src .venv/bin/python scripts/review_final_m1_results.py \
    --report reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json \
    --private-postconditions
srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G --time=00:15:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data --container-workdir="$PWD" \
  env PYTHONPATH=src .venv/bin/python scripts/validate_executed_notebook.py \
    outputs/private/executed_notebooks/06_final_m1_results.ipynb
```

To monitor later, reconnect to VPN and SSH, then run:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/prompt3_job_ids.env
squeue -j "${SMOKE_8B_JOB_ID},${FOUNDATION_JOB_ID},${THROUGHPUT_JOB_ID},${EMBED_ARRAY_JOB_ID},${FIT_JOB_ID}" \
  -o "%.18i %.12P %.28j %.2t %.10M %R"
for log in \
  "slurm-qwen3e-8b-smoke-${SMOKE_8B_JOB_ID}.out" \
  "slurm-tehr-final-foundation-${FOUNDATION_JOB_ID}.out" \
  "slurm-tehr-final-throughput-${THROUGHPUT_JOB_ID}.out" \
  slurm-tehr-final-m1-embed-${EMBED_ARRAY_JOB_ID}_*.out \
  "slurm-tehr-final-m1-fit-${FIT_JOB_ID}.out"; do
  [[ -e "${log}" ]] && { echo "===== ${log}"; tail -n 100 "${log}"; }
done
```

Do not treat a dependency cancellation, an empty queue, or an existing report as success. After
all jobs leave `squeue`, run the complete postcondition gate:

```bash
source outputs/private/prompt3_job_ids.env
SMOKE_REPORT="outputs/private/smoke/qwen_embedding_smoke-${SMOKE_8B_JOB_ID}.json"
SMOKE_REPORT="${SMOKE_REPORT}" PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

report = json.loads(Path(os.environ["SMOKE_REPORT"]).read_text(encoding="utf-8"))
assert report["status"] == "PASS"
assert report["model_repository"] == "Qwen/Qwen3-Embedding-8B"
assert report["max_seq_length"] == 8192
assert report["patient_data_used"] is False
assert report["local_files_only"] is True
for field in ("model_revision_hash", "model_artifact_hash", "tokenizer_hash"):
    assert len(report[field]) == 64
print("PASS: complete pinned 8B synthetic smoke")
PY

for job_id in "${FOUNDATION_JOB_ID}" "${THROUGHPUT_JOB_ID}" "${FIT_JOB_ID}"; do
  test -s "outputs/private/run_status/${job_id}/COMPLETE.json"
  test ! -e "outputs/private/run_status/${job_id}/FAILED.json"
done
for index in 0 1 2 3; do
  test -s "outputs/private/run_status/${EMBED_ARRAY_JOB_ID}_${index}/COMPLETE.json"
  test ! -e "outputs/private/run_status/${EMBED_ARRAY_JOB_ID}_${index}/FAILED.json"
done
STATUS_IDS="${FOUNDATION_JOB_ID} ${THROUGHPUT_JOB_ID} ${EMBED_ARRAY_JOB_ID}_0 ${EMBED_ARRAY_JOB_ID}_1 ${EMBED_ARRAY_JOB_ID}_2 ${EMBED_ARRAY_JOB_ID}_3 ${FIT_JOB_ID}" \
SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

for job_id in os.environ["STATUS_IDS"].split():
    payload = json.loads(
        Path(f"outputs/private/run_status/{job_id}/COMPLETE.json").read_text(encoding="utf-8")
    )
    assert payload["state"] == "COMPLETE"
    assert payload["exit_status"] == 0
    assert payload["git_commit"] == os.environ["SOURCE_GIT_COMMIT"]
print("PASS: every Prompt 3 scientific sentinel is COMPLETE/exit-0 at the submitted commit")
PY

FOUNDATION=reports/generated/final/locked_four_task_m1_v1/foundation.json
THROUGHPUT=reports/generated/final/locked_four_task_m1_v1/throughput.json
RESULTS=reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json
test -s "${FOUNDATION}"
test -s "${THROUGHPUT}"
test -s "${RESULTS}"
test -s outputs/private/executed_notebooks/05_final_foundation_review.ipynb
test -s outputs/private/executed_notebooks/06_final_m1_results.ipynb

srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data \
  --container-workdir="$PWD" \
  env PYTHONPATH=src .venv/bin/python scripts/review_final_foundation.py \
    --report "${FOUNDATION}" --private-postconditions
srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data \
  --container-workdir="$PWD" \
  env PYTHONPATH=src .venv/bin/python scripts/review_final_m1_results.py \
    --report "${RESULTS}" --private-postconditions
```

The foundation reviewer checks the exact panel, one official instance per patient/task, official
splits/times, strict censoring, seven-field deduplication, outcome/shortcut audits, windows,
content matching, label-index-only perturbations, metadata alignment, and private hashes. The final
reviewer additionally checks the exact 0.6B/8B identities, throughput-only 2x gate, B1-B4/C0-C3,
three C3 seeds, per-task and equal-weight metrics, paired bootstrap evidence, coverage/runtime
evidence, pilot isolation, and the continued test/M2/M3 prohibition.

Build the privacy-safe handoff only after both reviewers pass:

```bash
HANDOFF_ARGS=(
  --job-id "${FOUNDATION_JOB_ID}"
  --job-id "${THROUGHPUT_JOB_ID}"
  --job-id "${EMBED_ARRAY_JOB_ID}_0"
  --job-id "${EMBED_ARRAY_JOB_ID}_1"
  --job-id "${EMBED_ARRAY_JOB_ID}_2"
  --job-id "${EMBED_ARRAY_JOB_ID}_3"
  --job-id "${FIT_JOB_ID}"
  --output reports/generated/handoff/prompt3_handoff.json
)
PYTHONPATH=src .venv/bin/python scripts/build_handoff_bundle.py "${HANDOFF_ARGS[@]}"
```

Copy only the three reviewed aggregates and the handoff back to the Mac. Remain on VPN until the
transfer completes:

```bash
exit
mkdir -p /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/final/locked_four_task_m1_v1
mkdir -p /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/handoff
rsync -avP \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/final/locked_four_task_m1_v1/foundation.json \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/final/locked_four_task_m1_v1/throughput.json \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json \
  /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/final/locked_four_task_m1_v1/
rsync -avP \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/handoff/prompt3_handoff.json \
  /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/handoff/
```

Do not copy cohorts, patient-window tables, prepared text, manifests, embeddings, predictions,
executed notebooks, caches, or private status files. Disconnect VPN after the aggregate transfer.

## Prompt 4 M2/M3 validation and protocol freeze

Prompt 4 has three governed stages: reference compute, human-gated reference finalization, and
selected-approach sensitivity/freeze. M1 already has its complete C0-C3 matrix and baselines. M2
summarizes the same content-matched 8,192-token source history as M1, so cross-approach differences
are not caused by giving M2 additional clinical evidence. All summaries use only
`http://10.45.177.17:8000/v1` and model `nvidia/Qwen3.6-35B-A3B-NVFP4`; no governed text may leave
the on-premise network.

Before pulling source or submitting anything, inspect the current DGX state from the repository:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
git rev-parse HEAD
git status --short --branch
squeue -u "$USER" -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
test -x scripts/slurm/prompt4_status.sh && bash scripts/slurm/prompt4_status.sh
```

For every active job, inspect its source commit and dependency without modifying it:

```bash
for job_id in $(squeue -h -u "$USER" -o '%A' | sort -u); do
  scontrol show job "$job_id" | egrep 'JobId=|JobName=|JobState=|Dependency=|Command=|WorkDir='
done
```

Do not pull, cancel, or resubmit while a governed Prompt 4 job from another commit is active. An
empty `squeue` is also not completion evidence; the helper requires atomic sentinels and lists an
ended job as `ENDED_WITHOUT_FINAL_SENTINEL`, `FAILED`, or `COMPLETE`.

One known exception is the explicitly superseded `a28eab1` graph 456/458/459/460, which predates
the required Prompt 4 alignment repairs and must not finish as scientific evidence. Preserve its
failure log, cancel its remaining scheduler graph, and delete nothing:

```bash
tail -n 250 slurm-tehr-final-m2-summary-456_0.out
cat outputs/private/run_status/456_0/FAILED.json 2>/dev/null || true
scancel 456 458 459 460
squeue -u "$USER" -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
```

Its immutable summary rows may be resumed because their keys contain source, condition, prompt,
served-model identity, and decoding identity. Aligned M2 embeddings use a full-source-commit
namespace and cannot collide with any partial embedding left by job 456.

After the Prompt 4 source commit is pushed, connect to MedNet VPN, SSH to the DGX, load the GitHub
key if needed, and run:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github_dgx
git pull --ff-only origin main
git status --short --branch

export TEMPORAL_EHR_PROJECT_ROOT="$PWD"
export TEMPORAL_EHR_MODEL_ROOT="/data/shared73/projects/wdelery/model_cache/qwen3-embedding"
export EHRSHOT_ASSET_ROOT="/data/shared73/projects/wdelery/ehrshot-assets"
export EHRSHOT_DATA_PATH="${EHRSHOT_ASSET_ROOT}/data.parquet"

curl -fsS http://10.45.177.17:8000/v1/models
bash scripts/slurm/submit_prompt4_capacity.sh
source outputs/private/prompt4_capacity_job_id.env
bash scripts/slurm/prompt4_status.sh
```

The capacity probe uses synthetic text only. When it finishes, require its COMPLETE sentinel and
inspect `reports/generated/engineering/prompt4_capacity.json`. The committed defaults use four
concurrent local-LLM requests and M3 embedding batch size 32. Change either only before reference
compute, based on the synthetic throughput and peak-memory evidence, then rerun local checks,
commit, and push. Do not tune these settings from validation outcome performance.

Array concurrency is operational and may be selected at submission without changing model inputs.
The safe defaults are shown below; leave them unchanged unless the capacity report and Slurm node
inventory justify a higher value:

```bash
export PROMPT4_M2_SUMMARY_PARALLELISM=2
export PROMPT4_M2_EMBED_PARALLELISM=1
export PROMPT4_M3_PREPARE_PARALLELISM=1
export PROMPT4_M3_TRAIN_PARALLELISM=2
```

The submission record preserves these values. Hard caps are respectively 2, 2, 2, and 3, so an
accidental setting cannot schedule an unbounded array fan-out.

Start reference compute:

```bash
bash scripts/slurm/submit_prompt4_execution.sh reference
source outputs/private/prompt4_reference_job_ids.env
squeue -j "${M2_ARRAY_JOB_ID},${M2_EMBED_ARRAY_JOB_ID},${M2_REVIEW_PREP_JOB_ID},${M3_PREP_ARRAY_JOB_ID},${M3_ARRAY_JOB_ID}" \
  -o "%.18i %.12P %.28j %.2t %.10M %.10l %R"
```

The M2 summary array uses no DGX GPU, runs two tasks concurrently, and issues four concurrent
requests per task to the separate vLLM server. Its dependent M2 embedding array requests the DGX
GPU. M3 first prepares each task's seed-independent source and embedding cache once, then launches
the 12 task/seed GRU jobs with two active by default. Seed jobs never reload or re-encode Qwen, and
they project retained windows before padding to reduce memory. This lets remote summary generation,
M3 preparation, M2 embedding, and small-GRU training overlap where Slurm resources permit without
writing the same cache concurrently. Override only the documented `PROMPT4_*_PARALLELISM`
variables after the synthetic capacity report; defaults are 2/1/1/2 for M2 summary, M2 embedding,
M3 preparation, and M3 training. Once queued, SSH and VPN may be disconnected.

After reference compute leaves `squeue`, reconnect and require its sentinels:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/prompt4_reference_job_ids.env

for index in 0 1 2 3; do
  test -s "outputs/private/run_status/${M2_ARRAY_JOB_ID}_${index}/COMPLETE.json"
  test ! -e "outputs/private/run_status/${M2_ARRAY_JOB_ID}_${index}/FAILED.json"
  test -s "outputs/private/run_status/${M2_EMBED_ARRAY_JOB_ID}_${index}/COMPLETE.json"
  test ! -e "outputs/private/run_status/${M2_EMBED_ARRAY_JOB_ID}_${index}/FAILED.json"
done
test -s "outputs/private/run_status/${M2_REVIEW_PREP_JOB_ID}/COMPLETE.json"
test ! -e "outputs/private/run_status/${M2_REVIEW_PREP_JOB_ID}/FAILED.json"
for index in 0 1 2 3; do
  test -s "outputs/private/run_status/${M3_PREP_ARRAY_JOB_ID}_${index}/COMPLETE.json"
  test ! -e "outputs/private/run_status/${M3_PREP_ARRAY_JOB_ID}_${index}/FAILED.json"
done
for index in $(seq 0 11); do
  test -s "outputs/private/run_status/${M3_ARRAY_JOB_ID}_${index}/COMPLETE.json"
  test ! -e "outputs/private/run_status/${M3_ARRAY_JOB_ID}_${index}/FAILED.json"
done
```

Open these two governed files in the remote VS Code window; never copy either off DGX:

```text
outputs/private/final/locked_four_task_m2_m3_v1/m2/fidelity/manual_review_packet.json
outputs/private/final/locked_four_task_m2_m3_v1/m2/fidelity/manual_review_decisions.json
```

For all 40 items, compare `source_text` with `summary`, set each decision field explicitly to
`true` or `false`, and finally set top-level `review_complete` to `true`. Do not add free-text patient
notes. A completed FAIL makes M2 ineligible but does not block selection between M1 and M3.

Then finalize the reference stage:

```bash
bash scripts/slurm/submit_prompt4_execution.sh reference-finalize
source outputs/private/prompt4_reference_finalize_job_ids.env
squeue -j "${M2_FIT_JOB_ID},${M3_FIT_JOB_ID}" \
  -o "%.18i %.12P %.28j %.2t %.10M %.10l %R"
```

When both jobs leave the queue, require:

```bash
source outputs/private/prompt4_reference_finalize_job_ids.env
test -s "outputs/private/run_status/${M2_FIT_JOB_ID}/COMPLETE.json"
test ! -e "outputs/private/run_status/${M2_FIT_JOB_ID}/FAILED.json"
test -s "outputs/private/run_status/${M3_FIT_JOB_ID}/COMPLETE.json"
test ! -e "outputs/private/run_status/${M3_FIT_JOB_ID}/FAILED.json"

test -s reports/generated/final/locked_four_task_m2_m3_v1/m2_reference.json
test -s reports/generated/final/locked_four_task_m2_m3_v1/m2_manual_fidelity_review.json
test -s reports/generated/final/locked_four_task_m2_m3_v1/m3_reference.json
test -s reports/generated/final/locked_four_task_m2_m3_v1/approach_selection.json
test -s outputs/private/executed_notebooks/08_m2_summary_validation.ipynb
test -s outputs/private/executed_notebooks/09_m3_sequence_validation.ipynb

srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data --container-workdir="$PWD" --export=ALL \
  env PYTHONPATH=src .venv/bin/python scripts/review_final_m23_results.py
```

Only after that validator passes, submit the selected sensitivity and freeze stage. If M2 or M3
is selected, its conditions include the latest-window baseline and all three C3 seeds required for
the two co-primary held-out contrasts, plus C0 and C2 secondary sensitivities:

```bash
bash scripts/slurm/submit_prompt4_execution.sh sensitivity
source outputs/private/prompt4_sensitivity_job_ids.env
PROMPT4_JOBS="${FREEZE_JOB_ID}"
[[ -n "${SENSITIVITY_FIT_JOB_ID}" ]] && PROMPT4_JOBS="${SENSITIVITY_FIT_JOB_ID},${PROMPT4_JOBS}"
[[ -n "${SENSITIVITY_ARRAY_JOB_ID}" ]] && PROMPT4_JOBS="${SENSITIVITY_ARRAY_JOB_ID},${PROMPT4_JOBS}"
[[ -n "${SENSITIVITY_M3_PREP_ARRAY_JOB_ID}" ]] && PROMPT4_JOBS="${SENSITIVITY_M3_PREP_ARRAY_JOB_ID},${PROMPT4_JOBS}"
[[ -n "${SENSITIVITY_SUMMARY_ARRAY_JOB_ID}" ]] && PROMPT4_JOBS="${SENSITIVITY_SUMMARY_ARRAY_JOB_ID},${PROMPT4_JOBS}"
squeue -j "${PROMPT4_JOBS}" \
  -o "%.18i %.12P %.28j %.2t %.10M %.10l %R"
```

If M1 is selected, the two sensitivity IDs are intentionally empty because its full matrix is
already complete; only job 28 is submitted. If M2 or M3 is selected, the matching conditional array
and aggregate job must complete before job 28 can run. An empty queue is not evidence of success.
After the graph leaves `squeue`, verify:

```bash
source outputs/private/prompt4_sensitivity_job_ids.env
test -s "outputs/private/run_status/${FREEZE_JOB_ID}/COMPLETE.json"
test ! -e "outputs/private/run_status/${FREEZE_JOB_ID}/FAILED.json"
test -s reports/generated/final/locked_four_task_m2_m3_v1/FROZEN_EXPERIMENT_MANIFEST.json
test -s reports/generated/handoff/prompt4_handoff.json

srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data --container-workdir="$PWD" --export=ALL \
  env PYTHONPATH=src .venv/bin/python scripts/review_final_m23_results.py --require-frozen
```

Pass gates are: all submitted jobs have COMPLETE/exit 0 and no FAILED sentinel; the automated and
manual M2 fidelity audits cover exactly 40 validation histories across eight task/condition strata;
M3 covers all
four tasks, its mean-pool comparator, and all three seeds; selection follows the 0.005 macro-AUROC
rule; the selected approach has latest-window and three C3 conditions (or M1's existing results are
cited); the
frozen manifest hash validates; and every report states no held-out use. Copy back only
`m2_reference.json`, `m2_manual_fidelity_review.json`, `m3_reference.json`,
`approach_selection.json`, the selected sensitivity JSON if one exists,
`FROZEN_EXPERIMENT_MANIFEST.json`, and `reports/generated/handoff/prompt4_handoff.json`.
Never copy summaries, sequence rows, private predictions, embeddings, model weights, or sentinels.
