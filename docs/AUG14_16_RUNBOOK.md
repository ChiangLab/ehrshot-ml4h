# August 14–16 execution runbook

## What uses real versus synthetic data

- **Real downloaded EHRSHOT:** asset audit, label validation, cohort selection, censoring audit, all
  later serialization, embeddings, training, and scientific results.
- **Synthetic fixtures:** unit tests that must be safe to commit and run anywhere.
- **Synthetic text:** the first GPU embedding smoke test only. Its purpose is to prove that the
  model and CUDA stack work before governed data are loaded.

The synthetic fixtures never substitute for EHRSHOT results.

## August 14: finish the real-data foundation off VPN

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export EHRSHOT_DATA_PATH=../data/data.parquet
export EHRSHOT_ASSET_ROOT=../data

bash scripts/run_real_ehrshot_setup.sh
```

This runs the existing event audit against the real 41.7-million-event file, validates all real
task labels/splits, and creates the patient-balanced primary panel. The cohort file is a governed
patient-level derivative at `outputs/private/cohorts/primary_panel.parquet`; Git ignores it.

Review, without opening patient rows:

```bash
sed -n '1,220p' reports/generated/label_validation/label_validation.md
sed -n '1,220p' reports/generated/cohort/cohort_summary.md
bash scripts/git/pre_push_safety.sh
git status --short
```

## Move code through MedNet GitHub

While disconnected from VPN, use Codex only with source code, schemas, synthetic tests, and
aggregate output. Never paste a patient row, serialized patient history, label-level file, or
embedding into an OpenAI or Anthropic prompt.

Before connecting to VPN, finish edits and close the external AI session. On VPN, create the empty
private MedNet GitHub repository for account `wdelery`, then use the exact URL shown by that service:

```bash
export MEDNET_GIT_EMAIL='YOUR_VERIFIED_MEDNET_GITHUB_EMAIL'
bash scripts/git/configure_mednet_remote.sh 'EXACT_MEDNET_REPOSITORY_URL'
git add .
git diff --cached --stat
git diff --cached
bash scripts/git/pre_push_safety.sh
git commit -m 'Add EHRSHOT validation and DGX bootstrap workflow'
git branch -M main
git push -u origin main
```

Account ID `123689487` is informational; it is not a password, token, Git author email, or remote
URL. Do not put a personal access token in `.env`, a command argument, or a committed file.

## August 15: clone and validate the DGX environment

On VPN, clone the private repository into the project allocation and enter it:

```bash
cd /data/shared73/projects/wdelery
git clone 'EXACT_MEDNET_REPOSITORY_URL' temporal-ehr-ml4h
cd temporal-ehr-ml4h

export TEMPORAL_EHR_PROJECT_ROOT=/data/shared73/projects/wdelery/temporal-ehr-ml4h
export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
export EHRSHOT_DATA_PATH='DGX_PATH_TO_REAL_EHRSHOT/data.parquet'
export EHRSHOT_ASSET_ROOT='DGX_PATH_TO_REAL_EHRSHOT_ASSET_DIRECTORY'
export LOCAL_LLM_BASE_URL=http://10.45.177.17:8000/v1

sbatch --export=ALL scripts/slurm/01_bootstrap_validate.sbatch
squeue -u "$USER"
```

Do not call the 32B chat model an embedding model. Querying `/v1/models` records its actual served
ID, but the scientific encoder must be the separately pinned Qwen3-Embedding checkpoint.

When the bootstrap job passes, download and test the small embedding checkpoint:

```bash
sbatch --export=ALL scripts/slurm/02_download_qwen_06b.sbatch
# After that job succeeds:
sbatch --export=ALL scripts/slurm/03_smoke_qwen_06b.sbatch
```

## August 16: validate censoring and stage the primary 8B encoder

If the governed cohort was built on the Mac, transfer it through an approved governed mechanism;
do **not** put it in Git. Alternatively, rebuild it on the DGX with the same command and seed:

```bash
.venv/bin/python -m temporal_ehr.cli build-cohort \
  --asset-root "${EHRSHOT_ASSET_ROOT}" \
  --tasks lab_anemia new_acutemi guo_los \
  --selection-seed 20260814 \
  --private-output outputs/private/cohorts/primary_panel.parquet \
  --allow-private-output
```

Run the strict leakage audit and stage the primary model:

```bash
export EHRSHOT_COHORT_PATH="${TEMPORAL_EHR_PROJECT_ROOT}/outputs/private/cohorts/primary_panel.parquet"
sbatch --export=ALL scripts/slurm/05_validate_censoring.sbatch
sbatch --export=ALL scripts/slurm/04_download_qwen_8b.sbatch
# Only after the download completes:
sbatch --export=ALL scripts/slurm/06_smoke_qwen_8b.sbatch
```

The censoring rule is `event_time < prediction_time`, not `<=`. The real label audit found many
events at exact prediction timestamps, so strict inequality is the conservative leakage-safe rule.

## Completion criteria for August 16

- All 15 real label files and canonical splits pass core validation.
- The three primary tasks and their binary transformations are frozen.
- One label row per patient/task is chosen with the fixed, outcome-independent seed.
- Strict prediction-time censoring produces no post-prediction inputs.
- Qwen3-Embedding-0.6B loads and embeds invented text on CUDA.
- Qwen3-Embedding-8B is downloaded, revision-recorded, and preferably smoke-tested.
- No governed data, patient-level derivatives, embeddings, model weights, logs containing patient
  content, or credentials are tracked by Git.
