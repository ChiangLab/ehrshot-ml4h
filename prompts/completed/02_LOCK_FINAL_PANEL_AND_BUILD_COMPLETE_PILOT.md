Prompt 2 — Lock the final panel and build the complete pilot
Before running this prompt, copy the validated aggregate Phase 2 report back to the Mac.
Work in the current temporal-ehr-ml4h repository.

Read AGENTS.md, docs/CURRENT_SCIENTIFIC_PROTOCOL.md, docs/PROJECT_STATE.md, docs/DECISION_LOG.md, docs/DGX_BATCH_RUNBOOK.md, docs/NEXT_ACTION.md, the validated Phase 2 aggregate report, configs/tasks.yaml, and all existing experiment configurations before editing.

Goal: build the complete three-task engineering pilot while prospectively freezing the separate four-task final panel. The pilot must not be used to select or alter the final clinical tasks.

Scientific decisions authorized on 2026-08-21:

Engineering pilot:
- lab_anemia
- new_acutemi
- guo_los
- status remains PILOT
- Qwen3-Embedding-0.6B
- 4,096-token budget
- M1 direct serialization only
- train and validation only

Final submission panel, selected before pilot embedding results:
- lab_anemia
- guo_readmission
- new_hypertension
- chexpert

Update configs/experiments/locked_panel.yaml to freeze exactly those four final tasks. Record that the panel was prespecified before pilot embedding results and was informed by Jeff’s recommendation to span clinically different task families. Do not claim that Jeff explicitly approved wording he did not approve.

The pilot and final artifacts must remain versioned and separate. A pilot cohort or cache must never satisfy a final-panel gate.

Required pilot experiment:

Baselines:
- B4 training-prevalence baseline
- B3 governed code-count baseline
- B1 latest eligible task-family window
- B2 latest eligible occurrence of every code
- M1 C1 complete correctly timed history

Temporal conditions on the complete history:
- C0 order only
- C1 correct prediction-relative time
- C2 reversed earlier inter-window gaps, monotonic, same gap multiset/span, final label correct
- C3 shuffled earlier time-label indices, clinical blocks unchanged, final label correct
- C3 seeds: 20260815, 20260816, 20260817

All C0–C3 comparisons must use:
- identical patients and labels;
- identical source clinical events;
- identical event and window order;
- identical retained content under the 4,096-token budget;
- frozen Qwen encoder;
- identical logistic-regression procedure.

Implement Jeff’s patient-by-window index recommendation.

Create a private indexed table with at least:
- experiment identifier;
- instance key;
- private subject identifier;
- task and split;
- prediction time and target;
- task-family window width;
- source window index and bounds;
- event count;
- deduplicated clinical-content hash;
- correct time-label index;
- condition-specific displayed label index;
- perturbation seed;
- retained/omitted status under packing;
- tokenizer/model/instruction/budget hashes;
- embedding row index and cache key.

Requirements:
- private Parquet only;
- Git-ignored;
- deterministic sort order;
- unique primary key;
- one-to-one alignment assertions against cohort, texts, embeddings, labels, and predictions;
- public aggregate summaries contain no identifiers or timestamps.

The implementation should use reusable modules and CLIs. Notebooks must be thin aggregate review layers, not the only implementation.

Create:
- notebooks/04_pilot_results.ipynb
- scripts/slurm/12_pilot_prepare.sbatch
- scripts/slurm/13_pilot_throughput.sbatch
- scripts/slurm/14_pilot_embed_task_array.sbatch
- scripts/slurm/15_pilot_fit_report.sbatch

Batch design:
- use partition=batch;
- use the existing NVIDIA container;
- all Python executes inside the container;
- GPU jobs request the site-supported shared GPU resource;
- array index maps deterministically to the three pilot tasks;
- limit concurrency appropriately for one physical GB10 GPU;
- prepare, throughput, embedding, and report jobs use afterok dependencies;
- each job and array element has atomic RUNNING/COMPLETE/FAILED sentinels;
- every patient-level cache is resumable and fails closed on partial completion;
- aggregate report generation starts only after all task jobs complete;
- no internet/model download occurs during scientific jobs;
- model artifact and tokenizer revision are pinned and hashed.

Pilot reporting:
- train/validation sample counts and prevalence;
- AUROC, AUPRC, Brier score, log loss;
- calibration summaries where stable;
- paired patient bootstrap intervals for central contrasts;
- C3 seed mean and variability;
- eligible/perturbable history counts;
- token lengths, truncation, retained windows/events, historical span;
- content-matching audit;
- throughput, peak memory, failures and retries;
- no test predictions or performance.

Add tests for:
- explicit pilot-mode authorization without weakening the final locked-panel gate;
- complete isolation of pilot and final artifacts;
- patient-window index uniqueness/alignment;
- no future events;
- no test split;
- C0–C3 content identity;
- final label invariants;
- all three C3 seeds;
- latest-window/latest-code semantics;
- resumable caches;
- interrupted array jobs;
- deterministic embeddings-to-label joins;
- privacy-safe aggregate reports;
- notebook cleanliness;
- Slurm sentinel and dependency behavior.

Reuse existing code where correct. Do not duplicate serializers or metrics.

Do not implement M2, M3, the final four-task cohort, 8B confirmation, or test evaluation in this task.

Run all local synthetic tests, Ruff, shell syntax checks, and the pre-push privacy check. Do not run real EHRSHOT or models locally. Do not commit or push.

Finish with:
- exact files created/changed;
- exact pilot scientific scope;
- test results;
- exact 0.6B download/smoke commands if the model is absent;
- exact batch submission chain with captured job IDs and afterok dependencies;
- exact monitoring and completion-verification commands;
- exact aggregate handoff bundle to copy to the Mac;
- the next prompt filename.
