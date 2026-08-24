Prompt 3 — Build the true four-task foundation and M1 experiment
Work in the current temporal-ehr-ml4h repository.

Read AGENTS.md and every current context/state/decision document. Read the completed pilot aggregate report, but treat it only as engineering evidence. Do not change the final task panel, windows, conditions, model family, or clinical definitions in response to pilot AUROCs.

Hard preconditions:
- Phase 2 recovered validation is COMPLETE.
- The complete pilot is COMPLETE.
- configs/experiments/locked_panel.yaml contains exactly:
  lab_anemia
  guo_readmission
  new_hypertension
  chexpert
- The final panel was frozen before pilot embedding performance was used.
- Test performance remains untouched.

Goal: build the final four-task patient-balanced foundation and run the M1 validation experiment.

Final cohort:
- at most one official label instance per patient and task;
- deterministic selection independent of outcome, event history, record length, and model result;
- selection hash includes dataset/release version, task, split, patient, frozen seed, and sorted eligible prediction-time index;
- preserve official patient splits;
- use official task prediction times;
- exclude every event at or after T under the project’s conservative strict rule;
- exact seven-field duplicate removal;
- 30-day windows for lab, imaging, and operational tasks;
- 90-day windows for incident diagnosis;
- create a new versioned final-panel cohort and patient-window index;
- never overwrite or silently reuse pilot artifacts.

Add outcome-definition and shortcut audits:
- anemia: confirm prediction is the next qualifying hemoglobin result, not an incident anemia diagnosis;
- readmission: confirm the 30-day outcome horizon and exclude future admission evidence;
- hypertension: confirm the official new-diagnosis definition and one-year horizon;
- CheXpert: confirm prediction occurs before the radiology report/finding;
- report prior related predictors as legitimate pre-T information, while checking that target-defining future events are excluded.

Primary final M1 experiment:
- Qwen3-Embedding-0.6B;
- 8,192-token context;
- full embedding dimensionality and L2 normalization;
- frozen encoder;
- task-specific penalized logistic regression;
- B1 latest window;
- B2 latest code;
- B3 count baseline;
- B4 prevalence;
- C0 order only;
- C1 correct time;
- C2 reversed earlier gaps;
- C3 three shuffled-label seeds;
- content-matched C0–C3 packing;
- train/validation only.

Throughput gate:
- benchmark 0.6B at 8,192;
- benchmark the already-downloaded Qwen3-Embedding-8B at the prespecified confirmatory cells;
- use runtime/memory only, never outcome performance, for feasibility;
- retain a 2x time contingency.

Prespecified 8B confirmatory priority:
1. M1 C1 complete correct history
2. M1 C3 shuffled earlier labels
3. latest-window baseline
4. latest-code baseline if feasible

Create thin aggregate notebooks and reusable modules, plus:
- scripts/slurm/20_final_foundation.sbatch
- scripts/slurm/21_final_throughput.sbatch
- scripts/slurm/22_final_m1_embed_task_array.sbatch
- scripts/slurm/23_final_m1_fit_validation.sbatch

Use task-level array jobs with deterministic mappings and resumable condition-level caches. Do not reload/re-encode an input when an immutable verified cache already exists. Do not allow pilot caches to satisfy final cache keys.

Report:
- per-task and equal-weight panel macro-AUROC;
- AUPRC, Brier, log loss and calibration;
- paired patient bootstrap confidence intervals;
- C1-C0, C1-C2, C1-C3;
- C1-latest-window;
- C1-latest-code;
- perturbable-subset and complete-cohort estimates;
- C3 seed variability;
- history coverage, recurrence, truncation and perturbation strength;
- runtime, peak memory and storage;
- no test results.

The patient-window metadata table must mirror every embedding row and condition. Perturb C2/C3 by changing label indices only; never rearrange clinical event arrays.

Implement full automated tests and batch sentinels. Run local synthetic tests, Ruff, shell syntax, and privacy checks. Do not run real data or models locally. Do not commit or push.

Do not implement M2, M3, or held-out testing in this task.

Finish with exact batch submission commands, dependencies, expected outputs, pass/fail gates, and a privacy-safe handoff bundle.
