Prompt 5 — One-time test evaluation and paper artifacts

Completion note (2026-09-08): the frozen one-time M1 held-out evaluation and publication bundle
completed successfully. This prompt is retained as the immutable execution specification.

Authorized scope note (2026-09-07): M1 direct serialization is the primary one-time held-out
approach. M3 is reported as secondary validation-only evidence. M2 is incomplete future work and is
not a held-out or submission gate. The immutable repository authorization is
`configs/experiments/final_m1_submission.yaml`; all remaining instructions apply within that frozen
scope.
Work in temporal-ehr-ml4h.

This task is authorized only if:
- the final four-task cohort is locked;
- every validation-stage decision is frozen;
- the experiment manifest is complete and hashed;
- M1 C0–C3 and required baselines are complete;
- the selected M2/M3 scope is documented;
- no uncommitted scientific code exists;
- no held-out performance has previously been inspected.

If any gate fails, stop without reading test predictions.

Implement the one-time held-out evaluation exactly as frozen.

Required analyses:
- per-task AUROC;
- equal-weight four-task panel macro-AUROC;
- AUPRC, Brier score, log loss;
- calibration intercept/slope where stable;
- 2,000 paired patient-clustered bootstrap replicates;
- identical patient resamples across compared conditions;
- co-primary contrasts:
  complete correct history minus latest window
  complete correct history minus shuffled earlier timestamps
- Holm correction for the two co-primary tests;
- Benjamini-Hochberg correction for the prespecified per-task secondary family;
- C3 seed aggregation and variability;
- M3 seed aggregation and variability;
- complete-cohort and perturbable-subset results;
- runtime, coverage and perturbation diagnostics.

Never modify the protocol after seeing held-out results. A null or negative result must be reported as observed.

Create publication-ready aggregate artifacts:
- cohort/task table;
- main performance table;
- paired contrast table;
- confidence-interval forest plot;
- perturbation-strength figure;
- history-coverage figure;
- calibration figure where justified;
- CONSORT-like data/experiment flow diagram;
- machine-readable aggregate results;
- reproducibility manifest;
- methods and results Markdown draft;
- limitations and permitted-claims checklist.

No patient-level text, identifiers, timestamps, embeddings, labels, predictions, or private metadata may enter Git, external services, figures, or manuscripts.

Use a single batch job or dependency-locked workflow with atomic sentinels. Validate every artifact hash and row count before declaring COMPLETE.

Finish with:
- exact frozen Git commit;
- exact Slurm job IDs;
- exact test-access audit;
- primary results;
- all deviations;
- permissible conclusions;
- files ready for the ML4H submission.
