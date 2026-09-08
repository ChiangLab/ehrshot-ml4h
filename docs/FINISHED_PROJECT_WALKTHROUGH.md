# Finished project walkthrough

Status: scientifically complete through the frozen one-time M1 held-out evaluation on 2026-09-08. Administrative submission work remains: authorship, institutional ethics language, code availability, and final ML4H template checks.

## 1. The project in one sentence

This project asks whether a frozen text embedding of a longitudinal EHR actually uses the correct timing of earlier clinical events, or whether its predictions mostly reflect event content, order, recency, and frequency.

The careful conclusion is not “dates never help.” It is: **in this frozen, flat, text-serialized EHR setup, correct compact prediction-relative time labels did not reliably improve panel-level prediction.** A small anemia-specific effect suggests that the answer can be task dependent.

## 2. Motivation and the literature gap

The project grew out of an apparent tension in the literature.

- **EHRSHOT** established a public, de-identified benchmark with 6,739 patients, 15 tasks, four clinical task families, and canonical patient splits. It made it possible to compare EHR foundation-model representations under a common benchmark, but it did not itself isolate whether models understand longitudinal time (Wornow et al., 2023).
- **Large Language Models Are Powerful Electronic Health Record Encoders** showed that general-purpose language-model embeddings, especially Qwen3-Embedding-8B with simple logistic readouts, can be competitive on EHRSHOT. Its default reduction retained the latest occurrence of each code, and simple timestamp additions did not improve aggregate performance (Hegselmann et al., 2026). That result did not reveal whether time was intrinsically unhelpful, whether the encoder ignored the chosen timestamp representation, or whether deduplication and finite context removed the recurrence and event–time alignment needed to show a benefit.
- **Context Clues** found that longer context can help on many EHRSHOT tasks, while repetition, irregular intervals, and disease complexity are associated with poorer performance; artificial time tokens slightly reduced performance in its setting (Wornow et al., 2025). This motivated preserving recurrence and measuring truncation rather than treating every long history as equally informative.
- **CEHR-BERT** showed that an EHR-specific model can benefit from learned temporal components and artificial time tokens (Pang et al., 2021). This suggested that the usefulness of time may depend on how it is represented and learned, not merely on whether a date string is present.
- **Temporal Tokenization Strategies for Event Sequence Modeling with Large Language Models** and **Rethinking Tokenization for Clinical Time Series** report task- and distribution-dependent effects of temporal/value tokenization rather than one universally best strategy (Liu et al., 2025; Al Attrach et al., 2025). They reinforce the need to test representations rather than assume their semantics are used.
- **PaReGTA** combines visit-level templated text, contrastively adapted sentence embeddings, and explicit temporal pooling. Its results motivate structured temporal aggregation as an alternative to a single flat document, although its cohort and architecture differ from this project (Yoon et al., 2026).
- **MEDS** supplies the portable event-data framework used by the project and supports reproducible transformations across health-AI workflows (McDermott et al., 2026).
- **RAPTOR** inspired the general idea that abstraction can make long contexts more manageable, but M2 is not RAPTOR: it uses one summary pass and no recursive tree or retrieval system (Sarthi et al., 2024).
- **Time After Time** studies the causal effect of changing when an intervention occurs. It is included to mark a boundary: this project evaluates prediction from observed histories and does not estimate causal intervention-timing effects (Wald et al., 2025).

The gap Jeff and William identified was therefore methodological: a null result from “adding dates” does not distinguish whether earlier history helps, whether elapsed time helps beyond order, or whether correct event–time correspondence matters. Those questions require content-matched falsification controls, not only a dated versus undated comparison.

## 3. What was predicted

The four tasks were prospectively locked before pilot embedding performance was used. They cover the four EHRSHOT task families and different kinds of temporal dependence.

| Task | Target | Official prediction point | Window | Why it was chosen |
|---|---|---|---:|---|
| `lab_anemia` | Whether the next qualifying hemoglobin result is abnormal | Before the next qualifying lab result | 30 days | A recurrent, dynamic laboratory process in which recent values and trends should plausibly matter |
| `guo_readmission` | Readmission within 30 days | Official index-admission prediction time | 30 days | A rare operational outcome influenced by recent acuity and care transitions |
| `new_hypertension` | First hypertension diagnosis within one year | Official post-discharge prediction time | 90 days | An incident diagnosis with a longer horizon and potentially slower accumulation of risk evidence |
| `chexpert` | Chest radiograph has a finding rather than “No Finding” | 24 hours before the target radiology report | 30 days | An imaging task that tests whether the structured pre-report record anticipates a near-future finding |

Important clinical interpretation details:

- Anemia is a next-result task, not prediction of a new anemia diagnosis.
- Readmission excludes evidence from the future admission.
- Hypertension follows the official incident-diagnosis definition and one-year horizon.
- CheXpert excludes the target radiology report/finding at or after the prediction time. Its input can therefore look like general diagnoses, measurements, and procedures rather than report prose; that is expected and prevents leakage.

The final patient-balanced cohort contains at most one official instance per patient and task:

| Task | Train n (% positive) | Validation n (% positive) | Test n (% positive) |
|---|---:|---:|---:|
| Lab anemia | 2,092 (47.3%) | 1,991 (45.8%) | 2,002 (46.4%) |
| Readmission | 1,337 (5.8%) | 1,191 (5.4%) | 1,190 (5.3%) |
| New hypertension | 792 (14.5%) | 781 (15.4%) | 755 (15.9%) |
| CheXpert | 250 (64.4%) | 395 (66.6%) | 396 (67.4%) |
| Total | 4,471 | 4,358 | 4,343 |

## 4. Cohort construction and leakage protection

Official EHRSHOT patient splits and prediction times were preserved. For each task and patient, duplicate official prediction-time groups were excluded before selection, regardless of label agreement. One remaining eligible time was selected with an outcome-independent SHA-256 rank incorporating the dataset release, task, split, patient, frozen seed, and sorted eligible-time index. The selection rule never used the target, event history, record length, row order, or model result.

Only clinical events with `event_time < T` were retained under the conservative strict censoring rule. Exact duplicates were removed only when all seven fields matched: patient, event time, code, numeric value, text value, unit, and source table. Recurrence was otherwise preserved. Events were assigned to half-open prediction-relative windows: 30 days for anemia, readmission, and CheXpert, and 90 days for hypertension.

The most recent suffix that fit the token budget was retained. Packing was content matched across C0–C3 using the most token-expensive condition, so a condition could not win simply because shorter time labels allowed more clinical evidence into the model. The patient-window metadata table was required to mirror every embedding row.

## 5. The B baselines

“B” means a baseline that removes information or model complexity.

- **B1 — latest window:** only the most recent nonempty prediction-relative window. It asks whether older history adds value beyond the recent record.
- **B2 — latest code:** the most recent occurrence of each distinct code anywhere in the retained history. It approximates the reduction used by the closest predecessor and asks what recurrence contributes.
- **B3 — code count:** a sparse vector of `log1p` code frequencies with task-specific penalized logistic regression. It measures whether simple content and recurrence counts outperform dense text embeddings.
- **B4 — training prevalence:** a constant probability equal to training prevalence. It is the no-information reference and has AUROC 0.5.

B1 and B2 are separate baselines. Neither is a temporal falsification.

## 6. The C conditions

“C” means a controlled temporal representation of the same clinical history.

- **C0 — order only:** clinical events remain chronological, but no elapsed-time labels are supplied.
- **C1 — correct time:** compact labels give the correct window relative to prediction time, such as `[TIME=31_60_DAYS_BEFORE_T]`.
- **C2 — reversed earlier gaps:** the earlier inter-window gaps are assigned in reverse order while chronology, total span, gap multiset, and the final window remain correct. It is a monotonic sensitivity test.
- **C3 — shuffled earlier labels:** earlier time-label indices are shuffled with seeds 20260815, 20260816, and 20260817; the final window remains correct. It is the strongest direct falsification of earlier event–time correspondence.

C2 and C3 change only label indices. They never rearrange clinical event arrays. Histories whose labels could not actually be perturbed were excluded from the relevant perturbable-subset contrast.

A simplified example is:

```text
C0: prior hemoglobin 10.4 → iron study → recent hemoglobin 11.1
C1: [91–120 days] hemoglobin 10.4 → [31–60 days] iron study → [1–30 days] hemoglobin 11.1
C2: [31–60 days] hemoglobin 10.4 → [91–120 days] iron study → [1–30 days] hemoglobin 11.1
C3: shuffled earlier labels for hemoglobin and iron study → [1–30 days] hemoglobin 11.1
```

The clinical content and order are unchanged. Only the earlier mapping from events to elapsed-time labels differs.

## 7. The M interfaces

“M” means a representation/model interface, not a clinical outcome.

### M1 — direct serialized-history embedding

M1 is the completed primary experiment. The retained history is serialized into one flat document. Frozen Qwen3-Embedding-0.6B encodes up to 8,192 tokens into the full 1,024-dimensional embedding. Inference uses the model-supported reduced precision, but finite vectors are promoted to FP32 and explicitly L2-normalized before immutable caching. No encoder weights are updated.

Each task and representation receives its own penalized logistic-regression readout. Regularization is selected on validation AUROC from `C ∈ {0.01, 0.1, 1, 10}`, with the smallest C winning an exact tie. The readout is fit on training patients only; it is not refit on train plus validation before test.

Qwen3-Embedding-0.6B was used for the full matrix because a runtime-only throughput gate showed it was feasible within the DGX/Slurm limits. The already-downloaded 8B model was used only for the prespecified feasible confirmation cells, in the priority order C1, C3, B1, B2. That partial set cannot support a four-task 8B macro estimate and was never used to redesign the study.

### M2 — one-pass structured-summary embedding

M2 was intended to send each content-matched history to the on-premise `nvidia/Qwen3.6-35B-A3B-NVFP4` endpoint using deterministic decoding, require a bounded structured summary that preserved codes, values, units, recurrence, and exact time labels, and then pass the summary through the same embedding/logistic pipeline.

Several versioned runs exposed fidelity and runtime failures: null summaries, repetitive outputs, abrupt truncation, and inability to complete the required matrix under the scheduling deadline. Safeguards were progressively strengthened, but the interface did not pass the prespecified gates. M2 was therefore frozen as incomplete future work before held-out access. Partial caches are not results, and M2 never accessed the test outcomes.

### M3 — per-window sequence aggregation

M3 embeds up to the 64 most recent nonempty windows separately with the frozen 0.6B model, using at most 512 tokens per window. A fixed random projection (seed 20260819) maps each 1,024-dimensional embedding to 256 dimensions. A one-layer, unidirectional GRU with 128 hidden units and dropout 0.1 aggregates the ordered window sequence. It uses AdamW (`lr=10^-3`, weight decay 0.01, batch size 32), up to 50 epochs, and early stopping on validation AUROC with patience 5; validation log loss breaks AUROC ties. Training is repeated with seeds 20260815–17. Mean pooling plus logistic regression is the capacity comparator.

There is **no graph neural network** in this project. The learned temporal architecture is a GRU. The only graph-like object is the Slurm dependency DAG that staged data preparation, embedding, fitting, validation, and publication jobs.

M3 completed validation at macro-AUROC 0.622 versus 0.694 for M1. It therefore remained secondary validation evidence and never accessed the test set.

## 8. Statistical analysis and why it was used

- **AUROC** is the primary discrimination measure and is relatively interpretable across tasks with different prevalence.
- **AUPRC** is included because readmission and hypertension are imbalanced; it must be interpreted relative to outcome prevalence.
- **Brier score** measures squared probabilistic error and reflects discrimination and calibration.
- **Log loss** more heavily penalizes confident wrong predictions.
- **Calibration intercept and slope** evaluate whether predicted probabilities are systematically shifted or too extreme/too narrow. They are descriptive here, not evidence of clinical readiness.
- **Equal-weight macro averages** give each clinical task one-quarter weight, preventing the larger anemia cohort from dominating the panel result.
- **Paired patient-cluster bootstrap confidence intervals** use 2,000 shared resamples within task. Pairing preserves the fact that two representations predict the same patients, while patient clustering respects the sampling unit. The same resamples are used across estimands to stabilize comparisons.
- **Holm correction** controls family-wise error for the two co-primary panel contrasts, C1–B1 and C1–the mean C3 prediction.
- **Benjamini–Hochberg correction** controls false discovery rate over prespecified per-task secondary contrasts.

The 0.01 macro-AUROC threshold used during approach selection was a prespecified descriptive materiality threshold, not a new hypothesis test. Validation chose the interface and hyperparameters; the held-out set was then accessed once under a cryptographically frozen manifest, with no subsequent design change.

## 9. Main findings

### M1 validation

| Task | C0 | C1 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|
| Lab anemia | 0.813 | 0.823 | 0.820 | 0.789 | 0.828 |
| Readmission | 0.713 | 0.712 | 0.715 | 0.632 | 0.673 |
| New hypertension | 0.696 | 0.693 | 0.680 | 0.734 | 0.724 |
| CheXpert | 0.541 | 0.546 | 0.568 | 0.555 | 0.584 |
| Equal-weight macro | 0.690 | 0.694 | 0.696 | 0.678 | 0.702 |

### Frozen one-time held-out C1 performance

| Task / panel | n | AUROC (95% CI) | AUPRC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Lab anemia | 2,002 | 0.817 (0.797–0.835) | 0.768 | 0.174 | 0.528 |
| Readmission | 1,190 | 0.649 (0.581–0.713) | 0.085 | 0.050 | 0.202 |
| New hypertension | 755 | 0.613 (0.555–0.667) | 0.222 | 0.131 | 0.429 |
| CheXpert | 396 | 0.505 (0.446–0.566) | 0.695 | 0.229 | 0.650 |
| Equal-weight macro | 4,343 | 0.646 (0.618–0.672) | 0.442 | 0.146 | 0.452 |

CheXpert AUPRC appears high because the held-out prevalence is 67.4%; its AUROC near 0.5 shows that C1 provides essentially no useful ranking discrimination for this task.

### Held-out macro-AUROC by representation

| Representation | Macro-AUROC |
|---|---:|
| B1 latest window | 0.660 |
| B2 latest code | 0.622 |
| B3 code count | **0.711** |
| B4 prevalence | 0.500 |
| C0 order only | 0.645 |
| C1 correct time | 0.646 |
| C2 reversed gaps, perturbable subset | 0.663 |
| C3 mean shuffled-time prediction, perturbable subset | 0.662 |

C2/C3 raw macro values are computed on perturbable subsets and should not be compared naively with complete-cohort columns. The paired contrasts use aligned patients.

### Prespecified contrasts

- C1 minus B1: **−0.014** (95% CI −0.032 to 0.003; Holm-adjusted p=0.224).
- C1 minus mean C3: **+0.002** (95% CI −0.001 to 0.006; Holm-adjusted p=0.224).
- C1 minus C0: +0.001 (95% CI −0.005 to 0.007; unadjusted p=0.713).
- C1 minus C2: +0.002 (95% CI −0.002 to 0.005; unadjusted p=0.285).
- C1 minus B2: +0.024 (95% CI −0.008 to 0.056).
- C1 minus B3: **−0.066** (95% CI −0.097 to −0.035).
- C1 minus B4: +0.146 (95% CI 0.118 to 0.172).

Anemia was the only task with a detectable C1–C0 effect: +0.009 (95% CI 0.003–0.015; BH-adjusted q=0.012). The effect was not detectable for readmission, hypertension, or CheXpert.

## 10. What “different embeddings” means here

The embedding is a dense numeric vector, not a human-readable clinical explanation. C0 and C1 contain the same clinical events but produce slightly different vectors because C1 adds time-label tokens. C2 and C3 keep the same content and order while changing earlier time-label tokens, so differences in their vectors isolate sensitivity to event–time assignment more directly. B1 produces a vector from only the recent window; B2 produces one from a recurrence-reduced history.

The logistic readout then asks whether these vector differences make the outcome more separable. The near-zero C1–C0 and C1–C3 panel contrasts indicate that, after embedding and a linear readout, the correct time-label mapping did not produce a robust aggregate advantage. This does not prove that the vectors are identical, nor that the model contains no temporal information. It means the relevant difference was not reliably linearly accessible for these outcomes under this pipeline.

The strong B3 result is especially informative: simple code frequencies preserved predictive signal that the dense embedding/readout did not exploit as effectively. That can reflect recurrence burden, finite-context truncation, representation mismatch, or inadequate access through a linear probe—not necessarily a failure of the underlying language model in every possible architecture.

## 11. Interpretation for the field

The study contributes an evaluation lesson more than a new model. Adding a timestamp token is not evidence that a model uses time correctly. Studies should separate recency, order, elapsed time, recurrence, and correct event–time alignment with content-matched controls and report how token budgets change retained evidence.

The aggregate null result narrows the claim that can be made about flat general-purpose embeddings. The anemia result and prior work on learned temporal components leave open an important possibility: task-specific or explicitly trained temporal architectures may benefit, even when frozen flat embeddings do not. Future work should test richer learned encodings, stronger sequence objectives, external cohorts, more EHRSHOT tasks, and carefully gated summarization without changing the completed M1 result.

## 12. What is complete and what is not

Complete:

- Phase 1 data-contract and notebook foundation.
- Protocol alignment and recovery validation.
- Locked four-task patient-balanced cohort.
- Full M1 B1–B4 and C0–C3 validation matrix.
- Prespecified feasible 8B confirmation cells.
- M3 secondary validation analysis under three seeds.
- Frozen one-time M1 held-out evaluation, publication tables/figures, reproducibility manifest, and privacy-safe handoff.

Incomplete by design and excluded from claims:

- M2 did not pass fidelity/runtime gates.
- M3 did not access held-out outcomes.
- The 8B model did not complete a four-task matrix.
- No external validation, fairness analysis, prospective evaluation, or causal inference was performed.

The scientific computation is finished. The remaining work is manuscript quality control and submission administration, not another experimental iteration.

## References

The canonical citation metadata is in `paper/ml4h2026/references.bib`. The final paper uses EHRSHOT, Context Clues, Hegselmann et al., Qwen3 Embedding, CEHR-BERT, MEDS, RAPTOR, PaReGTA, the two recent temporal-tokenization studies, the original GRU paper, AdamW, Holm, Benjamini–Hochberg, and Time After Time.
