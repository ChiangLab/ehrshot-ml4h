# ML4H 2026 submission checklist

Status: scientific analyses and frozen held-out results completed 2026-09-08;
**administrative and final-format review remain before submission**.

## Recommended submission

- Track: **Proceedings** (archival, up to 8 content pages at submission).
- General area: **Applications and Practice — Investigation, Evaluation, Interpretation, and Deployment**.
- Suggested subject areas: structured EHR; foundation models and LLMs; representation learning; temporal and sequential modeling; robustness and model evaluation.
- Data modality: structured/tabular longitudinal EHR.
- Rationale: the contribution is a rigorous empirical evaluation and benchmark protocol applied to a practical health-ML limitation, not primarily a novel learning algorithm or a deployed end-to-end product. A Proceedings submission is automatically reconsidered for Findings if not accepted; the reverse is not possible.

## Official requirements checked

- [x] Uses the ML4H 2026 LaTeX structure and `\mlhtrack{proceedings}`.
- [x] Anonymous author block for double-blind review.
- [x] Main manuscript is one `.tex` file.
- [x] References are maintained in a `.bib` file.
- [x] Mandatory Data and Code Availability paragraph exists.
- [x] Mandatory IRB paragraph exists, but its placeholder is unresolved.
- [x] Draft is scoped to health-focused ML evaluation.
- [ ] Compile inside the official 2026 template and confirm at most 8 content pages, excluding references and appendices.
- [ ] Remove the visible draft warning and every `PENDING`/bracketed marker.
- [ ] Verify that the PDF, metadata, acknowledgments, supplement, repository, and file properties contain no identifying information.
- [ ] Finalize title, complete author list, track, subject areas, and data modality by **September 10, 2026, 11:59 PM AoE**; these metadata cannot be edited afterward and authors cannot be added.
- [ ] Confirm no simultaneous submission to another archival venue.
- [ ] Nominate at least one qualified author to review at least three papers, or submit the organizer exemption form if no author qualifies.
- [ ] Ensure at least one presenting author can register for and attend the December 6–7 in-person symposium in Sydney if accepted.

## Scientific completion and final checks

- [x] M1 completed the full C0--C3 matrix plus B1--B4 on train/validation.
- [x] M3 reference evidence was aggregated and semantically validated; its macro-AUROC (0.622) did not displace M1 (0.694).
- [x] M2 failed its prespecified fidelity/runtime gates and was frozen as incomplete future work before test access; partial M2 caches are not analyzed.
- [x] M1 was frozen as the sole held-out interface using the prespecified scope rule.
- [x] The selected M1 approach has B1--B4 and all three C3 seeds complete.
- [x] The exact held-out manifest was frozen before test access.
- [x] The one-time held-out graph completed with atomic sentinels and semantic validation (jobs 572, 573, 583, 585, 586, and 587).
- [x] The held-out tables report per-task and equal-weight macro-AUROC, AUPRC, Brier score, log loss, calibration, and paired patient-bootstrap intervals.
- [x] Complete-cohort and perturbable-subset estimates, C3 variability, coverage, recurrence, truncation, perturbation strength, runtime, memory, and storage are present in the aggregate/publication bundle.
- [x] The abstract, results, discussion, and conclusion use only hash-verified aggregate artifacts.
- [ ] Independent author cross-check: compare every manuscript number to `m1_held_out_results.json` and the publication CSVs.

## Ethics, privacy, and release blockers

- [ ] Obtain and insert the applicable IRB approval, exemption, NHSR determination, or institutionally supported non-applicability statement. Do not self-declare this when institutional review is required.
- [ ] Decide whether an anonymized code snapshot will accompany review. If not, explicitly state that code is not being shared.
- [ ] Remove Git history, usernames, institutional paths, commit authors, and remote URLs from any review supplement.
- [ ] Never include patient IDs, prediction timestamps, private manifest rows, histories, summaries, embeddings, labels, or predictions.
- [ ] Have every author verify the manuscript, figures, references, and any LLM-assisted writing; authors remain responsible for all content.

## Official links

- Call for Participation: https://ml4h.ahli.cc/submit/call-for-papers/
- 2026 template: https://www.overleaf.com/latex/templates/machine-learning-for-health-ml4h-2026-template/sqgwhtyswgcy
- Writing guidelines: https://ml4h.ahli.cc/resources/writing-guidelines/
- Review/IRB policy: https://ml4h.ahli.cc/resources/review-policy/
- FAQ: https://ml4h.ahli.cc/resources/faqs/
- OpenReview: https://openreview.net/group?id=ML4H/2026/Symposium
