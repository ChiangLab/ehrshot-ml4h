"""Generate privacy-safe publication artifacts from the immutable held-out aggregate."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import FINAL_TASKS
from temporal_ehr.final_m1.foundation import _write_immutable_json
from temporal_ehr.final_submission.protocol import load_frozen_manifest
from temporal_ehr.pilot.cache import file_sha256

TASK_LABELS = {
    "lab_anemia": "Lab anemia",
    "guo_readmission": "30-day readmission",
    "new_hypertension": "New hypertension",
    "chexpert": "CheXpert",
}


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ContractError(f"Refusing to overwrite a different publication artifact: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    if path.exists():
        raise ContractError(f"Refusing to overwrite publication table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _save_figure(path: Path) -> None:
    if path.exists():
        raise ContractError(f"Refusing to overwrite publication figure: {path.name}")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _cohort_table(report: dict[str, Any], root: Path) -> Path:
    path = root / "table_cohort.csv"
    rows = [
        [
            TASK_LABELS[group["task"]],
            group["instances"],
            group["positive_instances"],
            group["negative_instances"],
            f"{100 * group['prevalence']:.1f}",
        ]
        for group in report["held_out_cohort"]["groups"]
    ]
    _write_csv(path, ["Task", "Test n", "Positive n", "Negative n", "Positive (%)"], rows)
    return path


def _performance_table(report: dict[str, Any], root: Path) -> Path:
    path = root / "table_main_performance.csv"
    intervals = report["shared_patient_cluster_bootstrap"]["main_C1_metric_intervals_by_task"]
    rows = []
    for task in FINAL_TASKS:
        metrics = report["tasks"][task]["primary_scopes"]["complete_cohort"]["representations"][
            "C1_correct_compact_time"
        ]["metrics"]
        rows.append(
            [
                TASK_LABELS[task],
                metrics["auroc"],
                *intervals[task]["auroc"]["ci_95"],
                metrics["auprc"],
                metrics["brier_score"],
                metrics["log_loss"],
            ]
        )
    macro = report["equal_weight_panel_macro"]["C1_complete_cohort"]["metrics"]
    macro_ci = report["shared_patient_cluster_bootstrap"]["main_C1_panel_macro_intervals"]
    rows.append(
        [
            "Equal-weight panel macro",
            macro["auroc"],
            *macro_ci["auroc"]["ci_95"],
            macro["auprc"],
            macro["brier_score"],
            macro["log_loss"],
        ]
    )
    _write_csv(
        path,
        ["Task", "AUROC", "AUROC CI low", "AUROC CI high", "AUPRC", "Brier", "Log loss"],
        rows,
    )
    return path


def _contrast_table(report: dict[str, Any], root: Path) -> Path:
    path = root / "table_paired_contrasts.csv"
    bootstrap = report["shared_patient_cluster_bootstrap"]
    rows = []
    for name, item in bootstrap["equal_weight_panel_macro_contrasts"].items():
        rows.append(
            [
                "Panel macro",
                name,
                item["estimate"],
                *item["ci_95"],
                item["unadjusted_two_sided_p"],
                item.get("holm_adjusted_p", ""),
                "",
            ]
        )
    for task in FINAL_TASKS:
        for name, item in bootstrap["task_contrasts"][task].items():
            rows.append(
                [
                    TASK_LABELS[task],
                    name,
                    item["estimate"],
                    *item["ci_95"],
                    item["unadjusted_two_sided_p"],
                    "",
                    item.get("benjamini_hochberg_q", ""),
                ]
            )
    _write_csv(
        path,
        ["Level", "Contrast", "Estimate", "CI low", "CI high", "Unadjusted p", "Holm p", "BH q"],
        rows,
    )
    return path


def _performance_figure(report: dict[str, Any], root: Path) -> Path:
    path = root / "figure_main_performance.png"
    tasks = list(FINAL_TASKS)
    values = [
        report["tasks"][task]["primary_scopes"]["complete_cohort"]["representations"][
            "C1_correct_compact_time"
        ]["metrics"]["auroc"]
        for task in tasks
    ]
    intervals = report["shared_patient_cluster_bootstrap"]["main_C1_metric_intervals_by_task"]
    lower = [values[index] - intervals[task]["auroc"]["ci_95"][0] for index, task in enumerate(tasks)]
    upper = [intervals[task]["auroc"]["ci_95"][1] - values[index] for index, task in enumerate(tasks)]
    plt.figure(figsize=(7.2, 3.8))
    positions = np.arange(len(tasks))
    plt.errorbar(values, positions, xerr=[lower, upper], fmt="o", color="#2774AE", capsize=3)
    plt.axvline(0.5, color="0.6", linestyle="--", linewidth=1)
    plt.yticks(positions, [TASK_LABELS[task] for task in tasks])
    plt.xlabel("Held-out AUROC (95% patient-cluster bootstrap CI)")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", color="0.9")
    _save_figure(path)
    return path


def _contrast_figure(report: dict[str, Any], root: Path) -> Path:
    path = root / "figure_primary_contrasts.png"
    contrasts = report["shared_patient_cluster_bootstrap"]["equal_weight_panel_macro_contrasts"]
    names = ["C1_minus_B1", "C1_minus_C3_mean", "C1_minus_C0", "C1_minus_C2", "C1_minus_B2"]
    labels = [
        "C1 - latest window",
        "C1 - shuffled time",
        "C1 - order only",
        "C1 - reversed gaps",
        "C1 - latest code",
    ]
    values = [contrasts[name]["estimate"] for name in names]
    lower = [values[index] - contrasts[name]["ci_95"][0] for index, name in enumerate(names)]
    upper = [contrasts[name]["ci_95"][1] - values[index] for index, name in enumerate(names)]
    plt.figure(figsize=(7.2, 4.1))
    positions = np.arange(len(names))
    plt.errorbar(values, positions, xerr=[lower, upper], fmt="o", color="#2774AE", capsize=3)
    plt.axvline(0.0, color="0.4", linewidth=1)
    plt.yticks(positions, labels)
    plt.xlabel("Equal-weight panel macro-AUROC difference (95% CI)")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", color="0.9")
    _save_figure(path)
    return path


def _diagnostic_figures(report: dict[str, Any], root: Path) -> list[Path]:
    diagnostics = {item["task"]: item for item in report["held_out_preparation_diagnostics"]}
    coverage_path = root / "figure_history_coverage.png"
    values = [
        diagnostics[task]["representation_coverage"]["C1_correct_compact_time"]["eligible_histories"]
        for task in FINAL_TASKS
    ]
    plt.figure(figsize=(7.2, 3.7))
    plt.bar([TASK_LABELS[task] for task in FINAL_TASKS], values, color="#8BB8D8")
    plt.ylabel("Eligible held-out histories")
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", color="0.9")
    _save_figure(coverage_path)

    perturbation_path = root / "figure_perturbation_strength.png"
    changed = []
    for task in FINAL_TASKS:
        strength = diagnostics[task]["perturbation_strength"]
        c3 = [value for key, value in strength.items() if key.startswith("C3_shuffled")]
        changed.append(float(np.mean([value["mean_changed_labels_per_eligible_history"] for value in c3])))
    plt.figure(figsize=(7.2, 3.7))
    plt.bar([TASK_LABELS[task] for task in FINAL_TASKS], changed, color="#8BB8D8")
    plt.ylabel("Mean changed C3 time labels per eligible history")
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", color="0.9")
    _save_figure(perturbation_path)

    calibration_path = root / "figure_calibration_summary.png"
    slopes = []
    plotted_tasks = []
    for task in FINAL_TASKS:
        calibration = report["tasks"][task]["primary_scopes"]["complete_cohort"]["representations"][
            "C1_correct_compact_time"
        ]["calibration"]
        if calibration["status"] == "REPORTED":
            plotted_tasks.append(TASK_LABELS[task])
            slopes.append(calibration["slope"])
    plt.figure(figsize=(7.2, 3.7))
    plt.bar(plotted_tasks, slopes, color="#8BB8D8")
    plt.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    plt.ylabel("Calibration slope")
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", color="0.9")
    _save_figure(calibration_path)
    return [coverage_path, perturbation_path, calibration_path]


def _flow_figure(root: Path) -> Path:
    path = root / "figure_experiment_flow.png"
    labels = [
        "Official EHRSHOT\nprediction times",
        "One instance per\npatient and task",
        "Strict pre-T\n30/90-day windows",
        "M1: B1-B4 and\nC0-C3",
        "Frozen logistic\nreadouts",
        "One-time held-out\nevaluation",
    ]
    plt.figure(figsize=(11, 2.4))
    axis = plt.gca()
    axis.axis("off")
    positions = np.linspace(0.06, 0.94, len(labels))
    for index, (position, label) in enumerate(zip(positions, labels, strict=True)):
        axis.text(
            position,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "square,pad=0.45", "facecolor": "white", "edgecolor": "0.35"},
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(positions[index + 1] - 0.07, 0.5),
                xytext=(position + 0.07, 0.5),
                arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 1.2},
            )
    _save_figure(path)
    return path


def build_publication_artifacts(
    *,
    repository_root: Path,
    config_path: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Build tables, figures, and drafts from aggregate results only."""
    root = repository_root.resolve()
    spec, manifest = load_frozen_manifest(root, config_path, source_git_commit)
    report_path = spec.aggregate_root / "m1_held_out_results.json"
    if not report_path.is_file():
        raise ContractError("Publication artifacts require the completed held-out aggregate.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or report.get("protocol_decision")
        != "M1_PRIMARY_ONE_TIME_HELD_OUT_COMPLETE_NO_DESIGN_CHANGES_PERMITTED"
        or report.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
    ):
        raise ContractError("Held-out aggregate is not publication-authorized.")
    publication_root = spec.aggregate_root / "publication"
    if publication_root.exists() and any(publication_root.iterdir()):
        raise ContractError("Publication directory is nonempty; refusing partial overwrite.")
    if publication_root.exists():
        publication_root.rmdir()
    publication_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".publication.", dir=publication_root.parent))
    try:
        artifacts = [
            _cohort_table(report, staging_root),
            _performance_table(report, staging_root),
            _contrast_table(report, staging_root),
            _performance_figure(report, staging_root),
            _contrast_figure(report, staging_root),
            *_diagnostic_figures(report, staging_root),
            _flow_figure(staging_root),
        ]
        macro = report["equal_weight_panel_macro"]["C1_complete_cohort"]["metrics"]
        contrasts = report["shared_patient_cluster_bootstrap"]["equal_weight_panel_macro_contrasts"]
        methods = f"""# Frozen methods and held-out results draft

We evaluated a patient-balanced four-task EHRSHOT panel comprising laboratory anemia, 30-day
readmission, one-year incident hypertension, and CheXpert imaging findings. One official prediction
time was selected per patient and task using an outcome-independent hash rule. Events at or after
the official prediction time were excluded. Exact seven-field duplicates were removed, and retained
events were grouped into 30-day windows for laboratory, operational, and imaging tasks and 90-day
windows for incident hypertension.

The primary M1 interface serialized the complete strictly pre-prediction history and embedded it
with frozen Qwen3-Embedding-0.6B at 8,192 tokens. Penalized logistic readouts were fit on training
patients using regularization fixed by the hash-pinned validation analysis; models were not refit on
validation patients. Held-out evaluation was performed once. Conditions compared order only (C0),
correct prediction-relative time (C1), reversed earlier gaps (C2), and three fixed shuffled-time
seeds (C3), alongside latest-window, latest-code, count, and prevalence baselines. Confidence
intervals used 2,000 shared patient-cluster bootstrap resamples, with Holm correction for the two
co-primary panel contrasts and Benjamini-Hochberg correction for per-task secondary contrasts.

The held-out C1 equal-weight panel macro-AUROC was {macro['auroc']:.3f}, with macro-AUPRC
{macro['auprc']:.3f}, Brier score {macro['brier_score']:.3f}, and log loss {macro['log_loss']:.3f}.
The complete-history contrast versus latest-window was {contrasts['C1_minus_B1']['estimate']:.3f}
(95% CI {contrasts['C1_minus_B1']['ci_95'][0]:.3f} to
{contrasts['C1_minus_B1']['ci_95'][1]:.3f}); the contrast versus the mean of three shuffled-time
conditions was {contrasts['C1_minus_C3_mean']['estimate']:.3f} (95% CI
{contrasts['C1_minus_C3_mean']['ci_95'][0]:.3f} to
{contrasts['C1_minus_C3_mean']['ci_95'][1]:.3f}). Null or negative findings are retained as observed.

M3 is reported only as a prespecified secondary validation analysis. M2 did not complete its
fidelity and runtime gates and is described as incomplete future work; neither M2 nor M3 accessed
held-out outcomes.
"""
        methods_path = staging_root / "methods_and_results_draft.md"
        assert_privacy_safe_report(methods)
        _write_once(methods_path, methods)
        artifacts.append(methods_path)
        claims = """# Limitations and permitted claims

- Describe results as an EHRSHOT-derived patient-balanced four-task evaluation, not the official
  15-task leaderboard.
- Claim association and predictive sensitivity only; do not claim that time causes performance.
- Treat C3 as a temporal-label falsification and C2 as a monotonic sensitivity analysis.
- Report all null or negative held-out results without changing the protocol.
- State that M3 is validation-only and operationally underperformed M1 in this implementation.
- State that M2 was incomplete because it did not pass the required fidelity/runtime gates.
- Do not claim external validation, clinical utility, prospective performance, or fairness.
"""
        claims_path = staging_root / "limitations_and_permitted_claims.md"
        _write_once(claims_path, claims)
        artifacts.append(claims_path)
        reproducibility = {
            "status": "PASS",
            "experiment_id": spec.experiment_id,
            "source_git_commit": source_git_commit,
            "frozen_manifest_sha256": manifest["frozen_manifest_sha256"],
            "submission_config_sha256": manifest["submission_config_sha256"],
            "held_out_report_sha256": file_sha256(report_path),
            "validation_evidence": spec.raw["validation_evidence"],
            "bootstrap_replicates": spec.bootstrap_replicates,
            "bootstrap_seed": spec.bootstrap_seed,
            "patient_level_artifacts_included": False,
            "test_evaluated_once": True,
        }
        reproducibility_path = staging_root / "reproducibility_manifest.json"
        _write_immutable_json(reproducibility_path, reproducibility, aggregate=True)
        artifacts.append(reproducibility_path)
        complete = {
            "status": "PASS",
            "protocol_decision": "PUBLICATION_ARTIFACTS_COMPLETE_FROM_FROZEN_HELD_OUT_REPORT",
            "experiment_id": spec.experiment_id,
            "source_git_commit": source_git_commit,
            "frozen_manifest_sha256": manifest["frozen_manifest_sha256"],
            "held_out_report_sha256": file_sha256(report_path),
            "artifacts": [
                {
                    "path": str((publication_root / path.relative_to(staging_root)).relative_to(root)),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in artifacts
            ],
            "patient_level_artifacts_included": False,
        }
        _write_immutable_json(staging_root / "PUBLICATION_COMPLETE.json", complete, aggregate=True)
        staging_root.replace(publication_root)
        return complete
    finally:
        plt.close("all")
        if staging_root.exists():
            shutil.rmtree(staging_root)
