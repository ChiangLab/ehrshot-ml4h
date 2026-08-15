"""Deterministic aggregate-only quality-control figures."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

QC_NOTE = "Aggregate quality-control result; not a project outcome."


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finish_figure(fig: plt.Figure, path: Path, caption: str) -> None:
    fig.text(0.01, 0.01, f"{caption} {QC_NOTE}", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160, metadata={"Creator": "temporal-ehr-ml4h Phase 0 audit"})
    plt.close(fig)


def _histogram(
    values: Sequence[float],
    *,
    bins: np.ndarray,
    title: str,
    xlabel: str,
    path: Path,
    log_x: bool,
    caption: str,
) -> None:
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    rows = [
        {"bin_left": float(edges[index]), "bin_right": float(edges[index + 1]), "count": int(count)}
        for index, count in enumerate(counts)
    ]
    _write_csv(path.with_suffix(".csv"), rows)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    widths = np.diff(edges)
    ax.bar(edges[:-1], counts, width=widths, align="edge", color="#35618f", edgecolor="white")
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of subjects")
    ax.grid(axis="y", alpha=0.25)
    _finish_figure(fig, path, caption)


def _bar_chart(
    rows: Sequence[dict[str, Any]],
    *,
    category: str,
    value: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    caption: str,
) -> None:
    _write_csv(path.with_suffix(".csv"), rows)
    labels = [str(row[category]) for row in rows]
    values = [float(row[value]) for row in rows]
    fig_height = max(4.8, 0.36 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color="#4f7d61")
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.25)
    _finish_figure(fig, path, caption)


def generate_qc_figures(
    figures_dir: Path,
    subject_values: Sequence[dict[str, float]],
    table_counts: Sequence[dict[str, Any]],
    vocabulary_counts: Sequence[dict[str, Any]],
    missingness: Sequence[dict[str, Any]],
) -> list[str]:
    """Write five required plots and only their fully aggregated plot data."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )

    event_values = [float(row["events_per_subject"]) for row in subject_values]
    max_events = max(event_values)
    event_bins = np.geomspace(1, max_events + 1, 31)
    _histogram(
        event_values,
        bins=event_bins,
        title="Events per subject (log-scaled)",
        xlabel="Events per subject",
        path=figures_dir / "events_per_subject.png",
        log_x=True,
        caption="Distribution of aggregate per-subject event counts.",
    )

    duration_values = [float(row["timeline_duration_years"]) for row in subject_values]
    duration_max = max(duration_values)
    duration_bins = np.linspace(0, duration_max if duration_max > 0 else 1, 31)
    _histogram(
        duration_values,
        bins=duration_bins,
        title="Subject timeline duration",
        xlabel="Timeline duration (years)",
        path=figures_dir / "timeline_duration_years.png",
        log_x=False,
        caption="Durations use exact timestamps and a 365.2425-day year.",
    )

    _bar_chart(
        table_counts,
        category="omop_table",
        value="event_count",
        title="Event counts by OMOP source table",
        xlabel="Events",
        ylabel="OMOP source table",
        path=figures_dir / "events_by_omop_table.png",
        caption="Counts summarize source-table categories across the full event table.",
    )

    _bar_chart(
        vocabulary_counts,
        category="vocabulary",
        value="distinct_code_count",
        title="Distinct codes by public vocabulary prefix",
        xlabel="Distinct codes",
        ylabel="Vocabulary prefix",
        path=figures_dir / "distinct_codes_by_vocabulary.png",
        caption="Unrecognized prefixes are pooled into Other.",
    )

    _bar_chart(
        missingness,
        category="column",
        value="null_fraction",
        title="Missingness across MEDS-style event columns",
        xlabel="Null fraction",
        ylabel="Column",
        path=figures_dir / "column_missingness.png",
        caption="Missing optional values can be expected; core-key missingness is contract-checked.",
    )

    return [
        "figures/events_per_subject.png",
        "figures/timeline_duration_years.png",
        "figures/events_by_omop_table.png",
        "figures/distinct_codes_by_vocabulary.png",
        "figures/column_missingness.png",
    ]
