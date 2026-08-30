from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = {
    "00_ehrshot_foundation.ipynb": [
        "# 00 — EHRSHOT trusted foundation",
        "## Governance and pinned configuration",
        "## Environment and package versions",
        "## Existing aggregate audit and validation",
        "## Primary-panel aggregate counts",
        "## Source-row versus patient-balanced prevalence",
        "## Strict-censoring exclusions",
        "## Machine-readable readiness checklist",
    ],
    "01_count_baseline.ipynb": [
        "# 01 — Reproducible count baseline",
        "## Scope and safeguards",
        "## Configuration",
        "## Run the simple count-based reference",
        "## Validation estimates",
        "## Artifact manifest",
    ],
    "02_serialization_qc.ipynb": [
        "# 02 — Serialization QC before embeddings",
        "## Rules and condition matrix",
        "## Entirely synthetic timeline demonstration",
        "## Real-data aggregate QC configuration",
        "## Aggregate real-data distributions",
        "## QC interpretation",
    ],
    "03_phase1_protocol_review.ipynb": [
        "# 03 — Phase 1 protocol review",
        "## Protocol and unresolved decisions",
        "## Entirely synthetic C0-C3 demonstration",
        "## Governed aggregate review configuration",
        "## Duplicate removal and prediction-relative windows",
        "## Temporal-condition eligibility",
        "## Packing and information-reduction baselines",
        "## Machine-readable Phase 3 checklist",
    ],
    "04_pilot_results.ipynb": [
        "# 04 — Complete three-task engineering pilot",
        "## Frozen pilot scope and final-panel isolation",
        "## Train/validation cohort and preparation coverage",
        "## Validation metrics, calibration, and paired contrasts",
        "## Token coverage, throughput, failures, and retries",
        "## Machine-readable completion checklist",
    ],
    "05_final_foundation_review.ipynb": [
        "# 05 — Locked four-task final foundation review",
        "## Patient-balanced cohort and official split coverage",
        "## Outcome definitions, shortcut exclusion, and serializer gates",
        "## Boundary",
    ],
    "06_final_m1_results.ipynb": [
        "# 06 — Final four-task M1 validation",
        "## Equal-weight panel validation estimates",
        "## Prespecified temporal and baseline contrasts",
        "## Runtime-only 8B feasibility and operational evidence",
        "## Boundary",
    ],
}


def test_exact_source_notebook_pack_exists() -> None:
    actual = {path.name for path in Path("notebooks").glob("*.ipynb")}
    assert actual == set(NOTEBOOKS)


def test_source_notebooks_are_clean_and_have_required_heading_order() -> None:
    for name, required_headings in NOTEBOOKS.items():
        notebook = json.loads((Path("notebooks") / name).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        headings = [
            line.strip()
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            for line in cell["source"]
            if line.lstrip().startswith("#")
        ]
        positions = [headings.index(heading) for heading in required_headings]
        assert positions == sorted(positions)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_notebooks_keep_real_patient_content_out_of_source() -> None:
    for name in NOTEBOOKS:
        text = (Path("notebooks") / name).read_text(encoding="utf-8")
        assert "outputs/private/executed_notebooks" not in text
        assert "LOCAL_LLM_BASE_URL" not in text
        assert "32B" not in text
