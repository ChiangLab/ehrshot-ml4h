"""Read-only, private HTML comparison of saved development-set model inputs."""

from __future__ import annotations

import html
from collections import defaultdict
from typing import Any

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.integrity import stable_hash

REQUIRED = {"C0_order_only", "C1_correct_compact_time", "C2_reversed_earlier_gaps",
            "C3_shuffled_earlier_time_labels", "B1_latest_window_only", "B2_latest_code_occurrence"}


def select_example(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select by stable instance hash, never targets, predictions or performance."""
    if any(row["split"] not in {"train", "validation"} for row in rows):
        raise ContractError("Atlas refuses an input containing held-out rows.")
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            groups[row["instance_key"]].append(row)
    for key in sorted(groups, key=lambda value: stable_hash(("internal-atlas-v1", value))):
        group = groups[key]
        if {row["representation"] for row in group} >= REQUIRED:
            return sorted(group, key=lambda row: (row["representation"], str(row.get("perturbation_seed"))))
    raise ContractError("No training example has every requested saved representation.")


def render_atlas(examples: dict[str, list[dict[str, Any]]], commit: str) -> str:
    """No identifiers, outcome values or absolute prediction timestamps in HTML."""
    sections = []
    for task, rows in examples.items():
        if any(row["split"] != "train" for row in rows):
            raise ContractError("Only training examples may be displayed.")
        columns = []
        for row in rows:
            label = row["representation"]
            if row.get("perturbation_seed") is not None:
                label += f" (seed {row['perturbation_seed']})"
            columns.append(
                f'<details open><summary>{html.escape(label)}</summary>'
                f'<pre>{html.escape(str(row["text"]))}</pre></details>'
            )
        sections.append(f'<h2>{html.escape(task)}</h2><div class="compare">' + ''.join(columns) + '</div>')
    return '''<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Internal EHR input comparison</title><style>
body{font:17px system-ui;margin:28px;color:#222}h2{margin-top:36px}
.compare{display:flex;gap:16px;overflow-x:auto;align-items:flex-start}
details{min-width:420px;max-width:560px;border:1px solid #aaa;padding:12px}
summary{font-weight:bold;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px monospace}
@media print{.compare{display:block}details{max-width:none;page-break-before:always}}
</style></head><body><h1>Exact saved inputs: C0-C3, B1 and B2</h1>
<p>Governed internal view. Keep on DGX and view through an SSH tunnel. No test rows.
Each task uses one training instance, chosen by a fixed hash among instances with all conditions.
These illustrative instances are not a representative fidelity sample.</p>
<p>Scroll horizontally to compare complete saved inputs. C2 changes earlier gap labels.
C3 changes earlier label indices. Neither rearranges clinical event arrays. The final window stays correct.
B1 and B2 were independently packed from full pre-T history, so may differ from slices of C1.</p>
<p>B3 uses full pre-T code counts with log1p transformation and a training vocabulary.
B4 returns training prevalence. Neither has a text summary, so no synthetic text is shown here.</p>
<p>M1 encodes each displayed input. M2 separately selects source facts and encodes their compact rendering.
Its real source/output comparisons belong in the frozen 40-history fidelity packet after sealing.
M3 encodes windows separately and processes them with a GRU. This page does not invent M2 outputs
or display unseen M3 vectors.</p><p>Viewer source commit: ''' + html.escape(commit) + '</p>' + ''.join(sections) + (
        '</body></html>'
    )
