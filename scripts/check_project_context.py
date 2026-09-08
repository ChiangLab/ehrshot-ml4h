#!/usr/bin/env python3
"""Validate the repository-local phase prompt and handoff topology."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REQUIRED_CONTEXT = (
    Path("AGENTS.md"),
    Path("docs/CURRENT_SCIENTIFIC_PROTOCOL.md"),
    Path("docs/PROJECT_STATE.md"),
    Path("docs/DECISION_LOG.md"),
    Path("docs/DGX_BATCH_RUNBOOK.md"),
    Path("docs/NEXT_ACTION.md"),
    Path("docs/RESULTS_REGISTRY.md"),
)
PROMPT_STATES = ("completed", "next", "planned")
MODEL_SUGGESTION = re.compile(r"^Use GPT-", re.MULTILINE)


def validate_project_context(root: Path) -> list[str]:
    failures = [f"Missing context file: {path}" for path in REQUIRED_CONTEXT if not (root / path).is_file()]
    prompt_files = [
        path
        for state in PROMPT_STATES
        for path in sorted((root / "prompts" / state).glob("*.md"))
    ]
    active = sorted((root / "prompts" / "next").glob("*.md"))
    completed = sorted((root / "prompts" / "completed").glob("*.md"))
    planned = sorted((root / "prompts" / "planned").glob("*.md"))
    final_workflow_state = len(completed) == 5 and not active and not planned
    if len(active) != 1 and not final_workflow_state:
        failures.append(
            "Expected exactly one active prompt, or five completed prompts with no remaining "
            f"planned work; found {len(active)} active, {len(completed)} completed, and "
            f"{len(planned)} planned."
        )

    phases: dict[int, Path] = {}
    for path in prompt_files:
        match = re.match(r"(\d{2})_", path.name)
        if match is None:
            failures.append(f"Canonical prompt lacks a two-digit phase prefix: {path.relative_to(root)}")
            continue
        phase = int(match.group(1))
        if phase in phases:
            failures.append(
                f"Phase {phase} appears more than once: {phases[phase].relative_to(root)} and "
                f"{path.relative_to(root)}"
            )
        phases[phase] = path
        if MODEL_SUGGESTION.search(path.read_text(encoding="utf-8")):
            failures.append(f"Canonical prompt retains a model suggestion: {path.relative_to(root)}")
    if set(phases) != {1, 2, 3, 4, 5}:
        failures.append(f"Canonical prompt phases must be exactly 1-5; found {sorted(phases)}.")

    next_action = root / "docs/NEXT_ACTION.md"
    if len(active) == 1 and next_action.is_file():
        relative_active = str(active[0].relative_to(root))
        if relative_active not in next_action.read_text(encoding="utf-8"):
            failures.append(f"NEXT_ACTION.md does not name the active prompt: {relative_active}")
    if final_workflow_state and next_action.is_file():
        next_action_text = next_action.read_text(encoding="utf-8")
        if "No active implementation prompt remains" not in next_action_text:
            failures.append("NEXT_ACTION.md does not declare the completed five-prompt workflow.")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    failures = validate_project_context(args.root.resolve())
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: durable project context has five unique phases and a valid active-or-complete state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
