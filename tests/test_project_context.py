from __future__ import annotations

from pathlib import Path

from scripts.check_project_context import REQUIRED_CONTEXT, validate_project_context


def _write(root: Path, relative: Path, text: str = "context\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _valid_context(root: Path) -> None:
    for path in REQUIRED_CONTEXT:
        _write(root, path)
    _write(root, Path("prompts/completed/01_one.md"), "Prompt 1\n")
    _write(root, Path("prompts/completed/02_two.md"), "Prompt 2\n")
    _write(root, Path("prompts/next/03_three.md"), "Prompt 3\n")
    _write(root, Path("prompts/planned/04_four.md"), "Prompt 4\n")
    _write(root, Path("prompts/planned/05_five.md"), "Prompt 5\n")
    _write(root, Path("docs/NEXT_ACTION.md"), "prompts/next/03_three.md\n")


def test_project_context_accepts_one_unique_active_phase(tmp_path: Path) -> None:
    _valid_context(tmp_path)
    assert validate_project_context(tmp_path) == []


def test_project_context_rejects_duplicate_active_phase_and_model_suggestion(tmp_path: Path) -> None:
    _valid_context(tmp_path)
    _write(tmp_path, Path("prompts/next/04_duplicate.md"), "Use GPT-5.6 Terra\n")

    failures = validate_project_context(tmp_path)

    assert any("exactly one active prompt" in failure for failure in failures)
    assert any("appears more than once" in failure for failure in failures)
    assert any("retains a model suggestion" in failure for failure in failures)


def test_project_context_accepts_five_completed_phases(tmp_path: Path) -> None:
    for path in REQUIRED_CONTEXT:
        _write(tmp_path, path)
    for phase in range(1, 6):
        _write(tmp_path, Path(f"prompts/completed/{phase:02d}_phase.md"), f"Prompt {phase}\n")
    _write(tmp_path, Path("docs/NEXT_ACTION.md"), "No active implementation prompt remains.\n")

    assert validate_project_context(tmp_path) == []


def test_project_context_rejects_empty_next_before_final_phase(tmp_path: Path) -> None:
    for path in REQUIRED_CONTEXT:
        _write(tmp_path, path)
    for phase in range(1, 5):
        _write(tmp_path, Path(f"prompts/completed/{phase:02d}_phase.md"), f"Prompt {phase}\n")
    _write(tmp_path, Path("prompts/planned/05_phase.md"), "Prompt 5\n")

    failures = validate_project_context(tmp_path)

    assert any("exactly one active prompt" in failure for failure in failures)
