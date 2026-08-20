#!/usr/bin/env python3
"""Audit declared versus installed dependencies and local runtime assets."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

IMPORT_NAMES = {
    "PyYAML": "yaml",
    "scikit-learn": "sklearn",
    "huggingface-hub": "huggingface_hub",
    "sentence-transformers": "sentence_transformers",
    "temporal-ehr-ml4h": "temporal_ehr",
}
REQUIRED_ASSETS = {
    "TEMPORAL_EHR_PROJECT_ROOT": "directory",
    "EHRSHOT_DATA_PATH": "file",
    "EHRSHOT_ASSET_ROOT": "directory",
    "EHRSHOT_COHORT_PATH": "file",
    "QWEN_EMBEDDING_PATH": "directory",
}


def _package_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\s\[]", requirement, maxsplit=1)[0]


def _dependency_groups(pyproject: Path) -> dict[str, list[str]]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    return {
        "core": [_package_name(item) for item in project.get("dependencies", [])],
        **{
            name: [_package_name(item) for item in requirements]
            for name, requirements in project.get("optional-dependencies", {}).items()
        },
    }


def _installed(package: str) -> dict[str, Any]:
    import_name = IMPORT_NAMES.get(package, package.replace("-", "_"))
    try:
        installed_version = version(package)
    except PackageNotFoundError:
        installed_version = None
    try:
        importlib.import_module(import_name)
        import_success = True
        import_error = None
    except Exception as exc:
        import_success = False
        import_error = type(exc).__name__
    return {
        "declared_package": package,
        "import_name": import_name,
        "installed_version": installed_version,
        "import_success": import_success,
        "import_error_type": import_error,
    }


def build_audit(repository_root: Path) -> dict[str, Any]:
    groups = _dependency_groups(repository_root / "pyproject.toml")
    packages = sorted({package for values in groups.values() for package in values})
    installed = {package: _installed(package) for package in packages}
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        import torch

        torch_report: dict[str, Any] = {
            "installed_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "visible_device_count": int(torch.cuda.device_count()),
            "visible_device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        torch_report = {
            "installed_version": None,
            "cuda_available": False,
            "visible_device_count": 0,
            "visible_device_names": [],
            "import_error_type": type(exc).__name__,
        }

    assets: dict[str, Any] = {}
    for variable, expected_type in REQUIRED_ASSETS.items():
        raw = os.environ.get(variable)
        candidate = Path(raw).expanduser() if raw else None
        exists = bool(candidate and candidate.exists())
        correct_type = bool(
            candidate
            and (
                (expected_type == "file" and candidate.is_file())
                or (expected_type == "directory" and candidate.is_dir())
            )
        )
        assets[variable] = {
            "configured": raw is not None,
            "logical_name": candidate.name if candidate else None,
            "exists": exists,
            "expected_type": expected_type,
            "correct_type": correct_type,
        }

    tokenizer: dict[str, Any] = {
        "transformers_import_success": installed.get("transformers", {}).get("import_success", False),
        "local_path_configured": assets["QWEN_EMBEDDING_PATH"]["configured"],
        "local_path_exists": assets["QWEN_EMBEDDING_PATH"]["exists"],
        "local_load_success": False,
    }
    qwen_raw = os.environ.get("QWEN_EMBEDDING_PATH")
    if tokenizer["transformers_import_success"] and qwen_raw and Path(qwen_raw).is_dir():
        try:
            from transformers import AutoTokenizer

            loaded = AutoTokenizer.from_pretrained(qwen_raw, local_files_only=True)
            tokenizer["local_load_success"] = bool(loaded.encode("synthetic audit", add_special_tokens=False))
            tokenizer["class_name"] = type(loaded).__name__
        except Exception as exc:
            tokenizer["load_error_type"] = type(exc).__name__

    return {
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "dependency_groups_declared_in_pyproject": groups,
        "installed_dependency_audit": installed,
        "pip_check": {"passed": pip_check.returncode == 0, "exit_status": pip_check.returncode},
        "pytorch": torch_report,
        "transformers_and_tokenizer": tokenizer,
        "required_assets": assets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    report = build_audit(repository_root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    required_ok = all(value["correct_type"] for value in report["required_assets"].values())
    dependencies_ok = all(value["import_success"] for value in report["installed_dependency_audit"].values())
    tokenizer_ok = report["transformers_and_tokenizer"]["local_load_success"]
    return 0 if report["pip_check"]["passed"] and required_ok and dependencies_ok and tokenizer_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
