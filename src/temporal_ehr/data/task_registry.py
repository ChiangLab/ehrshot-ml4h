"""Task definitions and binary target transformations for EHRSHOT."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_task_registry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if not isinstance(registry, dict) or not isinstance(registry.get("tasks"), dict):
        raise ValueError("Task registry must contain a tasks mapping.")
    return registry


def binary_target(value: Any, task: Mapping[str, Any]) -> bool:
    """Apply the preregistered EHRSHOT binary target transformation."""
    rule = task["target_rule"]
    if value is None:
        raise ValueError("Cannot transform a null label.")
    if rule == "boolean_true":
        return bool(value)
    if rule == "integer_nonzero":
        return int(value) != 0
    if rule == "integer_not_equal":
        return int(value) != int(task["negative_value"])
    raise ValueError(f"Unsupported target rule: {rule}")


def validate_domain(value: Any, task: Mapping[str, Any]) -> bool:
    """Return whether a non-null source label belongs to its declared domain."""
    if "allowed_values" in task:
        return value in task["allowed_values"]
    if "allowed_min" in task and int(value) < int(task["allowed_min"]):
        return False
    return not ("allowed_max" in task and int(value) > int(task["allowed_max"]))
