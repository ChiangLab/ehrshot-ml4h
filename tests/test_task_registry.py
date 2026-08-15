from __future__ import annotations

import pytest

from temporal_ehr.data.task_registry import binary_target, validate_domain


@pytest.mark.parametrize(
    ("value", "task", "expected"),
    [
        (False, {"target_rule": "boolean_true"}, False),
        (True, {"target_rule": "boolean_true"}, True),
        (0, {"target_rule": "integer_nonzero"}, False),
        (3, {"target_rule": "integer_nonzero"}, True),
        (8192, {"target_rule": "integer_not_equal", "negative_value": 8192}, False),
        (16, {"target_rule": "integer_not_equal", "negative_value": 8192}, True),
    ],
)
def test_binary_target_rules(value: object, task: dict[str, object], expected: bool) -> None:
    assert binary_target(value, task) is expected


def test_declared_domain() -> None:
    task = {"allowed_values": [0, 1, 2, 3]}
    assert validate_domain(2, task)
    assert not validate_domain(4, task)
