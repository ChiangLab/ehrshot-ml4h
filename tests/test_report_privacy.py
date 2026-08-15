from __future__ import annotations

from pathlib import Path

import pytest

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report


def test_privacy_safe_aggregate_report_format() -> None:
    safe = '{"distinct_subject_count": 3, "schema": [{"column": "subject_id"}]}'
    assert_privacy_safe_report(safe)


@pytest.mark.parametrize(
    "unsafe",
    [
        "subject_id: 123456",
        'patient id = "98765"',
        str(Path.home() / "research" / "audit.json"),
    ],
)
def test_report_rejects_subject_like_values_and_user_paths(unsafe: str) -> None:
    with pytest.raises(ContractError, match="Privacy contract"):
        assert_privacy_safe_report(unsafe)
