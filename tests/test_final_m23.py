from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.experiments import FINAL_TASKS
from temporal_ehr.final_m23.m2 import (
    M2_EMBEDDING_MAX_TOKENS,
    M2_SOURCE_COMPACTION_SCHEMA,
    M2_SUMMARY_CACHE_SCHEMA,
    M2_SUMMARY_VALIDATION_SCHEMA,
    LocalSummaryTokenValidator,
    NonRetryableSummaryError,
    OnPremiseSummaryClient,
    SummaryCache,
    _generate_bounded_summary_with_attempts,
    _summarize_with_retries,
    audit_summary_fidelity,
    bounded_summary_payload_is_reusable,
    build_manual_fidelity_packet,
    compact_exact_duplicate_facts,
    exact_summary_payload_is_reusable,
    group_structured_facts,
    m2_decoding_variants,
    m2_embedding_root,
    m2_source_text,
    summary_cache_key,
    summary_shard_for_key,
    validate_bounded_summary,
    validate_manual_fidelity_decisions,
    validate_on_premise_api_base,
)
from temporal_ehr.final_m23.m3 import (
    fixed_projection,
    m3_stage_labels,
    pad_sequences,
    project_then_pad_sequences,
    split_serialized_windows,
    validation_auroc_improved,
)
from temporal_ehr.final_m23.protocol import (
    M3_SEEDS,
    load_final_m23_authorization,
    require_prompt4_hard_gates,
)
from temporal_ehr.final_m23.recovery import (
    _audit_invalid_cache_rows,
    _validate_canceled_or_unstarted_status,
    _validate_failed_status,
    _validate_fidelity_v2_config_change,
    _validate_fidelity_v3_config_change,
    _validate_fidelity_v4_config_change,
    _validate_fidelity_v5_config_change,
    _validate_fidelity_v7_config_change,
    _validate_only_allowed_config_change,
)
from temporal_ehr.final_m23.reporting import (
    _average_seed_predictions,
    aggregate_m3_validation,
    select_approach,
)
from temporal_ehr.serialization.integrity import stable_hash

CONFIG = Path("configs/experiments/final_m2_m3.yaml")


def _spec():
    return load_final_m23_authorization(CONFIG, repository_root=Path.cwd())


def test_prompt4_hard_gate_uses_verified_job451_handoff_and_no_test() -> None:
    result = require_prompt4_hard_gates(Path.cwd(), CONFIG)
    assert result["status"] == "PASS"
    assert result["prompt3_fit_recovery_job_id"] == 451
    assert result["test_predictions_or_performance_used"] is False
    spec = _spec()
    assert spec.tasks == FINAL_TASKS
    assert spec.splits == ("train", "validation")
    assert "test" not in spec.splits
    assert spec.m2["decoding"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert spec.m2["prompt_version"] == "m2_fidelity_v6_bounded_structured_summary"
    assert spec.m2["decoding"]["max_output_tokens"] == 6144
    assert "overflow_decoding" not in spec.m2
    assert "extended_decoding" not in spec.m2
    assert spec.m2["execution"]["generation_contract_attempts"] == 3
    assert spec.m2["execution"]["concurrent_requests"] == 8
    assert spec.m2["execution"]["summary_shards_per_task"] == 192


class _SyntheticSummaryClient(OnPremiseSummaryClient):
    def __init__(self, content: object) -> None:
        super().__init__(api_base="http://10.45.177.17:8000/v1", model="synthetic")
        self.content = content
        self.payload: dict[str, object] | None = None

    def _request(self, path: str, payload=None):  # type: ignore[no-untyped-def]
        self.payload = dict(payload or {})
        return {
            "choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12},
        }


def test_m2_client_disables_thinking_and_rejects_null_like_content() -> None:
    decoding = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260815,
        "max_output_tokens": 100,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    client = _SyntheticSummaryClient("- [TIME=1_30_DAYS_BEFORE_T]: fact")
    assert client.summarize(system_prompt="prompt", source_text="source", decoding=decoding)
    assert client.payload is not None
    assert client.payload["chat_template_kwargs"] == {"enable_thinking": False}
    for invalid in (None, "None", " null ", ""):
        with pytest.raises(RuntimeError, match=r"not text|null-like"):
            _SyntheticSummaryClient(invalid).summarize(
                system_prompt="prompt", source_text="source", decoding=decoding
            )


def test_m2_client_rejects_length_truncated_completion() -> None:
    client = _SyntheticSummaryClient("- CODE/A | occurrences=1")
    client._request = lambda path, payload=None: {  # type: ignore[method-assign]
        "choices": [{"message": {"content": client.content}, "finish_reason": "length"}]
    }
    with pytest.raises(RuntimeError, match="finish_reason='length'"):
        client.summarize(
            system_prompt="prompt",
            source_text="source",
            decoding={
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 20260815,
                "max_output_tokens": 100,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )


def test_m2_does_not_retry_deterministic_length_failure() -> None:
    client = _SyntheticSummaryClient("truncated")
    calls = 0

    def response(path, payload=None):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return {"choices": [{"message": {"content": "truncated"}, "finish_reason": "length"}]}

    client._request = response  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="finish_reason='length'"):
        _summarize_with_retries(
            client,
            system_prompt="prompt",
            source_text="source",
            decoding={
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 20260815,
                "max_output_tokens": 100,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            retries=3,
        )
    assert calls == 1


def test_m2_served_identity_ignores_vllm_process_timestamp() -> None:
    first = _SyntheticSummaryClient("unused")
    second = _SyntheticSummaryClient("unused")
    base = {
        "id": "synthetic",
        "root": "synthetic",
        "parent": None,
        "owned_by": "vllm",
        "max_model_len": 8192,
    }
    first._request = lambda path, payload=None: {"data": [{**base, "created": 1}]}  # type: ignore[method-assign]
    second._request = lambda path, payload=None: {"data": [{**base, "created": 2}]}  # type: ignore[method-assign]
    assert first.model_revision_hash() == second.model_revision_hash()


def test_m2_progress_logging_is_aggregate_only() -> None:
    source = Path("src/temporal_ehr/final_m23/m2.py").read_text(encoding="utf-8")
    assert '"event": "m2_summary_progress"' in source
    assert '"completed_new_cache_rows"' in source
    assert '"source_text"' not in source[source.index('"event": "m2_summary_progress"') :][:800]


def test_prompt4_recovery_changes_only_nonthinking_decoding_control() -> None:
    with pytest.raises(ContractError, match="more than"):
        _validate_only_allowed_config_change(
            Path.cwd(),
            "1a97d8f1a70c47e18f96a2f56e63a370d6ff9913",
            CONFIG,
        )


def test_prompt4_fidelity_v2_validator_rejects_superseding_v3_config() -> None:
    with pytest.raises(ContractError, match="more than"):
        _validate_fidelity_v2_config_change(
            Path.cwd(),
            "08a639b22b7cc916ec0f51faebe65a05a29de0ac",
            CONFIG,
            "1c64b4a5e45e8a09d5ef17b6115d6dc28815485ce60f6a29eddbb218ead05145",
        )


def test_prompt4_fidelity_v3_validator_rejects_superseding_v4_config() -> None:
    with pytest.raises(ContractError, match="more than"):
        _validate_fidelity_v3_config_change(
            Path.cwd(),
            "a316c9886ae10c1c5445ac7115c0f414fb6567b2",
            CONFIG,
            "acc28833d6c5ebf0635791967217d97005f3756f7a5fdb6d6949945eea7d02c6",
        )


def test_prompt4_fidelity_v4_validator_rejects_superseding_v5_config() -> None:
    with pytest.raises(ContractError, match="more than"):
        _validate_fidelity_v4_config_change(
            Path.cwd(),
            "818103ea4851081ab827529781efb9f526044765",
            CONFIG,
            "93e3aca0bcc504067b44a6d28b55dc5e7054d4ce22cdf9cbe1dbf81a9e1beca2",
        )


def test_prompt4_fidelity_v5_validator_rejects_superseding_v6_config() -> None:
    with pytest.raises(ContractError, match="more than"):
        _validate_fidelity_v5_config_change(
            Path.cwd(),
            "8682cb494b779efd0d6ea16785c08f61702bb4a0",
            CONFIG,
            "00a17a0f4c39b0e246bc3ed5e2fc8e26296275cf50d52d75881afc1a34a93980",
        )


def test_prompt4_fidelity_v7_changes_only_execution_scheduling() -> None:
    _validate_fidelity_v7_config_change(
        Path.cwd(),
        "38e2b7b8a7e17cc895c053b35f358784f90ed263",
        CONFIG,
        "434ef018e7db071986220e58b9f68b863830185c6dafb02f352cbaccf347675b",
    )


def test_recovery_cache_audit_is_aggregate_only_and_detects_null_rows(tmp_path: Path) -> None:
    root = (
        tmp_path
        / "outputs/private/final/locked_four_task_m2_m3_v1/m2/summaries/lab_anemia/rows"
    )
    root.mkdir(parents=True)
    (root / "invalid.json").write_text(json.dumps({"summary": "None"}), encoding="utf-8")
    (root / "valid.json").write_text(json.dumps({"summary": "synthetic fact"}), encoding="utf-8")
    result = _audit_invalid_cache_rows(tmp_path)
    assert result == {
        "rows_scanned": 2,
        "invalid_rows": 1,
        "nonempty_text_rows": 1,
        "patient_content_included": False,
    }


def test_fidelity_v3_recovery_validates_failed_status_without_overwrite(tmp_path: Path) -> None:
    directory = tmp_path / "outputs/private/run_status/519_0"
    directory.mkdir(parents=True)
    base = {
        "git_commit": "a" * 40,
        "job_id": "519_0",
        "run_kind": "final_m2_summary_reference",
    }
    running = {**base, "state": "RUNNING"}
    failed = {**base, "state": "FAILED", "exit_status": 143}
    running_path = directory / "RUNNING.json"
    failed_path = directory / "FAILED.json"
    running_path.write_text(json.dumps(running), encoding="utf-8")
    failed_path.write_text(json.dumps(failed), encoding="utf-8")
    before = (running_path.read_bytes(), failed_path.read_bytes())
    hashes = _validate_failed_status(
        tmp_path,
        "519_0",
        "a" * 40,
        "final_m2_summary_reference",
        143,
    )
    assert set(hashes) == {"RUNNING.json", "FAILED.json"}
    assert (running_path.read_bytes(), failed_path.read_bytes()) == before
    (directory / "COMPLETE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="provenance"):
        _validate_failed_status(
            tmp_path,
            "519_0",
            "a" * 40,
            "final_m2_summary_reference",
            143,
        )


def test_recovery_accepts_hashed_running_only_cancellation_but_never_complete(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "outputs/private/run_status/545_8"
    directory.mkdir(parents=True)
    running = {
        "state": "RUNNING",
        "git_commit": "a" * 40,
        "job_id": "545_8",
        "run_kind": "final_m2_summary_shard_reference",
    }
    running_path = directory / "RUNNING.json"
    running_path.write_text(json.dumps(running), encoding="utf-8")
    before = running_path.read_bytes()
    hashes = _validate_canceled_or_unstarted_status(
        tmp_path,
        "545_8",
        "a" * 40,
        "final_m2_summary_shard_reference",
    )
    assert set(hashes) == {"RUNNING.json"}
    assert running_path.read_bytes() == before
    (directory / "COMPLETE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="unexpectedly has COMPLETE"):
        _validate_canceled_or_unstarted_status(
            tmp_path,
            "545_8",
            "a" * 40,
            "final_m2_summary_shard_reference",
        )


def test_cross_commit_m3_reuse_requires_hashed_recovery_report() -> None:
    with pytest.raises(ContractError, match="exact hashed recovery report"):
        aggregate_m3_validation(
            repository_root=Path.cwd(),
            config_path=CONFIG,
            stage="reference",
            source_git_commit="b" * 40,
            evidence_source_git_commit="a" * 40,
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://10.45.177.17:8000/v1",
        "http://api.openai.com:80/v1",
        "http://127.0.0.1:8000/v1",
        "http://user:secret@10.45.177.17:8000/v1",
    ],
)
def test_m2_rejects_non_authorized_or_credentialed_summary_routes(url: str) -> None:
    with pytest.raises(ContractError):
        validate_on_premise_api_base(url)
    assert validate_on_premise_api_base("http://10.45.177.17:8000/v1") == (
        "http://10.45.177.17:8000/v1/"
    )


def test_summary_cache_key_covers_all_required_generation_inputs() -> None:
    base = {
        "source_hash": "a" * 64,
        "condition": "C1_correct_compact_time",
        "prompt_hash": "b" * 64,
        "model_revision_hash": "c" * 64,
        "decoding_hash": "d" * 64,
    }
    keys = {summary_cache_key(**{**base, field: f"changed-{field}"}) for field in base}
    keys.add(summary_cache_key(**base))
    assert len(keys) == len(base) + 1


def test_m2_bounded_decoding_uses_one_fixed_identity() -> None:
    variants = m2_decoding_variants(_spec().m2)
    assert [name for name, _ in variants] == ["bounded_6144"]
    assert [decoding["max_output_tokens"] for _, decoding in variants] == [6144]
    assert all(decoding["temperature"] == 0.0 for _, decoding in variants)
    assert len({stable_hash(decoding) for _, decoding in variants}) == 1
    assert OnPremiseSummaryClient(
        api_base="http://10.45.177.17:8000/v1", model="synthetic"
    ).timeout_seconds == 1200


def test_m2_summary_shards_are_deterministic_complete_and_disjoint() -> None:
    keys = [stable_hash({"synthetic": index}) for index in range(100)]
    assignments = [summary_shard_for_key(key, 16) for key in keys]
    assert assignments == [summary_shard_for_key(key, 16) for key in keys]
    assert set(assignments) == set(range(16))
    assert sum(assignments.count(index) for index in range(16)) == len(keys)
    with pytest.raises(ContractError):
        summary_shard_for_key("not-a-hash", 8)


def test_m2_reuses_only_exact_downstream_fit_summary_payloads() -> None:
    source = "[TIME=1_30_DAYS_BEFORE_T]\n- LOINC/X | values=1 | occurrences=1"
    payload = {"summary": source}
    assert exact_summary_payload_is_reusable(
        payload,
        source_text=source,
        token_counter=lambda text: len(text),
    )
    assert not exact_summary_payload_is_reusable(
        {"summary": source + "\n- ADDED/X | occurrences=1"},
        source_text=source,
        token_counter=lambda text: len(text),
    )

    def reject(_: str) -> int:
        raise NonRetryableSummaryError(
            "M2 summary exceeds the frozen downstream 8,192-token context."
        )

    assert not exact_summary_payload_is_reusable(
        payload,
        source_text=source,
        token_counter=reject,
    )


def test_m2_bounded_summary_rejects_unsupported_facts_and_preserves_time() -> None:
    source = (
        "[TIME=31_60_DAYS_BEFORE_T]\n"
        "- LOINC/X | values=10,11 | unit=g/dL | source=measurement | occurrences=2\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- CPT4/Y | source=procedure | occurrences=1"
    )
    summary = (
        "[TIME=31_60_DAYS_BEFORE_T]\n"
        "- LOINC/X | values=10,11 | unit=g/dL | occurrences=2 | trend=increasing\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- CPT4/Y"
    )
    validation = validate_bounded_summary(
        summary,
        source_text=source,
        token_counter=lambda value: len(value.split()),
        thresholds=_fidelity_thresholds(),
    )
    payload = {
        "summary": summary,
        "bounded_validation_passed": True,
        "bounded_validation": validation,
    }
    assert validation["status"] == "PASS"
    assert validation["time_label_sequence_exact"] is True
    assert bounded_summary_payload_is_reusable(
        payload,
        source_text=source,
        token_counter=lambda value: len(value.split()),
        thresholds=_fidelity_thresholds(),
    )
    with pytest.raises(NonRetryableSummaryError, match="unsupported"):
        validate_bounded_summary(
            summary.replace("CPT4/Y", "CPT4/INVENTED"),
            source_text=source,
            token_counter=lambda value: len(value.split()),
            thresholds=_fidelity_thresholds(),
        )
    with pytest.raises(NonRetryableSummaryError, match="time label"):
        validate_bounded_summary(
            summary.replace("31_60", "61_90"),
            source_text=source,
            token_counter=lambda value: len(value.split()),
            thresholds=_fidelity_thresholds(),
        )
    with pytest.raises(NonRetryableSummaryError, match="unsupported field"):
        validate_bounded_summary(
            summary.replace("- CPT4/Y", "- CPT4/Y | source=procedure"),
            source_text=source,
            token_counter=lambda value: len(value.split()),
            thresholds=_fidelity_thresholds(),
        )
    moved = (
        "[TIME=31_60_DAYS_BEFORE_T]\n"
        "- CPT4/Y\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- LOINC/X | values=10,11 | unit=g/dL | occurrences=2 | trend=increasing"
    )
    with pytest.raises(NonRetryableSummaryError, match="time block"):
        validate_bounded_summary(
            moved,
            source_text=source,
            token_counter=lambda value: len(value.split()),
            thresholds=_fidelity_thresholds(),
        )


def test_m2_bounded_summary_preserves_structured_text_values() -> None:
    source = (
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        '- CPT4/Y | text_values=["present","absent"] | source=procedure | occurrences=2'
    )
    summary = (
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        '- CPT4/Y | text_values=["present","absent"] | occurrences=2'
    )
    validation = validate_bounded_summary(
        summary,
        source_text=source,
        token_counter=lambda value: len(value.split()),
        thresholds=_fidelity_thresholds(),
    )
    assert validation["retention"]["text_values"] == 1.0
    with pytest.raises(NonRetryableSummaryError, match="unsupported"):
        validate_bounded_summary(
            summary.replace("absent", "invented"),
            source_text=source,
            token_counter=lambda value: len(value.split()),
            thresholds=_fidelity_thresholds(),
        )


def test_m2_bounded_summary_contract_attempts_are_fixed_and_bounded() -> None:
    source = "[TIME=1_30_DAYS_BEFORE_T]\n- LOINC/X | values=10 | unit=g/dL | occurrences=1"
    invalid = "[TIME=1_30_DAYS_BEFORE_T]\n- LOINC/INVENTED"
    valid = "[TIME=1_30_DAYS_BEFORE_T]\n- LOINC/X | values=10 | unit=g/dL"
    client = _SyntheticSummaryClient(invalid)
    responses = iter((invalid, valid))
    calls = 0

    def response(path, payload=None):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return {
            "choices": [
                {"message": {"content": next(responses)}, "finish_reason": "stop"}
            ]
        }

    client._request = response  # type: ignore[method-assign]
    generation, validation, attempt = _generate_bounded_summary_with_attempts(
        client,
        system_prompt="frozen prompt",
        source_text=source,
        decoding=dict(_spec().m2["decoding"]),
        request_retries=3,
        contract_attempts=3,
        token_counter=lambda value: len(value.split()),
        thresholds=_fidelity_thresholds(),
    )
    assert generation.summary == valid
    assert validation["status"] == "PASS"
    assert attempt == 2
    assert calls == 2

    client = _SyntheticSummaryClient(invalid)
    with pytest.raises(NonRetryableSummaryError, match="failed all fixed contract attempts"):
        _generate_bounded_summary_with_attempts(
            client,
            system_prompt="frozen prompt",
            source_text=source,
            decoding=dict(_spec().m2["decoding"]),
            request_retries=3,
            contract_attempts=1,
            token_counter=lambda value: len(value.split()),
            thresholds=_fidelity_thresholds(),
        )


def test_m2_embedding_namespace_isolates_source_commits() -> None:
    first = m2_embedding_root(
        _spec(),
        task=FINAL_TASKS[0],
        label="C1_correct_compact_time",
        source_git_commit="a" * 40,
    )
    second = m2_embedding_root(
        _spec(),
        task=FINAL_TASKS[0],
        label="C1_correct_compact_time",
        source_git_commit="b" * 40,
    )
    assert first != second
    assert "source_" + "a" * 40 in first.parts
    with pytest.raises(ContractError):
        m2_embedding_root(
            _spec(),
            task=FINAL_TASKS[0],
            label="C1_correct_compact_time",
            source_git_commit="short",
        )


def test_m2_receives_observation_windows_without_m1_prediction_instruction() -> None:
    instruction = "Represent this record for predicting a future outcome."
    prepared = f"{instruction}\n[TIME=1_30_DAYS_BEFORE_T]\n- code"
    assert m2_source_text(prepared, instruction) == (
        "[TIME=1_30_DAYS_BEFORE_T]\n- code | occurrences=1"
    )
    assert m2_source_text(instruction, instruction) == "[NO_PRE_T_EVENTS]"
    with pytest.raises(ContractError, match="locked task instruction"):
        m2_source_text(prepared, "different instruction")


def test_m2_deterministically_compacts_exact_repeats_within_each_window() -> None:
    source = (
        "[TIME=31_60_DAYS_BEFORE_T]\n"
        "- LOINC/X | value=0.1 | unit=g/dL | source=measurement\n"
        "- LOINC/X | value=0.1 | unit=g/dL | source=measurement\n"
        "- LOINC/X | value=0.2 | unit=g/dL | source=measurement\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- LOINC/X | value=0.1 | unit=g/dL | source=measurement\n"
        "- LOINC/X | value=0.1 | unit=g/dL | source=measurement"
    )
    assert compact_exact_duplicate_facts(source) == (
        "[TIME=31_60_DAYS_BEFORE_T]\n"
        "- LOINC/X | value=0.1 | unit=g/dL | source=measurement | occurrences=2\n"
        "- LOINC/X | value=0.2 | unit=g/dL | source=measurement\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- LOINC/X | value=0.1 | unit=g/dL | source=measurement | occurrences=2"
    )


def test_m2_source_compaction_is_versioned_and_rejects_double_counting() -> None:
    assert M2_SOURCE_COMPACTION_SCHEMA == "m2_grouped_facts_v2"
    source = (
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- LOINC/X | value=1 | occurrences=2\n"
        "- LOINC/X | value=1 | occurrences=2"
    )
    with pytest.raises(ContractError, match="already contains"):
        compact_exact_duplicate_facts(source)


def test_m2_groups_code_source_unit_and_preserves_values_text_and_order() -> None:
    source = (
        "[TIME=31_60_DAYS_BEFORE_T]\n"
        "- LOINC/X | value=0.1 | unit=g/dL | source=measurement\n"
        "- CPT4/Y | text_value=present | source=procedure\n"
        "- LOINC/X | value=0.2 | unit=g/dL | source=measurement\n"
        "- LOINC/X | value=0.1 | unit=g/dL | source=measurement\n"
        "- CPT4/Y | text_value=absent | source=procedure\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- LOINC/X | value=0.3 | unit=g/dL | source=measurement"
    )
    assert group_structured_facts(source) == (
        "[TIME=31_60_DAYS_BEFORE_T]\n"
        "- LOINC/X | values=0.1,0.2 | unit=g/dL | source=measurement | occurrences=3\n"
        '- CPT4/Y | text_values=["present","absent"] | source=procedure | occurrences=2\n'
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- LOINC/X | values=0.3 | unit=g/dL | source=measurement | occurrences=1"
    )


def test_m2_grouping_rejects_double_grouping_and_unknown_lines() -> None:
    with pytest.raises(ContractError, match="ungrouped"):
        group_structured_facts(
            "[TIME=1_30_DAYS_BEFORE_T]\n- LOINC/X | source=measurement | occurrences=2"
        )
    with pytest.raises(ContractError, match="window/fact grammar"):
        group_structured_facts("heading\n- LOINC/X | source=measurement")


def test_m2_downstream_token_gate_rejects_over_context() -> None:
    class FakeTokenizer:
        @staticmethod
        def encode(summary: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return list(range(len(summary.split())))

    validator = object.__new__(LocalSummaryTokenValidator)
    validator.tokenizer = FakeTokenizer()
    validator.maximum_tokens = 3
    assert validator("one two three") == 3
    with pytest.raises(RuntimeError, match="8,192-token"):
        validator("one two three four")
    assert M2_EMBEDDING_MAX_TOKENS == 8192
    assert M2_SUMMARY_CACHE_SCHEMA == "m2_summary_cache_v5_bounded_structured"
    assert M2_SUMMARY_VALIDATION_SCHEMA == "m2_bounded_summary_validation_v1"


def test_summary_cache_resumes_exact_rows_and_rejects_drift(tmp_path: Path) -> None:
    cache = SummaryCache(tmp_path)
    payload = {
        "cache_key": "a" * 64,
        "source_hash": "b" * 64,
        "condition": "C1_correct_compact_time",
        "prompt_hash": "c" * 64,
        "model_revision_hash": "d" * 64,
        "decoding_hash": "e" * 64,
        "summary": "[TIME=1_30_DAYS_BEFORE_T]\n- Hemoglobin [LOINC/718-7] | value=10 | unit=g/dL",
        "finish_reason": "stop",
        "prompt_tokens": 20,
        "completion_tokens": 10,
    }
    payload["summary_hash"] = stable_hash(payload["summary"])
    cache.store(payload["cache_key"], payload)
    expected = {key: value for key, value in payload.items() if key != "summary"}
    assert cache.load(payload["cache_key"], expected) == payload
    with pytest.raises(ContractError, match="immutable inputs"):
        cache.load(payload["cache_key"], {**expected, "prompt_hash": "f" * 64})

    for invalid in (None, "None", " null ", ""):
        invalid_payload = {**payload, "cache_key": stable_hash(invalid), "summary": invalid}
        invalid_payload["summary_hash"] = stable_hash(invalid)
        path = cache.path(invalid_payload["cache_key"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(invalid_payload), encoding="utf-8")
        with pytest.raises(ContractError, match="empty, truncated, or hash-mismatched"):
            cache.load(
                invalid_payload["cache_key"],
                {key: value for key, value in invalid_payload.items() if key != "summary"},
            )
    missing_finish = {key: value for key, value in payload.items() if key != "finish_reason"}
    missing_finish["cache_key"] = "f" * 64
    path = cache.path(missing_finish["cache_key"])
    path.write_text(json.dumps(missing_finish), encoding="utf-8")
    with pytest.raises(ContractError, match="truncated"):
        cache.load(
            missing_finish["cache_key"],
            {key: value for key, value in missing_finish.items() if key != "summary"},
        )


def _fidelity_records(*, repair_one: bool = False) -> list[dict[str, object]]:
    records = []
    tasks = FINAL_TASKS
    conditions = (
        "C1_correct_compact_time",
        "C3_shuffled_earlier_time_labels:seed=20260815",
    )
    for task in tasks:
        for condition in conditions:
            for index in range(5):
                source = (
                    "[TIME=31_60_DAYS_BEFORE_T]\n"
                    "- Hemoglobin [LOINC/718-7] | value=10 | unit=g/dL\n"
                    "[TIME=1_30_DAYS_BEFORE_T]\n"
                    "- Hemoglobin [LOINC/718-7] | value=11 | unit=g/dL"
                )
                summary = (
                    "[TIME=31_60_DAYS_BEFORE_T]\n"
                    "- LOINC/718-7 | values=10 | unit=g/dL | occurrences=1\n"
                    "[TIME=1_30_DAYS_BEFORE_T]\n"
                    "- LOINC/718-7 | values=11 | unit=g/dL | occurrences=1 | trend=increasing"
                )
                if repair_one and task == tasks[0] and "C3_" in condition and index == 0:
                    summary = summary.replace("[TIME=31_60_DAYS_BEFORE_T]", "[TIME=61_90_DAYS_BEFORE_T]")
                records.append(
                    {
                        "task": task,
                        "condition": condition,
                        "split": "validation",
                        "instance_key": f"{task}-{condition}-{index}",
                        "source_text": source,
                        "summary": summary,
                    }
                )
    return records


def _fidelity_thresholds() -> dict[str, float | int]:
    return {
        "minimum_concept_retention": 0.9,
        "minimum_numeric_retention": 0.95,
        "minimum_unit_retention": 0.95,
        "minimum_recurrence_retention": 0.8,
        "minimum_trend_retention": 0.8,
        "minimum_time_label_retention": 0.95,
        "maximum_unsupported_fact_rate": 0.01,
        "maximum_contradictions": 0,
        "maximum_timing_repairs": 0,
        "maximum_format_violations": 0,
        "maximum_generation_failure_rate": 0.0,
    }


def test_m2_fidelity_is_balanced_and_preserves_falsified_time() -> None:
    report = audit_summary_fidelity(
        _fidelity_records(),
        thresholds=_fidelity_thresholds(),
        histories_per_task_condition=5,
    )
    assert report["status"] == "PASS"
    assert report["sample_histories"] == 40
    assert report["task_condition_groups"] == 8
    assert report["falsified_timing_repairs"] == 0


def test_m2_fidelity_rejects_repaired_falsified_time() -> None:
    report = audit_summary_fidelity(
        _fidelity_records(repair_one=True),
        thresholds=_fidelity_thresholds(),
        histories_per_task_condition=5,
    )
    assert report["status"] == "FAIL"
    assert report["falsified_timing_repairs"] == 1


def test_m2_fidelity_reports_structured_fact_contradictions() -> None:
    records = _fidelity_records()
    records[0]["summary"] = str(records[0]["summary"]).replace("values=10", "values=99", 1)
    report = audit_summary_fidelity(
        records,
        thresholds=_fidelity_thresholds(),
        histories_per_task_condition=5,
    )
    assert report["status"] == "FAIL"
    assert report["contradictions"] >= 1


def test_m2_fidelity_parses_plain_codes_and_rejects_code_expansion_prose() -> None:
    records = _fidelity_records()
    records[0]["source_text"] = (
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- LOINC/60804-2 | value=0.1 | source=measurement\n"
        "- LOINC/60804-2 | value=0.1 | source=measurement\n"
        "- CPT4/82728 | source=measurement"
    )
    records[0]["summary"] = (
        "Respiratory Mechanics\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- LOINC/60804-2 (Static Compliance) | values=0.1 | "
        "source=measurement | occurrences=2"
    )
    report = audit_summary_fidelity(
        records,
        thresholds=_fidelity_thresholds(),
        histories_per_task_condition=5,
    )
    assert report["status"] == "FAIL"
    assert report["retention"]["concepts"] < 1.0
    assert report["structured_format_violations"] >= 1


def test_m2_fidelity_preserves_plain_codes_containing_spaces() -> None:
    records = _fidelity_records()
    records[0]["source_text"] = (
        "[TIME=1_30_DAYS_BEFORE_T]\n- Ethnicity/Not Hispanic | source=person"
    )
    records[0]["summary"] = (
        "[TIME=1_30_DAYS_BEFORE_T]\n"
        "- Ethnicity/Not Hispanic | source=person | occurrences=1"
    )
    report = audit_summary_fidelity(
        records,
        thresholds=_fidelity_thresholds(),
        histories_per_task_condition=5,
    )
    assert report["retention"]["concepts"] == 1.0


def test_m2_manual_fidelity_requires_all_40_explicit_reviews() -> None:
    packet, decisions = build_manual_fidelity_packet(
        _fidelity_records(),
        histories_per_task_condition=5,
    )
    assert len(packet["entries"]) == 40
    assert all("instance_key" not in row for row in packet["entries"])
    decisions["review_complete"] = True
    for row in decisions["decisions"]:
        for key in tuple(row):
            if key not in {"review_id", "task", "condition"}:
                row[key] = True
    report = validate_manual_fidelity_decisions(packet, decisions)
    assert report["status"] == "PASS"
    assert report["sample_histories"] == 40
    decisions["decisions"][0]["no_timing_repair"] = False
    report = validate_manual_fidelity_decisions(packet, decisions)
    assert report["status"] == "FAIL"
    assert report["criteria_failure_counts"]["no_timing_repair"] == 1


def test_m3_window_split_keeps_64_most_recent_without_reordering() -> None:
    text = "instruction\n" + "\n".join(
        f"[TIME={index * 30 + 1}_{(index + 1) * 30}_DAYS_BEFORE_T]\n- code-{index}"
        for index in range(70, -1, -1)
    )
    blocks = split_serialized_windows(text, maximum_windows=64)
    assert len(blocks) == 64
    assert "code-63" in blocks[0]
    assert "code-0" in blocks[-1]


def test_m3_padding_mask_and_fixed_projection_are_deterministic() -> None:
    first = np.arange(12, dtype=np.float32).reshape(3, 4)
    second = np.arange(8, dtype=np.float32).reshape(2, 4)
    padded, lengths, mask = pad_sequences([first, second], maximum_windows=4)
    assert padded.shape == (2, 4, 4)
    assert lengths.tolist() == [3, 2]
    assert mask.tolist() == [[True, True, True, False], [True, True, False, False]]
    projected_a = fixed_projection(padded, output_dimension=2, seed=20260819)
    projected_b = fixed_projection(padded, output_dimension=2, seed=20260819)
    projected_c = fixed_projection(padded, output_dimension=2, seed=20260820)
    assert np.array_equal(projected_a, projected_b)
    assert not np.array_equal(projected_a, projected_c)
    projected_then_padded, projected_lengths, projected_mask = project_then_pad_sequences(
        [first, second],
        maximum_windows=4,
        output_dimension=2,
        seed=20260819,
    )
    assert np.allclose(projected_then_padded, projected_a)
    assert np.array_equal(projected_lengths, lengths)
    assert np.array_equal(projected_mask, mask)


def test_m3_early_stopping_maximizes_auroc_and_uses_log_loss_only_for_ties() -> None:
    assert validation_auroc_improved(
        current_auroc=0.71,
        best_auroc=0.70,
        current_log_loss=0.70,
        best_log_loss=0.50,
    )
    assert validation_auroc_improved(
        current_auroc=0.70,
        best_auroc=0.70,
        current_log_loss=0.49,
        best_log_loss=0.50,
    )
    assert not validation_auroc_improved(
        current_auroc=0.69,
        best_auroc=0.70,
        current_log_loss=0.40,
        best_log_loss=0.50,
    )


def test_m3_three_seed_patient_alignment_and_probability_average() -> None:
    rows = {
        seed: [
            {
                "task": "lab_anemia",
                "condition": "C1_correct_compact_time",
                "instance_key": f"row-{index}",
                "target": bool(index),
                "probability": 0.1 * index + offset,
            }
            for index in range(2)
        ]
        for seed, offset in zip(M3_SEEDS, (0.0, 0.1, 0.2), strict=True)
    }
    averaged = _average_seed_predictions(rows)
    assert averaged[0]["probability"] == pytest.approx(0.1)
    assert averaged[1]["probability"] == pytest.approx(0.2)
    assert averaged[0]["seeds_averaged"] == list(M3_SEEDS)
    rows[M3_SEEDS[-1]] = rows[M3_SEEDS[-1]][:1]
    with pytest.raises(ContractError, match="identical validation patients"):
        _average_seed_predictions(rows)


def test_m3_sensitivity_is_blocked_before_reference_selection(tmp_path: Path) -> None:
    spec = _spec()
    assert m3_stage_labels(spec, "reference") == ("C1_correct_compact_time",)
    assert spec.m2["sensitivity_conditions"][0] == "B1_latest_window_only"
    assert spec.m3["sensitivity_conditions"][0] == "B1_latest_window_only"
    assert spec.m3["early_stopping_metric"] == "validation_auroc"
    assert spec.m3["embedding_batch_size"] == 32
    assert not (tmp_path / "approach_selection.json").exists()
    # The real spec points only to the Git-ignored governed namespace and currently has no selection.
    if not (spec.aggregate_root / "approach_selection.json").exists():
        with pytest.raises(ContractError, match="completed reference"):
            m3_stage_labels(spec, "sensitivity")


def test_approach_selection_margin_and_no_margin_priority() -> None:
    decisive = {
        "M1": {"macro_auroc": 0.70, "macro_log_loss": 0.60},
        "M2": {"macro_auroc": 0.72, "macro_log_loss": 0.59},
        "M3": {"macro_auroc": 0.71, "macro_log_loss": 0.58},
    }
    assert select_approach(
        decisive,
        minimum_margin=0.005,
        no_margin_priority=("M1", "M2", "M3"),
    )["selected_approach"] == "M2"
    close = {
        **decisive,
        "M2": {"macro_auroc": 0.703, "macro_log_loss": 0.40},
        "M3": {"macro_auroc": 0.702, "macro_log_loss": 0.58},
    }
    assert select_approach(
        close,
        minimum_margin=0.005,
        no_margin_priority=("M1", "M2", "M3"),
    )["selected_approach"] == "M1"


def test_prompt4_array_mapping_is_task_major_and_seed_minor() -> None:
    module_spec = importlib.util.spec_from_file_location("run_final_m23", "scripts/run_final_m23.py")
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.m3_array_mapping(0, FINAL_TASKS, M3_SEEDS) == ("lab_anemia", 20260815)
    assert module.m3_array_mapping(11, FINAL_TASKS, M3_SEEDS) == ("chexpert", 20260817)
    with pytest.raises(ValueError):
        module.m3_array_mapping(12, FINAL_TASKS, M3_SEEDS)


def test_prompt4_batch_scripts_use_atomic_sentinels_and_no_test_commands() -> None:
    scripts = sorted(Path("scripts/slurm").glob("2[4-8]_*.sbatch"))
    assert [path.name for path in scripts] == [
        "24_m2_summary_task_array.sbatch",
        "25_m2_fit_validation.sbatch",
        "26_m3_task_seed_array.sbatch",
        "27_m3_fit_validation.sbatch",
        "28_freeze_protocol.sbatch",
    ]
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "final_initialize" in text
        assert "final_mark_success" in text
        assert "final_container_python" in text
        assert "test_split" not in text
        assert "OpenAI" not in text
    manual = Path("scripts/slurm/25a_prepare_m2_fidelity_review.sbatch").read_text(
        encoding="utf-8"
    )
    assert "review_m2_fidelity_manual.py prepare" in manual
    assert "final_initialize" in manual
    assert "final_mark_success" in manual
    summary = Path("scripts/slurm/24_m2_summary_task_array.sbatch").read_text(encoding="utf-8")
    embed = Path("scripts/slurm/24b_m2_embed_task_array.sbatch").read_text(encoding="utf-8")
    m3_prepare = Path("scripts/slurm/26a_m3_prepare_task_array.sbatch").read_text(
        encoding="utf-8"
    )
    m3_train = Path("scripts/slurm/26_m3_task_seed_array.sbatch").read_text(encoding="utf-8")
    assert "--summaries-only" in summary
    assert "#SBATCH --gres" not in summary
    assert "#SBATCH --gres=mps:1" in embed
    assert "m3-prepare-task" in m3_prepare
    assert "#SBATCH --array=0-3%1" in m3_prepare
    assert "#SBATCH --array=0-11%2" in m3_train
    assert "TEMPORAL_EHR_MODEL_ROOT" not in m3_train


def test_prompt4_common_wrapper_jobs_define_the_container_image() -> None:
    for path in Path("scripts/slurm").glob("*.sbatch"):
        source = path.read_text(encoding="utf-8")
        if 'source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"' in source:
            assert "CONTAINER_IMAGE=" in source, f"{path.name} omits CONTAINER_IMAGE"


def test_prompt4_submission_splits_m3_preparation_from_parallel_seed_training() -> None:
    submit = Path("scripts/slurm/submit_prompt4_execution.sh").read_text(encoding="utf-8")
    assert "M3_PREP_ARRAY_JOB_ID" in submit
    assert 'afterok:${M3_PREP_ARRAY_JOB_ID}' in submit
    assert '0-11%${M3_TRAIN_PARALLELISM}' in submit
    assert "PROMPT4_M3_TRAIN_PARALLELISM" in submit
    assert "SENSITIVITY_M3_PREP_ARRAY_JOB_ID" in submit


def test_prompt4_m2_recovery_reruns_only_m2_and_reuses_canonical_m3() -> None:
    source = Path("scripts/slurm/submit_prompt4_m2_recovery.sh").read_text(encoding="utf-8")
    assert "validate_prompt4_m2_recovery.py" in source
    assert "--live-endpoint-check" in source
    assert "M3_PREP_ARRAY_JOB_ID=470" in source
    assert "M3_ARRAY_JOB_ID=471" in source
    assert "24_m2_summary_task_array.sbatch" in source
    assert "24b_m2_embed_task_array.sbatch" in source
    assert "25a_prepare_m2_fidelity_review.sbatch" in source
    assert "26a_m3_prepare_task_array.sbatch" not in source
    assert "26_m3_task_seed_array.sbatch" not in source


def test_prompt4_m2_fidelity_v2_recovery_is_fail_closed_and_reuses_m3() -> None:
    source = Path("scripts/slurm/submit_prompt4_m2_fidelity_v2.sh").read_text(
        encoding="utf-8"
    )
    assert "validate_prompt4_m2_fidelity_v2_recovery.py" in source
    assert "--live-endpoint-check" in source
    assert "24_m2_summary_task_array.sbatch" in source
    assert "24b_m2_embed_task_array.sbatch" in source
    assert "25a_prepare_m2_fidelity_review.sbatch" in source
    assert "M3_PREP_ARRAY_JOB_ID=470" in source
    assert "M3_ARRAY_JOB_ID=471" in source
    assert "26a_m3_prepare_task_array.sbatch" not in source
    assert "26_m3_task_seed_array.sbatch" not in source
    recovery = Path("src/temporal_ehr/final_m23/recovery.py").read_text(encoding="utf-8")
    assert "compact_exact_duplicate_facts(uncompressed_source)" in recovery
    assert "for attempt in range(3)" in recovery


def test_prompt4_m2_fidelity_v3_recovery_is_fail_closed_and_reuses_m3() -> None:
    source = Path("scripts/slurm/submit_prompt4_m2_fidelity_v3.sh").read_text(
        encoding="utf-8"
    )
    assert "validate_prompt4_m2_fidelity_v3_recovery.py" in source
    assert "--live-endpoint-check" in source
    assert "24_m2_summary_task_array.sbatch" in source
    assert "24b_m2_embed_task_array.sbatch" in source
    assert "25a_prepare_m2_fidelity_review.sbatch" in source
    assert "M3_PREP_ARRAY_JOB_ID=470" in source
    assert "M3_ARRAY_JOB_ID=471" in source
    assert "26a_m3_prepare_task_array.sbatch" not in source
    assert "26_m3_task_seed_array.sbatch" not in source
    assert "prompt4_m2_fidelity_v3_after_519.yaml" in source
    assert "M2_RECOVERY_REPORT=" in source
    assert "M2_RECOVERY_REPORT_SHA256=" in source
    recovery = Path("src/temporal_ehr/final_m23/recovery.py").read_text(encoding="utf-8")
    assert "preserved_job_519_status_sha256" in recovery
    assert "group_structured_facts(ungrouped)" in recovery
    assert "generation.summary.strip() != grouped" in recovery


def test_prompt4_m2_fidelity_v4_recovery_is_sharded_resumable_and_notified() -> None:
    submit = Path("scripts/slurm/submit_prompt4_m2_fidelity_v4.sh").read_text(
        encoding="utf-8"
    )
    shard = Path("scripts/slurm/24c_m2_summary_shard_array.sbatch").read_text(
        encoding="utf-8"
    )
    assert "validate_prompt4_m2_fidelity_v4_recovery.py" in submit
    assert "prompt4_m2_fidelity_v4_after_529.yaml" in submit
    assert "24c_m2_summary_shard_array.sbatch" in submit
    assert "M2_SUMMARY_SHARDS_PER_TASK=8" in submit
    assert '0-${LAST_SHARD_INDEX}%${M2_SUMMARY_PARALLELISM}' in submit
    assert "M3_PREP_ARRAY_JOB_ID=470" in submit
    assert "M3_ARRAY_JOB_ID=471" in submit
    assert "TIME_LIMIT_80,TIME_LIMIT_90" in submit
    assert "cleanup_partial_submission" in submit
    assert "reusing immutable fidelity-v4 preflight" in submit
    assert "--summary-shard-index" in shard
    assert "--summary-shard-count" in shard
    assert "#SBATCH --array=0-767%4" in shard
    assert "#SBATCH --time=12:00:00" in shard
    assert "final_initialize" in shard
    assert "final_mark_success" in shard
    assert "26_m3_task_seed_array.sbatch" not in submit
    general_submit = Path("scripts/slurm/submit_prompt4_execution.sh").read_text(
        encoding="utf-8"
    )
    assert "Direct reference submission is superseded" in general_submit
    assert "M2_REFERENCE_MAX_INDEX" in general_submit
    assert "M2_SENSITIVITY_MAX_INDEX" in general_submit
    assert "scripts/slurm/24c_m2_summary_shard_array.sbatch" in general_submit


def test_prompt4_m2_fidelity_v5_recovery_preserves_old_tiers_and_extends_generation() -> None:
    submit = Path("scripts/slurm/submit_prompt4_m2_fidelity_v5.sh").read_text(
        encoding="utf-8"
    )
    validator = Path("scripts/validate_prompt4_m2_fidelity_v5_recovery.py").read_text(
        encoding="utf-8"
    )
    recovery = Path("src/temporal_ehr/final_m23/recovery.py").read_text(encoding="utf-8")
    assert "prompt4_m2_fidelity_v5_after_536.yaml" in submit
    assert "validate_prompt4_m2_fidelity_v5_recovery.py" in submit
    assert "Job 536 must leave the queue" in submit
    assert "0-${LAST_SHARD_INDEX}%${M2_SUMMARY_PARALLELISM}" in submit
    assert "M3_PREP_ARRAY_JOB_ID=470" in submit
    assert "M3_ARRAY_JOB_ID=471" in submit
    assert "validate_prompt4_m2_fidelity_v5_recovery" in validator
    assert "finish_reason='length'" in recovery
    assert 'decoding_hashes["extended_16384"]' in recovery
    assert '"downstream_embedding_tokens": M2_EMBEDDING_MAX_TOKENS' in recovery


def test_prompt4_m2_fidelity_v6_uses_bounded_summaries_and_smaller_shards() -> None:
    submit = Path("scripts/slurm/submit_prompt4_m2_fidelity_v6.sh").read_text(
        encoding="utf-8"
    )
    validator = Path("scripts/validate_prompt4_m2_fidelity_v6_recovery.py").read_text(
        encoding="utf-8"
    )
    assert "prompt4_m2_fidelity_v6_after_545.yaml" in submit
    assert "validate_prompt4_m2_fidelity_v6_recovery.py" in submit
    assert "Jobs 545, 548, and 549 must leave the queue" in submit
    assert "M2_SUMMARY_SHARDS_PER_TASK=16" in submit
    assert "PROMPT4_M2_SUMMARY_PARALLELISM:-4" in submit
    assert "M2_MAXIMUM_TOTAL_ENDPOINT_REQUESTS" in submit
    assert "M3_PREP_ARRAY_JOB_ID=470" in submit
    assert "M3_ARRAY_JOB_ID=471" in submit
    assert "validate_prompt4_m2_fidelity_v6_recovery" in validator
    assert "--live-endpoint-check" in submit
    assert "26_m3_task_seed_array.sbatch" not in submit
    assert "M2_SUMMARY_SHARDS_PER_TASK=16" in submit


def test_prompt4_m2_fidelity_v7_is_capacity_selected_and_finely_sharded() -> None:
    submit = Path("scripts/slurm/submit_prompt4_m2_fidelity_v7.sh").read_text(
        encoding="utf-8"
    )
    validator = Path("scripts/validate_prompt4_m2_fidelity_v7_recovery.py").read_text(
        encoding="utf-8"
    )
    shard = Path("scripts/slurm/24c_m2_summary_shard_array.sbatch").read_text(
        encoding="utf-8"
    )
    assert "prompt4_m2_fidelity_v7_after_559.yaml" in submit
    assert "validate_prompt4_m2_fidelity_v7_recovery.py" in submit
    assert "M2_SUMMARY_SHARDS_PER_TASK=192" in submit
    assert "M2_REQUESTS_PER_SHARD=8" in submit
    assert "recommended_global_concurrency" in submit
    assert "MaxArraySize" in submit
    assert "{print $3; exit}" not in submit
    assert "END {if (value != \"\") print value}" in submit
    assert '0-${LAST_SHARD_INDEX}%${M2_SUMMARY_PARALLELISM}' in submit
    assert "Jobs 559, 560, and 561 must leave the queue" in submit
    assert "M3_PREP_ARRAY_JOB_ID=470" in submit
    assert "M3_ARRAY_JOB_ID=471" in submit
    assert "validate_prompt4_m2_fidelity_v7_recovery" in validator
    assert "#SBATCH --array=0-767%4" in shard
    assert "#SBATCH --cpus-per-task=4" in shard
    assert "#SBATCH --mem=16G" in shard
    assert "192 shards for each of four tasks" in shard
    assert "TASK_INDEX=$((ARRAY_INDEX % 4))" in shard
    assert "SHARD_INDEX=$((ARRAY_INDEX / 4))" in shard


def test_prompt4_m2_capacity_v7_is_synthetic_and_bounded_at_32() -> None:
    path = Path("scripts/dgx/benchmark_prompt4_m2_capacity_v7.py")
    source = path.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("prompt4_m2_capacity_v7", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CONCURRENCY_LADDER == (8, 16, 24, 32)
    assert module.SYNTHETIC_FACT_COUNT == 96
    assert "SYNTHETIC_ENGINEERING_CAPACITY_ONLY" in source
    assert "patient_data_used" in source
    assert "EHRSHOT_DATA_PATH" not in source
    assert module.recommend_concurrency(
        [
            {"concurrency": 8, "failed_requests": 0, "completion_tokens_per_second": 1.0},
            {"concurrency": 16, "failed_requests": 0, "completion_tokens_per_second": 1.8},
            {"concurrency": 24, "failed_requests": 0, "completion_tokens_per_second": 2.0},
            {"concurrency": 32, "failed_requests": 0, "completion_tokens_per_second": 1.9},
        ]
    ) == 32
    progress = Path("scripts/slurm/prompt4_m2_progress.sh").read_text(
        encoding="utf-8"
    )
    assert "Summary shards: complete=" in progress
    assert "Embedding tasks: complete=" in progress
    assert "m2_summary_progress" in progress
    assert "scancel" not in progress
    assert "sbatch" not in progress


def test_prompt4_m3_aggregate_supports_explicit_validated_compute_commit() -> None:
    source = Path("src/temporal_ehr/final_m23/reporting.py").read_text(encoding="utf-8")
    assert "evidence_source_git_commit" in source
    assert '"m3_compute_git_commit": evidence_commit' in source
    runner = Path("scripts/run_final_m23.py").read_text(encoding="utf-8")
    assert "PROMPT4_M3_EVIDENCE_SOURCE_GIT_COMMIT" in runner
    assert "m3-reference-report" in runner
    assert "select_reference_approach=args.command == \"m3-fit\"" in runner


def test_prompt4_overnight_launcher_advances_without_test_or_m3_recompute() -> None:
    arm = Path("scripts/slurm/arm_prompt4_m2_overnight_v7.sh").read_text(
        encoding="utf-8"
    )
    launcher = Path(
        "scripts/slurm/19c_launch_prompt4_m2_overnight_v7.sbatch"
    ).read_text(encoding="utf-8")
    m3_report = Path("scripts/slurm/27a_m3_reference_report.sbatch").read_text(
        encoding="utf-8"
    )
    status = Path("scripts/slurm/prompt4_overnight_status.sh").read_text(
        encoding="utf-8"
    )
    assert "afterany:${CAPACITY_JOB_ID}" in arm
    assert "automation is already armed" in arm
    assert "submit_prompt4_m2_capacity_v7.sh" in arm
    assert "submit_prompt4_m2_fidelity_v7.sh" in launcher
    assert "27a_m3_reference_report.sbatch" in launcher
    assert "26_m3_task_seed_array.sbatch" not in launcher
    assert "m3-reference-report" in m3_report
    assert "m3-task-seed" not in m3_report
    assert "prompt4_m2_progress.sh" in status
    combined = "\n".join((arm, launcher, m3_report, status))
    assert "test performance" not in combined.casefold()
    assert "held-out" not in combined.casefold()


def test_final_job_wrapper_attempts_atomic_status_on_signals_and_container_failure() -> None:
    source = Path("scripts/slurm/final_job_common.sh").read_text(encoding="utf-8")
    assert "trap 'exit 143' TERM" in source
    assert "trap 'exit 130' INT" in source
    assert "if ! final_container_python" in source
    assert 'PYTHONPATH=src "${PYTHON}" scripts/write_run_status.py' in source


def test_prompt4_status_helper_is_read_only_and_sentinel_aware() -> None:
    source = Path("scripts/slurm/prompt4_status.sh").read_text(encoding="utf-8")
    assert "squeue" in source
    assert "COMPLETE.json" in source
    assert "FAILED.json" in source
    assert "scancel" not in source
    assert "sbatch" not in source


def test_prompt4_freeze_authorizes_both_co_primary_contrasts() -> None:
    source = Path("src/temporal_ehr/final_m23/reporting.py").read_text(encoding="utf-8")
    assert "C1_correct_compact_time minus B1_latest_window_only" in source
    assert "C1_correct_compact_time minus each of three fixed C3" in source
    assert '"m2_completion_policy_hash"' in source
    submit = Path("scripts/slurm/submit_prompt4_execution.sh").read_text(encoding="utf-8")
    assert "reference-finalize" in submit
    assert "review_m2_fidelity_manual.py finalize" in submit


def test_prompt4_capacity_probe_is_synthetic_only() -> None:
    source = Path("scripts/dgx/benchmark_prompt4_capacity.py").read_text(encoding="utf-8")
    assert "SYNTHETIC_ENGINEERING_CAPACITY_ONLY" in source
    assert "patient_data_used" in source
    assert "EHRSHOT_DATA_PATH" not in source


def test_prompt4_source_contains_no_external_summary_provider_or_patient_artifact() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/temporal_ehr/final_m23").glob("*.py")
    )
    assert "api.openai.com" not in source
    assert "api.anthropic.com" not in source
    assert "outputs/private/final/locked_four_task_m2_m3_v1" in CONFIG.read_text(encoding="utf-8")
    assert "reports/generated/final/locked_four_task_m2_m3_v1" in CONFIG.read_text(encoding="utf-8")
    for path in Path("reports/generated/final/locked_four_task_m2_m3_v1").glob("*.json"):
        text = path.read_text(encoding="utf-8").lower()
        assert "subject_id" not in text
        assert "prediction_time" not in text
        assert "serialized_history" not in text


def test_handoff_allowlist_includes_only_prompt4_aggregates_and_config() -> None:
    text = Path("scripts/build_handoff_bundle.py").read_text(encoding="utf-8")
    assert "configs/experiments/final_m2_m3.yaml" in text
    assert "FROZEN_EXPERIMENT_MANIFEST.json" in text
    assert "outputs/private/final/locked_four_task_m2_m3_v1" not in text
