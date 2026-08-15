from __future__ import annotations

import duckdb

from temporal_ehr.data import queries


def _one(connection, sql: str, path) -> dict[str, object]:
    cursor = connection.execute(sql, [str(path)])
    names = [description[0] for description in cursor.description]
    return dict(zip(names, cursor.fetchone(), strict=True))


def test_basic_aggregate_query_correctness(synthetic_event_path) -> None:
    connection = duckdb.connect()
    try:
        result = _one(connection, queries.basic_stats_sql(), synthetic_event_path)
    finally:
        connection.close()
    assert result["event_count"] == 14
    assert result["distinct_subject_count"] == 3
    assert result["numeric_valued_events"] == 4
    assert result["text_valued_events"] == 1
    assert result["events_with_both_values"] == 0


def test_duplicate_recurrence_and_concurrency_counts(synthetic_event_path) -> None:
    connection = duckdb.connect()
    try:
        exact = _one(connection, queries.exact_duplicate_sql(), synthetic_event_path)
        same_day = _one(connection, queries.subject_date_code_duplicate_sql(), synthetic_event_path)
        recurrence = _one(connection, queries.recurrence_sql(), synthetic_event_path)
        concurrency = _one(connection, queries.concurrency_sql(), synthetic_event_path)
    finally:
        connection.close()
    assert exact["excess_duplicate_rows"] == 1
    assert same_day["duplicate_subject_date_code_groups"] == 1
    assert recurrence["recurring_subject_code_pairs"] == 2
    assert recurrence["subjects_with_repeated_codes_across_dates"] == 2
    assert concurrency["same_timestamp_groups"] == 2
    assert concurrency["subjects_with_same_timestamp_events"] == 2


def test_subject_and_order_integrity_aggregates(synthetic_event_path) -> None:
    connection = duckdb.connect()
    try:
        subject = _one(connection, queries.subject_integrity_sql(), synthetic_event_path)
        chronology = _one(connection, queries.chronological_order_sql(), synthetic_event_path)
        values = connection.execute(queries.subject_aggregate_values_sql(), [str(synthetic_event_path)] * 2).fetchall()
    finally:
        connection.close()
    assert subject["subjects_with_only_person_or_administrative_events"] == 1
    assert chronology["subjects_out_of_order_after_explicit_sort"] == 0
    assert len(values) == 3


def test_vocabulary_counts_pool_unknown_prefixes(synthetic_event_path) -> None:
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            queries.vocabulary_counts_sql(["LOINC", "SNOMED", "RxNorm", "Visit"]),
            [str(synthetic_event_path)],
        )
        rows = {row[0]: row[1:] for row in cursor.fetchall()}
    finally:
        connection.close()
    assert rows["LOINC"][0] == 4
    assert rows["Other"][0] == 3


def test_visit_code_prefix_counts_use_only_aggregate_public_names(synthetic_event_path) -> None:
    connection = duckdb.connect()
    try:
        rows = connection.execute(queries.visit_code_prefix_counts_sql(), [str(synthetic_event_path)]).fetchall()
    finally:
        connection.close()
    assert rows == [("Visit", 2, 2)]


def test_note_text_audit_does_not_infer_narrative_availability(synthetic_event_path) -> None:
    connection = duckdb.connect()
    try:
        result = _one(connection, queries.basic_stats_sql(), synthetic_event_path)
    finally:
        connection.close()
    assert result["note_events"] == 0
    assert result["note_events_with_text"] == 0
