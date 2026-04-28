from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def get_connection() -> sqlite3.Connection:
    config.ensure_directories()
    return sqlite3.connect(config.DATABASE_FILE)


def initialize_database() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction_label INTEGER NOT NULL,
                prediction_text TEXT NOT NULL,
                prediction_score REAL,
                risk_level TEXT,
                explanation TEXT,
                impact TEXT,
                suggestion TEXT,
                raw_features_json TEXT,
                key_features_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_detection_result(result: dict[str, Any]) -> None:
    initialize_database()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO detection_results (
                record_id, model_name, prediction_label, prediction_text,
                prediction_score, risk_level, explanation, impact, suggestion,
                raw_features_json, key_features_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["record_id"],
                result["model_name"],
                int(result["prediction_label"]),
                result["prediction_text"],
                result.get("prediction_score"),
                result.get("risk_level"),
                result.get("explanation"),
                result.get("impact"),
                result.get("suggestion"),
                json.dumps(result.get("raw_features", {}), ensure_ascii=False),
                json.dumps(result.get("key_features", {}), ensure_ascii=False),
                result["created_at"],
            ),
        )
        conn.commit()


def save_batch_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        save_detection_result(result)


def load_detection_history(limit: int | None = None) -> list[dict[str, Any]]:
    return query_detection_history(limit=limit)


def row_to_result(item: dict[str, Any]) -> dict[str, Any]:
    item["raw_features"] = json.loads(item.pop("raw_features_json") or "{}")
    item["key_features"] = json.loads(item.pop("key_features_json") or "{}")
    return item


def query_detection_history(
    limit: int | None = None,
    offset: int = 0,
    model_name: str | None = None,
    risk_level: str | None = None,
    prediction_text: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
) -> list[dict[str, Any]]:
    initialize_database()
    query = "SELECT * FROM detection_results"
    conditions: list[str] = []
    params: list[Any] = []

    if model_name:
        conditions.append("model_name = ?")
        params.append(model_name)
    if risk_level:
        conditions.append("risk_level = ?")
        params.append(risk_level)
    if prediction_text:
        conditions.append("prediction_text = ?")
        params.append(prediction_text)
    if created_from:
        conditions.append("created_at >= ?")
        params.append(created_from)
    if created_to:
        conditions.append("created_at <= ?")
        params.append(created_to)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    if offset:
        if limit is None:
            query += " LIMIT -1"
        query += " OFFSET ?"
        params.append(offset)

    with get_connection() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
        columns = [desc[0] for desc in conn.execute("SELECT * FROM detection_results").description]

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(zip(columns, row))
        results.append(row_to_result(item))
    return results


def summarize_detection_history(
    model_name: str | None = None,
    risk_level: str | None = None,
    prediction_text: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
) -> dict[str, Any]:
    results = query_detection_history(
        limit=None,
        model_name=model_name,
        risk_level=risk_level,
        prediction_text=prediction_text,
        created_from=created_from,
        created_to=created_to,
    )

    summary = {
        "total": len(results),
        "prediction_counts": {},
        "risk_counts": {},
        "model_counts": {},
    }

    for item in results:
        summary["prediction_counts"][item["prediction_text"]] = summary["prediction_counts"].get(item["prediction_text"], 0) + 1
        summary["risk_counts"][item["risk_level"]] = summary["risk_counts"].get(item["risk_level"], 0) + 1
        summary["model_counts"][item["model_name"]] = summary["model_counts"].get(item["model_name"], 0) + 1

    return summary


def count_detection_history(
    model_name: str | None = None,
    risk_level: str | None = None,
    prediction_text: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
) -> int:
    initialize_database()
    query = "SELECT COUNT(*) FROM detection_results"
    conditions: list[str] = []
    params: list[Any] = []

    if model_name:
        conditions.append("model_name = ?")
        params.append(model_name)
    if risk_level:
        conditions.append("risk_level = ?")
        params.append(risk_level)
    if prediction_text:
        conditions.append("prediction_text = ?")
        params.append(prediction_text)
    if created_from:
        conditions.append("created_at >= ?")
        params.append(created_from)
    if created_to:
        conditions.append("created_at <= ?")
        params.append(created_to)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with get_connection() as conn:
        return int(conn.execute(query, tuple(params)).fetchone()[0])


def export_results_to_json(results: list[dict[str, Any]]) -> None:
    config.DETECTION_RESULTS_JSON_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_results_to_csv(results: list[dict[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    for result in results:
        row = result.copy()
        row["raw_features"] = json.dumps(result.get("raw_features", {}), ensure_ascii=False)
        row["key_features"] = json.dumps(result.get("key_features", {}), ensure_ascii=False)
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(config.DETECTION_RESULTS_CSV_FILE, index=False)


def get_detection_result_by_record_id(record_id: str) -> dict[str, Any] | None:
    initialize_database()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM detection_results WHERE record_id = ? ORDER BY id DESC LIMIT 1",
            (record_id,),
        ).fetchone()
        columns = [desc[0] for desc in conn.execute("SELECT * FROM detection_results").description]
    if not row:
        return None
    return row_to_result(dict(zip(columns, row)))
