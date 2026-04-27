from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    total INTEGER NOT NULL,
    max_severity TEXT NOT NULL,
    distribution_json TEXT NOT NULL,
    decision_support_json TEXT NOT NULL,
    use_llm INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS log_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    line_no INTEGER NOT NULL,
    message TEXT NOT NULL,
    normalized TEXT NOT NULL,
    severity TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities_json TEXT NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dataset_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    rows INTEGER NOT NULL,
    sources_json TEXT NOT NULL,
    class_distribution_json TEXT NOT NULL,
    samples_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at ON analysis_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_log_predictions_analysis_id ON log_predictions(analysis_id);
CREATE INDEX IF NOT EXISTS idx_log_predictions_severity ON log_predictions(severity);
"""


class AnalysisStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def save_analysis(
        self,
        *,
        total: int,
        max_severity: str,
        distribution: dict[str, int],
        decision_support: dict[str, Any],
        use_llm: bool,
        predictions: list[Any],
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analysis_runs (
                    total,
                    max_severity,
                    distribution_json,
                    decision_support_json,
                    use_llm
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    total,
                    max_severity,
                    json.dumps(distribution, ensure_ascii=False),
                    json.dumps(decision_support, ensure_ascii=False),
                    int(use_llm),
                ),
            )
            analysis_id = int(cursor.lastrowid)
            rows = [
                (
                    analysis_id,
                    item.line_no,
                    item.message,
                    item.normalized,
                    item.severity,
                    item.source,
                    item.confidence,
                    json.dumps(item.probabilities, ensure_ascii=False),
                )
                for item in predictions
            ]
            connection.executemany(
                """
                INSERT INTO log_predictions (
                    analysis_id,
                    line_no,
                    message,
                    normalized,
                    severity,
                    source,
                    confidence,
                    probabilities_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return analysis_id

    def save_dataset_snapshot(self, dataset: dict[str, Any]) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO dataset_snapshots (
                    rows,
                    sources_json,
                    class_distribution_json,
                    samples_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    int(dataset["rows"]),
                    json.dumps(dataset["sources"], ensure_ascii=False),
                    json.dumps(dataset["class_distribution"], ensure_ascii=False),
                    json.dumps(dataset["samples"], ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list_history(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    total,
                    max_severity,
                    distribution_json,
                    decision_support_json,
                    use_llm
                FROM analysis_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "total": row["total"],
                "max_severity": row["max_severity"],
                "distribution": json.loads(row["distribution_json"]),
                "decision_support": json.loads(row["decision_support_json"]),
                "use_llm": bool(row["use_llm"]),
            }
            for row in rows
        ]

    def get_analysis(self, analysis_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            run = connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    total,
                    max_severity,
                    distribution_json,
                    decision_support_json,
                    use_llm
                FROM analysis_runs
                WHERE id = ?
                """,
                (analysis_id,),
            ).fetchone()
            if run is None:
                return None

            prediction_rows = connection.execute(
                """
                SELECT
                    line_no,
                    message,
                    normalized,
                    severity,
                    source,
                    confidence,
                    probabilities_json
                FROM log_predictions
                WHERE analysis_id = ?
                ORDER BY line_no
                """,
                (analysis_id,),
            ).fetchall()

        return {
            "id": run["id"],
            "created_at": run["created_at"],
            "summary": {
                "total": run["total"],
                "distribution": json.loads(run["distribution_json"]),
                "max_severity": run["max_severity"],
            },
            "decision_support": json.loads(run["decision_support_json"]),
            "use_llm": bool(run["use_llm"]),
            "predictions": [
                {
                    "line_no": row["line_no"],
                    "message": row["message"],
                    "normalized": row["normalized"],
                    "severity": row["severity"],
                    "source": row["source"],
                    "confidence": row["confidence"],
                    "probabilities": json.loads(row["probabilities_json"]),
                }
                for row in prediction_rows
            ],
        }

    def stats(self) -> dict[str, Any]:
        with self.connect() as connection:
            run_count = connection.execute("SELECT COUNT(*) AS count FROM analysis_runs").fetchone()["count"]
            log_count = connection.execute("SELECT COUNT(*) AS count FROM log_predictions").fetchone()["count"]
            severity_rows = connection.execute(
                """
                SELECT severity, COUNT(*) AS count
                FROM log_predictions
                GROUP BY severity
                ORDER BY count DESC
                """
            ).fetchall()
            last_dataset = connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    rows,
                    sources_json,
                    class_distribution_json
                FROM dataset_snapshots
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

        return {
            "analysis_runs": run_count,
            "stored_log_predictions": log_count,
            "severity_distribution": {row["severity"]: row["count"] for row in severity_rows},
            "last_dataset_snapshot": None
            if last_dataset is None
            else {
                "id": last_dataset["id"],
                "created_at": last_dataset["created_at"],
                "rows": last_dataset["rows"],
                "sources": json.loads(last_dataset["sources_json"]),
                "class_distribution": json.loads(last_dataset["class_distribution_json"]),
            },
        }
