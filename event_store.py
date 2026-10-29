"""SQLite persistence for operator-reviewable civilian activity events."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime, timezone


class EventStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    video_timestamp_seconds REAL NOT NULL,
                    frame_index INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    object_class TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending'
                )
            """)

    def migrate_from(self, legacy_database_path: Path) -> int:
        """Copy legacy runtime history into this durable store without deleting it."""
        if not legacy_database_path.exists() or legacy_database_path.resolve() == self.database_path.resolve():
            return 0
        with sqlite3.connect(legacy_database_path) as legacy_connection:
            legacy_connection.row_factory = sqlite3.Row
            try:
                rows = legacy_connection.execute(
                    "SELECT job_id, created_at, video_timestamp_seconds, frame_index, track_id, object_class, event_type, review_status FROM events"
                ).fetchall()
            except sqlite3.OperationalError:
                return 0
        copied = 0
        with self._connect() as connection:
            for row in rows:
                existing = connection.execute(
                    "SELECT id FROM events WHERE job_id = ? AND frame_index = ? AND track_id = ? AND event_type = ?",
                    (row["job_id"], row["frame_index"], row["track_id"], row["event_type"]),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """INSERT INTO events
                        (job_id, created_at, video_timestamp_seconds, frame_index, track_id, object_class, event_type, review_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        tuple(row),
                    )
                    copied += 1
            connection.commit()
        return copied

    def _connect(self):
        return sqlite3.connect(self.database_path)

    def add(self, event: dict) -> int:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM events WHERE job_id = ? AND frame_index = ? AND track_id = ? AND event_type = ?",
                (event["job_id"], event["frame_index"], event["track_id"], event["event_type"]),
            ).fetchone()
            if existing is not None:
                return int(existing[0])
            cursor = connection.execute(
                """INSERT INTO events
                (job_id, created_at, video_timestamp_seconds, frame_index, track_id, object_class, event_type, review_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (event["job_id"], created_at, event["timestamp_seconds"], event["frame_index"],
                 event["track_id"], event["object_class"], event["event_type"]),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list(self, job_id: str | None = None) -> list[dict]:
        query = "SELECT id, created_at, video_timestamp_seconds, frame_index, track_id, object_class, event_type, review_status FROM events"
        parameters: tuple = ()
        if job_id:
            query += " WHERE job_id = ?"
            parameters = (job_id,)
        query += " ORDER BY id DESC"
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(query, parameters)]

    def acknowledge(self, event_ids: list[int]) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        with self._connect() as connection:
            connection.execute(f"UPDATE events SET review_status = 'acknowledged' WHERE id IN ({placeholders})", event_ids)
            connection.commit()

    def job_ids(self) -> list[str]:
        with self._connect() as connection:
            return [row[0] for row in connection.execute("SELECT job_id FROM events GROUP BY job_id ORDER BY MAX(id) DESC")]
