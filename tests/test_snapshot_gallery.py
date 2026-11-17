"""Tests for the Event Snapshot Gallery feature (PT-007)."""

from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import uuid

import cv2
import numpy as np

# Allow ``python tests/test_snapshot_gallery.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection import Detection
from event_store import EventStore
from video_processor import VideoProcessor

# ---------------------------------------------------------------------------
# Project root (where video_processor.py lives) – snapshots land here.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StubDetector:
    def __init__(self):
        self.frame = 0

    def predict(self, frame, confidence_threshold):
        self.frame += 1
        return [Detection("person", 0.9, 18 + self.frame, 18, 38 + self.frame, 38)]


# ---------------------------------------------------------------------------
# Helper: build a tiny .mp4 that VideoProcessor can read
# ---------------------------------------------------------------------------
def _write_tiny_video(path: Path, frames: int = 15) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
    )
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


# ---------------------------------------------------------------------------
# Test 1 – schema migration adds snapshot_path to an old DB
# ---------------------------------------------------------------------------
def test_schema_migration_adds_snapshot_path():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "old.sqlite"
        # Create DB with OLD schema (no snapshot_path column)
        con = sqlite3.connect(db_path)
        try:
            con.execute("""
                CREATE TABLE events (
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
            con.commit()
        finally:
            con.close()

        # Open with EventStore — should auto-migrate
        EventStore(db_path)

        # Verify column now exists
        con = sqlite3.connect(db_path)
        try:
            cols = [row[1] for row in con.execute("PRAGMA table_info(events)").fetchall()]
        finally:
            con.close()
        assert "snapshot_path" in cols, f"snapshot_path not found in columns: {cols}"
    print("  test_schema_migration_adds_snapshot_path PASSED")


# ---------------------------------------------------------------------------
# Test 2 – snapshot .jpg is created on disk after processing
# ---------------------------------------------------------------------------
def test_snapshot_image_created():
    job_id = "test_snap_" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.mp4"
        _write_tiny_video(source)
        store = EventStore(root / "events.sqlite")
        snap_dir = root / "snapshots" / job_id
        VideoProcessor(StubDetector()).process(
            source,
            root / "out",
            0.4,
            zone=(0.2, 0.2, 0.8, 0.8),
            prolonged_presence_seconds=0.5,
            event_store=store,
            job_id=job_id,
            snapshot_dir=snap_dir,
        )
        jpg_files = list(snap_dir.glob("*.jpg"))
        assert len(jpg_files) > 0, f"No .jpg snapshots found in {snap_dir}"
    print("  test_snapshot_image_created PASSED")


# ---------------------------------------------------------------------------
# Test 3 – snapshot_path is persisted in the DB after processing
# ---------------------------------------------------------------------------
def test_snapshot_path_persisted():
    job_id = "test_snap_" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.mp4"
        _write_tiny_video(source)
        db_path = root / "events.sqlite"
        store = EventStore(db_path)
        snap_dir = root / "snapshots" / job_id
        VideoProcessor(StubDetector()).process(
            source,
            root / "out",
            0.4,
            zone=(0.2, 0.2, 0.8, 0.8),
            prolonged_presence_seconds=0.5,
            event_store=store,
            job_id=job_id,
            snapshot_dir=snap_dir,
        )
        # Fresh connection to verify persistence
        fresh_store = EventStore(db_path)
        events = fresh_store.list(job_id)
        assert len(events) > 0, "No events found after processing"
        paths = [e.get("snapshot_path") for e in events]
        non_empty = [p for p in paths if p]
        assert len(non_empty) > 0, f"No non-empty snapshot_path found; paths={paths}"
    print("  test_snapshot_path_persisted PASSED")


# ---------------------------------------------------------------------------
# Test 4 – events.csv written by VideoProcessor contains snapshot_path header
# ---------------------------------------------------------------------------
def test_events_csv_has_snapshot_path():
    job_id = "test_snap_" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.mp4"
        _write_tiny_video(source)
        store = EventStore(root / "events.sqlite")
        snap_dir = root / "snapshots" / job_id
        result = VideoProcessor(StubDetector()).process(
            source,
            root / "out",
            0.4,
            zone=(0.2, 0.2, 0.8, 0.8),
            prolonged_presence_seconds=0.5,
            event_store=store,
            job_id=job_id,
            snapshot_dir=snap_dir,
        )
        assert result.events_csv is not None and result.events_csv.exists(), \
            "events.csv not created"
        header = result.events_csv.read_text(encoding="utf-8").splitlines()[0]
        assert "snapshot_path" in header, \
            f"'snapshot_path' not in events.csv header: {header}"
    print("  test_events_csv_has_snapshot_path PASSED")


# ---------------------------------------------------------------------------
# Test 5 – duplicate add() returns same id and is_new=False, only 1 DB row
# ---------------------------------------------------------------------------
def test_duplicate_event_no_duplicate_row():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "events.sqlite"
        store = EventStore(db_path)
        event = {
            "job_id": "dup_test",
            "timestamp_seconds": 1.0,
            "frame_index": 5,
            "track_id": 1,
            "object_class": "person",
            "event_type": "zone_entry",
        }
        event_id1, is_new1 = store.add(event)
        event_id2, is_new2 = store.add(event)

        assert isinstance(event_id1, int), "event_id1 must be int"
        assert isinstance(event_id2, int), "event_id2 must be int"
        assert event_id1 == event_id2, f"IDs differ: {event_id1} != {event_id2}"
        assert is_new1 is True, "First add should be new"
        assert is_new2 is False, "Second add should NOT be new"

        # Confirm only 1 row in DB
        con = sqlite3.connect(db_path)
        try:
            count = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        finally:
            con.close()
        assert count == 1, f"Expected 1 row, got {count}"
    print("  test_duplicate_event_no_duplicate_row PASSED")


# ---------------------------------------------------------------------------
# Test 6 – missing snapshot file does not crash list() or gallery logic
# ---------------------------------------------------------------------------
def test_missing_snapshot_does_not_crash():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "events.sqlite"
        store = EventStore(db_path)
        event = {
            "job_id": "missing_snap",
            "timestamp_seconds": 0.1,
            "frame_index": 1,
            "track_id": 1,
            "object_class": "person",
            "event_type": "zone_entry",
            "snapshot_path": "nonexistent/path/snap.jpg",
        }
        event_id, _ = store.add(event)
        # Persist the bogus path via set_snapshot_path
        store.set_snapshot_path(event_id, "nonexistent/path/snap.jpg")

        # list() must not raise
        events = store.list("missing_snap")
        assert len(events) == 1

        # Gallery logic: check Path.exists() without raising
        for ev in events:
            snap_path_str = ev.get("snapshot_path")
            if snap_path_str:
                snap_path = Path(snap_path_str)
                # Simply checking .exists() must not raise an exception
                exists = snap_path.exists()
                assert exists is False, "File should not exist"
    print("  test_missing_snapshot_does_not_crash PASSED")


# ---------------------------------------------------------------------------
# Test 7 – add() returns tuple, acknowledge() still sets review_status
# ---------------------------------------------------------------------------
def test_acknowledgment_still_works():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "events.sqlite"
        store = EventStore(db_path)

        base = {
            "job_id": "ack_test",
            "timestamp_seconds": 1.0,
            "frame_index": 1,
            "track_id": 1,
            "object_class": "person",
        }

        result1 = store.add({**base, "event_type": "zone_entry"})
        result2 = store.add({**base, "frame_index": 2, "event_type": "prolonged_presence"})

        # Both should be tuples
        assert isinstance(result1, tuple) and len(result1) == 2, "add() must return (id, is_new)"
        assert isinstance(result2, tuple) and len(result2) == 2, "add() must return (id, is_new)"

        id1, is_new1 = result1
        id2, is_new2 = result2
        assert is_new1 is True
        assert is_new2 is True

        store.acknowledge([id1, id2])

        events = store.list("ack_test")
        statuses = {e["review_status"] for e in events}
        assert statuses == {"acknowledged"}, f"Expected all acknowledged, got {statuses}"
    print("  test_acknowledgment_still_works PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_schema_migration_adds_snapshot_path()
    test_snapshot_image_created()
    test_snapshot_path_persisted()
    test_events_csv_has_snapshot_path()
    test_duplicate_event_no_duplicate_row()
    test_missing_snapshot_does_not_crash()
    test_acknowledgment_still_works()
    print("All snapshot gallery tests passed")
