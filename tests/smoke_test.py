"""Minimal non-network smoke test for the independent video-processing path."""

from pathlib import Path
import tempfile
import sys

import cv2
import numpy as np

# Permit ``python tests/smoke_test.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection import Detection
from video_processor import VideoProcessor
from event_store import EventStore


class StubDetector:
    def __init__(self):
        self.frame = 0

    def predict(self, frame, confidence_threshold):
        self.frame += 1
        return [Detection("person", 0.9, 18 + self.frame, 18, 38 + self.frame, 38)]


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    source = root / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(12):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    store = EventStore(root / "events.sqlite")
    result = VideoProcessor(StubDetector()).process(source, root / "out", 0.4, zone=(0.2, 0.2, 0.8, 0.8), prolonged_presence_seconds=0.5, event_store=store, job_id="smoke")
    assert result.processed_frames == 12
    assert result.frame_level_detections == 12
    assert result.unique_tracked_objects == 1
    assert {event["event_type"] for event in store.list("smoke")} == {"zone_entry", "prolonged_presence"}
    # Separate connections model a Streamlit rerun / restart.
    reloaded_store = EventStore(root / "events.sqlite")
    persisted = reloaded_store.list("smoke")
    assert len(persisted) == 2
    reloaded_store.acknowledge([event["id"] for event in persisted])
    assert {event["review_status"] for event in EventStore(root / "events.sqlite").list("smoke")} == {"acknowledged"}
    # The same logical event must retain its database ID rather than duplicate on rerun.
    original_id = persisted[0]["id"]
    duplicate_id = EventStore(root / "events.sqlite").add({"job_id": "smoke", "timestamp_seconds": persisted[0]["video_timestamp_seconds"], "frame_index": persisted[0]["frame_index"], "track_id": persisted[0]["track_id"], "object_class": persisted[0]["object_class"], "event_type": persisted[0]["event_type"]})
    assert duplicate_id == original_id
    assert result.csv_file.exists() and result.annotated_video.exists()
print("PT-007 smoke test passed")
