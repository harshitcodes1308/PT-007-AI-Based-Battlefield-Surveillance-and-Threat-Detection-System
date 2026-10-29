"""Video and export services for PT-007, independent of the Streamlit UI."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from zone_monitor import zone_bounds, inspect_zone, draw_zone
from tracking import CentroidTracker
from typing import Callable, Protocol

import cv2
import pandas as pd

from detection import Detection


class Detector(Protocol):
    def predict(self, frame, confidence_threshold: float) -> list[Detection]: ...


@dataclass
class ProcessingResult:
    annotated_video: Path
    csv_file: Path
    total_frames: int
    processed_frames: int
    frame_level_detections: int
    fps: float
    codec: str
    browser_compatible: bool
    events_csv: Path | None = None
    event_count: int = 0
    processing_fps: float = 0.0
    unique_tracked_objects: int = 0
    events_by_class: dict[str, int] | None = None


ProgressCallback = Callable[[int, int, object, Counter], None]


def _open_writer(path: Path, fps: float, size: tuple[int, int]):
    """Prefer H.264, then an MP4 fallback that remains downloadable."""
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError("OpenCV could not create an MP4 writer on this system.")


def annotate_frame(frame, detections: list[Detection]):
    annotated = frame.copy()
    for item in detections:
        color = (0, 215, 255) if item.class_name == "person" else (80, 220, 110)
        cv2.rectangle(annotated, (item.x1, item.y1), (item.x2, item.y2), color, 2)
        id_text = f" ID {item.track_id}" if item.track_id is not None else ""
        label = f"{item.class_name}{id_text} {item.confidence * 100:.1f}%"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(item.y1, height + 6)
        cv2.rectangle(annotated, (item.x1, text_y - height - 6), (item.x1 + width + 6, text_y), color, -1)
        cv2.putText(annotated, label, (item.x1 + 3, text_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 20, 24), 1, cv2.LINE_AA)
    return annotated


class VideoProcessor:
    def __init__(self, detector: Detector) -> None:
        self.detector = detector

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        confidence_threshold: float,
        progress_callback: ProgressCallback | None = None,
        monitor_zone: bool = True,
        zone: tuple[float, float, float, float] = (0.30, 0.25, 0.70, 0.75),
        prolonged_presence_seconds: float = 5.0,
        event_store=None,
        job_id: str = "",
        max_frames: int | None = None,
    ) -> ProcessingResult:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Unsupported or unreadable video. Upload a valid MP4, MOV, or AVI file.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise ValueError("The uploaded video has invalid metadata (zero FPS or frame dimensions).")

        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "annotated.mp4"
        csv_path = output_dir / "detections.csv"
        writer, codec = _open_writer(video_path, fps, (width, height))
        rows: list[dict] = []
        events: list[dict] = []
        tracker = CentroidTracker(max_distance=max(width, height) * 0.12)
        zone_state: dict[int, dict] = {}
        bounds = zone_bounds(width, height, zone)
        start_time = perf_counter()
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break
                detections = tracker.update(self.detector.predict(frame, confidence_threshold))
                annotated = annotate_frame(frame, detections)
                if monitor_zone:
                    inside = inspect_zone(detections, bounds)
                    inside_ids = {item.track_id for item in inside if item.track_id is not None}
                    for item in detections:
                        if item.track_id is None:
                            continue
                        state = zone_state.setdefault(item.track_id, {"inside": False, "entered_at": None, "entry_sent": False, "prolonged_sent": False})
                        is_inside = item.track_id in inside_ids
                        now = frame_index / fps
                        if is_inside and not state["inside"]:
                            state.update({"inside": True, "entered_at": now})
                            if not state["entry_sent"]:
                                event = {"timestamp_seconds": round(now, 3), "frame_index": frame_index, "event_type": "zone_entry", "object_class": item.class_name, "track_id": item.track_id, "review_status": "pending", "job_id": job_id}
                                if event_store:
                                    event["event_id"] = event_store.add(event)
                                events.append(event)
                                state["entry_sent"] = True
                        elif not is_inside:
                            state["inside"] = False
                            state["entered_at"] = None
                        if is_inside and state["entered_at"] is not None and not state["prolonged_sent"] and now - state["entered_at"] >= prolonged_presence_seconds:
                            event = {"timestamp_seconds": round(now, 3), "frame_index": frame_index, "event_type": "prolonged_presence", "object_class": item.class_name, "track_id": item.track_id, "review_status": "pending", "job_id": job_id}
                            if event_store:
                                event["event_id"] = event_store.add(event)
                            events.append(event)
                            state["prolonged_sent"] = True
                    draw_zone(annotated, bounds, bool(inside))
                writer.write(annotated)
                for item in detections:
                    rows.append({
                        "timestamp_seconds": round(frame_index / fps, 3), "frame_index": frame_index,
                        "class": item.class_name, "confidence": round(item.confidence, 4),
                        "x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2, "track_id": item.track_id,
                    })
                frame_index += 1
                if progress_callback and (frame_index % 3 == 0 or frame_index == total_frames):
                    progress_callback(frame_index, total_frames, annotated, Counter(d.class_name for d in detections))
        finally:
            capture.release()
            writer.release()

        elapsed = max(perf_counter() - start_time, 1e-9)
        events_path = output_dir / "events.csv"
        pd.DataFrame(events, columns=["event_id", "timestamp_seconds", "frame_index", "event_type", "object_class", "track_id", "review_status", "job_id"]).to_csv(events_path, index=False)
        pd.DataFrame(rows, columns=["timestamp_seconds", "frame_index", "class", "confidence", "x1", "y1", "x2", "y2", "track_id"]).to_csv(csv_path, index=False)
        class_counts = Counter(event["object_class"] for event in events)
        return ProcessingResult(video_path, csv_path, total_frames, frame_index, len(rows), fps, codec, codec in {"avc1", "H264"}, events_path, len(events), frame_index / elapsed, len(tracker.seen_track_ids), dict(class_counts))
