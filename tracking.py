"""Lightweight class-aware multi-object tracking for uploaded civilian aerial footage."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from detection import Detection


@dataclass
class _Track:
    track_id: int
    class_name: str
    center: tuple[float, float]
    missed_frames: int = 0


class CentroidTracker:
    """Associate detections across adjacent frames using class and bounding-box center.

    This dependency-free tracker is intended for short uploaded videos. It is not an
    identity-verification system and IDs can change after occlusion or long gaps.
    """

    def __init__(self, max_distance: float = 90.0, max_missed_frames: int = 20) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}
        self.seen_track_ids: set[int] = set()

    @staticmethod
    def _center(item: Detection) -> tuple[float, float]:
        return ((item.x1 + item.x2) / 2, (item.y1 + item.y2) / 2)

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Return detections annotated with stable IDs where association succeeds."""
        unmatched_tracks = set(self._tracks)
        assignments: dict[int, int] = {}
        candidates: list[tuple[float, int, int]] = []
        for detection_index, item in enumerate(detections):
            cx, cy = self._center(item)
            for track_id, track in self._tracks.items():
                if track.class_name == item.class_name:
                    candidates.append((hypot(cx - track.center[0], cy - track.center[1]), detection_index, track_id))
        used_detections: set[int] = set()
        for distance, detection_index, track_id in sorted(candidates):
            if distance > self.max_distance or detection_index in used_detections or track_id not in unmatched_tracks:
                continue
            assignments[detection_index] = track_id
            used_detections.add(detection_index)
            unmatched_tracks.remove(track_id)

        for track_id in list(unmatched_tracks):
            self._tracks[track_id].missed_frames += 1
            if self._tracks[track_id].missed_frames > self.max_missed_frames:
                del self._tracks[track_id]

        tracked: list[Detection] = []
        for detection_index, item in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id, item.class_name, self._center(item))
            else:
                track = self._tracks[track_id]
                track.center = self._center(item)
                track.missed_frames = 0
            self.seen_track_ids.add(track_id)
            tracked.append(Detection(item.class_name, item.confidence, item.x1, item.y1, item.x2, item.y2, track_id))
        return tracked
