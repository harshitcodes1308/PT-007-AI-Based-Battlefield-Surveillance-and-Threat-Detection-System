"""YOLO inference adapter for PT-007. This module deliberately has no UI code."""

from __future__ import annotations

from dataclasses import dataclass
SUPPORTED_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}


@dataclass(frozen=True)
class Detection:
    """One frame-level object detection in pixel coordinates."""

    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int | None = None


class DetectionEngine:
    """Lazy YOLO11n loader with Apple Silicon MPS and CPU fallback."""

    def __init__(self, weights: str = "yolo11n.pt") -> None:
        self.weights = weights
        self._model = None
        self.device = "cpu"

    def load(self) -> str:
        """Load model weights, downloading the official pretrained file if needed."""
        if self._model is not None:
            return self.device
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Missing dependencies. Run: pip install -r requirements.txt") from exc

        try:
            self._model = YOLO(self.weights)
        except Exception as exc:
            hint = (
                "Could not load YOLO11n weights. Check your internet connection for the first "
                "run, or place yolo11n.pt beside this file."
            )
            raise RuntimeError(f"{hint} Details: {exc}") from exc

        self.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        return self.device

    def predict(self, frame, confidence_threshold: float) -> list[Detection]:
        """Run inference and return only the PT-007 supported COCO classes."""
        if self._model is None:
            self.load()
        try:
            result = self._model.predict(
                source=frame, conf=confidence_threshold, device=self.device, verbose=False
            )[0]
        except Exception as exc:
            if self.device == "mps":
                # Some macOS/PyTorch combinations lack an MPS operator. Retry safely on CPU.
                self.device = "cpu"
                try:
                    result = self._model.predict(
                        source=frame, conf=confidence_threshold, device="cpu", verbose=False
                    )[0]
                except Exception as fallback_exc:
                    raise RuntimeError(f"YOLO inference failed: {fallback_exc}") from fallback_exc
            else:
                raise RuntimeError(f"YOLO inference failed: {exc}") from exc

        detections: list[Detection] = []
        names = result.names
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_name = str(names[int(box.cls.item())])
            confidence = float(box.conf.item())
            if class_name not in SUPPORTED_CLASSES or confidence < confidence_threshold:
                continue
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            detections.append(Detection(class_name, confidence, x1, y1, x2, y2))
        return detections
