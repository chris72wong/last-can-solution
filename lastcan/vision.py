"""One locally trained YOLO model, loaded once in the inference worker.

The public frame contract is PIL RGB. Only this boundary converts RGB to
OpenCV/Ultralytics ndarray BGR. Results retain their exact source frame.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time

from .types import Detection, DetectionPacket, FramePacket


class Detector:
    def __init__(self, model_path, imgsz=640, confidence=0.25):
        self.model_path = Path(model_path).resolve()
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.available = False
        self.status = "Model not loaded (inference worker starts loading)"
        self.device = "cpu"
        self._model = None
        self._attempted = False
        self._lock = threading.RLock()
        if not self.model_path.is_file():
            self.status = f"Missing trained weights: {self.model_path.name}; train with train_cans.py"

    def load(self) -> bool:
        """Idempotent. Call in a worker, never on tkinter's thread.

        Missing/erroring models remain unavailable for this instance. Construct
        a new Detector (or restart the cockpit) after installing new weights.
        """
        with self._lock:
            if self._attempted:
                return self.available
            self._attempted = True
            if not self.model_path.is_file():
                self.status = f"Missing trained weights: {self.model_path.name}; train with train_cans.py"
                return False
            try:
                self.status = "Loading trained can detector..."
                cache = Path(__file__).resolve().parent.parent / ".cache"
                os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "ultralytics"))
                os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
                os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
                import numpy as np
                import torch
                from ultralytics import YOLO

                self.device = 0 if torch.cuda.is_available() else "cpu"
                model = YOLO(str(self.model_path), task="detect")
                if getattr(model, "task", "detect") != "detect":
                    raise ValueError("Expected YOLO detection weights, not classification/segmentation weights")
                if len(model.names) != 1:
                    raise ValueError("Expected a one-class can detector, not generic or MNIST weights")
                # Warm up before a live frame arrives; no network model fallback.
                model.predict(np.zeros((360, 640, 3), dtype=np.uint8),
                              imgsz=self.imgsz, conf=self.confidence,
                              device=self.device, verbose=False)
                self._model = model
                self.available = True
                self.status = f"Can detector ready: {'CUDA:0' if self.device == 0 else 'CPU (CUDA unavailable)'}"
            except Exception as exc:
                self.available = False
                self._model = None
                self.status = f"Detector unavailable: {type(exc).__name__}: {exc}"
            return self.available

    def predict(self, frame: FramePacket) -> DetectionPacket:
        with self._lock:
            if not self.load():
                return DetectionPacket(frame.frame_id, frame.captured_at, time.monotonic(),
                                       frame.image, (), error=self.status)
            try:
                import numpy as np

                rgb = np.asarray(frame.image.convert("RGB"))
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                result = self._model.predict(bgr, imgsz=self.imgsz,
                                             conf=max(0.01, min(1.0, self.confidence)),
                                             device=self.device, verbose=False)[0]
                width, height = frame.image.size
                boxes = []
                if result.boxes is not None:
                    for corners, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                        x1, y1, x2, y2 = (float(v) for v in corners)
                        conf = float(conf)
                        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, conf)):
                            continue
                        x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
                        y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
                        if x2 > x1 and y2 > y1 and conf >= self.confidence:
                            boxes.append(Detection(x1, y1, x2, y2, conf))
                self.status = f"Can detector ready: {'CUDA:0' if self.device == 0 else 'CPU (CUDA unavailable)'}"
                return DetectionPacket(frame.frame_id, frame.captured_at, time.monotonic(),
                                       frame.image, tuple(boxes))
            except Exception as exc:
                self.status = f"Inference failed: {type(exc).__name__}: {exc}"
                return DetectionPacket(frame.frame_id, frame.captured_at, time.monotonic(),
                                       frame.image, (), error=self.status)
