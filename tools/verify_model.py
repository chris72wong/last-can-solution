"""Verify exported weights through the exact GUI detector/frame API.

Uses one positive and one negative held-out frame, compares the PIL-RGB
boundary against Ultralytics' file-input path, and records actual outcomes.
No robot is contacted. Run: python -m tools.verify_model
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from lastcan.config import ROOT
from lastcan.types import FramePacket
from lastcan.vision import Detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "redbull_v5")
    parser.add_argument("--output", type=Path, default=ROOT / "verification" / "model-inference-check.json")
    args = parser.parse_args()
    from PIL import Image
    import numpy as np

    detector = Detector(args.model, imgsz=640, confidence=0.25)
    loaded = detector.load()
    report = {"model": "models/best.pt", "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model.is_file() else None,
              "available": detector.available, "load_status": detector.status, "device": detector.device,
              "color_source": "PIL RGB -> contiguous NumPy BGR in Detector.predict",
              "simulated": False, "robot_contacted": False, "frames": [], "passed": False}
    if loaded:
        candidates = {}
        for path in sorted((args.data / "test" / "images").glob("*.jpg")):
            label = args.data / "test" / "labels" / (path.stem + ".txt")
            positive = bool(label.read_text(encoding="utf-8-sig").strip())
            candidates.setdefault("positive" if positive else "negative", path)
            if len(candidates) == 2:
                break
        for index, (kind, path) in enumerate(sorted(candidates.items()), start=101):
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            frame = FramePacket(index, time.monotonic(), image)
            prediction = detector.predict(frame)
            via_file = detector._model.predict(str(path), conf=.25, imgsz=640,
                                               device=detector.device, verbose=False)[0]
            boxes = np.asarray([[b.x1, b.y1, b.x2, b.y2] for b in prediction.boxes], dtype=float).reshape(-1, 4)
            reference = np.asarray(via_file.boxes.xyxy.tolist(), dtype=float).reshape(-1, 4)
            same = boxes.shape == reference.shape and bool(np.allclose(boxes, reference, atol=.05, rtol=.001))
            actual = bool(prediction.boxes)
            report["frames"].append({"image": path.name, "split": "held-out test", "expected_positive": kind == "positive",
                "frame_id": prediction.frame_id, "image_size": list(image.size), "rgb_source_pixel": list(image.getpixel((0, 0))),
                "source_frame_identity_preserved": prediction.image is image and prediction.frame_id == frame.frame_id and prediction.captured_at == frame.captured_at,
                "box_count": len(prediction.boxes), "confidence": [b.confidence for b in prediction.boxes],
                "boxes": boxes.tolist(), "file_input_box_count": len(via_file.boxes),
                "file_input_and_rgb_interface_agree": same, "frame_level_expectation_met": actual == (kind == "positive"),
                "error": prediction.error})
        report["passed"] = len(report["frames"]) == 2 and all(
            row["source_frame_identity_preserved"] and row["file_input_and_rgb_interface_agree"] and not row["error"]
            for row in report["frames"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
