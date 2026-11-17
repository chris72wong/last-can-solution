"""Count hits/false alarms on NEW positive-only and negative-only robot frames.

Run from the project root:
python -m tools.evaluate_frames --positive captures/positives --negative captures/negatives
Keep these folders out of training for an honest test. Multi-can completeness
requires labelled object-level evaluation, not just these frame hit counts.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from lastcan.config import ROOT
from lastcan.types import FramePacket
from lastcan.vision import Detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--positive", type=Path)
    parser.add_argument("--negative", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=ROOT / "verification" / ("robot_frames_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    if not args.positive and not args.negative:
        parser.error("Provide --positive and/or --negative folders of previously unseen robot frames")
    from PIL import Image, ImageDraw
    detector = Detector(args.model, imgsz=args.imgsz, confidence=args.conf)
    if not detector.load():
        raise RuntimeError(detector.status)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"model": str(args.model), "confidence": args.conf, "imgsz": args.imgsz,
              "simulated": False, "note": "Frame-level counts, not object-level recall. Operator must confirm frames were unseen.",
              "splits": {}, "images": []}
    sequence = 0
    for title, folder in (("positive", args.positive), ("negative", args.negative)):
        if folder is None:
            continue
        images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not images:
            raise ValueError(f"No images in {folder}")
        counts = {"total": len(images), "frames_with_detections": 0}
        destination = args.output / title
        destination.mkdir()
        for path in images:
            sequence += 1
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            result = detector.predict(FramePacket(sequence, time.monotonic(), image))
            if result.error:
                raise RuntimeError(result.error)
            counts["frames_with_detections"] += int(bool(result.boxes))
            painted = image.copy()
            draw = ImageDraw.Draw(painted)
            for box in result.boxes:
                draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline="#18c99a", width=3)
                draw.text((box.x1, max(0, box.y1 - 14)), f"redbull {box.confidence:.2f}", fill="#18c99a")
            painted.save(destination / f"{sequence:05d}_{path.stem}.jpg", quality=92)
            report["images"].append({"source": str(path), "expected_positive": title == "positive",
                                     "detections": len(result.boxes)})
        report["splits"][title] = counts
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["splits"], indent=2))
    print(f"Saved predictions and report: {args.output}")


if __name__ == "__main__":
    main()
