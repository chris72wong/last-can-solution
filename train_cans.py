"""Train, validate, and evaluate our one-class Red Bull detector.

Adapted from the instructor's train_mnist.py: uses an existing labelled
dataset instead of generating digits, preserves its split, and records
held-out robot-frame evaluation. Heavy imports happen only inside main().
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_split(image_dir: Path) -> dict:
    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image split: {image_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    counts = {"images": len(images), "positive_images": 0, "negative_images": 0, "boxes": 0}
    if not images:
        raise ValueError(f"No images in {image_dir}")
    for image in images:
        label = image_dir.parent / "labels" / (image.stem + ".txt")
        if not label.exists():
            raise FileNotFoundError(f"Missing explicit label file (empty for negatives): {label}")
        lines = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        for fields in lines:
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError(f"Expected class 0 and four normalized coordinates in {label}")
            coords = [float(value) for value in fields[1:]]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                raise ValueError(f"Invalid normalized box in {label}: {fields}")
        counts["positive_images" if lines else "negative_images"] += 1
        counts["boxes"] += len(lines)
    counts["directory"] = str(image_dir)
    return counts


def inspect_dataset(data_yaml: Path) -> tuple[dict, dict]:
    import yaml
    description = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    names = description.get("names", [])
    if description.get("nc", len(names)) != 1 or len(names) != 1:
        raise ValueError("The can dataset must have exactly one class")
    base = data_yaml.parent
    if description.get("path"):
        configured = Path(description["path"])
        base = configured if configured.is_absolute() else base / configured
    splits = {}
    for key in ("train", "val", "test"):
        value = description.get(key)
        if value:
            if not isinstance(value, str):
                raise ValueError(f"This script expects one image directory for {key}")
            folder = Path(value)
            splits[key] = inspect_split(folder if folder.is_absolute() else base / folder)
    if "train" not in splits or "val" not in splits:
        raise ValueError("Dataset must define train and val image directories")
    return description, splits


def metric_values(result) -> dict:
    return {"mAP50": float(result.box.map50), "mAP50-95": float(result.box.map),
            "precision": float(result.box.mp), "recall": float(result.box.mr)}


def write_notes(report: dict) -> None:
    settings = report["settings"]
    split_rows = "\n".join(f"| {name} | {value['images']} | {value['positive_images']} | {value['negative_images']} | {value['boxes']} |"
                            for name, value in report["data_split"].items())
    metrics_rows = "\n".join(f"| {name} | {value['mAP50']:.5f} | {value['mAP50-95']:.5f} | {value['precision']:.5f} | {value['recall']:.5f} |"
                              for name, value in report["metrics"].items())
    counts = report.get("heldout_frame_counts", {})
    notes = f"""# Training notes

**Group:** fill in group member names before submission.
**Model file:** `models/best.pt` from `{report['run_name']}`.
**Recorded:** {report['completed_utc']}. Actual offline training; live robot testing is pending.

## 1. Model

Fine-tuned `{report['base_model']}` as a one-class YOLO detector. The nano
model is the lab's baseline and keeps live inference light. No larger model
comparison has been performed. The GUI loads the actual trained weights.

## 2. Settings

Command: `{report['command']}`

| Setting | Value | Reason |
| --- | --- | --- |
| Epoch limit / completed | {settings['epochs']} / {report['epochs_completed']} | Baseline 80; patience 20 permits early stopping. |
| Image size | {settings['imgsz']} | Preserve detail in 640-pixel-wide robot frames. |
| Batch / workers | {report['effective_batch']} / {report['effective_workers']} | GPU baseline, adjusted only for actual resource/startup errors. |
| Horizontal flip | {settings['fliplr']} | A mirrored can remains a can. |
| Patience / seed | {settings['patience']} / {settings['seed']} | Repeatable baseline and bounded early stopping. |
| Other augmentation | Ultralytics defaults | Preserve a simple starting point; compare future changes one at a time. |
| Runtime confidence | {report['confidence']} | Initial threshold; final live tuning is pending. |
| Device | CUDA:0 | Checked before training; see verification/environment.json. |

## 3. Data and split

Dataset directory: `{report['dataset_name']}`. Its existing
training/validation/test split was preserved; no test images were used for
training or parameter selection. For the supplied `redbull_v5` baseline,
the instructor's README says robot captures were split by time blocks and
test pictures are robot-camera frames. The v5 training split includes
rotated training-only copies. Record the provenance/splitting of any custom
dataset here after adding new group captures.

| Split | Images | With a target | Without a target | Labelled boxes |
| --- | --- | --- | --- | --- |
{split_rows}

## 4. Validation and held-out results

| Split | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- |
{metrics_rows}

On the held-out robot-frame split at confidence {report['confidence']}:
positive frames with at least one detection: **{counts.get('positive_detected', 'not evaluated')} / {counts.get('positive_total', 'not evaluated')}**;
negative frames with a false box: **{counts.get('negative_false_alarm', 'not evaluated')} / {counts.get('negative_total', 'not evaluated')}**.
These frame counts do not prove every can in a multi-can image was detected;
the object-level metrics above assess localization. Live lighting, motion,
wrong can variants, and new robot viewpoints still require physical testing.

Predictions, training curves, and batch visualizations remain in the local
run folder (excluded from submission). Machine-readable provenance and
metrics are in `verification/training-results.json`.

## 5. What we changed after testing on the robot

| Run | Robot test observation | Change | Result |
| --- | --- | --- | --- |
| {report['run_name']} | Pending: no live robot test performed | Initial provided-data baseline | Offline metrics above; live outcome pending |

Complete this section after real robot tests. Save failures through the GUI,
label all target cans and keep wrong cans unboxed, split new captures by
session, and retrain without mixing the held-out test frames into training.

## 6. Known weaknesses

This model has not been tested live by our group. Camera pose changes,
unseen can orientations/variants, motion blur, and multiple touching cans
remain unverified. The observed offline misses and false alarms are retained
in the prediction output and report rather than claimed as successful.
"""
    (ROOT / "TRAINING_NOTES.md").write_text(notes, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "redbull_v5", help="Dataset folder or data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--name", default="baseline_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--positive-frames", type=Path, help="Optional unseen positive-only robot-frame folder")
    parser.add_argument("--negative-frames", type=Path, help="Optional unseen negative-only robot-frame folder")
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run tools/setup.ps1 and tools/check_environment.py before training.")
    data_yaml = args.data.resolve()
    if data_yaml.is_dir():
        data_yaml = data_yaml / "data.yaml"
    _, splits = inspect_dataset(data_yaml)
    print("Dataset checks passed:", json.dumps(splits, indent=2), flush=True)
    project = ROOT / "runs" / "cans"
    if (project / args.name).exists():
        raise FileExistsError("Choose a unique --name; existing runs are never overwritten")
    effective_batch, effective_workers = args.batch, args.workers
    run_name = args.name
    for attempt in range(4):
        model = YOLO(args.model)
        try:
            model.train(data=str(data_yaml), epochs=args.epochs, patience=args.patience,
                        imgsz=args.imgsz, batch=effective_batch, workers=effective_workers,
                        device=0, project=str(project), name=run_name, exist_ok=False,
                        seed=0, plots=True, fliplr=0.5, amp=True)
            break
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message and effective_batch > 1:
                effective_batch = max(1, effective_batch // 2)
                run_name = args.name + f"_batch{effective_batch}"
                torch.cuda.empty_cache()
            elif effective_workers > 0 and any(word in message for word in ("worker", "multiprocessing", "bootstrapping")):
                effective_workers = 0
                run_name = args.name + "_workers0"
            else:
                raise
            print(f"Retrying after actual resource/startup error: batch={effective_batch}, workers={effective_workers}", flush=True)
    else:
        raise RuntimeError("Training failed after resource/startup recovery attempts")

    run_dir = Path(model.trainer.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    best = YOLO(str(best_path))
    metrics = {}
    for split in ("val", "test"):
        if split in splits:
            metrics[split] = metric_values(best.val(data=str(data_yaml), split=split,
                imgsz=args.imgsz, device=0, batch=effective_batch, workers=effective_workers,
                project=str(project), name=run_name + "_" + split, plots=True, exist_ok=False))

    frame_counts = {"positive_total": 0, "positive_detected": 0, "negative_total": 0, "negative_false_alarm": 0}
    predictions = []
    test_dir = Path(splits.get("test", splits["val"])["directory"])
    images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    predictions_dir = run_dir / "heldout_predictions"
    predictions_dir.mkdir(exist_ok=True)
    examples_saved = {True: 0, False: 0}
    for index, result in enumerate(best.predict([str(p) for p in images], conf=args.conf,
            imgsz=args.imgsz, device=0, stream=True, verbose=False)):
        source = Path(result.path)
        label = test_dir.parent / "labels" / (source.stem + ".txt")
        is_positive = bool(label.read_text(encoding="utf-8-sig").strip())
        detected = len(result.boxes) > 0
        frame_counts["positive_total" if is_positive else "negative_total"] += 1
        if detected:
            frame_counts["positive_detected" if is_positive else "negative_false_alarm"] += 1
        predictions.append({"image": source.name, "positive": is_positive,
                            "detections": len(result.boxes), "confidence": result.boxes.conf.tolist()})
        if examples_saved[is_positive] < 12 or (is_positive and not detected) or (not is_positive and detected):
            result.save(filename=str(predictions_dir / source.name))
            examples_saved[is_positive] += 1

    extra_counts = {}
    for title, folder in (("positive", args.positive_frames), ("negative", args.negative_frames)):
        if folder:
            total = hits = 0
            for result in best.predict(str(folder.resolve()), conf=args.conf, imgsz=args.imgsz,
                    device=0, stream=True, save=True, project=str(project), name=run_name + "_robot_" + title):
                total += 1
                hits += int(len(result.boxes) > 0)
            extra_counts[title] = {"total": total, "frames_with_detections": hits}

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    shutil.copy2(best_path, models_dir / "best.pt")
    import csv
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        epochs_completed = len(list(csv.DictReader(handle)))
    report = {"completed_utc": datetime.now(timezone.utc).isoformat(), "run_name": run_dir.name,
        "run_directory": str(run_dir), "base_model": args.model,
        "model_file": "models/best.pt", "dataset_name": data_yaml.parent.name,
        "command": "python train_cans.py " + " ".join(sys.argv[1:]),
        "settings": {"epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz, "fliplr": .5, "seed": 0},
        "epochs_completed": epochs_completed, "effective_batch": effective_batch, "effective_workers": effective_workers,
        "confidence": args.conf, "data_split": splits, "metrics": metrics,
        "heldout_frame_counts": frame_counts, "extra_robot_frame_counts": extra_counts,
        "per_image_predictions": predictions, "live_robot_tested": False}
    verification = ROOT / "verification"
    verification.mkdir(exist_ok=True)
    (verification / "training-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_notes(report)
    print("ACTUAL FINAL RESULTS", json.dumps({"metrics": metrics, "heldout_frames": frame_counts}, indent=2), flush=True)
    print(f"Trained can weights: {models_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
