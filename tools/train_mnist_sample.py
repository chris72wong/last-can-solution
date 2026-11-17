#!/usr/bin/env python3
"""
Sample: train a YOLO object detector (Ultralytics) on handwritten digits.

A complete, working example of the training workflow you need for the cans, on
a problem where the data is free and training takes a few minutes: find every
digit in a picture, draw a box around it and say which digit it is.

What it does, in order:
    1. Builds a small detection dataset out of MNIST (first run only; it
       downloads MNIST, about 10 MB, so do this while you have internet).
    2. Fine-tunes a pretrained YOLO model (yolo11n.pt) on it.
    3. Scores the best weights on the validation pictures.
    4. Runs the trained model on a few validation pictures and saves them with
       the boxes drawn, so you can see what it learned.

Usage (from this folder, with your virtual environment active):
    python train_mnist.py
    python train_mnist.py --epochs 20 --name longer
    python train_mnist.py --model yolo11s.pt --set degrees=10 scale=0.5

Everything it makes goes in this folder:
    data/mnist_yolo/            the dataset it built (look inside!)
    runs/mnist/<name>/          training curves, example batches, weights/best.pt
    runs/mnist/<name>_predict/  validation pictures with predicted boxes

It will not find cans. FROM_DIGITS_TO_CANS.md lists what has to change, and why.
"""

import argparse
import glob
import json
import os
import random
import sys

import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "mnist_yolo")

NO_GPU = """
PyTorch cannot see an NVIDIA GPU, so this would train on the CPU.

  installed PyTorch build: {version}
  (a build that can use the GPU has "+cu" in its version, e.g. 2.11.0+cu128)

Fix it before you train:
  1. Is the virtual environment active? The prompt should start with (.venv).
  2. Reinstall PyTorch with CUDA support:
       pip uninstall -y torch torchvision
       pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  3. Check:  python -c "import torch; print(torch.cuda.is_available())"   -> True
  4. Still False? Run  nvidia-smi . If that fails too, tell your TA.

(--cpu runs on the CPU anyway. It is far slower; use it only if your TA says so.)
"""


def pick_device(allow_cpu=False):
    """Device 0 = the first NVIDIA GPU. Anything else is a broken install, so
    the script stops and says why instead of quietly training on the CPU at a
    fraction of the speed."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}, {gpu.total_memory / 1024 ** 3:.0f} GB   "
              f"(PyTorch {torch.__version__})")
        return 0
    if allow_cpu:
        print("WARNING: no GPU visible - training on the CPU because of --cpu.")
        return "cpu"
    sys.exit(NO_GPU.format(version=torch.__version__))


# ── Step 1: the dataset ────────────────────────────────────────────────────
# MNIST is a *classification* dataset: 28x28 pictures, one digit each, and the
# answer is just "which digit". A detector needs something else: bigger
# pictures with objects somewhere in them, and for each picture a list of
# boxes. So this pastes a few MNIST digits onto a blank picture and writes
# down where each one went. For the cans there is nothing to generate: the
# pictures are photos and the boxes were drawn by a person.
def build_dataset(out_dir, n_train=2000, n_val=400, size=320, seed=0):
    """Write a YOLO-format dataset:

        out_dir/data.yaml              where the pictures are, and the class names
        out_dir/train/images/*.jpg     pictures the model learns from
        out_dir/train/labels/*.txt     one text file per picture, same file name
        out_dir/valid/images/*.jpg     pictures kept aside to test it on
        out_dir/valid/labels/*.txt

    Each line of a label file is one box:   class x_centre y_centre width height
    with the four numbers as fractions of the picture's width and height (0 to 1).
    A picture with nothing in it has an empty label file.
    """
    from PIL import Image
    from torchvision.datasets import MNIST

    rng = random.Random(seed)
    raw = os.path.join(HERE, "data", "mnist_raw")
    # MNIST comes already divided: 60,000 training digits and 10,000 test
    # digits written by different people. Validation pictures are made only
    # from the second group, so the model is tested on handwriting it has
    # never seen.
    for split, n, digits in (("train", n_train, MNIST(raw, train=True, download=True)),
                             ("valid", n_val, MNIST(raw, train=False, download=True))):
        img_dir = os.path.join(out_dir, split, "images")
        lbl_dir = os.path.join(out_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(n):
            canvas = Image.new("L", (size, size), 0)
            boxes, lines = [], []
            # about one picture in ten has no digit at all: a "negative"
            for _ in range(0 if rng.random() < 0.1 else rng.randint(1, 4)):
                digit, cls = digits[rng.randrange(len(digits))]
                digit = digit.crop(digit.point(lambda p: 255 * (p > 40)).getbbox())   # trim the black border
                scale = rng.uniform(1.5, 4.5)
                w, h = max(int(digit.width * scale), 4), max(int(digit.height * scale), 4)
                digit = digit.resize((w, h), Image.BILINEAR)
                for _ in range(20):                       # find a spot that doesn't cover another digit
                    x, y = rng.randint(0, size - w), rng.randint(0, size - h)
                    if all(x + w < bx or bx + bw < x or y + h < by or by + bh < y
                           for bx, by, bw, bh in boxes):
                        canvas.paste(digit, (x, y), digit)
                        boxes.append((x, y, w, h))
                        lines.append(f"{cls} {(x + w / 2) / size:.6f} {(y + h / 2) / size:.6f} "
                                     f"{w / size:.6f} {h / size:.6f}")
                        break
            canvas.convert("RGB").save(os.path.join(img_dir, f"{split}_{i:05d}.jpg"), quality=90)
            with open(os.path.join(lbl_dir, f"{split}_{i:05d}.txt"), "w") as f:
                f.write("\n".join(lines))
    # No "path:" line: Ultralytics then looks for train/ and valid/ next to this file,
    # so the dataset folder can be moved or copied to another computer.
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write("train: train/images\n"
                "val: valid/images\n"
                "nc: 10\n"
                "names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']\n")
    print(f"built {out_dir}: {n_train} training and {n_val} validation pictures")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n.pt",
                    help="pretrained weights to start from (yolo11n/s/m.pt, yolov8n.pt, ...)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence cut-off for the saved example pictures")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None, help="run name (default: model stem)")
    ap.add_argument("--rebuild", action="store_true", help="build the dataset again")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra Ultralytics train args, e.g. scale=0.9 degrees=10")
    ap.add_argument("--cpu", action="store_true",
                    help="train on the CPU if no GPU is visible (slow; only if your TA says so)")
    args = ap.parse_args()
    DEVICE = pick_device(allow_cpu=args.cpu)

    extra = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        extra[k] = v
    # Ultralytics mirrors half the training pictures left-to-right by default.
    # A mirrored digit is not that digit, so it is switched off here.
    extra.setdefault("fliplr", 0.0)

    data_yaml = os.path.join(DATA_DIR, "data.yaml")
    if args.rebuild or not os.path.exists(data_yaml):
        build_dataset(DATA_DIR)

    # ── Step 2: train ──────────────────────────────────────────────────────
    name = args.name or os.path.splitext(os.path.basename(args.model))[0]
    project = os.path.join(HERE, "runs", "mnist")
    model = YOLO(args.model)                 # downloaded the first time it is used
    model.train(data=data_yaml,
                epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=DEVICE, workers=args.workers, project=project, name=name,
                exist_ok=True, patience=20, seed=0, plots=True, **extra)

    # ── Step 3: score the best weights on the validation pictures ─────────
    run_dir = os.path.join(project, name)
    best = YOLO(os.path.join(run_dir, "weights", "best.pt"))
    val = best.val(data=data_yaml,
                   imgsz=args.imgsz, device=DEVICE, project=project, name=name + "_val",
                   exist_ok=True)
    results = {"val": {"mAP50": float(val.box.map50), "mAP50-95": float(val.box.map),
                       "precision": float(val.box.mp), "recall": float(val.box.mr)}}
    print(f"\nbest val: {results['val']}")

    # ── Step 4: use the model on pictures, the way your GUI will ──────────
    imgs = sorted(glob.glob(os.path.join(DATA_DIR, "valid", "images", "*.jpg")))[:12]
    for r in best.predict(imgs, conf=args.conf, imgsz=args.imgsz, device=DEVICE,
                          stream=True, save=True, project=project,
                          name=name + "_predict", exist_ok=True, verbose=False):
        found = [f"{best.names[int(c)]} ({p:.2f})" for c, p in zip(r.boxes.cls, r.boxes.conf)]
        print(f"  {os.path.basename(r.path)}: {', '.join(found) or 'nothing'}")
    print(f"pictures with boxes drawn: {os.path.join(project, name + '_predict')}")

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"results: {os.path.join(run_dir, 'results.json')}")
    print(f"trained weights: {os.path.join(run_dir, 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
