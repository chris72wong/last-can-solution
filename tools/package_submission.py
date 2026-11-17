"""Create a portable submission ZIP from an explicit allowlist, never datasets/venv."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ("lastcan", "config", "models", "tests", "tools")
ROOT_FILES = ("main.py", "train_cans.py", "setup.ps1", "README.md", "HOW_TO_RUN.md",
              "HARDWARE_CHECKLIST.md", "TRAINING_NOTES.md", "INTERFACES.md", ".gitignore",
              "requirements.txt", "requirements-demo.txt", "requirements-cuda.txt", "requirements-lock.txt")
EVIDENCE = ("environment.json", "training-results.json", "test-results.txt", "gui-smoke.json",
            "training-batch-review.md", "model-inference-check.json", "BUILD_REPORT.md",
            "mnist-sample-results.json", "dataset-summary.json", "training-review.json",
            "gui-smoke-normal.json")


def included_files(root: Path = ROOT):
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for folder in FOLDERS:
        base = root / folder
        if not base.exists():
            continue
        for directory, subdirs, filenames in os.walk(base):
            subdirs[:] = [name for name in subdirs if not name.startswith(".") and
                          name not in {"__pycache__", "data", "runs", "captures", "dist"}]
            for name in filenames:
                path = Path(directory) / name
                if not name.startswith(".") and path.suffix.lower() in (".py", ".json", ".md", ".txt", ".ps1", ".pt"):
                    files.append(path)
    for name in EVIDENCE:
        path = root / "verification" / name
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "last-can-submission.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = included_files()
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"):
                {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in files}
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "last-can-solution/" + path.relative_to(ROOT).as_posix())
        archive.writestr("last-can-solution/SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(files)} files: {args.output}")
    print("Trained model included:", (ROOT / "models" / "best.pt") in files)


if __name__ == "__main__":
    main()
