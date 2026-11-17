"""Check ZIP contents/hashes and launch the extracted cockpit without a robot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "dist" / "last-can-submission.zip")
    args = parser.parse_args()
    destination = ROOT / ".cache" / "submission-check" / uuid.uuid4().hex
    destination.mkdir(parents=True)
    forbidden = {".venv", ".cache", "__pycache__", "data", "runs", "captures"}
    with zipfile.ZipFile(args.archive) as archive:
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()) or ".." in parts:
                raise ValueError(f"Unsafe archive member: {name}")
            if any(part in forbidden for part in parts):
                raise ValueError(f"Excluded training/environment artifact in ZIP: {name}")
            if name.endswith("/yolo11n.pt"):
                raise ValueError("Generic base weights must not be shipped as the final detector")
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        manifest = json.loads(archive.read("last-can-solution/SUBMISSION_MANIFEST.json"))
        for relative, expected in manifest.items():
            data = archive.read("last-can-solution/" + relative)
            if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError(f"Manifest mismatch: {relative}")
        if "models/best.pt" not in manifest:
            raise ValueError("The genuine trained detector is missing from the submission")
        archive.extractall(destination)  # every resolved target was checked above
    extracted = destination / "last-can-solution"
    checks = {}
    for name, arguments in (("demo", ["main.py", "--demo", "--smoke-seconds", "1"]),
                            ("normal_model", ["tools/gui_smoke.py", "--normal"])):
        result = subprocess.run([sys.executable, *arguments], cwd=extracted,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", timeout=40)
        checks[name] = {"exit_code": result.returncode, "output": result.stdout.strip()}
        if result.returncode:
            raise RuntimeError(f"Extracted {name} check failed: {result.stdout}")
    normal = json.loads((extracted / "verification" / "gui-smoke-normal.json").read_text(encoding="utf-8"))
    if normal["connected"] or not normal["model_available"] or not normal["real_configuration_unchanged"]:
        raise AssertionError("Extracted normal cockpit did not load weights while staying disconnected")
    report = {"passed": True, "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "manifest_files_verified": len(manifest), "trained_model_sha256": manifest["models/best.pt"]["sha256"],
        "excluded_artifacts_absent": True, "extracted_to": str(extracted), "checks": checks}
    output = ROOT / "verification" / "package-verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
