"""Download the instructor's public ZIP and safely extract only redbull_v5.

Uses OpenCloud's public WebDAV endpoint, as the /s/... page is an HTML app.
No annotation, redistribution, or splitting is performed here.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

PUBLIC_SHARE = "https://opencloud.colecorbett.ca/s/WrRfxtOfPprYiSj"
DOWNLOAD_URL = "https://opencloud.colecorbett.ca/remote.php/dav/public-files/WrRfxtOfPprYiSj/last_can_data.zip"
EXPECTED_SHA1 = "f41af221b8b6357e02025ca387af1baa1e771b32"
ROOT = Path(__file__).resolve().parent.parent


def extract_dataset(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            pieces = Path(member.filename).parts
            if ".." in pieces or Path(member.filename).is_absolute():
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if "redbull_v5" in pieces:
                relative = Path(*pieces[pieces.index("redbull_v5"):])
            elif member.filename.replace("\\", "/") in ("data/README.md", "README.md"):
                relative = Path("DATASET_README.md")
            else:
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive path escapes dataset destination")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    yaml_file = destination / "redbull_v5" / "data.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError("Archive did not contain redbull_v5/data.yaml")
    return count, yaml_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "can_dataset.zip")
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.extract_only:
        temporary = args.archive.with_suffix(".part")
        print(f"Downloading instructor dataset: {PUBLIC_SHARE}", flush=True)
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=60) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        actual = hashlib.sha1(temporary.read_bytes()).hexdigest()
        if actual != EXPECTED_SHA1:
            raise ValueError(f"Dataset checksum mismatch: {actual}; archive not extracted")
        temporary.replace(args.archive)
    count, yaml_file = extract_dataset(args.archive, args.destination)
    print(f"Extracted {count} files; dataset: {yaml_file}")


if __name__ == "__main__":
    main()
