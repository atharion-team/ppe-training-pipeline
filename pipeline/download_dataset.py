"""Download and extract the dataset named in the data config

Two sources:
  roboflow:  the API. Pins an exact dataset version, needs ROBOFLOW_API_KEY.
  download:  a signed export URL. No key needed, but the link expires.

Either way the export's own class list is checked against ours. A silent class
mismatch trains correct-looking weights that predict the wrong labels.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "css-data.yaml"
ENV_FILE = REPO_ROOT / ".env"
API_KEY_VAR = "ROBOFLOW_API_KEY"
SPLIT_KEYS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the dataset for a data config")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--url", default=None, help="signed export url, skips the roboflow api")
    p.add_argument("--api-key", default=None, help=f"overrides {API_KEY_VAR} and .env")
    p.add_argument("--version", type=int, default=None, help="overrides the config's roboflow version")
    p.add_argument("--force", action="store_true", help="re-download over an existing dataset")
    p.add_argument("--keep-zip", action="store_true", help="keep the downloaded archive")
    p.add_argument("--no-settings", action="store_true", help="do not point Ultralytics at the repo root")
    return p.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_api_key(explicit: str | None) -> str | None:
    """Check the flag, then the environment, then .env. Never the data config."""
    if explicit:
        return explicit
    if os.environ.get(API_KEY_VAR):
        return os.environ[API_KEY_VAR]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == API_KEY_VAR:
                return value.strip().strip('"').strip("'")
    return None


def fetch_roboflow(config: dict, target: Path, api_key: str, version_override: int | None) -> None:
    try:
        from roboflow import Roboflow
    except ImportError as err:
        raise SystemExit("roboflow is not installed. pip install -r requirements.txt") from err

    version_number = version_override or config["version"]
    print(f"roboflow: {config['workspace']}/{config['project']} v{version_number} as {config['format']}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(config["workspace"]).project(config["project"])
    version = project.version(version_number)
    version.download(config["format"], location=str(target), overwrite=True)


def fetch_url(url: str, target: Path, keep_zip: bool) -> None:
    archive = target.parent / "dataset.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": "ppe-training-pipeline"})
    try:
        with urllib.request.urlopen(request) as response, archive.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            read = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                read += len(chunk)
                if total:
                    print(f"\r  {read / 1e6:7.1f} / {total / 1e6:.1f} MB  ({read * 100 // total:3d}%)", end="")
                else:
                    print(f"\r  {read / 1e6:7.1f} MB", end="")
    except urllib.error.HTTPError as err:
        raise SystemExit(f"download failed: HTTP {err.code}. Signed links expire, regenerate it.") from err
    except urllib.error.URLError as err:
        raise SystemExit(f"download failed: {err.reason}") from err
    print("\nextracting")

    extract(archive, target)
    if not keep_zip:
        archive.unlink(missing_ok=True)


def extract(archive: Path, target: Path) -> None:
    """Extract into target, flattening a single wrapper directory if present."""
    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)

        entries = list(staging.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging

        target.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            destination = target / item.name
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            shutil.move(str(item), str(destination))


def normalise_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names or [])}


def check_classes(config: dict, dataset_dir: Path) -> bool:
    """Compare the export's own data.yaml class list against ours."""
    exported = next((p for p in (dataset_dir / "data.yaml", dataset_dir / "data.yml") if p.exists()), None)
    if exported is None:
        print("\nno data.yaml in the export, skipping the class check")
        return True

    theirs = normalise_names(yaml.safe_load(exported.read_text(encoding="utf-8")).get("names"))
    ours = normalise_names(config.get("names"))

    if theirs == ours:
        print(f"\nclass check: {len(ours)} classes match the export")
        return True

    print("\nCLASS MISMATCH between the export and the data config")
    for index in sorted(set(theirs) | set(ours)):
        mine, exported_name = ours.get(index), theirs.get(index)
        flag = "  " if mine == exported_name else "->"
        print(f"  {flag} {index:>2}  ours={mine!s:<16} export={exported_name!s}")
    print("\nFix data/css-data.yaml (and data/vocabulary.yaml) to match the export before training.")
    return False


def check_splits(config: dict, dataset_dir: Path) -> bool:
    ok = True
    print()
    for key in SPLIT_KEYS:
        relative = config.get(key)
        if not relative:
            continue
        images = dataset_dir / relative
        labels = images.parent / "labels"
        count = len(list(images.glob("*.*"))) if images.is_dir() else 0
        if count:
            print(f"  {key:<6} {count:>6} images   labels={'yes' if labels.is_dir() else 'MISSING'}")
        else:
            print(f"  {key:<6} MISSING at {images}")
            ok = False
    return ok


def fetch(args, config: dict, dataset_dir: Path) -> None:
    """Pick a source and populate dataset_dir."""
    url = args.url or (config.get("download") or "").strip()
    roboflow_config = config.get("roboflow") or {}

    if args.url:
        return fetch_url(url, dataset_dir, args.keep_zip)

    if roboflow_config:
        api_key = resolve_api_key(args.api_key)
        if api_key:
            return fetch_roboflow(roboflow_config, dataset_dir, api_key, args.version)
        if not url:
            raise SystemExit(
                f"no api key. Set {API_KEY_VAR} in {ENV_FILE.name}, export it, or pass --api-key.\n"
                "Find it under Roboflow > Settings > API Keys."
            )
        print(f"no {API_KEY_VAR} found, falling back to the signed url")

    if not url:
        raise SystemExit(f"no roboflow config and no download url in {args.data}")
    fetch_url(url, dataset_dir, args.keep_zip)


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"data config not found: {data_path}")
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_dir = resolve_path(config.get("path", "datasets/dataset"))

    if dataset_dir.exists() and any(dataset_dir.iterdir()) and not args.force:
        print(f"dataset already present at {dataset_dir}")
        print("re-run with --force to replace it")
    else:
        print(f"downloading to {dataset_dir}")
        fetch(args, config, dataset_dir)

    classes_ok = check_classes(config, dataset_dir)
    splits_ok = check_splits(config, dataset_dir)

    if not args.no_settings:
        try:
            from ultralytics import settings

            settings.update({"datasets_dir": str(REPO_ROOT)})
            print(f"\nultralytics datasets_dir -> {REPO_ROOT}")
        except Exception as err:  # not installed yet, or settings locked
            print(f"\ncould not update ultralytics settings ({err}), set datasets_dir manually if training cannot find the data")

    if classes_ok and splits_ok:
        print("\nready:   python pipeline/train.py --epochs 1 --fraction 0.05 --batch 2 --device cpu --workers 0 --name smoke_test")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
