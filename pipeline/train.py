"""Finetunes a YOLO detector on the construction PPE dataset.

For full list of arguments, run:
    python pipeline/train.py -h
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "css-data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "train"
WANDB_PROJECT = "ppe-training-pipeline"

# Online augmentation until actual dataset is provided 
# Reduce these if the dataset is already augmented offline
AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "mosaic": 1.0,
}


def init_wandb(name: str) -> None:
    """
    Pre-init a wandb run so Ultralytics' callback reuses it.
    Make sure you are logged in using python -m wandb login
    """
    try:
        from ultralytics import settings

        if not settings.get("wandb"):
            return
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        wandb.finish()
    wandb.init(project=WANDB_PROJECT, name=name)


def train(
    model: str = "yolo26n.pt",
    data: str | Path = DEFAULT_DATA,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 24,
    device: str | int | None = None,
    workers: int = 8,
    project: str | Path = DEFAULT_PROJECT,
    name: str = "ppe",
    patience: int = 30,
    seed: int = 42,
    fraction: float = 1.0,
    resume: str | None = None,
    exist_ok: bool = False,
    **overrides,
):
    """Finetune the detector and return the Ultralytics results object"""
    if resume:
        init_wandb(Path(resume).resolve().parent.parent.name)
        return YOLO(resume).train(resume=True)

    init_wandb(name)
    return YOLO(model).train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        patience=patience,
        seed=seed,
        fraction=fraction,
        exist_ok=exist_ok,
        cos_lr=True,
        save_period=10,
        plots=True,
        **{**AUGMENTATION, **overrides},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune a YOLO PPE detector")
    p.add_argument("--model", default="yolo26n.pt", help="base checkpoint, downloaded on first run")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=24, help="-1 auto-fits to available VRAM, but conservatively")
    p.add_argument("--device", default="0", help='"0" for the first GPU, "cpu" to force CPU')
    p.add_argument("--workers", type=int, default=8, help="use 0 if the Windows dataloader hangs")
    p.add_argument("--project", default=str(DEFAULT_PROJECT))
    p.add_argument("--name", default="ppe")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of the train set, for smoke tests")
    p.add_argument("--resume", default=None, help="path to a last.pt to resume from")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def main() -> None:
    results = train(**vars(parse_args()))

    save_dir = Path(results.save_dir)
    weights = save_dir / "weights" / "best.pt"
    print(f"\nrun:     {save_dir}")
    print(f"weights: {weights}")
    print(f"\nnext:    python pipeline/evaluate.py --weights {weights} --split test")


if __name__ == "__main__":
    main()
