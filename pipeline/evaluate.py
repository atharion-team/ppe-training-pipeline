"""Validate trained weights and write a metrics summary.

Usable two ways, same as train.py:

    python pipeline/evaluate.py --weights runs/train/ppe/weights/best.pt --split test

    from pipeline.evaluate import evaluate
    summary = evaluate(weights="runs/train/ppe/weights/best.pt", split="test")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "css-data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "val"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained PPE detector")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--project", default=str(DEFAULT_PROJECT))
    p.add_argument("--name", default="ppe")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def per_class_metrics(box, names: dict[int, str]) -> list[dict]:
    """Build a per-class row for every class the validation run reported."""
    rows = []
    # box.ap_class_index maps each row of the per-class arrays back to a class id
    class_ids = getattr(box, "ap_class_index", None)
    class_ids = [] if class_ids is None else class_ids
    for i, class_id in enumerate(class_ids):
        precision = float(box.p[i]) if i < len(box.p) else 0.0
        recall = float(box.r[i]) if i < len(box.r) else 0.0
        rows.append(
            {
                "class_id": int(class_id),
                "name": names.get(int(class_id), str(class_id)),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1(precision, recall), 4),
                "map50": round(float(box.ap50[i]), 4) if i < len(box.ap50) else None,
                "map50_95": round(float(box.ap[i]), 4) if i < len(box.ap) else None,
            }
        )
    return sorted(rows, key=lambda r: r["map50_95"] or 0.0)


def evaluate(
    weights: str | Path,
    data: str | Path = DEFAULT_DATA,
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    device: str | int | None = None,
    project: str | Path = DEFAULT_PROJECT,
    name: str = "ppe",
    exist_ok: bool = False,
) -> dict:
    """Validate weights and return the same summary dict written to evaluation_summary.json."""
    model = YOLO(weights)
    results = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        exist_ok=exist_ok,
        plots=True,
    )

    box = results.box
    names = results.names or {}
    precision, recall = float(box.mp), float(box.mr)

    summary = {
        "weights": str(weights),
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "save_dir": str(results.save_dir),
        "overall": {
            "map50_95": round(float(box.map), 4),
            "map50": round(float(box.map50), 4),
            "map75": round(float(box.map75), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1(precision, recall), 4),
        },
        "per_class": per_class_metrics(box, names),
    }

    out_dir = Path(results.save_dir)
    out_path = out_dir / "evaluation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = evaluate(**vars(parse_args()))

    o = summary["overall"]
    print(f"\nmAP50-95 {o['map50_95']}   mAP50 {o['map50']}   P {o['precision']}   R {o['recall']}   F1 {o['f1']}")
    print("\nweakest classes by mAP50-95:")
    for row in summary["per_class"][:5]:
        print(f"  {row['name']:<16} {row['map50_95']}")
    print(f"\nsummary: {Path(summary['save_dir']) / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
