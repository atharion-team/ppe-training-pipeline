"""Track persons and associate PPE to each worker, per frame.

Binds each PPE box to at most one worker by matching against a body zone (not the whole person box)

Usable two ways, same as train.py:

    python pipeline/associate.py --weights runs/train/ppe/weights/best.pt --source clip.mp4

    from pipeline.associate import associate
    summary = associate(weights="...", source="clip.mp4")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allows `python pipeline/associate.py` direct invocation
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.detect import load_ppe_vocabulary, reencode_h264  # noqa: E402
from pipeline.track import resolve_person_id, track_frame  # noqa: E402

DEFAULT_VOCAB = REPO_ROOT / "data" / "vocabulary.yaml"
DEFAULT_TRACKER = REPO_ROOT / "pipeline" / "trackers" / "bytetrack_ppe.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "associate"
DEFAULT_MIN_CONTAINMENT = 0.5

# BGR
COLOR_ASSOCIATED = (80, 190, 90)   # positive PPE, worn and confirmed
COLOR_VIOLATION = (60, 60, 220)    # negative class (NO-*), confirmed on a worker
COLOR_UNCERTAIN = (150, 150, 150)  # couldn't confidently bind to any worker
COLOR_ZONE = (200, 160, 60)


def zone_box(person_box: list[float], region: str, zones: dict[str, list[float]]) -> tuple[float, float, float, float]:
    """person_box = [x1, y1, x2, y2]. Returns the zone's (x1, y1, x2, y2)."""
    if region not in zones:
        raise SystemExit(f"no zone geometry for region '{region}' in data/vocabulary.yaml, known regions: {sorted(zones)}")
    x1, y1, x2, y2 = person_box
    top_frac, bottom_frac = zones[region]
    height = y2 - y1
    return (x1, y1 + top_frac * height, x2, y1 + bottom_frac * height)


def containment(ppe_box: list[float], zone: tuple[float, float, float, float]) -> float:
    """(PPE area inside zone) / (PPE area). Not IoU."""
    ix1, iy1 = max(ppe_box[0], zone[0]), max(ppe_box[1], zone[1])
    ix2, iy2 = min(ppe_box[2], zone[2]), min(ppe_box[3], zone[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-9, (ppe_box[2] - ppe_box[0]) * (ppe_box[3] - ppe_box[1]))
    return inter / area


def assign(
    ppe_boxes: list[dict], tracks: list[dict], region: str, min_score: float, zones: dict[str, list[float]]
) -> list[dict]:
    """Hungarian-assign one region's PPE boxes to tracks. Returns per-ppe-box
    dicts: {"track_id", "score", "associated"}, one per input ppe_box."""
    if not ppe_boxes or not tracks:
        return [{"track_id": None, "score": 0.0, "associated": False} for _ in ppe_boxes]

    zone_boxes = [zone_box(t["box"], region, zones) for t in tracks]
    scores = np.zeros((len(ppe_boxes), len(tracks)))
    for i, ppe in enumerate(ppe_boxes):
        for j, zone in enumerate(zone_boxes):
            scores[i, j] = containment(ppe["box"], zone)

    cost = np.where(scores >= min_score, 1.0 - scores, 1.0)
    row_idx, col_idx = linear_sum_assignment(cost)

    results = [{"track_id": None, "score": 0.0, "associated": False} for _ in ppe_boxes]
    for r, c in zip(row_idx, col_idx):
        score = float(scores[r, c])
        if score >= min_score:
            results[r] = {"track_id": tracks[c]["track_id"], "score": round(score, 4), "associated": True}
    return results


def draw_label(frame, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    """Filled color background behind white text, readable against any footage."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_associations(
    frame, tracks: list[dict], ppe_boxes: list[dict], scored: dict[int, dict],
    negative_names: set[str], zones: dict[str, list[float]],
) -> None:
    for t in tracks:
        for region in zones:
            x1, y1, x2, y2 = (int(v) for v in zone_box(t["box"], region, zones))
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ZONE, 2)
        x1, y1, x2, y2 = (int(v) for v in t["box"])
        draw_label(frame, x1, y1, f"id {t['track_id']}", COLOR_ZONE)

    for idx, ppe in enumerate(ppe_boxes):
        result = scored.get(idx, {"associated": False, "score": 0.0})
        if not result["associated"]:
            color = COLOR_UNCERTAIN
        elif ppe["cls"] in negative_names:
            color = COLOR_VIOLATION  # confirmed: this worker is missing PPE
        else:
            color = COLOR_ASSOCIATED  # confirmed: this worker is wearing it
        x1, y1, x2, y2 = (int(v) for v in ppe["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        draw_label(frame, x1, y1, f"{ppe['cls']} {result['score']:.2f}", color)


def associate(
    weights: str | Path,
    source: str,
    vocabulary: str | Path = DEFAULT_VOCAB,
    tracker: str | Path = DEFAULT_TRACKER,
    min_containment: float | None = None,
    conf: float = 0.35,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str | int | None = None,
    output: str | Path = DEFAULT_OUTPUT,
    show: bool = False,
) -> dict:
    """Track persons and associate PPE per frame. Returns the summary dict written to associate_summary.json.
    min_containment=None (the default, and the CLI's default) means "use data/vocabulary.yaml's min_containment"; pass a number to override it."""
    
    vocab = load_ppe_vocabulary(Path(vocabulary))
    zones = vocab["zones"]
    if min_containment is None:
        min_containment = vocab["min_containment"] if vocab["min_containment"] is not None else DEFAULT_MIN_CONTAINMENT
    model = YOLO(weights)
    ppe_model = YOLO(weights)
    person_id = resolve_person_id(model, vocab["subject"])

    by_name = {name: idx for idx, name in ppe_model.names.items()}
    ppe_names = list(vocab["ppe"])
    negative_names = list(vocab["negative"])
    ppe_and_negative_ids = [by_name[n] for n in ppe_names + negative_names if n in by_name]
    missing = [n for n in ppe_names + negative_names if n not in by_name]
    if missing:
        print(f"warning: vocabulary classes not in model, skipping: {missing}")
        
    # positive classes this detector can actually emit
    known_ppe_classes = [n for n in ppe_names if n in by_name]

    violation_to_region = {entry["violation"]: entry["region"] for entry in vocab["ppe"].values()}
    class_info: dict[str, dict] = {
        name: {"region": entry["region"], "violation": entry["violation"]} for name, entry in vocab["ppe"].items()
    }
    for name, violation in vocab["negative"].items():
        region = violation_to_region.get(violation)
        if region is None:
            print(f"warning: negative class '{name}' has no matching ppe region for violation '{violation}', skipping")
            continue
        class_info[name] = {"region": region, "violation": violation}

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"could not open source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = "webcam" if source.isdigit() else Path(source).stem
    video_path = out_dir / f"{stem}_associated.mp4"
    records_path = out_dir / "associations.jsonl"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames = 0
    associated_count = 0
    try:
        with records_path.open("w", encoding="utf-8") as records_file:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                tracks, _ = track_frame(model, frame, person_id, str(tracker), conf, iou, imgsz, device)

                ppe_result = ppe_model.predict(
                    frame, classes=ppe_and_negative_ids, conf=conf, iou=iou, imgsz=imgsz, device=device,
                    verbose=False,
                )[0]
                ppe_boxes = [
                    {"cls": ppe_result.names[int(box.cls)], "conf": float(box.conf),
                     "box": [float(v) for v in box.xyxy[0]]}
                    for box in ppe_result.boxes
                ]

                timestamp = round(frames / fps, 3)
                records_file.write(
                    json.dumps({"frame": frames, "timestamp": timestamp, "tracks_present": [t["track_id"] for t in tracks]}) + "\n"
                )

                scored: dict[int, dict] = {}
                by_region: dict[str, list[int]] = {}
                for idx, ppe in enumerate(ppe_boxes):
                    name = ppe["cls"]
                    region = (class_info.get(name) or {}).get("region")
                    if region is None:
                        continue
                    by_region.setdefault(region, []).append(idx)

                for region, indices in by_region.items():
                    region_boxes = [ppe_boxes[i] for i in indices]
                    results = assign(region_boxes, tracks, region, min_containment, zones)
                    for local_i, idx in enumerate(indices):
                        scored[idx] = results[local_i]
                        result = results[local_i]
                        if not result["associated"]:
                            continue
                        associated_count += 1
                        name = ppe_boxes[idx]["cls"]
                        violation = class_info[name]["violation"]
                        records_file.write(
                            json.dumps(
                                {
                                    "frame": frames, "timestamp": timestamp,
                                    "track_id": result["track_id"], "ppe_class": name,
                                    "region": region, "violation": violation,
                                    "score": result["score"], "associated": True,
                                }
                            )
                            + "\n"
                        )

                draw_associations(frame, tracks, ppe_boxes, scored, set(negative_names), zones)
                frames += 1
                writer.write(frame)
                if show:
                    cv2.imshow("associate", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        capture.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    reencode_h264(video_path)

    summary = {
        "source": source,
        "output": str(video_path),
        "records": str(records_path),
        "frames": frames,
        "associations": associated_count,
        "min_containment": min_containment,
        "known_ppe_classes": known_ppe_classes,
    }
    summary_path = out_dir / "associate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{stem}: {frames} frames, {associated_count} PPE associations (min_containment={min_containment})")
    print(f"\nsummary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track persons and associate PPE to workers")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="video, webcam index, or stream")
    p.add_argument("--vocabulary", default=str(DEFAULT_VOCAB))
    p.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    p.add_argument(
        "--min-containment", type=float, default=None, dest="min_containment",
        help="overrides data/vocabulary.yaml's min_containment; omit to use the vocabulary's value",
    )
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    associate(**vars(parse_args()))


if __name__ == "__main__":
    main()
