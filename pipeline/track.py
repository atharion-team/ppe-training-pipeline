"""Tracks people across a video, webcam, or stream. 
Persons only, per the project's tracking design: mixing PPE classes into the tracker causes id switches,

Usable two ways, same as train.py:

    python pipeline/track.py --weights runs/train/ppe/weights/best.pt --source clip.mp4

    from pipeline.track import track
    summary = track(weights="...", source="clip.mp4")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allows `python pipeline/track.py` direct invocation
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.detect import load_ppe_vocabulary, reencode_h264  # noqa: E402

DEFAULT_VOCAB = REPO_ROOT / "data" / "vocabulary.yaml"
DEFAULT_TRACKER = REPO_ROOT / "pipeline" / "trackers" / "bytetrack_ppe.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "track"

# BGR
COLOR_TRACK = (80, 190, 90)


def resolve_person_id(model: YOLO, subject: str) -> int:
    """Look up the subject's class id from the model's own names, never hardcoded."""
    by_name = {name: idx for idx, name in model.names.items()}
    if subject not in by_name:
        raise SystemExit(f"vocabulary subject '{subject}' not in model classes: {sorted(by_name)}")
    return by_name[subject]


def track_frame(
    model: YOLO, frame, person_id: int, tracker: str, conf: float, iou: float, imgsz: int, device
) -> tuple[list[dict], int]:
    """Track persons in one frame. Returns (records, dropped) where records
    covers boxes with a confirmed track id only; dropped counts boxes the
    tracker could not match this frame."""
    result = model.track(
        frame,
        persist=True,
        tracker=tracker,
        classes=[person_id],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    records = []
    dropped = 0
    ids = result.boxes.id
    for i, box in enumerate(result.boxes):
        track_id = ids[i] if ids is not None else None
        if track_id is None:
            dropped += 1
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        records.append(
            {
                "track_id": int(track_id),
                "cls": result.names[int(box.cls)],
                "conf": float(box.conf),
                "box": [x1, y1, x2, y2],
            }
        )
    return records, dropped


def draw_tracks(frame, records: list[dict]) -> None:
    for r in records:
        x1, y1, x2, y2 = (int(v) for v in r["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TRACK, 2)
        caption = f"id {r['track_id']} {r['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), COLOR_TRACK, -1)
        cv2.putText(frame, caption, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def track(
    weights: str | Path,
    source: str,
    vocabulary: str | Path = DEFAULT_VOCAB,
    tracker: str | Path = DEFAULT_TRACKER,
    conf: float = 0.35,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str | int | None = None,
    output: str | Path = DEFAULT_OUTPUT,
    show: bool = False,
) -> dict:
    """Track persons across a video/webcam source. Returns the same summary
    dict written to track_summary.json."""
    vocab = load_ppe_vocabulary(Path(vocabulary))
    model = YOLO(weights)
    person_id = resolve_person_id(model, vocab["subject"])

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"could not open source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = "webcam" if source.isdigit() else Path(source).stem
    video_path = out_dir / f"{stem}_tracked.mp4"
    records_path = out_dir / "tracks.jsonl"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames = 0
    tracks_seen: set[int] = set()
    boxes_dropped_no_id = 0
    try:
        with records_path.open("w", encoding="utf-8") as records_file:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_records, dropped = track_frame(model, frame, person_id, str(tracker), conf, iou, imgsz, device)
                boxes_dropped_no_id += dropped

                timestamp = frames / fps
                for r in frame_records:
                    tracks_seen.add(r["track_id"])
                    records_file.write(json.dumps({"frame": frames, "timestamp": round(timestamp, 3), **r}) + "\n")

                draw_tracks(frame, frame_records)
                frames += 1
                writer.write(frame)
                if show:
                    cv2.imshow("track", frame)
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
        "tracks_seen": len(tracks_seen),
        "boxes_dropped_no_id": boxes_dropped_no_id,
    }
    summary_path = out_dir / "track_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{stem}: {frames} frames, {len(tracks_seen)} tracks seen, {boxes_dropped_no_id} boxes dropped (no id)")
    print(f"\nsummary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track persons across a video")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="video, webcam index, or stream")
    p.add_argument("--vocabulary", default=str(DEFAULT_VOCAB))
    p.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    summary = track(**vars(parse_args()))
    print(f"\nnext:    python pipeline/associate.py --weights ... --source {summary['source']}")


if __name__ == "__main__":
    main()
