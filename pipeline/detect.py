"""Per-frame PPE detection on an image, video, folder, or webcam.

This is the per-frame baseline: every violation box is counted in every frame it appears,
with no worker identity and no deduplication.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = REPO_ROOT / "data" / "vocabulary.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "inference"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# BGR
COLOR_VIOLATION = (60, 60, 220)
COLOR_PPE = (80, 190, 90)
COLOR_SUBJECT = (200, 160, 60)
COLOR_IGNORE = (150, 150, 150)


def load_vocabulary(path: Path) -> tuple[set[str], set[str], str]:
    """Return (violation class names, positive PPE names, subject name)."""
    vocab = yaml.safe_load(path.read_text(encoding="utf-8"))
    violations = set(vocab.get("negative", {}) or {})
    ppe = set(vocab.get("ppe", {}) or {})
    return violations, ppe, vocab.get("subject", "Person")


def load_ppe_vocabulary(path: Path) -> dict:
    """Full vocabulary: subject, ppe (region+violation per class), negative
    violation ids, ignore list, body zones, min containment score, and
    compliance state-machine thresholds. Used by track.py/associate.py/compliance.py."""
    vocab = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "subject": vocab.get("subject", "Person"),
        "ppe": vocab.get("ppe", {}) or {},
        "negative": vocab.get("negative", {}) or {},
        "ignore": list(vocab.get("ignore", []) or []),
        "zones": vocab.get("zones", {}) or {},
        "min_containment": vocab.get("min_containment"),
        "compliance": vocab.get("compliance", {}) or {},
    }


def reencode_h264(path: Path) -> None:
    """Re-encode a video in place to H.264. cv2.VideoWriter's mp4v codec runs 4-5x larger than H.264 for the same content.
    No-ops, leaving the mp4v file as-is, if imageio-ffmpeg isn't installed."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print(f"note: imageio-ffmpeg not installed, leaving {path.name} as mp4v (larger file)")
        return

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = path.with_suffix(".h264" + path.suffix)
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-c:a", "copy", str(tmp)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not tmp.exists():
        print(f"warning: h264 re-encode failed for {path.name}, keeping original mp4v ({result.stderr[-300:]})")
        tmp.unlink(missing_ok=True)
        return
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run per-frame PPE detection")
    p.add_argument("--weights", required=True)
    p.add_argument(
        "--source", required=True, help="image, video, folder, or webcam index"
    )
    p.add_argument("--vocabulary", default=str(DEFAULT_VOCAB))
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


class Annotator:
    def __init__(self, violations: set[str], ppe: set[str], subject: str):
        self.violations = violations
        self.ppe = ppe
        self.subject = subject

    def color_for(self, label: str) -> tuple[int, int, int]:
        if label in self.violations:
            return COLOR_VIOLATION
        if label in self.ppe:
            return COLOR_PPE
        if label == self.subject:
            return COLOR_SUBJECT
        return COLOR_IGNORE

    def draw(self, frame, result) -> tuple[int, int]:
        """Draw every box on the frame. Returns (detections, violations)."""
        names = result.names
        detections = violations = 0

        for box in result.boxes:
            label = names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            color = self.color_for(label)

            detections += 1
            if label in self.violations:
                violations += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            caption = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame,
                caption,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return detections, violations

    def draw_panel(self, frame, lines: list[str]) -> None:
        y = 24
        for line in lines:
            cv2.putText(
                frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
            )
            cv2.putText(
                frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            y += 24


def resolve_sources(source: str) -> tuple[list[Path], bool]:
    """Return (paths, is_stream). A webcam index yields an empty path list."""
    if source.isdigit():
        return [], True

    path = Path(source)
    if path.is_dir():
        return (
            sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
            False,
        )
    return [path], path.suffix.lower() in VIDEO_SUFFIXES


def process_images(
    model, annotator, paths: list[Path], out_dir: Path, args
) -> list[dict]:
    records = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"skipped unreadable image: {path}")
            continue

        result = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        detections, violations = annotator.draw(frame, result)
        annotator.draw_panel(
            frame, [f"detections {detections}", f"violations {violations}"]
        )

        out_path = out_dir / f"{path.stem}_detected{path.suffix}"
        cv2.imwrite(str(out_path), frame)
        if args.show:
            cv2.imshow("ppe", frame)
            cv2.waitKey(0)

        records.append(
            {
                "source": str(path),
                "output": str(out_path),
                "frames": 1,
                "detections": detections,
                "violations": violations,
            }
        )
        print(f"{path.name}: {detections} detections, {violations} violations")
    return records


def process_video(model, annotator, source: str, out_dir: Path, args) -> list[dict]:
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"could not open source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = "webcam" if source.isdigit() else Path(source).stem
    out_path = out_dir / f"{stem}_detected.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frames = total_detections = total_violations = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            detections, violations = annotator.draw(frame, result)

            frames += 1
            total_detections += detections
            total_violations += violations
            annotator.draw_panel(
                frame,
                [
                    f"frame {frames}",
                    f"detections {total_detections}",
                    f"violations {total_violations}",
                ],
            )

            writer.write(frame)
            if args.show:
                cv2.imshow("ppe", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()

    reencode_h264(out_path)

    print(
        f"{stem}: {frames} frames, {total_detections} detections, {total_violations} raw violation counts"
    )
    return [
        {
            "source": source,
            "output": str(out_path),
            "frames": frames,
            "detections": total_detections,
            "violations": total_violations,
        }
    ]


def main() -> None:
    args = parse_args()

    violations, ppe, subject = load_vocabulary(Path(args.vocabulary))
    annotator = Annotator(violations, ppe, subject)
    model = YOLO(args.weights)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, is_stream = resolve_sources(args.source)
    if is_stream:
        records = process_video(model, annotator, args.source, out_dir, args)
    else:
        records = process_images(model, annotator, paths, out_dir, args)

    if args.show:
        cv2.destroyAllWindows()

    summary_path = out_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
