"""Turn associations per-frame facts into deduplicated violation events.

Three filters in series, per docs/pipeline/compliance.md:
  Smoothing     rolling window, majority vote, absorbs single-frame dropouts
  Hysteresis    on-delay to raise a violation, off-delay to clear it
  Deduplication at most one open event per (track_id, violation), a cleared-then-recurring violation within a cooldown is the
                same incident, not a new one

Primary signal is positive-PPE absence (ADR 0001's decided direction): a
worker is non-compliant when no positive PPE item is confirmed on them that
frame, regardless of whether a NO-* box also fired. NO-* detections are in
associate.py's output already but are not wired into this decision, "which
signal compliance trusts" is explicitly left to event-level eval, not code,
see docs/decisions/0001-positive-only-detection.md.

All thresholds are placeholders reasoned from the docs' own design intent,
not tuned against labelled footage yet: see the module-level defaults below
for the reasoning behind each one.

Usable two ways, same as train.py:

    python pipeline/compliance.py --associations runs/associate/ppe/associations.jsonl

    from pipeline.compliance import compliance
    summary = compliance(associations="...")
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allows `python pipeline/compliance.py` direct invocation
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.detect import load_ppe_vocabulary  # noqa: E402

DEFAULT_VOCAB = REPO_ROOT / "data" / "vocabulary.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "compliance"

WINDOW_SECONDS = 0.5
PRESENT_RATIO = 0.5
ON_DELAY_SECONDS = 3.0
OFF_DELAY_SECONDS = 2.0
COOLDOWN_SECONDS = 5.0
TRACK_LOSS_CLOSE_SECONDS = 3.0


def resolve_known_classes(associations_path: Path, ordered: list[dict], violation_of: dict[str, str]) -> set[str]:
    """Which positive PPE classes to run absence-judgement on. Prefers
    associate.py's own record of what the detector can actually emit
    (associate_summary.json's known_ppe_classes, written next to
    associations.jsonl), since a class never positively seen this session
    is ambiguous: either the model doesn't know it, or every worker is
    genuinely violating it, and only the model's own class list can tell
    those apart. Falls back to session-observed presence, with a warning,
    only when no summary is available (e.g. a hand-built associations.jsonl)."""
    summary_path = associations_path.parent / "associate_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        known = summary.get("known_ppe_classes")
        if known is not None:
            return set(known)
    print(
        f"warning: no known_ppe_classes in {summary_path.name} next to {associations_path.name}, falling back to "
        f"session-observed presence (can't tell a class the model never learned from one everyone is violating)"
    )
    return {name for f in ordered for (_, name) in f["assoc"] if name in violation_of}


def load_records(path: Path) -> dict[int, dict]:
    """Group associations.jsonl by frame: {frame: {timestamp, tracks_present, assoc}}."""
    frames: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        frame = frames.setdefault(rec["frame"], {"timestamp": rec["timestamp"], "tracks_present": [], "assoc": {}})
        if "tracks_present" in rec:
            frame["tracks_present"] = rec["tracks_present"]
        else:
            frame["assoc"][(rec["track_id"], rec["ppe_class"])] = rec
    return frames


def compliance(
    associations: str | Path,
    vocabulary: str | Path = DEFAULT_VOCAB,
    window_seconds: float | None = None,
    present_ratio: float | None = None,
    on_delay_seconds: float | None = None,
    off_delay_seconds: float | None = None,
    cooldown_seconds: float | None = None,
    track_loss_close_seconds: float | None = None,
    output: str | Path = DEFAULT_OUTPUT,
) -> dict:
    """Run the smoothing/hysteresis/dedup pipeline over one associations.jsonl.
    """
    vocab = load_ppe_vocabulary(Path(vocabulary))
    violation_of = {name: entry["violation"] for name, entry in vocab["ppe"].items()}

    cfg = vocab["compliance"]
    if window_seconds is None:
        window_seconds = cfg.get("window_seconds", WINDOW_SECONDS)
    if present_ratio is None:
        present_ratio = cfg.get("present_ratio", PRESENT_RATIO)
    if on_delay_seconds is None:
        on_delay_seconds = cfg.get("on_delay_seconds", ON_DELAY_SECONDS)
    if off_delay_seconds is None:
        off_delay_seconds = cfg.get("off_delay_seconds", OFF_DELAY_SECONDS)
    if cooldown_seconds is None:
        cooldown_seconds = cfg.get("cooldown_seconds", COOLDOWN_SECONDS)
    if track_loss_close_seconds is None:
        track_loss_close_seconds = cfg.get("track_loss_close_seconds", TRACK_LOSS_CLOSE_SECONDS)

    frames = load_records(Path(associations))
    ordered = [frames[f] for f in sorted(frames)]

    known_classes = resolve_known_classes(Path(associations), ordered, violation_of)
    undetected = [name for name in violation_of if name not in known_classes]
    if undetected:
        print(f"warning: not a known class for this detector, skipping absence judgement for: {undetected}")

    windows: dict[tuple[int, str], deque] = {}
    states: dict[tuple[int, str], dict] = {}
    last_seen: dict[int, float] = {}
    recent_closures: dict[tuple[int, str], tuple[dict, float]] = {}
    events: list[dict] = []
    next_event_id = [0]

    def open_or_reuse_event(track_id: int, violation: str, start_ts: float) -> dict:
        key = (track_id, violation)
        recent = recent_closures.get(key)
        if recent is not None and start_ts - recent[1] <= cooldown_seconds:
            event, _ = recent
            event["end"] = None
            event["status"] = "open"
            event["reopens"] += 1
            del recent_closures[key]
            return event
        next_event_id[0] += 1
        event = {
            "event_id": next_event_id[0], "track_id": track_id, "violation": violation,
            "start": round(start_ts, 3), "end": None, "status": "open", "reopens": 0,
        }
        events.append(event)
        return event

    def close_event(key: tuple[int, str], event: dict, end_ts: float, status: str) -> None:
        event["end"] = round(end_ts, 3)
        event["status"] = status
        recent_closures[key] = (event, end_ts)

    def advance(track_id: int, ppe_name: str, violation: str, ts: float, worn: bool) -> None:
        key = (track_id, violation)
        win = windows.setdefault(key, deque())
        win.append((ts, worn))
        while win and ts - win[0][0] > window_seconds:
            win.popleft()
        smoothed_worn = (sum(1 for _, w in win if w) / len(win)) >= present_ratio

        state = states.setdefault(key, {"status": "compliant", "since": ts, "event": None})
        status = state["status"]

        if status == "compliant":
            if not smoothed_worn:
                state["status"], state["since"] = "pending", ts
        elif status == "pending":
            if smoothed_worn:
                state["status"] = "compliant"
            elif ts - state["since"] >= on_delay_seconds:
                event = open_or_reuse_event(track_id, violation, state["since"])
                state["status"], state["since"], state["event"] = "violation", ts, event
        elif status == "violation":
            if smoothed_worn:
                state["status"], state["since"] = "clearing", ts
        elif status == "clearing":
            if not smoothed_worn:
                state["status"], state["since"] = "violation", ts
            elif ts - state["since"] >= off_delay_seconds:
                close_event(key, state["event"], ts, "closed_normal")
                state["status"], state["since"], state["event"] = "compliant", ts, None

    for frame in ordered:
        ts = frame["timestamp"]
        present = set(frame["tracks_present"])

        for track_id in present:
            last_seen[track_id] = ts
            for ppe_name, violation in violation_of.items():
                if ppe_name not in known_classes:
                    continue
                worn = (track_id, ppe_name) in frame["assoc"]
                advance(track_id, ppe_name, violation, ts, worn)

        # track-loss grace: force-close open events for tracks gone too long
        for (track_id, violation), state in states.items():
            if track_id in present or state["status"] == "compliant":
                continue
            if ts - last_seen.get(track_id, ts) >= track_loss_close_seconds:
                if state["event"] is not None:
                    close_event((track_id, violation), state["event"], ts, "closed_track_lost")
                state["status"], state["event"] = "compliant", None

    # end of video: force-close whatever is still open
    end_ts = ordered[-1]["timestamp"] if ordered else 0.0
    for (track_id, violation), state in states.items():
        if state["event"] is not None:
            close_event((track_id, violation), state["event"], end_ts, "closed_end_of_video")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    events_path.write_text("\n".join(json.dumps(e) for e in events) + ("\n" if events else ""), encoding="utf-8")

    by_violation: dict[str, int] = {}
    durations = []
    for e in events:
        by_violation[e["violation"]] = by_violation.get(e["violation"], 0) + 1
        if e["end"] is not None:
            durations.append(e["end"] - e["start"])

    summary = {
        "associations": str(associations),
        "frames": len(ordered),
        "tracks_monitored": len({k[0] for k in states}),
        "events": len(events),
        "by_violation": by_violation,
        "undetected_classes": undetected,
        "avg_duration_seconds": round(sum(durations) / len(durations), 2) if durations else None,
        "params": {
            "window_seconds": window_seconds, "present_ratio": present_ratio,
            "on_delay_seconds": on_delay_seconds, "off_delay_seconds": off_delay_seconds,
            "cooldown_seconds": cooldown_seconds, "track_loss_close_seconds": track_loss_close_seconds,
        },
    }
    summary_path = out_dir / "compliance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{len(ordered)} frames, {summary['tracks_monitored']} tracks monitored, {len(events)} events")
    print(f"by violation: {by_violation}")
    print(f"\nsummary: {summary_path}")
    print(f"events:  {events_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoothing + hysteresis + dedup over associate.py's output")
    p.add_argument("--associations", required=True, help="path to associations.jsonl from associate.py")
    p.add_argument("--vocabulary", default=str(DEFAULT_VOCAB))
    p.add_argument(
        "--window-seconds", type=float, default=None, dest="window_seconds",
        help="overrides data/vocabulary.yaml's compliance.window_seconds; omit to use the vocabulary's value",
    )
    p.add_argument(
        "--present-ratio", type=float, default=None, dest="present_ratio",
        help="overrides data/vocabulary.yaml's compliance.present_ratio; omit to use the vocabulary's value",
    )
    p.add_argument(
        "--on-delay-seconds", type=float, default=None, dest="on_delay_seconds",
        help="overrides data/vocabulary.yaml's compliance.on_delay_seconds; omit to use the vocabulary's value",
    )
    p.add_argument(
        "--off-delay-seconds", type=float, default=None, dest="off_delay_seconds",
        help="overrides data/vocabulary.yaml's compliance.off_delay_seconds; omit to use the vocabulary's value",
    )
    p.add_argument(
        "--cooldown-seconds", type=float, default=None, dest="cooldown_seconds",
        help="overrides data/vocabulary.yaml's compliance.cooldown_seconds; omit to use the vocabulary's value",
    )
    p.add_argument(
        "--track-loss-close-seconds", type=float, default=None, dest="track_loss_close_seconds",
        help="overrides data/vocabulary.yaml's compliance.track_loss_close_seconds; omit to use the vocabulary's value",
    )
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def main() -> None:
    compliance(**vars(parse_args()))


if __name__ == "__main__":
    main()
