# Offline dashboard demo

A real, end-to-end example of this pipeline's output: one video in, one `events.jsonl` out. Built to be built against, this is the reference for the dashboard's video-sync and alert-popup integration. See `docs/pipeline/schemas.md` ("Offline upload workflow") for the full write-up this package accompanies.

## What's here

| File | What it is |
|---|---|
| `source_video.mp4` | The original, unprocessed clip (26s, 1080x1920, 30fps) |
| `annotated_video.mp4` | Same clip with tracking + association overlays: `id N` labels per worker, head/torso zone outlines, PPE boxes color-coded (green = confirmed worn, red = confirmed violation, gray = detected but not confidently matched to a worker). Useful for visually understanding what produced the events below, not meant for the dashboard itself. |
| `events.jsonl` | The actual schema output, one JSON object per line. This is what the dashboard consumes. |
| `compliance_summary.json` | Aggregate stats for this run: 36 events (19 `no_hardhat`, 17 `no_vest`), average duration 12.0s, 36 snapshots, and the exact thresholds used (below). |
| `snapshots/` | One evidence JPEG per event, `event_<id>.jpg`, grabbed from `source_video.mp4` at that event's `start` timestamp. Every `events.jsonl` row's `snapshot` field points here with a path relative to this folder. |


## How this was generated

Detector: `yolo26m`, trained on the old 25-class Roboflow set (`ppe_v1_s-2`'s dataset), mAP50-95 0.437 on held-out test images, evaluated 2026-09. **Not yet retrained on the the real 8-class dataset**, that's the next planned step, see the caveats below.

Every other setting used the script defaults: `conf=0.35`, `iou=0.45`, `imgsz=640`, tracker = `pipeline/trackers/bytetrack_ppe.yaml` (`fuse_score=False`, `match_thresh=0.9`, `track_buffer=60`), `min_containment=0.5`. Compliance thresholds, from `compliance_summary.json`'s `params`:

| Threshold | Value |
|---|---|
| Smoothing window | 0.5s |
| Present ratio | 0.5 |
| On-delay (raise) | 3.0s |
| Off-delay (clear) | 2.0s |
| Dedup cooldown | 5.0s |
| Track-loss grace | 3.0s |

## Known rough edges, read before assuming a dashboard bug

- **Tracking fragmentation is real and significant.** This clip has roughly 10-15 real workers; the pipeline reports 83 distinct `track_id` values (`tracks_monitored` in the summary). A single real worker can appear under more than one `track_id`, which means one real violation can occasionally surface as two separate events. This is a known, open issue (occlusion behind steel bars, distant/small workers), not a dashboard-side bug if you see what looks like a duplicate event. Deferred until after the detector retrain.
- **Every compliance threshold above is a reasoned starting value, not one tuned against labelled ground truth.** Event timing (`start`/`end`) should be directionally correct but not trusted to the second.

Full field-by-field schema and more UI patterns (timeline markers, per-worker grouping): `docs/pipeline/schemas.md`.
