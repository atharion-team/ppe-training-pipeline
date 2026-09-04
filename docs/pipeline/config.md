# Configuration

!!! abstract "Orientation"
    The pipeline's runtime config surface: what lives in a file, what lives in code, and why.

## Principle

Config lives in files, not in code, so a tuning change is a diff and the same source runs on every site and GPU. Two people running parallel experiments need this.

## What Exists

| File | Owns | Read by |
|---|---|---|
| `data/css-data.yaml` | Dataset paths, Roboflow version, the 25 class names, currently out of sync with the team's 8-class vocabulary (see [vocabulary](vocabulary.md)) | `train.py`, `evaluate.py`, `download_dataset.py` |
| `data/vocabulary.yaml` | The PPE definition, and the config for three runtime layers: association's zone geometry and minimum containment score, and every compliance timing threshold. See [vocabulary](vocabulary.md#schema) for the full shape | `detect.py`, `associate.py`, `compliance.py` |
| `pipeline/trackers/bytetrack_ppe.yaml` | ByteTrack thresholds, tuned for this project | `track.py`, `associate.py`, via `model.track` ([tracking](tracking.md)) |
| `.env` | `ROBOFLOW_API_KEY` only, gitignored | `download_dataset.py:49` |

Training hyperparameters are deliberately ==not== in a config file. They live in `pipeline/train.py` so that a run started from the CLI and one started from the notebook are the same run, with the Makefile exposing the few that change per machine:

```
make train BATCH=16 WORKERS=4 NAME=ppe_v2
```


## Experiment Tracking

Optional, off by default. `train.py:33` calls `init_wandb()` before every run, which no-ops unless both are true:

1. `wandb` is installed (`pip install wandb`, already in `requirements.txt`)
2. The integration is enabled, a one-time per-machine setting, not a repo file:
   ```powershell
   python -c "from ultralytics import settings; settings.update({'wandb': True})"
   ```

Every run then logs to the `ppe-training-pipeline` W&B project under the runner's account, named to match the local run directory (`ppe_v1` locally is `ppe_v1` on W&B too).



## Results and Output

Every script writes under `runs/<stage>/`. `train` and `evaluate` follow Ultralytics' `--project`/`--name` convention: each run gets its own numbered subdirectory, never overwritten unless `--exist-ok` is passed. `detect`, `track`, `associate`, and `compliance` take a flat `--output` instead, no separate `--name`, so organizing multiple runs (`runs/associate/demo`, as in [the offline demo](schemas.md#reproducing-event-emission)) means pointing `--output` at a subdirectory yourself.

=== "train"

    ```yaml
    runs/train/<name>/:
      args.yaml:            # every resolved argument
      results.csv:          # one row per epoch
      weights/:
        best.pt:             # highest fitness so far
        last.pt:             # most recent epoch, the resume point
        epoch{N}.pt:         # every save_period epochs
      "*.png / *.jpg":       # curves, confusion matrix, batch previews
    ```

=== "val"

    ```yaml
    runs/val/<name>/:
      evaluation_summary.json:  # overall + per-class metrics, pipeline/evaluate.py's own output
      confusion_matrix.png:
      "*_curve.png":             # PR, F1, precision, recall curves
      "val_batch*.jpg":          # label vs prediction previews
    ```

    Everything but `evaluation_summary.json` is Ultralytics' standard `model.val(plots=True)` output, written to the same directory.

=== "detect"

    ```yaml
    runs/inference/:
      "<stem>_detected.jpg":   # or .mp4 for a video/webcam source
      inference_summary.json:
    ```

    `pipeline/detect.py`'s output: per-frame boxes, no identity, no dedup, the baseline everything downstream is measured against.

=== "track"

    ```yaml
    runs/track/:
      "<stem>_tracked.mp4":
      tracks.jsonl:              # one line per tracked box per frame
      track_summary.json:
    ```

=== "associate"

    ```yaml
    runs/associate/:
      "<stem>_associated.mp4":
      associations.jsonl:        # one line per associated PPE box
      associate_summary.json:
    ```

=== "compliance"

    ```yaml
    runs/compliance/:
      events.jsonl:               # the schema this project ships; see event schema
      compliance_summary.json:
    ```

## Related

- [Detector](detector.md)
- [Tracking & association](tracking.md)
- [Compliance state](compliance.md)
- [Event schema](schemas.md)
