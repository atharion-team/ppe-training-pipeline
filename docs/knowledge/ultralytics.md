# Ultralytics

!!! abstract "Orientation"
    How the Ultralytics framework behaves in general: the object model, how it resolves dataset paths, how a run directory is laid out, what AutoBatch and mixed precision actually do, why dataloader workers matter, and what resume does and does not restore. Not an API reference, that is [upstream](https://docs.ultralytics.com). This page covers the behaviours that are easy to get surprised by.

!!! info "Checked Against the Installed Version"
    Read from `ultralytics 8.4.133`. Source-file citations below are to the installed package itself, not to this project's own code. Line references drift, re-check when the pin moves. Where this project's own measured numbers and gotchas live, on the hardware actually used, is [Configuration](../pipeline/config.md).

## The Object Model

Everything routes through one class. `YOLO(weights)` loads a checkpoint and exposes four verbs:

| Call | Does |
|---|---|
| `.train(...)` | Finetune, write a run directory |
| `.val(...)` | Evaluate on a split, return metrics |
| `.predict(...)` | Per-frame inference |
| `.track(...)` | Predict plus identity across frames |

The argument surface is one flat namespace shared across all four calls, so code that forwards a bundle of keyword arguments to any of them does not need to enumerate every argument by name.

## Dataset Resolution

The single most confusing part of the framework, and worth understanding before it bites.

A dataset config declares a ==relative== root:

```yaml
path: some/dataset
train: train/images
val: valid/images
```

Relative to what? Not the config file, and not the working directory. Ultralytics resolves it against a global `datasets_dir` setting, stored in a settings file in the user's profile, outside any repository entirely[^ultralytics-settings].

!!! warning "Machine State, Not Repo State"
    `datasets_dir` lives on the machine, not in the repo, so it is not captured by a clone, a commit, or a Docker build. A fresh machine, or one that last worked on a different project, resolves a relative dataset path against whatever that setting happens to hold, and training either fails to find the data or silently trains on the wrong data.

    How this project pins it before every run is on [Configuration](../pipeline/config.md).

## Run Directories

Every `train` or `val` call writes to `project/name`, and by default ==never overwrites==: a colliding name gets a numeric suffix instead. `exist_ok=True` opts into overwriting.

The standard layout:

```
runs/train/<name>/
    args.yaml          every resolved argument
    results.csv        one row per epoch
    weights/
        best.pt        highest fitness so far
        last.pt        most recent epoch, the resume point
        epoch{N}.pt     every save_period epochs
    *.png / *.jpg       curves, confusion matrix, batch previews
```

!!! tip "args.yaml Records the Request, Not the Result"
    If an argument is auto-resolved at runtime rather than fixed up front, batch size under AutoBatch is the clearest example below, `args.yaml` still shows the value that was requested, not the one the framework actually chose. The resolved value lives inside the checkpoint's own `train_args` instead. See [Configuration](../pipeline/config.md) for a worked example of the gap between the two.

### Fitness, Not mAP

`best.pt` is selected by a weighted ==fitness== scalar, by default dominated by mAP50-95 with a small mAP50 contribution, not by mAP50 alone. A run whose mAP50 peaks at one epoch and mAP50-95 at another keeps the weights from the latter.

## AutoBatch

Passing `batch=-1` profiles the model at several batch sizes and extrapolates to a memory target. From the installed source (`utils/autobatch.py:46`):

```python
fraction=batch if 0.0 < batch < 1.0 else 0.6
```

So `-1` targets ==60% of GPU memory==, and passing a fraction directly sets that target explicitly: `batch=0.85` aims for 85%.

!!! warning "The Estimate Tends to Run Conservative"
    The profiling pass runs before the dataloader is warm, and it adds safety margin sized for the largest-object case it samples, so the batch size AutoBatch settles on is often smaller than what the card can actually sustain once training is under way. The practical guidance: use `batch=-1` for a safe starting point on unfamiliar hardware, but ==verify by hand== once it matters, either against a specific measured number or by watching GPU memory through a few epochs. What this looked like on this project's own hardware is on [Configuration](../pipeline/config.md).

## Mixed Precision

`amp=True` is the default and runs most convolutions in FP16, halving activation memory. The speed benefit specifically comes from ==tensor cores==, hardware present from the Volta/Turing generation onward (compute capability 7.0+); earlier GPU generations (Pascal and before, compute capability 6.x and below) get the memory saving with close to no speedup, since there is no faster FP16 path on that hardware to use.

Ultralytics also runs an AMP capability check at startup that downloads a small checkpoint. On a machine with no network access this stalls training before epoch 1, which presents as a hang rather than a clear network error.

## Dataloader Workers

`workers` sets the number of subprocesses feeding the GPU, and it is the parameter most likely to be the real bottleneck: too few, and JPEG decode plus augmentation run serially in the training process while the GPU sits idle between batches.

!!! danger "Windows Plus a Notebook Needs workers=0"
    Windows has no `fork`, so each worker subprocess re-imports the parent module from scratch. Inside a Jupyter kernel, which has no guarded `if __name__ == "__main__":` entry point the way a plain script does, that reliably hangs. This is a Windows and multiprocessing fact, not an Ultralytics-specific one, but it bites hardest here because the library's default assumes several workers.

    Running from a terminal script does not have this restriction. The workaround has a real cost: with `workers=0`, the GPU idles between batches waiting on serial decode and augmentation. What this measured as on this project's own hardware is on [Configuration](../pipeline/config.md).

## Resume

`train(resume=True)` on a `last.pt` restores optimizer state, EMA weights, the learning-rate schedule position, and the epoch counter, then continues into the ==same run directory==.

!!! warning "Resume Ignores New Arguments"
    Arguments are reread from inside the checkpoint. A resumed run cannot change `batch`, `imgsz`, or `workers`. This is deliberate: changing batch size invalidates the optimizer state and LR schedule that resume exists to preserve. Changing those settings requires a fresh run.

`patience` also counts from the ==best== epoch, not from the resume point, so a resumed run can early-stop sooner than the remaining epoch count on its own would suggest.

## Augmentation

Ultralytics augments online, per batch, during training: HSV jitter, horizontal flip, and mosaic are common defaults. Mosaic stitches four images into one, which manufactures scale variety and partial occlusion for free, and produces the odd-looking grids sometimes seen in a training-batch preview image, which are correct output, not corrupted data.

!!! warning "Do Not Stack Online and Offline Augmentation"
    Some dataset export tools, Roboflow among them, offer to augment a dataset offline at export time. Applying a full online augmentation pipeline on top of an already-augmented export compounds the distortion rather than adding independent variety. Worth confirming whether a dataset was pre-augmented before tuning online augmentation values. Whether this project's own dataset export was, and how the training script handles it, is on [Configuration](../pipeline/config.md).

## Related

- [YOLO](yolo.md) - the architecture being trained here
- [Detector](../pipeline/detector.md) - our model and hyperparameter choices
- [Configuration](../pipeline/config.md) - this project's own measured numbers, gotchas, and the knobs it exposes

## References

[^ultralytics-settings]: Ultralytics. Settings and configuration documentation. <https://docs.ultralytics.com/>
