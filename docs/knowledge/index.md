# Knowledge

!!! abstract "Orientation"
    Background reference for the techniques this project builds on. These pages explain ==how the methods work==, independent of what we decided to do with them.

## Scope

The other sections describe ==this project==. This one describes the ==field==. The split is deliberate: a design decision ages with the project, but the definition of IoU does not.

| Section | Answers | Changes when |
|---|---|---|
| **Knowledge** | How does the method work? | Rarely, the theory is stable |
| [Architecture](../architecture.md) / [Pipeline](../pipeline/stages.md) | What did we build, and why? | Every time we build something |
| [Decisions](../decisions/README.md) | What did we choose, and what did we give up? | Append-only, superseded not rewritten |

So a claim like "containment beats IoU for a small object inside a large box" belongs on [association](../pipeline/tracking.md), because it is a choice we made. The definition of IoU it rests on belongs [here](object-detection.md).

## Contents

<div class="grid cards" markdown>

-   :material-vector-square:{ .lg .middle } __Object detection__

    ---

    What the task is, the model families used to solve it, boxes, IoU, NMS, and how mAP is actually computed.

    [:octicons-arrow-right-24: Object detection](object-detection.md)

-   :material-chart-bell-curve:{ .lg .middle } __Precision, recall, F1__

    ---

    What each one actually asks, why F1 is a harmonic mean and not an average, and where they stop being enough.

    [:octicons-arrow-right-24: Precision, recall, F1](precision-recall.md)

-   :material-eye-outline:{ .lg .middle } __YOLO__

    ---

    The single-shot idea, anchor-free heads, the loss terms, and what YOLO26 changed.

    [:octicons-arrow-right-24: YOLO](yolo.md)

-   :material-wrench-outline:{ .lg .middle } __Ultralytics__

    ---

    The framework we actually run: dataset resolution, AutoBatch, AMP, resume.

    [:octicons-arrow-right-24: Ultralytics](ultralytics.md)

-   :material-radar:{ .lg .middle } __Multi-object tracking__

    ---

    Kalman prediction, Hungarian assignment, and what makes ByteTrack different.

    [:octicons-arrow-right-24: Tracking](multi-object-tracking.md)

</div>

## Conventions

- Math renders through MathJax; write LaTeX inline as `$\dots$` and display as `$$\dots$$`
- Notation is shared across pages: $B$ is a box, $\hat{B}$ a prediction, $\mathcal{T}$ a track set
