# Precision, recall, and F1

!!! abstract "Orientation"
    What precision and recall each actually ask, in plain terms first, then the formulas, then a worked example built from this project's own numbers. This page exists because the two are easy to mix up, and the fix is a rule, not repetition.

## The Intuition

Two different questions, about two different ways a detector fails.

- **Precision** asks: of everything I called a hardhat, how much was actually a hardhat? Low precision means false alarms, the detector cries wolf.
- **Recall** asks: of every hardhat actually in the frame, how many did I find? Low recall means misses, the detector stays silent on something real.

The same shape as a smoke detector. Precision is "when it goes off, is there really a fire". Recall is "if there's a fire, does it go off". A detector can fail either way independently, and they are not the same failure.

| | Detector says hardhat | Detector says nothing |
|---|---|---|
| **Actually a hardhat** | True Positive (TP) | False Negative (FN) — a miss |
| **Not actually a hardhat** | False Positive (FP) — a false alarm | True Negative (TN) |

!!! info "TN Barely Matters Here"
    Detection has no well-defined count of "places where correctly nothing was found", an image doesn't have a fixed number of non-hardhat locations to count. Precision and recall are built only from TP, FP, and FN. TN drops out of both formulas and is not used again on this page.

## The Formulas

$$P = \frac{TP}{TP+FP} \qquad R = \frac{TP}{TP+FN}$$

Both share the numerator, TP, correct positive calls. The denominator is what differs, and that difference is the whole trick to keeping them straight: precision divides by what you claimed, recall divides by what was actually there.

!!! tip "The Rule That Never Fails"
    Precision's denominator is your own output ($TP + FP$). Recall's denominator is the ground truth ($TP + FN$). If you remember nothing else about this page, remember that one line.

How a box qualifies as a TP in the first place, matched to ground truth by class and by an IoU threshold, is defined on [object detection](object-detection.md#true-positive-criteria); this page assumes that definition and starts from the counts it produces.

## Why They Trade Off

Lowering the confidence threshold makes the detector fire more often: recall can only rise or hold (more of the real objects get caught somewhere in the extra output), while precision usually falls (the extra output brings false alarms with it). Raising the threshold does the reverse. Neither number alone says whether a detector is good, a detector that flags everything scores recall $= 1$ and precision near $0$, and is useless.

!!! warning "Both Are Threshold-Dependent"
    Quoting a precision or a recall without the confidence threshold it was measured at says almost nothing, the same detector can report very different numbers at different thresholds. [mAP](object-detection.md#average-precision) exists specifically to remove this dependence, by sweeping the threshold instead of fixing it.

## F1 Score

$$F_1 = \frac{2PR}{P+R}$$

This is the harmonic mean of $P$ and $R$[^rijsbergen], not the arithmetic mean, and the difference is the point. The harmonic mean is dragged down hard by whichever of the two is smaller:

| P | R | Arithmetic mean | F1 (harmonic mean) |
|---|---|---|---|
| 1.00 | 0.00 | 0.50 | 0.00 |
| 0.90 | 0.10 | 0.50 | 0.18 |
| 0.80 | 0.80 | 0.80 | 0.80 |

A detector cannot buy a good F1 by being excellent at one and useless at the other. That is deliberate: a detector nobody can trust, or a detector that misses most real objects, is not "half good", it isn't useful, and F1 is built to say so.

## A Worked Example

Suppose a hardhat detector is run over a set of test images. Of every box it labels "hardhat", just over half turn out to be correct: precision $\approx 0.53$. Of every hardhat actually present in those images, it finds a little under half: recall $\approx 0.42$.

$$P = 0.53, \qquad R = 0.42$$

$$F_1 = \frac{2 \times 0.53 \times 0.42}{0.53 + 0.42} = \frac{0.445}{0.95} \approx 0.47$$

Notice $F_1$ sits close to the plain average of $0.53$ and $0.42$ (which is $0.475$), the two inputs here are not far apart, so the harmonic and arithmetic means nearly agree. Re-run the same arithmetic on the $1.0 / 0.0$ row of the earlier table to see the gap open up when the two disagree sharply, that gap is what F1 is for.

!!! tip "Real Numbers, Not a Hypothetical"
    This project's own measured precision, recall, and per-class breakdown, computed on held-out test images rather than the illustrative figures above, are on [detector](../pipeline/detector.md).

## Related

- [Object detection](object-detection.md) - how a box qualifies as TP, FP, or FN, and how mAP builds on P/R across thresholds
- [Compliance state](../pipeline/compliance.md) - why a good frame-level F1 still doesn't guarantee a usable event stream

## References

[^rijsbergen]: van Rijsbergen, C. J. (1979). *Information Retrieval* (2nd ed.). Butterworths. Origin of the F-measure as a weighted harmonic mean of precision and recall.
