# ADR 0001: Positive-Only Detection & Compliance Logic

!!! info
    **Status**: Proposed

    **Created**: 2026-08-29 - **Updated**: 2026-08-29

    **Author**: Project team

    **Supersedes**: -

!!! abstract "Executive Summary"
    The ppe dataset offers both positive PPE classes (`Hardhat`) and negative ones (`NO-Hardhat`). This ADR proposes detecting ==positive PPE only== and deriving absence in logic, ==tracking persons only==, and gating events with ==hysteresis plus per-track deduplication==. 


## Context

The dataset ships `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest` as detection targets, which makes it tempting to have the model output violations directly. But "NO-Hardhat" is the absence of an object, a context-dependent, weakly-defined concept that tends to inflate false positives and couples detection to compliance policy.

Separately, PPE detections flicker frame to frame while workers persist. A naive emit-on-detection design would flood the external system with contradictory alerts. Where identity lives and where the violation judgement is made must be decided up front, because the layers downstream conform to that choice.

## Decision

### ADD_1: Detect Positive PPE Only, Derive Violations in Logic [Proposed]

Train on `person` + positive PPE. Drop the `NO-*` classes. Compute a violation downstream: a worker with no associated hardhat in the head zone is a `no_hardhat`.

**Consequences**

- Pro: Model learns well-defined visual objects; likely higher precision
- Pro: Compliance policy becomes tunable config, not a retrain
- Con: Requires a reliable association step to decide ==missing==

### ADD_2: Track Persons Only [Proposed]

ByteTrack receives person boxes exclusively; PPE is associated per frame, never tracked.

**Consequences**

- Pro: Stable identity anchor; PPE flicker cannot cause id churn
- Con: PPE must be re-associated every frame

### ADD_3: Hysteresis + Per-Track Deduplication [Proposed]

Raise an event only after a violation persists an on-delay, clear only after compliance persists an off-delay, and keep at most one open event per `(track_id, violation_type)`.

**Consequences**

- Pro: Kills boundary flicker and alert floods
- Con: Introduces alert latency equal to the on-delay
- Other: Exact delays need validation on site footage

??? note "Options Considered"
    **Detect NO-* classes directly**: emit a violation when a `NO-Hardhat` box appears.

    - Pro: No association step needed
    - Con: Weak visual concept, high false-positive rate; policy baked into the weights

    **Track PPE items too**: give hardhats their own tracks.

    - Pro: Could smooth PPE without association
    - Con: PPE flicker causes constant id churn; still needs association to a worker

    **Wire NO-* associations into the compliance decision**: since, only positive-PPE absence drives the state machine, so a confirmed `NO-Hardhat` box currently has no effect on when an event opens or closes.

    - Pro: A confirmed `NO-Hardhat` detection is stronger evidence than mere absence, e.g. could shorten the on-delay when both signals agree
    - Con: A real logic change to compliance's state machine, not a config knob; reintroduces the weak, context-dependent NO-* signal ADD_1 already chose to avoid, without measured evidence it helps


## Additional Information

- [PPE vocabulary](../pipeline/vocabulary.md) - the definition ADD_1 would read
- [Compliance state](../pipeline/compliance.md) - where ADD_3 would live
- [Multi-object tracking](../knowledge/multi-object-tracking.md) - the algorithms behind ADD_2
- [Object detection](../knowledge/object-detection.md) - why per-class metrics decide question 1
- Revisit when: first training run is done, or site footage is labelled
