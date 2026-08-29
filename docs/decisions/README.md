# Decisions

An Architecture Decision Record captures one significant, hard-to-reverse choice: the context that forced it, what was decided, and the consequences accepted. One decision per record. Records are append-only, a choice that changes is superseded, not rewritten.

!!! tip "Starting a new ADR"
    Copy `template.md` to `NNNN-short-slug.md` with the next zero-padded number. Then register it in **two** places or it will not appear: the log table below, and `nav:` in `mkdocs.yml`.

## Log

| ADR | Title | Status |
|---|---|---|
| [0001](0001-positive-only-detection.md) | Positive-only detection & compliance logic | Proposed |

## Conventions

- Filename `NNNN-short-slug.md`, zero-padded from `0001`
- One decision per ADR; a single ADR may hold several numbered sub-decisions (`ADD_1`, `ADD_2`), each with its own status
- Supersede rather than rewrite, keep the history readable
- Allowed status values: ==Proposed==, ==Under Review==, ==Accepted==, ==Rejected==, ==Superseded==, ==Deprecated==
- Keep the executive summary current with the decision's real state

## Status values

| Status | Meaning |
|---|---|
| Proposed | Written, not yet reviewed |
| Under Review | Being weighed by the team |
| Accepted | Decided and in force |
| Rejected | Considered and declined, kept for the record |
| Superseded | Replaced by a later ADR (link it) |
| Deprecated | No longer relevant, not replaced |
