# ADR 0002: Build the redesign before the understand phase

Status: accepted, 2026-11-10 (owner). Recorded by `site-baseline`.

## Context

The owner's takeover process orders the work: stop the bleeding, understand (`docs/ARCHITECTURE.md`), tidy, then
build. A repo's phase is the earliest one whose check is not met. After `site-baseline` merges, this repo has CI, a
protected `main` and a project `CLAUDE.md`, so its phase is understand. The approved redesign in
`CAPABILITY-MAP.md` (versions 3 and 4) is build work.

## Decision

The redesign modules may be built while the repo is in understand. The phase line stays
`Takeover phase: understand`, so the gap stays visible, and each redesign PR opens with
`Takeover phase: build (owner-approved exception; repo is in understand)`.

## Limits

- Covers only the modules in `CAPABILITY-MAP.md` versions 3 and 4. Any other feature waits for the normal order.
- Every gate still applies; the exception skips a phase, not a check.
- Deferred, not dropped: `docs/ARCHITECTURE.md` and a profile in the owner's unattended-run file. There are no
  unattended runs on this repo until that profile exists.

## Consequences

- The redesign starts now, on top of CI, instead of after a full architecture write-up of pages it will rewrite.
- Risk: builders work without a written architecture. The module specs list the code facts they rely on, and
  `CLAUDE.md` lists the known traps, to cover the gap until `docs/ARCHITECTURE.md` lands.
