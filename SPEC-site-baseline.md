# Spec: site-baseline

Module of `CAPABILITY-MAP.md` (version 3). Depends on nothing. Status: **approved** 2026-11-10.

## Objective

Every change to this public site is checked by a machine before it reaches `main`, and a fresh session can pick the
repo up from files alone. It comes first because the redesign regenerates all 90 reading pages, and `main` is the
live site.

Users: the owner, who approves each merge, and future sessions, which start from the repo with no chat history.

| Requirement | Done means |
|---|---|
| One command runs every gate | `sh scripts/gates.sh` runs all six `CONSTRAINTS.md` gates, locally and in CI, and exits non-zero if any fails |
| CI runs it | Every pull request to `main` gets a `gates` check |
| Private terms stay private | The denylist still runs in CI, and no term ever appears in a public log |
| `main` is protected | A PR and a green `gates` check are required, for admins too; no force push |
| A fresh session can resume | Project `CLAUDE.md` with the phase line, `docs/learning-log.md`, and two ADRs |

## What the code forces

| Finding | Consequence |
|---|---|
| `scripts/check.py:196` exits when the gitignored denylist is missing | CI writes it from an Actions secret, and fails closed if the secret is absent |
| `scripts/check.py:141` prints each matched term; Actions logs on a public repo are public | Add `--redact`: CI reports `private term #3`, the local run keeps the term |
| The Python blocks in posts import numpy, scipy, sklearn, torch, pandas and threadpoolctl, and no file pins them | Add `requirements-ci.txt`, pinned to the versions the posts pass on today |
| The "Rebuild is clean" gate is a bare `git status`; `.codex/` is untracked on the owner's machine | Scope it to generated paths, so a local tool folder cannot fail it |
| The "Post wording" gate allows exactly one hit | `gates.sh` compares the hits with that one allowed file, so a second hit fails |
| `.nojekyll` makes Pages serve every file in the repo | `CLAUDE.md`, the learning log and the ADRs are public pages (see Boundaries) |
| `SPEC.md:72` deploys with `git push origin main` | Protection ends that route; change the line to "merge a PR" |
| Numeric asserts have only ever run on macOS | Unknown until the first Linux run; see Boundaries for a failure here |

Baseline on 2026-11-10 at 7af1270, on macOS: post build 153 s, LeetCode build 2 s, `check.py` ok, 37 check tests pass,
one allowed wording hit, no change to generated pages.

## Tech Stack

- GitHub Actions on `ubuntu-24.04`, Python 3.12, job timeout 30 minutes (the builds take about 3 minutes locally; the
  rest allows for installing torch on a cold cache), one job named `gates`.
- `permissions: contents: read`. Actions pinned to a full commit SHA, as GitHub's hardening guide advises, because a
  moved tag would otherwise run new code with access to the denylist secret.
- `requirements-ci.txt`: numpy 1.26.4, scipy 1.15.3, scikit-learn 1.8.0, torch 2.11.0 (CPU wheel), pandas 2.3.3,
  threadpoolctl 3.6.0, with pip's cache on.
- Pages keeps deploying from `main` as it does now. CI checks; it does not deploy.

## Commands

```
All gates:   sh scripts/gates.sh
Check tests: (cd scripts && python3 -m unittest test_check)
Set secret:  gh secret set SITE_DENYLIST < scripts/denylist.local.txt
Read back:   gh api repos/r03922123/r03922123.github.io/branches/main/protection
Preview:     python3 -m http.server 8000
```

## Project Structure

```
CLAUDE.md                              → commands, traps, current state, the takeover phase line (under ~200 lines)
docs/learning-log.md                   → the three whys of each change, newest first
docs/adr/0001-stay-on-github-pages.md  → hosting decision and what it gives up
docs/adr/0002-redesign-before-understand.md → the phase exception and its limits
scripts/gates.sh                       → the six gates, the only place they are spelled out
scripts/check.py                       → gains --redact
scripts/test_check.py                  → gains the redaction case
requirements-ci.txt                    → pinned packages for the post code
.github/workflows/gates.yml            → runs scripts/gates.sh
CONSTRAINTS.md                         → gates now point at gates.sh; "there is no CI" removed
```

Phase line once this module merges: `Takeover phase: understand (next when: docs/ARCHITECTURE.md is merged)`.
Reading-page PRs then open with `Takeover phase: build (owner-approved exception; repo is in understand)`.

## Code Style

Match the repo: POSIX `sh` and stdlib Python, comments that give the reason.

```sh
# gates.sh: each gate stops the run with its own name, so a red CI log says which rule broke.
gate() { name=$1; shift; echo "== $name"; "$@" || { echo "FAILED: $name"; exit 1; }; }
gate "builds"  python3 -I tools/posts/build.py .
gate "builds"  python3 -I tools/leetcode/build.py .
```

## Testing Strategy

- Test first: a `test_check.py` case proves `--redact` still fails the run on a denylisted term and that the term
  is absent from the output. Watch it fail, then add the flag.
- Each gate shown red once: on the PR, a throwaway commit adds a broken link, the `gates` run fails on that gate's
  name, and the commit is reverted. Link both runs in the PR.
- Stability: three consecutive green runs on the final head, re-runs included, because a flaky required check blocks
  every later merge.
- Protection: read back through the API and paste the JSON into the PR.

## Boundaries

- Always: the same `gates.sh` locally and in CI, so the two cannot drift. Keep `CLAUDE.md`, the learning log and the
  ADRs free of absolute paths, emails and machine names, since Pages serves them. One branch per change, each PR
  about 300 lines.
- Ask first: setting the secret and turning on protection (named approvals at that moment, since both change the
  repo's settings); adding a package; any change to a post's code.
- Never: put a denylist term in a log, commit, PR or chat. Use `pull_request_target`, which would expose the secret
  to untrusted code. Loosen an assert or a gate to get green: if a post fails only on Linux, stop and report it.

Out of scope: `docs/ARCHITECTURE.md` and an unattended-run profile (ADR-0002 defers them), Lighthouse, deploying from
CI, all visual work.

## Success Criteria

- [ ] `sh scripts/gates.sh` exits 0 on the branch tip locally
- [ ] The redaction test failed before `--redact` and passes after it
- [ ] A PR shows a red `gates` run for a broken link and a green run after the revert
- [ ] Three consecutive green `gates` runs on the final head
- [ ] CI logs contain no denylist term (checked locally against the downloaded log, reporting only a count)
- [ ] The API read-back shows a required PR, required check `gates`, admins enforced, force pushes off
- [ ] `CLAUDE.md` has the phase line, and both ADRs and the first learning-log entry are merged

## Open Questions

None. The two approvals under "Ask first" are requested when each step is reached.
