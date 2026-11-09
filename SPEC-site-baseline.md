# Spec: site-baseline

Module of `CAPABILITY-MAP.md` (version 3). Depends on nothing. Status: **approved** 2026-11-10; **amended** the same
day after two pre-merge review rounds, and the amendment is approved.

## Objective

Every change to this public site is checked by a machine before it reaches `main`, and a fresh session can pick the
repo up from files alone. It comes first because the redesign rewrites all 90 reading pages (86 generated, 4 by
hand), and `main` is the live site.

Users: the owner, who approves each merge, and future sessions, which start from the repo with no chat history.

| Requirement | Done means |
|---|---|
| One command runs every gate | `sh scripts/gates.sh` runs all six `CONSTRAINTS.md` gates, locally and in CI, and exits non-zero if any fails |
| CI runs it | Every pull request to `main` gets a `gates` check |
| Private terms stay private | The denylist runs in CI with real terms, and no term can reach a public log by any route |
| `main` is protected | A PR with 0 required approvals and a green `gates` check, for admins too; no force push |
| A fresh session can resume | Project `CLAUDE.md` with the phase line, `docs/learning-log.md`, and two ADRs |

0 approvals is deliberate: GitHub does not let authors approve their own PRs, so requiring 1 would lock out a solo
owner. The owner's merge is the human approval.

## What the code forces

| Finding | Consequence |
|---|---|
| `scripts/check.py:196` catches a missing denylist, but an empty or comments-only one prints `ok` | `main()` exits non-zero when no terms load, so an unset secret fails closed |
| `check.py:141` prints each matched term, and `:148`, `:153`, `:156` and `:190` echo page text; Actions logs on a public repo are public | Every line of output passes through one scrubber. Redaction is the default; `--show-terms` is for local debugging only |
| `tools/leetcode/build.py:177` prints stderr with source lines; the old wording gate's `grep -n` prints whole lines | The denylist runs as the first gate and stops the run, so later steps only see text already known clean. The wording gate uses `grep -l` and `grep -c` |
| `text_files` (`check.py:56`) reads contents but not names and skips binary files; `term_rx` (`check.py:53`) separates words by whitespace only, so `acme corp` misses `acme-corp.md` | The denylist gate also matches every file name, binary files included, after turning `-`, `_`, `/` and `%20` into spaces, since Pages serves those names and the builds print them |
| `check.py:197` tests `#` before stripping, so `  # note` loads as a term | Strip first, so a file of indented comments also counts as empty |
| The Python blocks in posts import numpy, scipy, sklearn, torch, pandas and threadpoolctl, and no file pins them | Add `requirements-ci.txt`, pinned to the versions the posts pass on today |
| PyPI's `torch==2.11.0` on Linux requires the CUDA packages | Pin `torch==2.11.0+cpu` from `https://download.pytorch.org/whl/cpu`, which has a cp312 x86-64 wheel |
| `actions/setup-python` caches on `requirements.txt` by default | Set `cache-dependency-path: requirements-ci.txt` |
| `check.py:58` walks the whole tree | Any virtualenv sits outside the workspace; otherwise site-packages HTML produces false findings |
| The "Post wording" gate allows exactly one hit | Require that the matching file list is that one file, its matching line count is 1, and that line contains `InterviewQuery`. A swapped line then fails too |
| The "Rebuild is clean" gate is a bare `git status --short`; `.codex/` is untracked on the owner's machine | Keep the gate unchanged and add `.codex/` to `.gitignore`. Narrowing a gate to get past a local folder would weaken it |
| `CONSTRAINTS.md:29` escapes `\|` for its table | Spell the pattern without backslashes in `gates.sh`; copied literally it matches nothing |
| `.nojekyll` makes Pages serve every file in the repo | `CLAUDE.md`, the learning log and the ADRs are public pages (see Boundaries) |
| On a public repo a pushed branch is published before CI runs | CI cannot be the first check: run `gates.sh` before every push (see Boundaries) |
| `SPEC.md:72` deploys with `git push origin main` | Protection ends that route; change the line to "merge a PR" |
| Numeric asserts have only ever run on macOS | Unknown until the first Linux run; see Boundaries for a failure here |

Baseline on 2026-11-10 at 7af1270, on macOS: post build 153 s, LeetCode build 2 s, `check.py` ok, 37 check tests pass,
one allowed wording hit, no change to generated pages.

## Tech Stack

- GitHub Actions on `ubuntu-24.04`, Python 3.12, job timeout 30 minutes (the builds take about 3 minutes locally; the
  rest allows for installing torch on a cold cache), one job named `gates`.
- `permissions: contents: read`. Actions pinned to a full commit SHA, as GitHub's hardening guide advises, because a
  moved tag would otherwise run new code with access to the denylist secret.
- The secret is exposed only through step-level `env` on the one step that writes the denylist, never job-level and
  never `${{ }}` inside `run:`, because `build.py:173` forwards the whole environment to every post script.
- `requirements-ci.txt`: numpy 1.26.4, scipy 1.15.3, scikit-learn 1.8.0, torch 2.11.0+cpu, pandas 2.3.3,
  threadpoolctl 3.6.0.
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
scripts/check.py                       → scrubbed output by default, --show-terms, --denylist-only, fails on no terms
scripts/test_check.py                  → gains the scrubbing and empty-denylist cases
requirements-ci.txt                    → pinned packages for the post code
.github/workflows/gates.yml            → writes the denylist, runs scripts/gates.sh
.gitignore                             → gains .codex/
CONSTRAINTS.md                         → gates now point at gates.sh; "there is no CI" removed
SPEC.md                                → the deploy line
```

`CLAUDE.md` traps include: the Actions secret is a copy, so re-run "Set secret" whenever the local denylist changes.

Phase line once this module merges: `Takeover phase: understand (next when: docs/ARCHITECTURE.md is merged)`.
Reading-page PRs then open with `Takeover phase: build (owner-approved exception; repo is in understand)`.

## Code Style

Match the repo: POSIX `sh` and stdlib Python, comments that give the reason.

```sh
# gates.sh: one named step per gate, so a red log says which rule broke. Privacy runs first and stops the run,
# so no later step can echo a private term out of a file.
gate() { name=$1; shift; echo "== $name"; "$@" || { echo "FAILED: $name"; exit 1; }; }
check_tests() { (cd scripts && python3 -m unittest test_check); }
gate "privacy"     python3 scripts/check.py --denylist-only
gate "post build"  python3 -I tools/posts/build.py .
gate "leetcode build" python3 -I tools/leetcode/build.py .
gate "check tests" check_tests
```

## Testing Strategy

- Test first, watching each fail: scrubbed output still fails the run on a denylisted term and omits it, including
  from a broken-link finding, a hyphenated file path and a case variant of the term; `main()` exits non-zero on an
  empty, a comments-only and an indented-comments denylist. Paste the failing output into the PR. These tests are the
  proof of scrubbing, because the Actions runner masks each literal line of a secret, so a clean CI log proves nothing.
- One planted fault per step, each shown red through `gates.sh` with the step's name in the output, then reverted.
  `CONSTRAINTS.md`'s "Privacy and site" gate is two steps here, and Site and Home weight share the `site` step:

| Step | Planted fault | Where |
|---|---|---|
| `privacy` | A harmless canary word in a page, and temporarily in the secret only, so local gates still pass | CI, which proves the real secret reaches the job |
| `site` | A broken link | Local, because `CONSTRAINTS.md` forbids pushing a failing tip |
| `site` | `index.html` padded past 150 KB | Local |
| `post build` | A failing assert in one post's Python block | Local |
| `leetcode build` | A failing assert in one LeetCode post's Python block | Local |
| `check tests` | A failing assertion in `test_check.py` | Local |
| `post wording` | A second matching line inside the one allowed file | Local |
| `rebuild clean` | A committed hand edit to a generated page; the build restores it, so the tree differs from HEAD | Local |

- Stability: three consecutive green runs on the final head, re-runs included, because a flaky required check blocks
  every later merge.
- Protection: read back through the API and paste the JSON into the PR.

## Boundaries

- Always: the same `gates.sh` locally and in CI, so the two cannot drift. Run it before every push, since a pushed
  branch is already public. Keep `CLAUDE.md`, the learning log and the ADRs free of absolute paths, emails and machine
  names, since Pages serves them. One branch per change, each PR about 300 lines.
- Ask first: setting the secret and turning on protection (named approvals at that moment, since both change the
  repo's settings); adding a package; any change to a post's code, other than the planted assert, which is reverted.
- Never: put a denylist term in a log, commit, PR or chat, the planted canary excepted. Use `pull_request_target`, which
  would expose the secret to untrusted code. Loosen an assert or a gate to get green: if a post fails only on Linux,
  stop and report it.
- If a real term is ever pushed: treat it as exposed, tell the owner, and remove the branch with the owner's OK before
  anything else.

Out of scope: `docs/ARCHITECTURE.md` and an unattended-run profile (ADR-0002 defers them), Lighthouse, deploying from
CI, all visual work.

## Success Criteria

- [ ] `sh scripts/gates.sh` exits 0 on the branch tip locally
- [ ] The scrubbing and empty-denylist tests failed before the change and pass after it, with the failing output in the PR
- [ ] Every row of the planted-fault table was shown red, each named by its step, and reverted; the CI row links its run
- [ ] An empty `SITE_DENYLIST` makes the CI run fail
- [ ] The CI run with the canary fails at the `privacy` step and runs no later step
- [ ] Three consecutive green `gates` runs on the final head
- [ ] The API read-back shows a required PR with 0 approvals, required check `gates`, admins enforced, force pushes off
- [ ] `CLAUDE.md` has the phase line, and both ADRs and the first learning-log entry are merged

## Open Questions

None. The two approvals under "Ask first" are requested when each step is reached.
