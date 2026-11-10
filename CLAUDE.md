# CLAUDE.md

Takeover phase: understand (next when: docs/ARCHITECTURE.md is merged)

Personal site, public repo, served by GitHub Pages from `main` with `.nojekyll`: every file here is a public page,
this one included. A Python build turns Markdown into HTML, and the generated HTML is committed.

## Commands

```
All gates:   sh scripts/gates.sh          # run before every push; about 4 min, mostly the post build
Check tests: (cd scripts && python3 -m unittest test_check)
Site check:  python3 scripts/check.py     # --denylist-only for terms only; --show-terms only when debugging locally
Builds:      python3 -I tools/posts/build.py . && python3 -I tools/leetcode/build.py .
Set secret:  base64 < scripts/denylist.local.txt | gh secret set SITE_DENYLIST   # base64: see gates.yml
Read back:   gh api repos/r03922123/r03922123.github.io/branches/main/protection
Preview:     python3 -m http.server 8000
```

## Where things are decided

- `CONSTRAINTS.md`: the workflow rules and the six gates. Do not loosen it to make a change pass.
- `CAPABILITY-MAP.md`: the index of modules and their build order; each module has a `SPEC-<id>.md`.
- `docs/adr/`: decisions and what they give up. `docs/learning-log.md`: the three whys of each change.

## Traps

- A pushed branch is public before CI runs. CI is the second check, not the first: run `gates.sh` first.
- The `SITE_DENYLIST` Actions secret is a copy of `scripts/denylist.local.txt` (gitignored). Re-run "Set secret"
  whenever the local file changes, or CI checks against an old list.
- Never print a denylist term anywhere: not in a commit, PR, log or chat. `check.py` scrubs its own output, but
  your own commands do not. Use `grep -l` or `grep -c`, never `grep -n`, on text that might hold a term.
- Commit with `-c user.name=r03922123 -c user.email=<the account's GitHub noreply address>` (as on `main`'s
  commits), and check `git log --format='%ae %ce'` before every push: the machine's default email is personal.
- Merge PRs with rebase, not squash, since squash can take the account's default email.
- `check.py` walks the whole folder. Keep virtualenvs and worktrees outside the repo, or their files become findings.
- A new branch made from `origin/main` tracks it; unset the upstream so a bare `git push` cannot target `main`.
- The post build forwards its whole environment to every post script, so no secret may be job-level in CI.
- `--show-terms` refuses to run when `CI` is set, so a gates.sh edit cannot unscrub the public log.
- Detection misses two term shapes the list does not hold today: a phone number written in `+886` form (write it
  in national form) and a CJK term inside CJK text. Fix `term_rx` in `check.py` before adding either.
- `README.md` still describes version 1 (the desk scene); `docs/ARCHITECTURE.md` will replace its layout section.

## Current state (2026-11-10)

- `site-baseline` is built: gates script, CI, scrubbed checker, this file, ADRs 0001 and 0002, the learning log.
- Next, in the build order of `CAPABILITY-MAP.md`: `post-feeds`, `blog-theme`, `post-voice`, each in its own session.
  Reading-page PRs open with `Takeover phase: build (owner-approved exception; repo is in understand)` (ADR 0002).
- Parked branches: none.
