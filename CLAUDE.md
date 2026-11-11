# CLAUDE.md

Takeover phase: understand (next when: docs/ARCHITECTURE.md is merged)

Personal site, public repo, served by GitHub Pages from `main` with `.nojekyll`: every file here is a public page,
this one included. A Python build turns Markdown into HTML, and the generated HTML is committed.

## Commands

```
All gates:   sh scripts/gates.sh          # run before every push; about 4 min, mostly the post build
Check tests: (cd scripts && python3 -m unittest test_check)
Search tests: node --test scripts/test_search.mjs
Site check:  python3 scripts/check.py     # --denylist-only for terms only; --show-terms only when debugging locally
Builds:      python3 -I tools/posts/build.py . && python3 -I tools/leetcode/build.py . && python3 -I tools/search/build.py .
Lock CI:     uv pip compile requirements-ci.in --generate-hashes --python-version 3.12 \
               --python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match \
               --emit-index-url -o requirements-ci.txt   # after any change to requirements-ci.in
Set secret:  base64 < scripts/denylist.local.txt | gh secret set SITE_DENYLIST   # base64: see gates.yml
Read back:   gh api repos/r03922123/r03922123.github.io/branches/main/protection
Preview:     python3 -m http.server 8000
```

## Where things are decided

- `CONSTRAINTS.md`: the workflow rules and the gates. Do not loosen it to make a change pass.
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
  in national form) and a CJK term inside CJK text, which the scrubber does not hide either. Fix `term_rx` and
  `scrubber` in `check.py` before adding either.
- `README.md` still describes version 1 (the desk scene); `docs/ARCHITECTURE.md` will replace its layout section.

## Current state (2026-11-11)

- Merged: `site-baseline` (PR #3: gates script, CI, scrubbed checker, this file, ADRs 0001 and 0002, the learning
  log; `main` is protected: a PR with a green `gates` check, for admins too), CI hardening (hash-locked wheels-only
  install, base64 denylist secret), `post-feeds` (feed files, `index.md` copies, page head tags), `blog-theme`
  (reading tokens at `:root`, the room's under `body.home`), `reading-shell` (fragment check, then the layout)
  and an expert accuracy pass over all 90 posts (PR #9, which lists the doubts left unedited).
- `posts-index` (merged): both list pages are topic cards, a count and `section.group`s in `.groups`, built by
  `topic_list` in `tools/leetcode/build.py`; the `#fragment` is the filter, with no JavaScript. `check.py` matches
  each card's count to its section's rows.
- Reading pages: the bar sits outside `.wrap`; the home keeps its own header inside it. After a layout change, run
  `python3 tools/dev/same_articles.py origin/main` to show no article text moved.
- `site-search` (merged): `search.mjs` inserts a box on the two list pages and ranks with `search-core.mjs`
  over `search-index.json`, written by `tools/search/build.py` after the post builds (gates.sh runs both, and
  `node --test scripts/test_search.mjs`). `check.py` fails on a listed post missing from the index.
- `post-voice` (merged, 8 PRs from #14): all 59 posts open on a situation, stake or result, say "I" only for what
  their code or derivation does, and name their limits; topic posts also end on what is open. The voice check's
  `--final` passed before PR 8 removed the check, its baseline and its gate. The sentences PRs #16 to
  #21 listed as overstating their code were fixed, prose only, in `post-claims` (PRs #22 to #24). Edited sources
  need `tools/posts/build.py . <slug>`, then the feeds and search builds.
- Waiting for the owner's OK, since each is a code change: `transformers-and-llms-key-ideas`'s comment "the bug
  ~0.15" (runs to 0.118), and the LeetCode `hash-table` note's unseeded assert (fails about 2 runs in 300).
- Next: every module in `CAPABILITY-MAP.md` is merged, so the understand phase's check, `docs/ARCHITECTURE.md`, comes
  next.
- Parked branches: none.
