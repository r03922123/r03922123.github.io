# CLAUDE.md

Personal site, public repo, served by GitHub Pages from `main` with `.nojekyll`: every file here is a public page,
this one included. A Python build turns Markdown into HTML, and the generated HTML is committed.

## Commands

```
All gates:   sh scripts/gates.sh          # run before every push; about 4 min, mostly the post build
Check tests: (cd scripts && python3 -m unittest test_check)
Search tests: node --test scripts/test_search.mjs
Site check:  python3 scripts/check.py     # --denylist-only for terms only; --show-terms only when debugging locally
Builds:      python3 -I tools/posts/build.py . && python3 -I tools/leetcode/build.py . && python3 -I tools/book/build.py . \
               && python3 -I tools/feeds/build.py . && python3 -I tools/search/build.py .   # gates.sh order; skip one and rebuild clean fails
One post:    python3 -I tools/posts/build.py . <slug>   # after editing its source; then the feeds and search builds
One check:   (cd scripts && python3 -m unittest test_check.CheckTest.test_phone_numbers)   # any Class.test_name
One search:  node --test --test-name-pattern='<test name>' scripts/test_search.mjs
Screenshot:  sh tools/dev/shoot.sh <name> light|dark desktop|phone <site path>   # needs Preview on 8765; prints the PNG
Merge:       gh pr merge <n> --rebase --match-head-commit <sha>
Identity:    git log --format='%ae%n%ce' origin/main..HEAD | grep -c -v noreply   # before every push: must print 0
Layout:      python3 tools/dev/same_articles.py origin/main   # after a layout change: shows no article text moved
Lock CI:     uv pip compile requirements-ci.in --generate-hashes --python-version 3.12 \
               --python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match \
               --emit-index-url -o requirements-ci.txt   # after any change to requirements-ci.in
Set secret:  base64 < scripts/denylist.local.txt | gh secret set SITE_DENYLIST   # base64: see gates.yml
Read back:   gh api repos/r03922123/r03922123.github.io/branches/main/protection
Preview:     python3 -m http.server 8765
```

## Where things are decided

- `docs/ARCHITECTURE.md`: where each page, build and gate lives, and how a change reaches the live site.
- `CONSTRAINTS.md`: the workflow rules and the gates. Do not loosen it to make a change pass.
- `CAPABILITY-MAP.md`: the index of modules and their build order; each module has a `SPEC-<id>.md`.
- `docs/adr/`: decisions and what they give up.

## Traps

- A pushed branch is public before CI runs. CI is the second check, not the first: run `gates.sh` first.
- The `SITE_DENYLIST` Actions secret is a copy of `scripts/denylist.local.txt` (gitignored). Re-run "Set secret"
  whenever the local file changes, or CI checks against an old list.
- Never print a denylist term anywhere: not in a commit, PR, log or chat. `check.py` scrubs its own output, but
  your own commands do not. Use `grep -l` or `grep -c`, never `grep -n`, on text that might hold a term.
- Commit with `-c user.name=r03922123 -c user.email=<the account's GitHub noreply address>` (as on `main`'s
  commits), and run Identity before every push: the machine's default email is personal. Never print the addresses.
- `main` is protected, admins included: a PR with a green `gates` check. Merge with rebase, not squash, since squash
  can take the account's default email.
- `gates.sh` ends with `rebuild clean`, which fails on any uncommitted or untracked file, your own edits included:
  commit first.
- A new worktree has no `scripts/denylist.local.txt` (gitignored), so `check.py` refuses to run: copy it in from the
  main checkout.
- Post code runs on macOS here and Linux in CI, and float results differ: write asserts as ranges that hold on both.
  If an existing assert fails only on Linux, stop and report it.
- `check.py` walks the whole folder. Keep virtualenvs and worktrees outside the repo, or their files become findings.
- A new branch made from `origin/main` tracks it; unset the upstream so a bare `git push` cannot target `main`.
- The post build forwards its whole environment to every post script, so no secret may be job-level in CI.
- `--show-terms` refuses to run when `CI` is set, so a gates.sh edit cannot unscrub the public log.
- Detection misses two term shapes the list does not hold today: a phone number written in `+886` form (write it
  in national form) and a CJK term inside CJK text, which the scrubber does not hide either. Fix `term_rx` and
  `scrubber` in `check.py` before adding either.

## Status and decisions in force

- History is the git log. Known gaps, and the next thing to pick, are the Follow-ups list in `docs/ARCHITECTURE.md`;
  PR #9 lists the post-accuracy doubts left unedited.
- Every post follows `SPEC-post-voice.md`: open on a situation, stake or result; say "I" only for what its code or
  derivation does; name its limits.
- The résumé book is unlisted and `noindex` (ADR 0003): `check.py` fails if a feed or index names it, or any page
  but the home's one door links it.
- The post wording gate fails only on obsidian or vault (PR #37). Interview and MusicGen wording in posts is a
  review question, not a gate.
