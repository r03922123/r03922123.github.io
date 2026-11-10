# Learning log

The three most important whys of each change, newest first. Each heading reads `<date> · <phase>: <title>`.

## 2026-11-11 · understand: CI hardening from the site-baseline review

1. **Give a pattern one way to match.** The scrubber read `< >` as one tag or as three characters, so each such gap
   doubled the work, like a maze where every fork leads to the same room. A `<...>` now counts as a tag only if
   it holds a letter or digit that no other branch can read, so `< >`, `<%20>` and `<&nbsp;>` are read piece by
   piece, and each gap has one reading. Trade-off: a harder pattern, split into named parts. Risk: three quicker
   fixes each fell short. A possessive repeat missed `%20`, "a tag starts with a letter" missed `<1>` (a punctuated
   term printed in clear text), and "a tag holds a letter" left `<&nbsp;>` slow. Tests and two fresh reviews
   caught them. The final rule accepts the same gaps as the old one on 111,150 test strings.
2. **Measure the slow case in a test.** The new test times crafted lines: seconds to minutes before (90 s for 24
   `<%20>` gaps on one machine), milliseconds after.
   Trade-off: a timing test can flake on a slow machine, so it allows 1 s for a case that takes milliseconds.
   Risk: the bound is loose enough that a smaller slowdown would pass.
3. **Cancel stale PR runs, never `main`'s.** A newer push to a PR makes the older run pointless. Each commit on
   `main` is live, though, so each one gets its own concurrency group and finishes its own check. One shared
   group was not enough: GitHub keeps one waiting run per group and cancels it when a third push arrives.
   Trade-off: quick merges run in parallel and cost more minutes. Risk: none beyond those minutes.

## 2026-11-10 · stop the bleeding: site-baseline (gates script, CI, protected main)

1. **Fail closed.** A check that cannot see its input should say no, not yes, like a smoke alarm that beeps when its
   battery is gone. `check.py` now exits non-zero when the denylist has no terms, so an unset CI secret turns the
   run red. Trade-off: a PR from a fork, which gets no secrets, always fails. Risk: someone "fixes" that by
   allowing an empty list, which would make the privacy gate pass on nothing.
2. **Scrub at the one exit, not at each print.** Every line `check.py` prints, tracebacks included, goes through
   one scrubber, like one security gate at an airport instead of a guard at every shop. A term shows as
   `[term #3]`. Trade-off: local debugging needs `--show-terms`. Risk: commands outside `check.py` are not
   scrubbed, so privacy runs first and stops the run, and the wording gate prints names and counts, never lines.
   Hiding too much leaks as well: a label inside `planning` would mark `ann` as private, so the scrubber hides only
   whole words, with the same word gaps that detection uses. Making those gaps match found a term already on `main`.
3. **The same script locally and in CI.** `scripts/gates.sh` is the only place the gates are written, like one
   recipe card used at home and in the restaurant. Trade-off: CI must install the post code's exact packages
   (`requirements-ci.txt`, every package locked by hash, because post code runs while the denylist is on disk). Risk: numbers can differ on Linux; if a post fails only there, stop and report it
   rather than loosen an assert.

## 2026-11-10 · stop the bleeding: capability map v4 and seven module specs (PR #2)

1. **Specs merge before any build.** Each build session starts from one written source instead of a chat.
   Trade-off: a large docs PR. Risk: specs drift from the code, so each spec lists the code facts it relies on.
2. **Freeze the evidence, then reword.** Like sealing lab results before editing the paper's prose: `post-voice`
   checks by hash that code, tables and numbers stay the same while the framing changes. Trade-off: a temporary
   check and a baseline file. Risk: a new sentence can still be untrue, so a fresh reviewer reads every one.
3. **Measure our own content before copying a layout.** Like checking a recipe against what is in your kitchen:
   530 of 585 code blocks are wider than the reference's 720 px column, so code gets a wider column. Trade-off: the
   design departs from the reference where the data differs. Risk: a few measures rest on small samples; the specs
   say which.
