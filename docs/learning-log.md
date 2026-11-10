# Learning log

The three most important whys of each change, newest first. Each heading reads `<date> · <phase>: <title>`.

## 2026-11-10 · stop the bleeding: site-baseline (gates script, CI, protected main)

1. **Fail closed.** A check that cannot see its input should say no, not yes, like a smoke alarm that beeps when its
   battery is gone. `check.py` now exits non-zero when the denylist has no terms, so an unset CI secret turns the
   run red. Trade-off: a PR from a fork, which gets no secrets, always fails. Risk: someone "fixes" that by
   allowing an empty list, which would make the privacy gate pass on nothing.
2. **Scrub at the one exit, not at each print.** Every line `check.py` prints, tracebacks included, goes through
   one scrubber, like one security gate at an airport instead of a guard at every shop. A term shows as
   `[term #3]`. Trade-off: local debugging needs `--show-terms`. Risk: commands outside `check.py` are not
   scrubbed, so privacy runs first and stops the run, and the wording gate prints names and counts, never lines.
3. **The same script locally and in CI.** `scripts/gates.sh` is the only place the gates are written, like one
   recipe card used at home and in the restaurant. Trade-off: CI must install the post code's exact packages
   (`requirements-ci.txt`). Risk: numbers can differ on Linux; if a post fails only there, stop and report it
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
