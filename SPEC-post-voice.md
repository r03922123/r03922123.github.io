# Spec: post-voice

Module of `CAPABILITY-MAP.md` (version 4). Depends on `site-baseline`. Status: **approved** 2026-11-10.

## Objective

Posts read like reports of work: they open on a problem, state what was found, say what the result does not show, and
end on what is still open. The code and numbers that make these posts worth reading do not change.

Users: readers deciding within two paragraphs whether a post is worth their time, and the owner, whose name is on it.

| Requirement | Done means |
|---|---|
| 59 posts are reframed | The 55 topic posts and 4 long-form posts follow the moves below; the 31 LeetCode notes and the external MusicGen posts are untouched |
| Evidence is frozen | Every code block and table is byte-identical; a machine check says so |
| No invented claims | No number appears that was not already in that post, and none disappears |
| Posts do not grow | Each post's prose grows by at most 10% |
| The voice is the owner's | One author, "I" only for what the post itself does; no sentence from any reference |

## The moves

Taken from the 9-post sample in `CAPABILITY-MAP.md`. Heading style is not a rule, because the sample disagrees on it.

| Move | Rule | Where |
|---|---|---|
| Open on the problem | First sentences give a situation, a stake or a result. The definition comes next | Paragraph 1 |
| State the finding early | Say what the post shows, in first person. About 43 of the 55 topic posts already say "this post" in their first three paragraphs (a regex count); those lines convert | Paragraphs 1–3 |
| Say what it does not show | Name the conditions the evidence rests on, using facts already in the post | Before the last section ends |
| Make evidence speak | A results table or key assert is followed by one sentence on what it shows, if it lacks one | Anywhere |
| End on what is open | One closing paragraph after Pitfalls: what remains unknown or where to go next | Before credits or References |
| Reference pages | The 5 cheat sheets and quick references take only the opening and limits moves, since they argue no single claim | |

First person: "I" for what the post's code or derivation does ("I fit", "I check"). Never for jobs, production
experience or other people's systems, because the posts were not written to claim those, and a false claim on a
public page is the one mistake this module could make that a test cannot catch.

Not changed: titles, summaries, headings (their ids carry the contents and links), sentence length (already 19.6 words,
the same as the sample), code, tables, citations.

## Tech Stack

Markdown sources in `tools/posts/src/` and the four hand-written pages in `posts/<slug>/index.html`, stdlib Python only.
`scripts/voice_check.py` compares each post with `scripts/voice-baseline.json`, generated once from `main` before the
first edit. On the 4 long-form pages, prose means the text inside `<article>`, leaving out `header`, `nav`, `aside`,
`pre` and `table`, so `reading-shell`'s markup cannot disturb the comparison.

## Commands

```
All gates:      sh scripts/gates.sh
Voice check:    python3 scripts/voice_check.py
Final check:    python3 scripts/voice_check.py --final
Voice tests:    (cd scripts && python3 -m unittest test_voice_check)
Build one post: python3 -I tools/posts/build.py . <slug>
```

## Project Structure

```
scripts/voice_check.py        → compares posts with the baseline; a gate until this module ends
scripts/voice-baseline.json   → per post: hashes of code blocks, tables and prose; numbers; prose word count
scripts/test_voice_check.py   → one failing case per rule
scripts/gates.sh              → gains the voice gate; the last PR removes it and the baseline
tools/posts/src/<slug>.md     → the 55 edited sources; generated pages rebuilt
posts/<slug>/index.html       → the 4 long-form pages, edited by hand
```

## Rollout

Each PR opens with `Takeover phase: build (owner-approved exception; repo is in understand)`.

| PR | Posts | Count |
|---|---|---|
| 1 | The check and baseline, no post edited | 0 |
| 2, pilot | `dropout`, `ab-testing`, `position-bias`: a mechanism, an experiment, a long-form page | 3 |
| 3 | Machine learning fundamentals | 11 |
| 4 | Statistics and experimentation | 8 |
| 5 | Probability | 9 |
| 6 | LLMs and generative AI | 10 |
| 7 | Recommender systems, ML systems | 11 |
| 8 | Coding and data, the other long-form pages; removes the gate | 7 |

The 300-line limit counts sources; generated pages are mirrors. The owner reads the pilot before PR 3 starts, because
a wrong voice is cheap to fix on 3 posts and expensive on 59.

## Code Style

A before and after from `dropout`, to show the form. Facts come from the post's own table and code.

```markdown
Before: Dropout is the simplest regularizer in deep learning to write down: on every training step, set each
        unit's output to zero with probability \(p\)…

After:  Fit a network with about 17,000 weights to 20 noisy points and it memorizes the noise: training error
        0.03, test error 0.17. Dropout moves those to 0.13 and 0.12, by setting each unit's output to zero with
        probability \(p\) on every training step.

Limit:  These numbers come from one sine-wave task averaged over 4 seeds, so they show the mechanism, not how much
        dropout will help on your data.
```

## Testing Strategy

- Test first: `test_voice_check.py` has one case each for a changed code block, a changed table, a new number in
  prose, a dropped number, 11% growth, and an unchanged post, which passes normally and fails under `--final`. A number
  already in that post's code or tables may appear in prose, which is what the "4 seeds" line above needs.
- Every post still builds and its asserts pass (`gates.sh`), on Linux in CI.
- Fresh review per PR: a reviewer that never saw the edits' reasoning checks every new sentence against that post's
  code and tables, and checks each "I" against the first-person rule. Machines verify the evidence is intact; only a
  reader can verify the framing is true.

## Boundaries

- Always: derive every limit from what the post already contains. Keep credits and citations.
- Ask first: any change to a post's code; rewriting a paragraph the moves do not cover.
- Stop and report: a claim in an existing post that looks wrong. It gets its own fix, so it cannot hide in a style diff.
- Never: use "we", hiring calls or a reference's sentences. Add an experiment or figure. Edit the baseline to get green.

Out of scope: new figures, cover images, real per-post dates, LeetCode notes, the external MusicGen posts.

## Success Criteria

- [ ] Each rule's test failed before the check existed, and all pass now
- [ ] The owner approved the pilot
- [ ] In PR 8, before the commit that removes the gate, `python3 scripts/voice_check.py --final` passes: all 59 posts
  meet the rules, and each post's prose hash differs from the baseline
- [ ] `sh scripts/gates.sh` exits 0 on every PR
- [ ] Every PR has a fresh review of its new sentences, with no open finding
- [ ] `grep -c` for the post-wording gate still shows only the one allowed hit

## Open Questions

None.
