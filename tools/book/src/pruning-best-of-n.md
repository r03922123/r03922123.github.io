---
title: Pruning best-of-N music generation: a 42% that is derived, not measured
summary: The story behind cutting text-to-music best-of-N compute by about 42%: score partial candidates every 10 seconds, drop the weaker half, and say plainly that the number follows from the schedule and that the early score was never validated.
group: Project stories
order: 1
tags: generative audio · inference · best-of-N · CLAP
---
Best-of-N generation pays for N songs and keeps one. At KKCompany the text-to-music setup generated eight 30-second candidates with MusicGen and picked the one whose CLAP embedding sat closest to the prompt's, so seven full generations were paid for and thrown away. I changed the generation loop to drop weak candidates early. The compute saving is about 42%, and that figure comes from the schedule, not from a stopwatch; this page is about telling it that way.

The fuller technical write-ups are the two [MusicGen posts](/posts/#musicgen). This page is the version for telling the project.

## The one sentence

"I cut best-of-N music generation compute by about 42% by scoring partial candidates against the prompt with CLAP every 10 seconds and dropping the weaker half."

## Context and constraint

The goal was a text-to-music feature for KKBOX's listeners, who would pick from a few generated songs. I worked on it alone, with my manager reviewing direction, and owned the inference code. The constraint was cost: every candidate ran to the full 30 seconds before anything judged it, so the batch paid full price for songs it would discard.

## The decision

At each 10-second mark I embed what every surviving candidate has produced so far with CLAP, score it against the embedding of the prompt text, and keep the top half. So the batch shrinks from eight to four to two over the clip, and the final pick is the stronger of the last pair. CLAP is a model trained to place audio and text in one embedding space, so the similarity says how well the audio matches the words.

## The number, and why it is derived

Count 10-second segments generated. The baseline runs eight candidates for three segments; the pruned schedule runs eight, then four, then two.

```python
baseline = 8 * 3
pruned = 8 + 4 + 2
saving = 1 - pruned / baseline
assert (baseline, pruned) == (24, 14)
assert round(saving, 3) == 0.417            # "about 42%"

# However cheap the smaller batches get, the first 10 s still runs all eight candidates, which is a third of
# the baseline's three full-batch segments: pruning alone cannot make this schedule more than 3x faster.
ceiling = baseline / 8
assert ceiling == 3
```

That is why the claim is compute, not wall-clock time. A decoding step has a large fixed cost that does not shrink with the batch, so the clock gains less than the segment count suggests. At the time I timed a wall-clock speed-up above that ceiling of 3x, which pruning cannot produce on its own, so the baseline must have differed in something else, and I could not separate the two. I do not quote that figure.

A later reproduction on my own laptop, not KKCompany's pipeline, shows the same gap between compute and clock: the derived compute ratio was 1.71x, the measured end-to-end speed-up 1.59x (104 s down to 65 s). Halving the batch did not halve the step time, which is the fixed cost at work.

## What quality claim it supports

We compared pruned and unpruned outputs side by side, on final CLAP score and by listening: my manager, the project lead and I. We heard no noticeable difference. There was no blind test and no statistical test, so the honest phrase is "no noticeable drop", never "no significant drop".

## Limits, said first

- It stayed in research. No product owner took it on, so there was never a definition of good enough to ship. The lesson I took is to find a product counterpart at the start.
- The main gap: I never checked that a candidate's score at 10 seconds predicts its score at 30. If it does not, pruning throws away songs that would have won.
- A product would stop at k candidates for the listener to choose from, not one; the experiment pruned to one to measure the largest saving.
- Every 10 seconds and half each time is a fixed schedule I did not tune.

## The two likeliest follow-ups

**"Does the early score predict the final one?"** That is untested. The check is cheap: run a few hundred prompts with no pruning, record each candidate's CLAP score at 10 seconds and at the end, then look at how well the two rankings agree and how often the song that would have won is cut early. The sketch below shows that measurement on synthetic scores, where I choose the correlation; it says nothing about real music.

```python
import numpy as np

def best_survives(rho, prompts=4000, n=8, seed=0):
    """Share of prompts where the candidate with the best final score survives the 8 -> 4 -> 2 schedule,
    if each 10-second score correlates rho with the final score (synthetic Gaussian scores)."""
    rng = np.random.default_rng(seed)
    final = rng.standard_normal((prompts, n))
    def early():
        return rho * final + np.sqrt(1 - rho**2) * rng.standard_normal((prompts, n))
    alive = np.ones((prompts, n), bool)
    for keep in (4, 2):                       # cut at 10 s and at 20 s, each on a fresh noisy early score
        score = np.where(alive, early(), -np.inf)
        cut = np.sort(score, axis=1)[:, -keep][:, None]
        alive &= score >= cut
    return alive[np.arange(prompts), final.argmax(axis=1)].mean()

rates = [best_survives(r) for r in (0.0, 0.5, 0.9)]
assert abs(rates[0] - 2 / 8) < 0.03         # no signal: the best survives as often as any 2 of 8
assert rates[0] < rates[1] < rates[2]       # more correlation, fewer good songs lost
```

With the real logs, the same function's question, how often the eventual best is still alive at the end, is the number that would justify or kill the schedule, and a curve of it against when and how hard to cut would replace my fixed choice.

**"Why not quantize or use a smaller model?"** Those make each step cheaper; pruning skips steps. They stack.

## What I would do today

I would score with a learned music reward model trained on human preference pairs instead of raw CLAP, since such models are now used to pick the best of N generated songs, and I would validate the early-score assumption before pruning on it. Open-source text-to-music has also moved on to full songs with vocals, so a 30-second instrumental setup is an old baseline; the pruning idea does not depend on it.
