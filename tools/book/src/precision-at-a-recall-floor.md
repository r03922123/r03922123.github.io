---
title: Raising precision at a recall floor: an ad-review detector
summary: The story of a smoking-violation detector that screened ads before human review: why recall was fixed and precision moved, two levels of class imbalance, hard negatives as the main gain, and the arithmetic that checks "+25 points" against "two thirds fewer false positives".
group: Project stories
order: 3
tags: object detection · class imbalance · hard negative mining · evaluation
---
Every ad the model flagged went to a human reviewer, so a missed violation was the expensive mistake: it skipped review and went live. A false alarm cost a reviewer a few seconds. At TikTok, in Trust and Safety, I worked on the detector for smoking in ad creatives, used across more than 50 markets, and the job had a fixed shape: hold recall at the floor policy set, and raise precision so reviewers see fewer false alarms. Precision went from the mid-50s to around 80% at that floor.

## The one sentence

"At a recall floor set by policy, I raised precision from the mid-50s to about 80% on a pre-review ad filter, cutting false positives by about two thirds, mainly through hard negative mining."

## Why recall was fixed

The model was a screen in front of people. A false negative was never seen by anyone, so nothing downstream could catch it. A false positive was caught by the reviewer. So policy fixed recall first; then, because reviewer time is finite, every false positive removed was reviewer time returned. Precision at a fixed recall is the metric that matches that cost.

## Three decisions

**A detector, not a classifier.** A cigarette is often a few pixels in one corner of a frame, and a whole-image classifier can lose that after downsampling. A detector (YOLOX, over frames sampled from each video) keeps small objects and gives the reviewer a box as evidence. I did not benchmark it against a whole-image classifier at the time; that belongs in the reflection.

**Two levels of imbalance.** Inside each image, background locations far outnumber object locations; YOLOX's label assignment handles that by design. Across the dataset, violating ads were very rare, which YOLOX does not address, so I oversampled positives to keep a fixed ratio in each training batch. Evaluation kept the real class distribution, because precision measured on a rebalanced set would be inflated.

**Hard negatives, the main gain.** Most false positives were things that look like cigarettes. I collected them two ways: by reviewing sampled data by hand, and by running the previous model version over the negative pool and taking its highest-scoring negatives. Every mined example was checked by a person before it went back into training, so an unlabeled real violation could not be taught as a negative.

## Checking the numbers against each other

Fix recall, and the number of caught violations is fixed. Call it 100.

```python
tp = 100
fp_before = tp / 0.55 - tp          # precision = tp / (tp + fp)
fp_after = tp / 0.80 - tp
cut = 1 - fp_after / fp_before
assert round(fp_before) == 82 and fp_after == 25
assert 0.65 < cut < 0.72            # "about two thirds" fewer false positives
assert round((0.80 - 0.55) * 100) == 25   # "+25 points"
```

The three claims agree, which is worth checking before saying any of them.

## From training to traffic

The loop ran on the company's internal ML platform: reviewer decisions became training labels; I trained with oversampling and evaluated offline at the recall floor on the real distribution; the new model then scored live traffic in shadow mode, where its output affected nothing, so old and new could be compared on the same ads; then it rolled out gradually to full traffic. Reviewer decisions on the new model's flags fed the next round of hard negatives. I went through this loop first with a weapon classifier, then with the smoking detector.

What I would watch at each stage, as my own judgment rather than a written internal rule: in shadow mode, the flag volume and whether the new model still catches what the old one caught and reviewers confirmed; during the ramp, reviewer-confirmed precision, flag volume against reviewer capacity, and a breakdown by market so no market quietly regresses.

## Why online recall is hard to measure

Missed violations are, by definition, not reviewed. If one ad in a thousand violates, an audit of 10,000 passed ads finds about ten, and ten events give a wide interval.

```python
import math

rate, audited = 0.001, 10_000
expected = rate * audited
assert expected == 10
# A rough 95% interval on a count near 10 (normal approximation to the Poisson): about 4 to 16 misses.
low, high = expected - 1.96 * math.sqrt(expected), expected + 1.96 * math.sqrt(expected)
assert 3.5 < low < 4.0 and 16.0 < high < 16.5
```

So recall after launch leans on random audits, user reports and later re-review, not on one clean number.

## Limits, said first

- No ablation: the main gain came from hard negatives, but I did not record each step's separate contribution.
- No benchmark against a whole-image classifier or a vision-language model.
- I do not quote internal policy thresholds or exact rollout percentages.

## The two likeliest follow-ups

**"Were the high-scoring negatives really negative?"** Each was checked by a person before training, for exactly that reason.

**"Did recall drop?"** Every version was evaluated at the same recall floor, so the comparison is precision at equal recall.

## What I would do today

Keep a light detector for the high-volume screen, and add a multimodal model that writes a judgment with a reason for the flagged ads, so reviewers decide faster. Whether that reason helps reviewers or just adds reading time is the experiment to run first.
