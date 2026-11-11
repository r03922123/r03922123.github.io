---
title: Stroke triage on head CT from 500 labeled volumes
summary: The story of a CT triage model where the label budget drove every decision: a 2.5D input instead of 3D, an encoder started from my own hemorrhage-competition model, attention across slices, and a 0.88 AUC told with the interval a small test set implies.
group: Project stories
order: 4
tags: medical imaging · transfer learning · small data · evaluation
---
Labels on head CT need a radiologist, so the stroke triage project at HTC had 500 labeled volumes and no way to get many more. A volume is not one image: it is dozens of slices, and early ischemic stroke shows on only a few of them, as a faint darkening. The goal was to flag suspicious scans so they get read sooner. Every modeling choice in the story follows from the label budget, and the result, 0.88 AUC, needs the uncertainty a small test set brings.

## The one sentence

"With only 500 labeled CT volumes, I fine-tuned a 2.5D encoder started from my own hemorrhage-detection model and added attention across slices, reaching 0.88 slice-level AUC on a held-out set split by patient."

## Three decisions, one constraint

**2.5D, not 3D.** A 3D network has far more parameters, and 500 volumes would let it memorize. Stacking a slice with its neighbours as input channels keeps a 2D model's size while giving the first layer some through-plane context. It also made the next decision possible, because large labeled head-CT datasets are 2D.

**Start from my own hemorrhage model.** In 2019 I placed 30th of 1,345 (top 3%) in Kaggle's RSNA intracranial hemorrhage competition, on about 750,000 labeled CT slices. Hemorrhage is a different disease, but it is the same modality, anatomy and physics, which is a much shorter step than from ImageNet photographs. The risk is real and worth saying: bleeding is bright on CT and early ischemia is a subtle dark change, so the pretraining may favour bright findings.

**Attention across slices.** Each slice's decision should use the slices around it, because a faint change that continues across neighbours is more convincing than one that appears once. An attention layer over the slice embeddings learns that context from the labels instead of a fixed rule.

## The number, with its interval

0.88 is the measured AUC on the held-out test set, which was split by patient, so no patient's slices were in both training and test. I do not remember the test set's size, so the interval below uses an assumed one, 30 positive and 70 negative cases, with the Hanley and McNeil approximation for the standard error of an AUC.

```python
import math

def auc_interval(a, n_pos, n_neg):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    se = math.sqrt(var)
    return a - 1.96 * se, a + 1.96 * se

low, high = auc_interval(0.88, 30, 70)                 # assumed counts, not the project's
assert round(low, 2) == 0.80 and round(high, 2) == 0.96
low3, high3 = auc_interval(0.88, 300, 700)              # ten times the data: about a third as wide
assert 0.30 < (high3 - low3) / (high - low) < 0.33
```

At that size, 0.88 cannot be told apart from 0.84. For a slice-level AUC the formula is too kind, because slices from one patient are not independent; the right interval resamples whole patients, a bootstrap by patient.

## Who did what

I owned the model end to end: preprocessing, the 2.5D input, the transfer from my competition model and the attention over slices. Radiologists made the labels. It stayed a research prototype, so the integration into a reading worklist and any clinical validation never happened.

## Limits, said first

- No confidence interval was reported at the time; the one above is an illustration with assumed counts.
- No external validation. A model trained on one hospital's scanners often loses several AUC points at another.
- No comparison with the same model started from ImageNet at the same 500 volumes, which is the baseline that would show how much the pretraining bought.

## The two likeliest follow-ups

**"Sample-efficient compared with what?"** I do not have that baseline number, and I should. The right comparison is the same architecture initialized from ImageNet, trained on the same 500 volumes.

**"How was the data split?"** By patient. A split by slice would put neighbouring slices of one scan on both sides and inflate the AUC.

## What a deployment would measure

AUC is not what triage is for. A deployment would fix a sensitivity at an operating point the clinicians accept, then report specificity there; measure how much sooner true positives get read, which is the point of triage; and validate on scans from another hospital before any of it counts.

## What I would do today

Pretrain without labels first, with contrastive or masked-image objectives on unlabeled head CT, which uses far more scans and does not import the hemorrhage bias. Whether that beats starting from a supervised hemorrhage model at 500 labels is the question I would test.
