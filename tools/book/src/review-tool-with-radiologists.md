---
title: A review tool built with radiologists, and a story with no numbers
summary: The story of a heatmap review tool, windowing presets and a multi-reader labeling protocol designed in weekly sessions with radiologists: why a single score is useless to a reader, why a heatmap is a lead and not an explanation, and how to tell a result that was never measured.
group: Project stories
order: 5
tags: medical imaging · explainability · labeling · human factors
---
A model that gives a radiologist one score per scan saves no time: "0.83" still leaves the whole study to read. Alongside the stroke triage model at HTC, I met board-certified radiologists every week to make the model's output usable and its labels trustworthy. Out of those sessions came a review tool with heatmaps, windowing presets designed together, and a multi-reader labeling protocol that the team then used for later datasets. None of it was measured, and the story says so.

## The one sentence

"I built a heatmap review tool and co-designed windowing presets and a multi-reader labeling protocol with radiologists in weekly sessions; the protocol became the team's labeling standard."

## Three things, three reasons

**Show where, not just how much.** The tool overlaid Grad-CAM heatmaps on each slice, so a radiologist could see where the model responded and check that region, which is faster than reading from scratch. Grad-CAM weights the last convolutional layer's feature maps by the gradient of the score, so its map is that layer's coarse grid stretched to the slice.

**Windowing presets, chosen by eye.** CT stores intensity in Hounsfield units (HU), and a display window picks which range of HU maps onto the screen's gray levels. Early ischemic change is a difference of a few HU, and a wide window squeezes it into a couple of gray levels. Whether a reader can see a finding is a question about human perception, so no loss function can tune it; the only source of that signal is a radiologist looking. We designed the presets in the sessions and then used the same windows as the model's input.

The arithmetic of the squeeze, for an 8-bit display (these widths illustrate the effect; they are not our presets):

```python
def gray_levels(hu_difference, window_width, levels=256):
    return hu_difference * levels / window_width

assert gray_levels(4, 80) == 12.8      # a 4 HU change in an 80 HU window
assert gray_levels(4, 400) == 2.56     # the same change in a 400 HU window: almost invisible
```

**More than one reader per label.** Early ischemia is a judgment call, and readers disagreed exactly at the boundary the model has to learn. That disagreement, not the model, was the real data-quality problem. The protocol had several readers label each case and set a clear path for settling disagreements, instead of trusting one reader. It is slower and costs more reader time per case, so the same budget labels fewer cases.

## What a heatmap does not show

A heatmap shows where the model responded, not why it decided. At the resolution of the last convolutional layer it is coarse for a subtle finding, and saliency methods can produce similar maps even after a model's weights are randomized. So I presented it as a lead to check, never as an explanation; clinicians, like everyone, tend to trust anything that looks like a heatmap.

## How to tell a result with no number

There was no reader study, no timing and no agreement statistic. An earlier résumé line carried an efficiency gain that was an impression from the sessions, not a measurement, and I removed it. What I can say is what happened: the tool and presets were used in the prototype, and the labeling protocol became how the team built later datasets.

The number that was easiest to collect and was not is agreement between readers before and after the protocol. Cohen's kappa corrects raw agreement for the agreement chance alone would give:

```python
def cohen_kappa(agree, positive_rate_a, positive_rate_b):
    chance = positive_rate_a * positive_rate_b + (1 - positive_rate_a) * (1 - positive_rate_b)
    return (agree - chance) / (1 - chance)

# Two readers agree on 80 of 100 cases, and each calls half of them positive.
assert round(cohen_kappa(0.80, 0.5, 0.5), 2) == 0.60
# The same 80% agreement means less when both readers call nearly everything negative.
assert round(cohen_kappa(0.80, 0.1, 0.1), 2) < 0
```

## Limits, said first

- It was a research prototype and demo; it never went through a regulatory pathway or clinical deployment.
- No reader study: I cannot say the heatmaps made reading faster, or that they did not add misses.
- No before-and-after agreement figure for the protocol.

## The two likeliest follow-ups

**"Was it deployed?"** No. What was adopted inside the team was the labeling protocol.

**"Does Grad-CAM prove the model saw the lesion?"** No; it is evidence to check, and a coarse one.

## What I would measure now

A small reader study: several radiologists read the same cases in random order, with and without the tool, with a washout period between sessions, recording both time and accuracy. And kappa before and after the protocol, which costs almost nothing once two readers label the same cases.
