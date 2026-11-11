---
title: Fixing product names in live Japanese captions, without an accuracy number
summary: The story of a correction layer after speech recognition: morphemes because Japanese has no spaces, katakana normalization so spelling variants do not eat the match budget, fuzzy lookup in the client's error table, and why the honest result has a latency figure and no accuracy figure.
group: Project stories
order: 2
tags: speech recognition · Japanese · fuzzy matching · latency
---
A client's live captions kept misspelling its own product names, and those were exactly the words the audience had come to hear. The captions came from a commercial speech-to-text service, streamed at a product event, and ran on the client's own laptop. Retraining the recognizer was out: there was no audio to train on, no GPU, and no time. At KKCompany I designed a correction layer that sits after recognition and fixes those terms within a 50 ms streaming budget; the platform team deployed it.

The result has no accuracy figure, and the story has to own that.

## The one sentence

"I designed a correction layer after speech recognition that fixed a client's mis-transcribed product names inside a 50 ms streaming budget, and the platform team deployed it."

## Who did what

| Part | Owner |
|---|---|
| The correction algorithm | me |
| The streaming recognizer and its integration | the platform team |
| The caption display and deployment | the platform team |
| The table of known errors and correct terms | the client, who kept it up to date |

I can explain how the recognizer's streaming events feed my layer; I did not build that part.

## Three steps

**Split into morphemes.** Japanese text has no spaces, so there are no words to compare against. A dictionary-based morphological analyzer (Janome) splits each segment into morphemes. Product names often split into two or three of them, so the layer tries windows of one to three adjacent morphemes as candidate spans.

**Normalize katakana.** The same foreign word has several accepted katakana spellings: ヴァ or バ, a long-vowel mark or not, half-width or full-width characters. To an edit distance these are real differences. If the threshold allows one edit and a spelling variant already costs one, nothing is left for the recognition error itself. So both sides are normalized first, and what distance remains is signal.

**Fuzzy-match the error table.** Each normalized span is compared with the known error forms in the client's table, and above a threshold it is replaced by the correct term. Below it, the text is left alone. The comparison used RapidFuzz's ratio, a similarity from 0 to 100 based on how many characters must be inserted or deleted to make two strings equal.

The sketch below rebuilds the idea with invented product names and hand-written morphemes, so it runs without the tokenizer. The ratio is computed from scratch with the same definition.

```python
import unicodedata

def normalize(s):
    s = unicodedata.normalize("NFKC", s)                 # half-width katakana to full width
    for a, b in (("ヴァ", "バ"), ("ヴィ", "ビ"), ("ヴェ", "ベ"), ("ヴォ", "ボ"), ("ヴ", "ブ")):
        s = s.replace(a, b)
    return s.replace("ー", "")                           # drop long-vowel marks (a choice with a cost: see below)

def ratio(a, b):
    """100 * (1 - insertions and deletions needed / total length), via the longest common subsequence."""
    lcs = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            lcs[i + 1][j + 1] = lcs[i][j] + 1 if x == y else max(lcs[i][j + 1], lcs[i + 1][j])
    return 100 * (1 - (len(a) + len(b) - 2 * lcs[-1][-1]) / (len(a) + len(b)))

# The client's table: known mis-transcriptions -> the correct product name (both invented here).
TABLE = {"ノバリス": "ノヴァリス", "ゼフィラン": "ゼフィランEX"}
THRESHOLD = 70

def correct(morphemes):
    out, i = [], 0
    while i < len(morphemes):
        best = None
        for width in (3, 2, 1):                          # windows of 1 to 3 morphemes, longest first
            span = normalize("".join(morphemes[i:i + width]))
            for wrong, right in TABLE.items():
                score = ratio(span, normalize(wrong))
                if score >= THRESHOLD and (best is None or score > best[0]):
                    best = (score, width, right)
        if best:
            out.append(best[2]); i += best[1]
        else:
            out.append(morphemes[i]); i += 1
    return "".join(out)

assert round(ratio("ノバリズ", "ノバリス")) == 75            # one wrong character in four: 2 edits out of 8
assert normalize("ﾉｳﾞｧﾘｰｽ") == normalize("ノヴァリス")      # width, ヴァ and the long vowel all fold away
assert correct(["新しい", "ノ", "バリズ", "を", "発表"]) == "新しいノヴァリスを発表"
assert correct(["ノート", "を", "見る"]) == "ノートを見る"     # an ordinary word is left alone
```

## The number I can give, and the one I cannot

The client never gave us a measured error rate, and we had no held-out captions with ground truth, so I have no accuracy number for the deployed layer. An earlier version of my résumé carried a coverage percentage and a latency figure that were estimates, not measurements, and I took both off.

What I can give is latency on a reproduction: I rebuilt the layer later and timed it, at a median of about 1.2 ms per segment, mostly spent in tokenization, far inside the 50 ms budget. That is my rebuild on my machine, not the client's system, and I say so.

## Limits, said first

- A lookup table does not generalize: an error not in the table is never fixed. In practice the client added new errors to the table.
- Dropping long-vowel marks can merge two different words; a real list of normalizations needs checking against the client's vocabulary.
- Nothing measured the false substitutions, where a correct word is "fixed" into a wrong one, which is the most embarrassing error on a live screen.
- The threshold in the sketch is for the example. Choosing it properly means sweeping it on held-out captions against both fixes and false substitutions.

## How I would measure it now

Three numbers, in this order: the error rate on the product terms the client cares about, since overall word error barely moves when a dozen names are fixed; the false-substitution rate; and caption latency from speech to screen.

## The two likeliest follow-ups

**"Why not bias the recognizer instead?"** Biasing the recognizer with a phrase list or a custom model beats patching its output, and it should come first. If I rebuilt this, I would bias first and correct whatever remains with this layer.

**"Why a table and not a learned model?"** There was no training data, no GPU on the laptop, and the client wanted to control the vocabulary themselves. The cost is zero generalization beyond the table.

## What stays open

Correcting captions with a language model that reads the surrounding sentence is now common, and it would catch errors outside the table. Whether it fits a live 50 ms budget on a laptop is the question I would test before choosing it.
