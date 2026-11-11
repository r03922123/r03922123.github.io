# Evaluating LLM applications: golden sets, judges, noise and guardrails

A prompt change that is truly 3 points better scores the same or worse on a 60-question golden set in about 30% of runs. To detect the same change with 80% power, the arithmetic below needs about 865 questions. That is the gap between how LLM applications are often evaluated (a few dozen hand-picked questions, eyeballed before and after a change) and what the numbers can support.

An **evaluation loop** is the routine that turns "the new prompt looks better" into a measured decision: a fixed set of labelled questions, metrics that say which part of the system moved, a grader that is itself checked, an interval around every difference, and a gate that runs the whole thing on every change. **Guardrails** are the checks that run on live traffic and block an unsafe input or output. If you know classifier evaluation, most of this is familiar: a judge is a classifier to be validated against labels, and a guardrail is a classifier with an operating point. What is new is that the system under test produces text, so the labels themselves come from a model or a person.

Everything here is simulated with numpy and scipy, with no model calls, so every number is an assert. I build one piece per section, in the order a real loop needs them.

## A golden set, scored in two halves
A **golden set** is a fixed list of questions with what a correct answer needs: for a RAG system, the chunks that hold the evidence and a reference answer. [RAG architecture](../rag-architecture/) builds such a set and argues for checking retrieval first; this section shows why one end-to-end number is not enough to act on.

Score the two halves separately. **Recall@k** asks whether the evidence reached the prompt: the fraction of a question's relevant chunks in the top \(k\). **Faithfulness** asks whether the answer is supported by the retrieved text, the idea behind the RAGAS faithfulness score [1]. **Answer correctness** compares the answer with the reference. Correctness and faithfulness differ exactly when the model answers from memory: right by luck, not grounded, and not something to rely on next month.

The simulation gives 300 questions one or two relevant chunks each, places each chunk at a geometric rank, and lets the generator answer correctly with probability \(g\) when all the evidence is in the top 5. With evidence missing, it answers from memory, right 35% of the time and unfaithful by construction. Two changes run on the same questions: a better retriever and a better prompt.

```python
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

rng = np.random.default_rng(0)
N, K = 300, 5
n_rel = np.where(rng.random(N) < 0.3, 2, 1)                  # 30% of questions need two chunks
u_rank, u_gen, u_mem = rng.random((N, 2)), rng.random(N), rng.random(N)   # shared: same questions, same luck


def run_rag(p_top, g, memory=0.35):
    ranks = stats.geom.ppf(u_rank, p_top)                      # rank of each relevant chunk
    found = ((ranks <= K) & (np.arange(2) < n_rel[:, None])).sum(1)
    complete = found == n_rel
    faithful = complete & (u_gen < g)                           # grounded answer from complete evidence
    correct = faithful | (~complete & (u_mem < memory))         # or a lucky answer from memory
    return {"recall@5": np.mean(found / n_rel), "complete": complete.mean(), "correct": correct.mean(),
            "faithful": faithful.mean(), "correct | complete": correct[complete].mean(),
            "lucky share": (correct & ~faithful).sum() / correct.sum()}


R = {name: {k: round(v, 3) for k, v in run_rag(*args).items()}
     for name, args in {"baseline": (0.3, 0.85), "retriever v2": (0.42, 0.85), "prompt v2": (0.3, 0.91)}.items()}
assert [R[n]["correct"] for n in R] == [0.773, 0.823, 0.83]          # both changes gain 5 to 6 points
assert [R[n]["recall@5"] for n in R] == [0.823, 0.932, 0.823]        # only one moved retrieval
assert [R[n]["correct | complete"] for n in R] == [0.866, 0.862, 0.937]   # only the other moved generation
assert [R[n]["faithful"] for n in R] == [0.69, 0.793, 0.747]
assert R["baseline"]["lucky share"] == 0.108                          # correct but not grounded
```

| Change | Recall@5 | Correct | Faithful | Correct when evidence complete |
|---|---|---|---|---|
| Baseline | 0.823 | 0.773 | 0.690 | 0.866 |
| Retriever v2 | 0.932 | 0.823 | 0.793 | 0.862 |
| Prompt v2 | 0.823 | 0.830 | 0.747 | 0.937 |

The two changes gain about the same end to end, and the component columns say which half each one fixed: recall@5 moves only for the retriever, correctness given complete evidence only for the prompt. In the baseline, 10.8% of the correct answers are lucky guesses from memory, which correctness alone counts as successes and faithfulness does not. For agents the same split applies to the path and the final answer; [LLM agents](../llm-agents-langchain-langgraph/) scores trajectories against a reference.

## An LLM judge is a classifier: validate it
Faithfulness and correctness on free text are usually graded by another model, an **LLM-as-judge**. Before trusting it, treat it like any classifier: label a few hundred outputs by hand and measure agreement. Raw agreement misleads when one label dominates, for the same reason accuracy misleads on imbalanced classes. **Cohen's kappa** [2] corrects for the agreement two raters would reach by chance given how often each says "pass":

\[ \kappa = \frac{p_o - p_e}{1 - p_e}, \qquad p_e = p_1 q_1 + (1 - p_1)(1 - q_1), \]

where \(p_o\) is observed agreement and \(p_1, q_1\) are the two raters' pass rates. Zero means chance-level agreement; one means perfect.

```python
rng = np.random.default_rng(1)
n = 300
human = (rng.random(n) < 0.7).astype(int)                     # 1 = the answer passes, by a person
judge = np.where(human == 1, rng.random(n) < 0.92, rng.random(n) < 0.30).astype(int)
always_pass = np.ones(n, int)


def kappa(a, b):
    p_o = np.mean(a == b)
    p_e = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (p_o - p_e) / (1 - p_e)


assert np.isclose(kappa(human, judge), cohen_kappa_score(human, judge))
assert (round(np.mean(judge == human), 3), round(kappa(human, judge), 2)) == (0.867, 0.67)
assert (round(np.mean(always_pass == human), 2), kappa(human, always_pass)) == (0.71, 0.0)

tpr, fpr = judge[human == 1].mean(), judge[human == 0].mean()  # the judge's operating point
rng = np.random.default_rng(2)
truth = rng.random(2000) < 0.55                                 # a weaker system: 55% truly pass
judged = np.where(truth, rng.random(2000) < 0.92, rng.random(2000) < 0.30)
raw = judged.mean()
corrected = (raw - fpr) / (tpr - fpr)                           # invert the judge's error rates
assert (round(truth.mean(), 3), round(raw, 3), round(corrected, 3)) == (0.55, 0.654, 0.595)
assert corrected - truth.mean() < (raw - truth.mean()) / 2 and (human == 0).sum() < 100
```

A judge that passes everything agrees with the person 71% of the time and has a kappa of exactly zero; the real judge agrees 86.7% of the time with a kappa of 0.67. Kappa only says the judge is useful, not that its numbers are unbiased. This judge passes 30% of failing answers, so on a system that truly passes 55% it reports 65.4%: its false-pass rate inflates every headline number. Inverting the error rates measured on the labelled set brings the estimate to 59.5%, more than half the way back; the rest is noise in a false-pass rate estimated from under a hundred failing answers. Report the judge's kappa and its error rates next to any number it produces, and re-label a sample whenever the judge model or its prompt changes.

## Position bias in pairwise judging
Pairwise judging asks which of two answers is better, which is easier than an absolute score and is how system-versus-system comparisons are often run. Judges favour one position. Zheng et al. [3] measured this and proposed calling the judge twice with the answers swapped, counting a win only when both orders agree and a tie otherwise. Below, A and B have the same quality distribution and the judge adds a fixed push toward whichever answer it reads first.

```python
rng = np.random.default_rng(3)
n = 1000
qa, qb = rng.normal(size=n), rng.normal(size=n)                 # true quality, same distribution


def prefers_first(q1, q2, u, bias=1.0):
    return u < 1 / (1 + np.exp(-(3 * (q1 - q2) + bias)))


b_first = prefers_first(qb, qa, rng.random(n))                  # B shown first; True = B wins
a_first = prefers_first(qa, qb, rng.random(n))                  # A shown first; True = A wins
b_wins, a_wins = b_first & ~a_first, a_first & ~b_first        # consistent in both orders
ties = ~(b_wins | a_wins)
swapped = (b_wins.sum() + 0.5 * ties.sum()) / n
assert (round(b_first.mean(), 2), round(a_first.mean(), 2)) == (0.57, 0.6)   # whoever goes first "wins"
assert (round(swapped, 3), round(np.mean(qb > qa), 3), round(ties.mean(), 3)) == (0.484, 0.489, 0.226)
```

Shown first, B wins 57% of comparisons; shown first, A wins 60%. With both orders and ties for disagreement, B's win rate is 48.4%, close to the 48.9% of pairs where B is truly better. The 22.6% of pairs that flip with order are worth logging on their own: they measure how much the judge's verdicts depend on layout rather than content. Length and self-preference biases [3] need their own controls, such as length-matched pairs.

## Small eval sets are noisy
A golden set is a sample, so every score on it has sampling error, and the bridge from classifier evaluation is direct: a difference between two systems needs an interval. Because both systems answer the same questions, the right interval is **paired**: resample questions, and with each question take both systems' outcomes together. Most questions are answered the same way by both, so only the questions where they disagree add variance.

Take 60 questions on which the candidate fixes 4 and breaks 2: 78.3% to 81.7%.

```python
def bootstrap_ci(d, rng, B=10_000):
    """95% percentile interval for mean(d). Resampling a vector with few distinct values is a multinomial draw."""
    vals, counts = np.unique(d, return_counts=True)
    means = rng.multinomial(len(d), counts / len(d), size=B) @ vals / len(d)
    return np.quantile(means, [0.025, 0.975])


base = np.r_[np.ones(45), np.ones(2), np.zeros(4), np.zeros(9)]   # both right, base only, candidate only, both wrong
cand = np.r_[np.ones(45), np.zeros(2), np.ones(4), np.zeros(9)]
rng = np.random.default_rng(4)
lo, hi = bootstrap_ci(cand - base, rng)
assert (round(base.mean(), 3), round(cand.mean(), 3), round(lo, 3), round(hi, 3)) == (0.783, 0.817, -0.05, 0.117)
ulo, uhi = np.quantile(rng.binomial(60, cand.mean(), 10_000) / 60 - rng.binomial(60, base.mean(), 10_000) / 60, [0.025, 0.975])
assert (round(ulo, 3), round(uhi, 3)) == (-0.117, 0.183)          # ignoring the pairing
assert uhi - ulo > 1.7 * (hi - lo) and hi - lo > 0.16

CELLS = np.array([0.745, 0.035, 0.065, 0.155])                  # a true +3 points, 10% of questions disagree
k = rng.multinomial(60, CELLS, size=20_000)
assert round(np.mean(k[:, 2] <= k[:, 1]), 2) == 0.30             # the opening: no better on 60 questions

z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)                 # two-sided 5% test, 80% power


def items_needed(gain, discordant):
    return int(np.ceil(z**2 * (discordant - gain**2) / gain**2))


assert [items_needed(g, 0.10) for g in (0.02, 0.03, 0.05, 0.10)] == [1955, 865, 307, 71]
n = items_needed(0.03, 0.10)
k = rng.multinomial(n, CELLS, size=4000)
d = (k[:, 2] - k[:, 1]) / n
power = np.mean(d / np.sqrt(((k[:, 2] + k[:, 1]) / n - d**2) / n) > stats.norm.ppf(0.975))
assert 0.78 < power < 0.83                                      # the formula delivers the power it promised
```

The paired interval on the +3.3 points runs from −5.0 to +11.7, so this eval cannot tell the candidate from the baseline; ignoring the pairing makes it nearly twice as wide, −11.7 to +18.3. The sample-size formula is the normal approximation for paired binary outcomes: the variance of the per-question difference is the discordant rate minus the squared gain. With 10% of questions disagreeing, a 10-point gain needs 71 questions, a 5-point gain 307, a 3-point gain 865 and a 2-point gain 1955; a simulation at 865 confirms about 80% power. Two levers follow: more questions, or fewer discordant ones, which in practice means a less noisy grader (sampling the system at temperature zero, or several judge calls per answer).

## Guardrails are classifiers with an operating point
A **guardrail** scores an input or output and blocks it above a threshold: a policy classifier, a PII detector, a prompt-injection detector. The choice is the one from [ROC and PR curves](../roc-auc-vs-pr-auc/): pick the operating point from the cost of each error. For unsafe outputs that usually means a **recall floor** (catch at least 95% of unsafe outputs on the labelled set), then reading off what that costs on benign traffic in false refusals.

The hard negatives matter most. A benign question about medication doses or a security patch looks unsafe to a lexical or shallow classifier. In the simulation, 10% of benign traffic is such hard negatives.

```python
rng = np.random.default_rng(6)
unsafe = rng.normal(3.5, 1, 500)                                 # guardrail scores on labelled unsafe outputs
hard = rng.random(5000) < 0.1
benign = np.where(hard, rng.normal(1.5, 1, 5000), rng.normal(0, 1, 5000))

rows = {}
for floor in (0.90, 0.95, 0.99):
    t = np.quantile(unsafe, 1 - floor)                           # lowest threshold that keeps the floor
    recall, fpr = np.mean(unsafe >= t), np.mean(benign >= t)
    per_catch = [fpr * (1 - pi) / (recall * pi) for pi in (0.01, 0.001)]   # benign blocked per unsafe caught
    rows[floor] = [round(x, 3) for x in (fpr, np.mean(benign[hard] >= t), np.mean(benign[~hard] >= t))] + \
                  [round(x, 1) for x in per_catch]
assert rows == {0.90: [0.038, 0.24, 0.015, 4.2, 42.0],
                0.95: [0.059, 0.331, 0.029, 6.2, 62.3],
                0.99: [0.197, 0.657, 0.146, 19.7, 198.6]}
assert round(stats.norm.ppf(0.975) * np.sqrt(0.95 * 0.05 / len(unsafe)), 3) == 0.019   # 95% half-width on the floor
assert round(np.mean(hard[benign >= np.quantile(unsafe, 0.05)]), 2) == 0.56   # hard share of benign refusals
```

| Recall floor | False refusals, all benign | Hard negatives | Easy benign | Blocked benign per unsafe caught, 1% unsafe | 0.1% unsafe |
|---|---|---|---|---|---|
| 0.90 | 3.8% | 24.0% | 1.5% | 4.2 | 42.0 |
| 0.95 | 5.9% | 33.1% | 2.9% | 6.2 | 62.3 |
| 0.99 | 19.7% | 65.7% | 14.6% | 19.7 | 198.6 |

At a 95% floor the guardrail refuses 5.9% of benign requests overall, and a third of the hard negatives: 56% of its benign refusals land on the 10% of traffic that asks sensitive but legitimate questions. Base rate decides the rest: at 1% unsafe traffic each caught unsafe output costs about 6 benign refusals, at 0.1% about 62. Report false refusals by slice next to the recall floor, and measure the floor itself with an interval, since 500 unsafe examples pin a 95% recall only to within about 1.9 points either way.

## A regression gate that respects noise
The last piece runs the golden set on every prompt or model change in CI and decides whether to block. The naive gate blocks whenever the score drops. The gate below blocks only when the paired bootstrap interval of candidate minus baseline lies entirely below zero, that is, when the drop is outside noise.

```python
def gate(base, cand, rng):
    return bootstrap_ci(cand - base, rng, B=2000)[1] < 0           # block only if the whole interval is a drop


def draw(cells, n, rng):
    c = rng.choice(4, size=n, p=cells)                             # both right, base only, candidate only, both wrong
    return np.isin(c, (0, 1)).astype(int), np.isin(c, (0, 2)).astype(int)


SAME = np.array([0.745, 0.05, 0.05, 0.155])                       # a change with no effect
DROP = np.array([0.745, 0.065, 0.015, 0.175])                     # a change that is 5 points worse
rng = np.random.default_rng(7)
block = {}
for name, cells in (("same", SAME), ("drop", DROP)):
    for n in (60, 200, 1000):
        runs = [draw(cells, n, rng) for _ in range(400)]
        block[name, n] = (sum(gate(b, c, rng) for b, c in runs), sum(c.mean() < b.mean() for b, c in runs))
assert block == {("same", 60): (4, 179), ("same", 200): (17, 205), ("same", 1000): (7, 186),   # blocks out of 400:
                 ("drop", 60): (98, 358), ("drop", 200): (294, 397), ("drop", 1000): (400, 400)}  # (noise-aware, naive)
```

The naive gate blocks 45% to 51% of changes that do nothing, which teaches a team to override it. The noise-aware gate blocks 1% to 4%. Its cost is the other error: a real 5-point drop is blocked only 24.5% of the time on 60 questions, 73.5% on 200 and every time on 1000. A gate this lenient on a small set must be paired with the sample-size arithmetic above; the stricter alternative, a non-inferiority gate that passes only when the interval's lower end is above a tolerated drop, would block most changes on 60 questions, where the paired interval above spans more than 16 points, which is the honest answer when the set cannot tell.

## The loop in one table

| Problem | Naive fix | How it breaks | Real fix | How it's measured |
|---|---|---|---|---|
| Answers got worse | Tune the prompt | The evidence never reached the prompt | Score retrieval and generation separately | Recall@k, then correctness given complete evidence |
| Too many outputs to grade by hand | An LLM judge | Biased toward one label, inflates pass rates | Validate on human labels, report error rates | Cohen's kappa, false-pass and false-fail rates |
| Pairwise comparison | One call per pair | First position wins | Judge both orders, disagreement is a tie | Swapped win rate, flip rate |
| "+3 points on 60 questions" | Ship it | A true +3 looks no better 30% of the time | Paired bootstrap interval, sized eval set | Interval width, items needed for the gain |
| Unsafe outputs | Block at a high score | Misses, or refuses legitimate users | Threshold from a recall floor | False refusals by slice and base rate |
| Regressions in CI | Block any drop | Blocks half of harmless changes | Block only drops outside the interval | False-block rate and detection rate by set size |

Each row is a classifier-evaluation habit applied to a system whose output is text.

## Pitfalls
- **A golden set that never changes.** Real traffic drifts and a fixed set gets tuned to. Add failing production cases (with labels) on a schedule and keep a held-out slice no one tunes against.
- **Judging with the model under test.** A judge from the same family may prefer its own style [3]. Use a different model, or at least measure its kappa on that system's outputs.
- **One judge call per answer at high temperature.** Grader noise adds to the discordant rate and so to the items needed. Fix the temperature, or average several calls.
- **A single guardrail threshold for every surface.** Base rates differ between a public chat box and an internal tool, and so does the cost of a false refusal.
- **Reporting the point estimate only.** Every number from a golden set needs its interval and its \(n\).

Every judge, retriever and guardrail here is a random draw with error rates I chose, so the asserts show how the quantities behave, not what a real LLM judge's kappa is, how position bias varies with the judge's prompt, or how a real guardrail's scores separate unsafe from benign text. Grader noise is also modelled as independent per question, while a real judge errs more on some kinds of question than others, which widens intervals further. Open for a real system: how many human labels a judge needs before its error rates are stable enough to correct with, and how to keep a golden set representative as traffic changes; [Evaluating ML systems](../evaluating-ml-systems/) covers where these offline numbers sit among online tests and product metrics.

## References
1. Es et al., [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217), EACL 2024 (demo track).
2. Cohen, A Coefficient of Agreement for Nominal Scales, Educational and Psychological Measurement 20(1), 1960.
3. Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685), NeurIPS 2023 Datasets and Benchmarks Track.
