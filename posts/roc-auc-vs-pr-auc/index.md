# ROC AUC vs PR AUC: what each curve sees

On rare positives (fraud, disease, abusive content) the ROC curve and the precision-recall curve can rank the same pair of models in opposite orders. In the simulation below, at 1% positives, model A leads on ROC AUC by 0.856 to 0.794 and model B leads on average precision by 0.34 to 0.12. A classifier that outputs a score, not a label, has no single accuracy. Every threshold you could cut the scores at gives a different trade between catching positives and raising false alarms. Both curves sweep that threshold and summarize the whole sweep as an area. They look like two views of the same thing, and on balanced data they mostly agree.

I build both curves from scratch, check what each area means against brute force and scikit-learn, and show which part of the ranking each one rewards, so choosing between them stops being a rule of thumb.

## Four counts at one threshold
Fix a threshold \(t\) and call everything with score \(s \ge t\) positive. Against the true labels that gives four counts: true positives \(TP\), false positives \(FP\), true negatives \(TN\) and false negatives \(FN\). Three rates built from them are all I need:

| Rate | Formula | Question it answers |
|---|---|---|
| Recall, true positive rate (TPR) | \(TP / (TP + FN)\) | Of the real positives, what fraction did we flag? |
| False positive rate (FPR) | \(FP / (FP + TN)\) | Of the real negatives, what fraction did we flag? |
| Precision | \(TP / (TP + FP)\) | Of what we flagged, what fraction is really positive? |

TPR and FPR each look inside one class only: TPR is computed from positives alone and FPR from negatives alone. Precision mixes the two classes, and that one difference drives everything below.

## The ROC curve
The ROC curve plots TPR against FPR as the threshold falls from above the highest score to below the lowest. It starts at \((0, 0)\), where nothing is flagged, and ends at \((1, 1)\), where everything is. Sorting the scores once gives every point: walk down the sorted list, and each positive moves the curve up by \(1/P\) while each negative moves it right by \(1/N\), where \(P\) and \(N\) are the class counts. Items with equal scores must cross the threshold together, so a run of ties becomes one diagonal step.

```python
import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


def trapezoid(y, x):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def roc_points(y, s):
    """FPR and TPR after each distinct threshold, highest score first."""
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    run_end = np.r_[s[1:] != s[:-1], True]      # ties cross the threshold together
    tp, fp = np.cumsum(y)[run_end], np.cumsum(1 - y)[run_end]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]


rng = np.random.default_rng(0)
y = (rng.random(2000) < 0.2).astype(int)        # 20% positives
s = np.round(rng.normal(1.2 * y, 1.0), 1)       # positives score higher on average; rounding makes ties

fpr, tpr = roc_points(y, s)
ref_fpr, ref_tpr, _ = roc_curve(y, s, drop_intermediate=False)
assert np.allclose(fpr, ref_fpr) and np.allclose(tpr, ref_tpr)
roc_auc = trapezoid(tpr, fpr)
assert np.isclose(roc_auc, roc_auc_score(y, s))
```

The area under the curve is a number between 0 and 1. A model that scores at random traces the diagonal and gets 0.5; a model that puts every positive above every negative goes straight up and then across, and gets 1.

## The area is a ranking probability
ROC AUC has a reading that needs no curve at all. Pick one positive and one negative at random. The AUC is the probability that the positive gets the higher score, counting a tie as half:

\[ \text{AUC} = P(s^+ > s^-) + \tfrac{1}{2} P(s^+ = s^-). \]

The reason is that each negative moves the curve one step right, and the height at that step is the fraction of positives already above it, so the area adds up exactly the fraction of correctly ordered pairs. That fraction is the Mann–Whitney \(U\) statistic divided by \(PN\), which also gives an \(O(n \log n)\) formula through ranks: with \(R^+\) the sum of the positives' ranks (ties given their average rank),

\[ \text{AUC} = \frac{R^+ - P(P+1)/2}{P N}. \]

```python
pos, neg = s[y == 1], s[y == 0]
diff = pos[:, None] - neg[None, :]                  # every (positive, negative) pair
pair_auc = np.mean(diff > 0) + 0.5 * np.mean(diff == 0)
assert np.isclose(pair_auc, roc_auc)

P, N = len(pos), len(neg)
rank_auc = (stats.rankdata(s)[y == 1].sum() - P * (P + 1) / 2) / (P * N)
assert np.isclose(rank_auc, roc_auc)
assert np.isclose(stats.mannwhitneyu(pos, neg).statistic / (P * N), roc_auc)

# only the order of the scores matters
assert np.isclose(roc_auc_score(y, np.exp(3 * s) - 7), roc_auc)
assert np.isclose(roc_auc_score(y, -s), 1 - roc_auc)
```

Two consequences follow. Any strictly increasing transform of the scores leaves the AUC unchanged, so it says nothing about calibration: a model whose "probabilities" are all between 0.40 and 0.41 can have AUC 0.99. And since the pair probability depends only on how scores are spread within each class, it does not depend on how many positives there are relative to negatives.

## The precision-recall curve
The precision-recall (PR) curve plots precision against recall over the same sweep. Its usual summary is **average precision** (AP): walk down the ranking, and at each positive record the precision of the list cut just below it; AP is the mean of those precisions. Equivalently it is the area under a step curve where precision is held constant across each increase in recall.

Precision is where class balance enters. Write \(\pi = P/(P + N)\) for the prevalence. Dividing the counts by the class sizes gives

\[ \text{precision} = \frac{\text{TPR}\cdot \pi}{\text{TPR}\cdot \pi + \text{FPR}\cdot(1 - \pi)}, \]

which is Bayes' rule with \(\pi\) as the prior ([conditional probability and Bayes](../conditional-probability-bayes/) works through the same formula for medical tests). Hold TPR and FPR fixed, make positives rarer, and precision falls. A model that scores at random has precision \(\pi\) on average at every threshold, and any model that flags everything has precision exactly \(\pi\), so the baseline of the PR curve is the prevalence, not 0.5.

```python
def average_precision(y, s):
    """Mean precision at the rank of each positive (scores assumed distinct)."""
    hits = y[np.argsort(-s)]
    precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return precision_at_k[hits == 1].mean()


s_cont = rng.normal(1.2 * y, 1.0)                    # same model without rounding: no ties
assert np.isclose(average_precision(y, s_cont), average_precision_score(y, s_cont))

# precision at a threshold is Bayes' rule on TPR, FPR and prevalence
t, prev = 0.8, y.mean()
tpr_t, fpr_t = np.mean(s_cont[y == 1] >= t), np.mean(s_cont[y == 0] >= t)
direct = np.mean(y[s_cont >= t])
assert np.isclose(direct, tpr_t * prev / (tpr_t * prev + fpr_t * (1 - prev)))

# a random scorer: ROC AUC near 0.5, average precision near the prevalence
noise = [rng.random(len(y)) for _ in range(20)]
assert abs(np.mean([roc_auc_score(y, r) for r in noise]) - 0.5) < 0.01
assert abs(np.mean([average_precision_score(y, r) for r in noise]) - prev) < 0.01
```

## Same model, rarer positives
Take two models, hold each one's score distributions fixed, and change only how common positives are. Model A shifts every positive's score up by the same moderate amount. Model B is very sure about 30% of the positives and only weakly separates the rest. The code keeps one sample per class and reweights the negatives to reach each prevalence, so any change in a metric comes from the prevalence alone.

```python
n_pos, n_neg = 20_000, 200_000
neg_scores = rng.normal(0, 1, n_neg)
sure = rng.random(n_pos) < 0.3
models = {"A": rng.normal(1.5, 1, n_pos),
          "B": np.where(sure, rng.normal(4, 0.5, n_pos), rng.normal(0.75, 1, n_pos))}
labels = np.r_[np.ones(n_pos), np.zeros(n_neg)]

table = {}
for prevalence in (0.5, 0.1, 0.01, 0.001):
    w_neg = n_pos * (1 - prevalence) / prevalence / n_neg      # weight per negative
    weights = np.r_[np.ones(n_pos), np.full(n_neg, w_neg)]
    for name, pos_scores in models.items():
        scores = np.r_[pos_scores, neg_scores]
        table[name, prevalence] = (roc_auc_score(labels, scores, sample_weight=weights),
                                   average_precision_score(labels, scores, sample_weight=weights))

for name in models:     # ROC AUC does not move at all
    assert np.allclose([table[name, p][0] for p in (0.5, 0.1, 0.01, 0.001)], table[name, 0.5][0])
assert table["A", 0.5][0] > table["B", 0.5][0] + 0.05          # ROC AUC prefers A
assert table["A", 0.5][1] > table["B", 0.5][1]                 # so does AP when classes are balanced
assert table["B", 0.01][1] > 2.5 * table["A", 0.01][1]         # at 1% prevalence AP prefers B by far
```

| Prevalence | ROC AUC, A | ROC AUC, B | AP, A | AP, B |
|---|---|---|---|---|
| 50% | 0.856 | 0.794 | 0.854 | 0.827 |
| 10% | 0.856 | 0.794 | 0.48 | 0.53 |
| 1% | 0.856 | 0.794 | 0.12 | 0.34 |
| 0.1% | 0.856 | 0.794 | 0.018 | 0.25 |

The ROC columns do not move, as the pairwise reading promised. The AP columns collapse as positives get rarer, at different speeds. At 1% prevalence, an FPR of a few percent costs model A a few hundred false alarms per hundred positives, while model B's confident 30% sit above almost every negative and keep precision near 1 at the top of the list. The two metrics now disagree about which model is better.

## Which model is better, then?
Neither metric is wrong; they weight the ranking differently. ROC AUC counts every misordered pair equally, wherever it sits in the list. AP is dominated by the top of the list, because at low prevalence that is the only region where precision is far from zero. The right question is which part of the list you will act on.

```python
def precision_at_recall(y, s, w, target):
    """Best precision among thresholds that reach the target recall."""
    precision, recall, _ = precision_recall_curve(y, s, sample_weight=w)
    return precision[recall >= target].max()


weights = np.r_[np.ones(n_pos), np.full(n_neg, n_pos * 99 / n_neg)]   # 1% prevalence
p_at = {(name, r): precision_at_recall(labels, np.r_[ps, neg_scores], weights, r)
        for name, ps in models.items() for r in (0.25, 0.8)}
assert p_at["B", 0.25] > 0.9 and p_at["A", 0.25] < 0.2       # a short review queue: B wins
assert p_at["A", 0.8] > 1.4 * p_at["B", 0.8]                 # catching 80% of positives: A wins
```

If a team can review only a thin slice of the highest-scoring cases, B finds a quarter of all positives at better than 90% precision, while at the same recall A's flags are more than 80% false alarms. If the requirement is to catch 80% of positives, A does it at about 3% precision and B at under 2%, so B raises about two-thirds more false alarms for every case it catches. ROC AUC happens to side with the second use and AP with the first. When the operating point is known, report precision at that recall (or recall at that review budget) directly.

## Computing PR AUC honestly
The area under a PR curve is easy to get wrong in ways that the ROC curve forgives.

- **Do not connect PR points with straight lines.** On the ROC curve a straight segment between two thresholds is achievable by random tie-breaking. On the PR curve it is not: precision along that segment is a ratio of two linear functions, not a line. When precision falls as recall rises, which is the usual case, the achievable curve sags below the chord, so the straight line overstates the area. Jesse Davis and Mark Goadrich showed this, and also that one curve dominates another in ROC space exactly when it dominates in PR space, even though the areas can order models differently.
- **Coarse scores make it worse.** A model that outputs a handful of distinct values gives a handful of PR points, and a trapezoid through them, plus the conventional point at recall 0 and precision 1, can be far too generous.

```python
n_p, n_n = 200, 19_800                                          # 1% prevalence
levels = np.r_[rng.choice(3, n_p, p=[0.3, 0.4, 0.3]), rng.choice(3, n_n, p=[0.01, 0.1, 0.89])]
y3, s3 = np.r_[np.ones(n_p), np.zeros(n_n)], 2.0 - levels      # three score levels, many ties

precision, recall, _ = precision_recall_curve(y3, s3)
trapezoid_area = trapezoid(precision[::-1], recall[::-1])
ap = average_precision_score(y3, s3)


def expected_pr_area(y, s, grid=10_000):
    """Area under the PR curve, interpolating inside each tied run as Davis and Goadrich do."""
    tp = fp = area = 0.0
    x = (np.arange(grid) + 0.5) / grid                          # fraction of the tied run let in
    for v in np.unique(s)[::-1]:
        dtp, dfp = y[s == v].sum(), (1 - y[s == v]).sum()
        area += np.mean((tp + x * dtp) / (tp + fp + x * (dtp + dfp))) * dtp / y.sum()
        tp, fp = tp + dtp, fp + dfp
    return area


honest = expected_pr_area(y3, s3)
assert trapezoid_area > 2 * honest                              # the trapezoid more than doubles it
assert abs(ap - honest) < 0.03                                  # the step sum stays close

# independent check: break the ties at random many times and average the untied AP
random_breaks = np.mean([average_precision(y3, s3 + 1e-6 * rng.random(len(s3))) for _ in range(100)])
assert abs(random_breaks - honest) < 0.01 and trapezoid_area > 2 * random_breaks
```

Here the trapezoid reports about 0.24, the area under the correctly interpolated curve is about 0.11 (averaging AP over many random tie-breaks gives the same 0.11) and AP is about 0.09. That is why scikit-learn's `average_precision_score` uses the step sum and not a trapezoid, and why "PR AUC" numbers from different libraries are often not comparable.

## Pitfalls
- **AP depends on the test set's prevalence.** Downsample negatives to speed up evaluation and AP rises while ROC AUC stays put. Compare AP only across test sets with the same base rate, and always quote the base rate next to it, since AP 0.3 is weak at 25% prevalence and strong at 0.1%.
- **A high ROC AUC on rare positives can still mean an unusable model.** An FPR of 1% sounds small, but against a million negatives it is ten thousand false alarms. Convert to precision at the operating point before calling a model good.
- **Neither area measures calibration.** Both depend only on the order of the scores. If the scores are used as probabilities (to set prices, to combine with other models, to compute expected cost), check calibration separately, for example with log loss; see [cross-entropy vs MSE](../cross-entropy-vs-mse/).
- **Few positives make AP noisy.** With a few dozen positives, AP can move by a factor of two or three between resamples of the same test set (the code below). Put a bootstrap interval on it, resampling positives and negatives separately; [simulation-based evaluation](../simulation-based-evaluation/) covers the method.
- **"Use PR AUC for imbalanced data" is a rule of thumb, not a theorem.** As the two models above show, AP rewards the top of the ranking. That is right for a review queue and wrong when you must catch most positives. Matthew McDermott and colleagues (2024) make this case in detail, including how AP can favour gains on the subgroups that already have the most positives.

```python
# model A on a small test set: 40 positives among 4,000
y_small = np.r_[np.ones(40), np.zeros(3960)]
s_small = np.r_[rng.normal(1.5, 1, 40), rng.normal(0, 1, 3960)]
pos_i, neg_i = np.flatnonzero(y_small == 1), np.flatnonzero(y_small == 0)
boot = []
for _ in range(300):                       # resample each class separately, keeping the prevalence
    idx = np.r_[rng.choice(pos_i, len(pos_i)), rng.choice(neg_i, len(neg_i))]
    boot.append(average_precision_score(y_small[idx], s_small[idx]))
low, high = np.percentile(boot, [5, 95])
assert high > 2 * low                      # the 90% interval spans more than a factor of two
```

## When to use which
| Situation | Report |
|---|---|
| Roughly balanced classes, the whole ranking matters | ROC AUC |
| Rare positives, you act only on the top of the list | Average precision, and precision at the review budget |
| Rare positives, you must catch most of them | Precision at the required recall; AP alone can mislead |
| The threshold is already fixed | Precision and recall at that threshold, with intervals |
| Scores are used as probabilities | Add a calibration check or log loss; neither area covers it |

Both models here are built from normal scores I chose, reweighted to each prevalence, so the prevalence table shows how prevalence moves each metric, not which kind of model wins on real data. For a real model the open question is which part of the list will be acted on; once that operating point is known, precision there is the number to report. For where these numbers sit among product and online metrics, see [evaluating ML systems](/posts/evaluating-ml-systems/).

Sources: the pairwise reading of AUC goes back to Hanley and McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve" (Radiology, 1982); Tom Fawcett's "An introduction to ROC analysis" (Pattern Recognition Letters, 2006) is a good general reference. The PR interpolation result is from Jesse Davis and Mark Goadrich, "The Relationship Between Precision-Recall and ROC Curves" (ICML 2006). Takaya Saito and Marc Rehmsmeier argue for PR plots on imbalanced data in "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets" (PLoS ONE, 2015), and McDermott et al. question that rule in "A Closer Look at AUROC and AUPRC under Class Imbalance" (NeurIPS 2024). The structure of the first sections follows Sinyi Chou's blog posts on [AUC and the ROC curve](https://sinyi-chou.github.io/classification-auc/) and [the PR curve](https://sinyi-chou.github.io/classification-pr-curve/), rewritten here.
