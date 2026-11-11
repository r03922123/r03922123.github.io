# Conditional probability and Bayes' rule, by counting

When a detector catches 90% of real events, fires on 20% of normal cases, and real events make up 10% of traffic, only one flag in three is real. Conditional probability explains why. It answers the question "given that I now know B happened, how likely is A?" Knowing B rules out every outcome where B did not happen; among the outcomes that are left you count how often A also happens. From that one idea I derive the rules for "or" and "and", Bayes' rule, and the base-rate effect that makes a good detector of a rare event produce mostly false alarms.

Most claims below are checked in code, on a small sample space that can be enumerated exactly, or by simulation.

## Counting on a small sample space
Roll two fair dice. There are 36 equally likely outcomes, so the probability of any event is the number of outcomes in it divided by 36. Writing events as functions of an outcome, and conditioning as a filter on the outcomes, gives exact answers with `Fraction`:

```python
from fractions import Fraction
from itertools import product

omega = list(product(range(1, 7), repeat=2))          # 36 equally likely (first, second) rolls


def P(event, given=lambda w: True):
    """Probability of `event` among the outcomes where `given` holds: conditioning is filtering."""
    kept = [w for w in omega if given(w)]
    return Fraction(sum(event(w) for w in kept), len(kept))


def AND(e1, e2):
    return lambda w: e1(w) and e2(w)


def OR(e1, e2):
    return lambda w: e1(w) or e2(w)


def NOT(e):
    return lambda w: not e(w)


first6 = lambda w: w[0] == 6
second6 = lambda w: w[1] == 6
high = lambda w: w[0] + w[1] >= 11                     # (5, 6), (6, 5), (6, 6)

assert P(first6) == Fraction(1, 6) and P(high) == Fraction(3, 36)
```

## "Or" and "and"
The **addition rule** for "at least one of A, B" counts both events and subtracts the overlap, which was counted twice:

\[
P(A \cup B) = P(A) + P(B) - P(A \cap B)
\]

The **multiplication rule** for "both" is the definition of conditional probability rearranged, and it holds for any two events:

\[
P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)
\]

Two events are **independent** when learning one does not change the probability of the other, \(P(A \mid B) = P(A)\). Only then does the product simplify to \(P(A)\,P(B)\). Independence is a property of the numbers, not of the story: you check it, you do not assume it because the events "seem unrelated".

A common shortcut for "at least one" goes through the complement. By De Morgan's law, "not (A or B)" is "not A and not B", so

\[
P(A \cup B) = 1 - P(A^c \cap B^c)
\]

That part is always true. The popular next step, \(1 - P(A^c)\,P(B^c)\), replaces the joint probability with a product and so is valid **only when A and B are independent**. For the two separate dice it works. For "first die is 6" and "the sum is at least 11" it does not, because the second event makes the first much more likely:

```python
# Independent events: the shortcut and the addition rule agree
assert P(OR(first6, second6)) == P(first6) + P(second6) - P(AND(first6, second6)) == Fraction(11, 36)
assert P(AND(first6, second6)) == P(first6) * P(second6)
assert P(OR(first6, second6)) == 1 - (1 - P(first6)) * (1 - P(second6))

# Dependent events: De Morgan still holds, the product shortcut does not
assert P(first6, given=high) == Fraction(2, 3)                     # four times P(first6)
assert P(OR(first6, high)) == P(first6) + P(high) - P(AND(first6, high)) == Fraction(7, 36)
assert P(OR(first6, high)) == 1 - P(AND(NOT(first6), NOT(high)))
assert 1 - (1 - P(first6)) * (1 - P(high)) == Fraction(51, 216) != Fraction(7, 36)

# The definition of conditioning, and the multiplication rule, on any pair
assert P(first6, given=high) == P(AND(first6, high)) / P(high)
assert P(AND(first6, high)) == P(high, given=first6) * P(first6)
```

The shortcut overstates the answer here (51/216 against 42/216) because it treats the overlap as smaller than it is.

## Bayes' rule
Writing the multiplication rule both ways round and dividing gives **Bayes' rule**, which turns a known probability into the wanted one:

\[
P(H \mid E) = \frac{P(E \mid H)\,P(H)}{P(E)}
\]

Read \(H\) as a hypothesis ("this transaction is fraud") and \(E\) as evidence ("the model flagged it"). \(P(H)\) is the **prior**, \(P(E \mid H)\) the **likelihood**, and \(P(H \mid E)\) the **posterior**. The denominator is rarely given directly. The **law of total probability** builds it by splitting on whether \(H\) is true, since every flag is either a true or a false one:

\[
P(E) = P(E \mid H)\,P(H) + P(E \mid H^c)\,P(H^c)
\]

With more than two hypotheses, the sum runs over any partition of the outcomes into disjoint cases.

## Worked example: the base-rate effect
A detector flags rare events. It catches 90% of real ones, \(P(E \mid H) = 0.9\), the **recall** or true positive rate. It also fires on 20% of normal cases, \(P(E \mid H^c) = 0.2\), the false positive rate. Real events make up 10% of traffic. When the detector fires, how likely is it that the event is real?

\[
P(H \mid E) = \frac{0.9 \times 0.1}{0.9 \times 0.1 + 0.2 \times 0.9} = \frac{0.09}{0.27} = \frac{1}{3}
\]

Exactly one flag in three is real. In this setting the posterior is the detector's **precision**. The cause is visible in the denominator: of every 100 cases, 9 are real and caught, while 18 are normal and wrongly flagged. The false positive rate is small, but it applies to the 90% majority.

It is tempting to call this "a 90% accurate detector", but 90% is the recall. Accuracy counts correct answers over all cases, \(0.9 \times 0.1 + 0.8 \times 0.9 = 0.81\). Recall conditions on the true state, precision conditions on the flag, and accuracy conditions on nothing at all. They are three different probabilities, and quoting the wrong one is the usual way this effect surprises people. A simulation with scikit-learn's metric functions agrees with the formulas:

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def posterior(prior, p_e_given_h, p_e_given_not_h):
    evidence = p_e_given_h * prior + p_e_given_not_h * (1 - prior)   # law of total probability
    return p_e_given_h * prior / evidence


assert posterior(Fraction(1, 10), Fraction(9, 10), Fraction(1, 5)) == Fraction(1, 3)

rng = np.random.default_rng(0)
n = 1_000_000
y = rng.random(n) < 0.10                                        # real events
flag = np.where(y, rng.random(n) < 0.9, rng.random(n) < 0.2)    # detector output
assert abs(precision_score(y, flag) - 1 / 3) < 0.005
assert abs(recall_score(y, flag) - 0.9) < 0.005
assert abs(accuracy_score(y, flag) - 0.81) < 0.005
```

The rarer the event, the worse it gets. Holding the detector fixed and varying the base rate:

| Base rate | Precision, FPR 20% | Precision, FPR 1% |
|---|---|---|
| 50% | 0.818 | 0.989 |
| 10% | 0.333 | 0.909 |
| 1% | 0.043 | 0.476 |
| 0.1% | 0.004 | 0.083 |

```python
table = {0.5: (0.818, 0.989), 0.1: (0.333, 0.909), 0.01: (0.043, 0.476), 0.001: (0.004, 0.083)}
for base_rate, expected in table.items():
    got = tuple(round(posterior(base_rate, 0.9, fpr), 3) for fpr in (0.2, 0.01))
    assert got == expected, (base_rate, got)
```

For an event that occurs once in a thousand, even a 1% false positive rate leaves more than 90% of flags false. This is why precision depends on the base rate while recall and the false positive rate do not, and why the precision-recall curve is the more honest summary for rare positives ([ROC AUC vs PR AUC](../roc-auc-vs-pr-auc/)).

## The odds form
Bayes' rule is easiest to apply in your head when written in odds, \(p / (1 - p)\). Dividing the posterior for \(H\) by the posterior for \(H^c\) cancels \(P(E)\):

\[
\frac{P(H \mid E)}{P(H^c \mid E)} = \frac{P(H)}{P(H^c)} \times \frac{P(E \mid H)}{P(E \mid H^c)}
\]

Posterior odds are prior odds times the **likelihood ratio**. In the example, the prior odds are 1 to 9 and a flag has likelihood ratio \(0.9 / 0.2 = 4.5\), so the posterior odds are 4.5 to 9, that is 1 to 2, which is probability 1/3. In log space the update is addition, which is why a logistic regression's logit has the same shape, an intercept in the role of log prior odds plus a sum of evidence terms, and why [naive Bayes](../naive-bayes/) with binary or count features is a linear model in disguise.

## Updating twice, and when not to
The posterior after one piece of evidence becomes the prior for the next. With two flags from detectors that err **independently given the true state** (conditionally independent), multiplying the likelihood ratios is exact: \(1/9 \times 4.5^2\) gives odds 2.25, a posterior of 0.69.

That condition fails whenever both signals share an error source: the same model run on overlapping windows, two rules built on the same feature, two reviewers reading the same misleading summary. Suppose the second flag copies the first 80% of the time and is an independent draw otherwise. Each flag alone still has recall 0.9 and false positive rate 0.2, so each looks exactly as informative as before, yet two flags together lift the posterior only to 0.37. Enumerating the joint distribution exactly:

```python
def joint_two_flags(prior, tpr, fpr, copy):
    """P(H = h, F1 = a, F2 = b): F2 repeats F1 with probability `copy`, else is a fresh draw."""
    out = {}
    for h, a, b in product((0, 1), repeat=3):
        q = tpr if h else fpr                                    # P(flag | H = h)
        p_a = q if a else 1 - q
        p_b_fresh = q if b else 1 - q
        out[h, a, b] = (prior if h else 1 - prior) * p_a * (copy * (a == b) + (1 - copy) * p_b_fresh)
    return out


def posterior_both_flagged(joint):
    return joint[1, 1, 1] / (joint[1, 1, 1] + joint[0, 1, 1])


prior, tpr, fpr = Fraction(1, 10), Fraction(9, 10), Fraction(1, 5)
naive_odds = prior / (1 - prior) * (tpr / fpr) ** 2
naive = naive_odds / (1 + naive_odds)

independent = joint_two_flags(prior, tpr, fpr, copy=0)
correlated = joint_two_flags(prior, tpr, fpr, copy=Fraction(4, 5))

for joint in (independent, correlated):                         # each flag alone is unchanged
    for h, rate, p_h in ((1, tpr, prior), (0, fpr, 1 - prior)):
        p_f1 = sum(v for (hh, a, b), v in joint.items() if hh == h and a == 1)
        p_f2 = sum(v for (hh, a, b), v in joint.items() if hh == h and b == 1)
        assert p_f1 / p_h == p_f2 / p_h == rate                  # recall 0.9, false positive rate 0.2
assert posterior_both_flagged(independent) == naive == Fraction(9, 13)          # 0.692
assert posterior_both_flagged(correlated) == Fraction(49, 133)                 # 0.368
```

Treating correlated evidence as independent counts the same information twice and makes the system overconfident. This is the same failure that makes naive Bayes probabilities extreme when features are redundant.

## Bayes inside a classifier: correcting for the base rate
A probabilistic classifier estimates \(P(y = 1 \mid x)\), which by Bayes' rule depends on the class prior. Train on data rebalanced to 50/50 and deploy where positives are 5%, and the model's probabilities are too high everywhere. The odds form gives the fix: the likelihood ratio \(p(x \mid y=1) / p(x \mid y=0)\) is unchanged by resampling, so only the prior odds need swapping. On the logit scale,

\[
\text{logit}\,p_{\text{deploy}}(x) = \text{logit}\,p_{\text{train}}(x) - \text{logit}\,\pi_{\text{train}} + \text{logit}\,\pi_{\text{deploy}}
\]

where \(\pi\) is the positive rate. This assumes only the class mix changed, not what each class looks like. With two Gaussian classes the true posterior is known in closed form, so I can check the correction directly:

```python
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

pi_train, pi_deploy = 0.5, 0.05
m = 20_000
x_train = np.concatenate([rng.normal(1, 1, m), rng.normal(-1, 1, m)])   # x | y=1 ~ N(1, 1), x | y=0 ~ N(-1, 1)
y_train = np.concatenate([np.ones(m), np.zeros(m)])
clf = LogisticRegression(C=1e6).fit(x_train[:, None], y_train)

grid = np.linspace(-3, 3, 61)
p_train = clf.predict_proba(grid[:, None])[:, 1]
p_fixed = expit(logit(p_train) - logit(pi_train) + logit(pi_deploy))
p_true = expit(2 * grid + logit(pi_deploy))          # log likelihood ratio of these Gaussians is 2x

assert np.abs(p_train - p_true).max() > 0.4          # uncorrected: far too high
assert np.abs(p_fixed - p_true).max() < 0.02         # corrected: matches the true posterior

# On simulated deployment traffic, the corrected scores are calibrated on average
y_dep = rng.random(200_000) < pi_deploy
x_dep = np.where(y_dep, rng.normal(1, 1, y_dep.size), rng.normal(-1, 1, y_dep.size))
p_dep = expit(logit(clf.predict_proba(x_dep[:, None])[:, 1]) - logit(pi_train) + logit(pi_deploy))
assert abs(p_dep.mean() - y_dep.mean()) < 0.003

# The shift is monotone, so the ranking and ROC AUC are untouched
from sklearn.metrics import roc_auc_score
p_raw = clf.predict_proba(x_dep[:, None])[:, 1]
assert roc_auc_score(y_dep, p_raw) == roc_auc_score(y_dep, p_dep)
```

The ranking of cases does not change, so ROC AUC is identical before and after; only the probabilities and any threshold set on them move. When the class-conditional distributions also drift, this shortcut no longer suffices and the model needs recalibrating on fresh labelled data.

## Pitfalls
- **Swapping the condition.** \(P(E \mid H)\) and \(P(H \mid E)\) are different numbers; the first is a property of the detector, the second also depends on the base rate. Confusing them is the confusion of the inverse; in a courtroom, with \(H\) as innocence, it is called the prosecutor's fallacy.
- **Quoting one number for a detector.** Ask for recall, the false positive rate and the base rate. On a rare event, accuracy can be high and recall can be high while precision is poor, as in the table above.
- **The product shortcut without independence.** \(1 - P(A^c)\,P(B^c)\) is the probability of "at least one" only for independent events. De Morgan's form \(1 - P(A^c \cap B^c)\) is always correct.
- **Multiplying likelihood ratios from correlated signals.** Two flags that share an error source carry much less than twice the evidence.
- **Reusing probabilities across populations.** A score learned at one base rate is miscalibrated at another. Shift the logit if only the class mix changed; recalibrate if more than that changed.
- **Conditioning on an event of probability zero.** \(P(A \mid B)\) is undefined when \(P(B) = 0\). For continuous variables, the conditional density takes its place, which is what the Gaussian example above uses.

The detector numbers come from fixed recall and false positive rates, and the prior-shift check from two Gaussian classes whose log likelihood ratio is exactly linear, so they show the arithmetic of Bayes' rule, not how a real detector or classifier behaves. On a real score, the open question is whether only the class mix changed or more than that, and the second case needs recalibrating on fresh labelled data. The same rule, applied to a parameter instead of an event, is the basis of [MLE and MAP](../statistical-estimation-mle-map/) and of the Beta-Binomial updates in [quantifying uncertainty with the Beta distribution](../beta-uncertainty/). For more of the core identities on one page, see the [probability cheatsheet](../probability-cheatsheet/).

The opening rules follow the Day 3 tutorial of HackerRank's [10 Days of Statistics](https://www.hackerrank.com/domains/tutorials/10-days-of-statistics); chapter 2 of Blitzstein and Hwang's Introduction to Probability (2nd edition, 2019) covers conditioning, Bayes' rule and the odds form in depth. The prior-shift correction is the standard adjustment described by Saerens, Latinne and Decaestecker (Neural Computation, 2002).
