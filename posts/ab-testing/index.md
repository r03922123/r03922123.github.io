# A/B testing: sample size, peeking, variance reduction and overlapping experiments

An A/B test splits users at random into a control group that sees the current product and a treatment group that sees a change, then asks whether a metric differs by more than chance would explain. Random assignment makes the groups alike in everything except the change, so a difference can be blamed on the change.

The significance test at the end is the easy part; the two-proportion z-test, the unit-of-analysis trap and sample ratio mismatch are in the online testing section of [Evaluating ML systems](/posts/evaluating-ml-systems/). This post covers the decisions around that test: how many users to collect, when you may look, how to get the same answer with fewer users, and how to run many experiments on the same users. Every claim is checked by simulation.

## The test, and an A/A check
For conversion rates, the statistic is the difference in rates over its standard error under the null of no difference, using the pooled rate. Its square is the Pearson chi-squared statistic on the 2×2 table, a free cross-check. Report a confidence interval for the difference too; it uses the unpooled standard error, since it no longer assumes the rates are equal.

Before trusting a pipeline, run **A/A tests**, where both groups get the same experience. About 5% should come out significant at \(\alpha = 0.05\); many more means the test or the data are broken.

```python
import hashlib
import math
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)
Z95 = stats.norm.ppf(0.975)


def two_proportion_test(conv_a, n_a, conv_b, n_b):
    p_a, p_b = conv_a / n_a, conv_b / n_b
    pooled = (conv_a + conv_b) / (n_a + n_b)
    z = (p_b - p_a) / np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    return z, 2 * stats.norm.sf(np.abs(z))


def diff_ci(conv_a, n_a, conv_b, n_b):
    p_a, p_b = conv_a / n_a, conv_b / n_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)   # unpooled
    return p_b - p_a - Z95 * se, p_b - p_a + Z95 * se


z, p = two_proportion_test(1_000, 20_000, 1_120, 20_000)          # 5.0% vs 5.6%
chi2, p_chi, _, _ = stats.chi2_contingency([[1_000, 19_000], [1_120, 18_880]], correction=False)
assert math.isclose(z ** 2, chi2) and math.isclose(p, p_chi)
assert round(z, 2) == 2.68 and p < 0.01
lo, hi = diff_ci(1_000, 20_000, 1_120, 20_000)
assert 0.0015 < lo < 0.002 and 0.0100 < hi < 0.0105                  # +0.6 points, somewhere in [0.2, 1.0]

aa = rng.binomial(20_000, 0.05, size=(20_000, 2))                   # 20,000 A/A tests
_, p_aa = two_proportion_test(aa[:, 0], 20_000, aa[:, 1], 20_000)
assert abs((p_aa < 0.05).mean() - 0.05) < 0.005

_, p_tiny = two_proportion_test(50_000, 1_000_000, 51_000, 1_000_000)  # 5.0% vs 5.1%, a million per arm
assert p_tiny < 0.01                                                  # significant, possibly not important
```

## How many users: power before you start
A test can make two mistakes. A **false positive** (rate \(\alpha\)) declares a difference that is not there; a **false negative** (rate \(\beta\)) misses one that is. **Power** is \(1 - \beta\), the chance of detecting a real effect of a given size. The convention is \(\alpha = 0.05\) and power 0.8.

You plan for the **minimum detectable effect** (MDE), the smallest change worth shipping: a business decision, not a statistical one. To detect a change from rate \(p_A\) to \(p_B\), the difference has to sit about \(z_{1-\alpha/2} + z_{1-\beta}\) standard errors from zero, which gives the sample size **per arm**:

\[ n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \,\big(p_A(1-p_A) + p_B(1-p_B)\big)}{(p_B - p_A)^2}. \]

For a 5% baseline and a 10% relative lift to 5.5%, that is about 31,200 users per arm. Two consequences matter more than the formula:

- **The MDE enters squared.** Halving it costs four times the users. Small effects on rare events are expensive.
- **An underpowered test says almost nothing.** With 10,000 users per arm, the same real lift is detected only about 35% of the time. A "not significant" result from that test is mostly a statement about the sample size.

```python
def n_per_arm(p_a, p_b, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return math.ceil(z ** 2 * (p_a * (1 - p_a) + p_b * (1 - p_b)) / (p_b - p_a) ** 2)


def power_at(n, p_a, p_b, alpha=0.05):
    se = math.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / n)
    return stats.norm.cdf(abs(p_b - p_a) / se - stats.norm.ppf(1 - alpha / 2))


n = n_per_arm(0.05, 0.055)
assert 31_000 < n < 31_500
assert 3.8 < n_per_arm(0.05, 0.0525) / n < 4.0                       # half the MDE, ~4x the users
assert round(power_at(10_000, 0.05, 0.055), 2) == 0.35

sims = 4_000                                                         # check the formula by simulation
_, p_sim = two_proportion_test(rng.binomial(n, 0.05, sims), n, rng.binomial(n, 0.055, sims), n)
assert abs((p_sim < 0.05).mean() - 0.80) < 0.03
_, p_small = two_proportion_test(rng.binomial(10_000, 0.05, sims), 10_000, rng.binomial(10_000, 0.055, sims), 10_000)
assert abs((p_small < 0.05).mean() - 0.35) < 0.03
```

For a continuous metric such as revenue per user, replace the two Bernoulli variances with twice the metric's variance; a few big spenders inflate it, and the sample size with it.

## Peeking
A daily dashboard invites a natural mistake: check the p-value every day and stop as soon as it dips below 0.05. Each look is another chance for noise to cross the line. With 20 looks at an A/A test, about a quarter of experiments are declared winners at some point, five times the rate you thought you were running. The simulation follows the z-statistic as data arrive in 20 equal batches; under the null it is a scaled random walk.

There are two honest fixes. Fix the sample size in advance and look once, or use a **sequential design** that plans the looks and uses a stricter threshold at each. Pocock's design uses the same threshold every time, chosen so the chance of ever crossing it under the null is 5%. For 20 looks that threshold is about 2.67 instead of 1.96. O'Brien and Fleming's design is very strict early and keeps the final threshold close to the fixed-sample 1.96 (about 2.12 with 20 looks), and always-valid p-values (Johari et al., 2017) allow looking at any time. All pay for the freedom with some power at a given final sample size.

Switching to Bayesian statistics does not escape this. "Stop when the probability that B beats A exceeds 95%" is still a stopping rule, and in the A/A simulation below it declares several times as many false winners as a single look. The [Beta posterior](../beta-uncertainty/) is a good way to report a test, not a licence to peek.

```python
looks, n_exp = 20, 100_000
walk = np.cumsum(rng.standard_normal((n_exp, looks)), axis=1)
z_path = walk / np.sqrt(np.arange(1, looks + 1))                    # z-statistic after each batch, under H0

assert abs((np.abs(z_path[:, -1]) > Z95).mean() - 0.05) < 0.005    # one look at the end: 5%
peek_fpr = (np.abs(z_path) > Z95).any(axis=1).mean()
assert 0.22 < peek_fpr < 0.28                                       # stop at the first p < 0.05: ~25%

c = np.quantile(np.abs(z_path).max(axis=1), 0.95)                   # a constant boundary for 20 looks
assert 2.62 < c < 2.72
fresh = np.cumsum(rng.standard_normal((n_exp, looks)), axis=1) / np.sqrt(np.arange(1, looks + 1))
assert abs((np.abs(fresh) > c).any(axis=1).mean() - 0.05) < 0.005   # back to 5% on new data

shrink = np.sqrt(looks / np.arange(1, looks + 1))                   # O'Brien-Fleming: c_obf * shrink[k] at look k
c_obf = np.quantile((np.abs(z_path) / shrink).max(axis=1), 0.95)
assert 2.07 < c_obf < 2.17 and shrink[0] * c_obf > 9                # ~2.12 at the end, ~9.5 at the first look

def prob_b_beats_a(c_a, c_b, n, draws=4_000):                       # Beta(1, 1) priors, Monte Carlo
    return (rng.beta(1 + c_b, 1 + n - c_b, draws) > rng.beta(1 + c_a, 1 + n - c_a, draws)).mean()


batch, false_win_once, false_win_peek = 1_000, 0, 0
for _ in range(400):                                                # A/A tests, Bayesian stopping rule
    c_a = np.cumsum(rng.binomial(batch, 0.05, looks))
    c_b = np.cumsum(rng.binomial(batch, 0.05, looks))
    probs = [prob_b_beats_a(c_a[k], c_b[k], batch * (k + 1)) for k in range(looks)]
    false_win_once += probs[-1] > 0.95
    false_win_peek += max(probs) > 0.95
assert false_win_peek > 2.5 * false_win_once                         # peeking hurts here too
```

## CUPED: same answer, fewer users
Much of a metric's variance has nothing to do with the experiment: heavy users spend a lot in both arms. Knowing what each user spent **before** the experiment, you can subtract the predictable part. CUPED (Deng, Xu, Kohavi and Walker, 2013) does this with a pre-experiment covariate \(X\):

\[ Y_{\text{adj}} = Y - \theta\,(X - \bar X), \qquad \theta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)}. \]

Because \(X\) was measured before assignment, it has the same mean in both arms in expectation, so subtracting it leaves the expected difference alone. It does remove variance: the adjusted metric has variance \((1 - \rho^2)\operatorname{Var}(Y)\), with \(\rho\) the correlation of \(X\) and \(Y\). With \(\rho^2 \approx 0.7\) the standard error falls to about 55% of its old size, and the same power needs under a third of the users. The estimate almost exactly matches the treatment coefficient in a regression of \(Y\) on the treatment flag and the centred covariate.

"Before" is the whole condition. A covariate measured during the experiment can be moved by the treatment, and subtracting it subtracts part of the effect. The simulation also tries a covariate that tracks the outcome itself: the estimated effect almost vanishes.

```python
def experiment(n_users, effect=1.0):
    x = rng.gamma(2.0, 10.0, n_users)                                # pre-period spend
    t = rng.integers(0, 2, n_users)                                  # random assignment
    y = 5 + 0.9 * x + rng.normal(0, 8, n_users) + effect * t         # spend during the test
    return x, t, y


def diff(y, t):
    return y[t == 1].mean() - y[t == 0].mean()


def cuped(y, x, t):
    theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)                   # same ddof on both sides
    return diff(y - theta * (x - x.mean()), t)


plain, adjusted, leaky = [], [], []
for _ in range(500):
    x, t, y = experiment(4_000)
    during = y + rng.normal(0, 4, len(y))                            # measured during the test: moved by treatment
    plain.append(diff(y, t))
    adjusted.append(cuped(y, x, t))
    leaky.append(cuped(y, during, t))
plain, adjusted, leaky = map(np.array, (plain, adjusted, leaky))

rho2 = np.corrcoef(x, y)[0, 1] ** 2
assert 0.65 < rho2 < 0.78
assert abs(plain.mean() - 1.0) < 0.1 and abs(adjusted.mean() - 1.0) < 0.05   # both unbiased
assert abs(adjusted.std() / plain.std() - math.sqrt(1 - rho2)) < 0.06       # SE shrinks by sqrt(1 - rho^2)
assert leaky.mean() < 0.2                                                   # the effect is mostly erased

design = np.column_stack([np.ones_like(y), t, x - x.mean()])                # regression cross-check
beta = np.linalg.lstsq(design, y, rcond=None)[0]
assert abs(beta[1] - cuped(y, x, t)) < 0.02
```

## Many experiments on the same users
With dozens of teams, giving each test its own slice of traffic leaves too few users to power any of them. The standard answer is **overlapping layers** (Tang et al., 2010): experiments are grouped into layers, and each layer hashes the user id with its own salt, so a user's bucket in one layer is unrelated to their bucket in another, and experiments in different layers each use all the traffic. Hashing is also deterministic, so a returning user keeps their variant with no stored state. Two experiments that share a salt and the same bucket range are perfectly confounded: everyone in one treatment is in the other.

```python
def bucket(user_id, salt, buckets=100):
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


users = range(100_000)
in_b = np.array([bucket(u, "ranking-v2") < 50 for u in users])     # experiment 1: A vs B
in_d = np.array([bucket(u, "new-layout") < 50 for u in users])     # experiment 2: C vs D
assert abs(in_b.mean() - 0.5) < 0.01 and abs(in_d.mean() - 0.5) < 0.01
assert abs((in_b & in_d).mean() - 0.25) < 0.01                      # independent: a quarter in both
table = [[(~in_b & ~in_d).sum(), (~in_b & in_d).sum()], [(in_b & ~in_d).sum(), (in_b & in_d).sum()]]
assert stats.chi2_contingency(table)[1] > 0.001
assert bucket(42, "ranking-v2") == bucket(42, "ranking-v2")          # deterministic

same_salt = np.array([bucket(u, "shared") < 50 for u in users])
assert (same_salt == np.array([bucket(u, "shared") < 50 for u in users])).all()   # fully confounded
```

Overlap makes the design a 2×2 **factorial**, and the changes might interact: a new ranking model and a new layout might each help alone and hurt together. The interaction is a difference of differences, the effect of B with D on minus its effect with D off. Suppose each cell has 1,000 users:

| cell | conversions | rate |
|---|---|---|
| A, C | 100 | 10% |
| A, D | 120 | 12% |
| B, C | 110 | 11% |
| B, D | 160 | 16% |

B adds 4 points with D and 1 point without it, so the interaction is \((16 - 12) - (11 - 10) = 3\) points. The cells are independent, so its variance is the sum of the four cell variances \(p(1-p)/n\): a standard error of 2.07 points, \(z = 1.45\), p-value 0.15, and a 95% interval of about \(-1.1\) to \(+7.1\) points. Plausible, but not distinguishable from noise. Its standard error is twice that of B's main effect averaged over C and D, so detecting an interaction takes four times the data of a main effect of the same size.

A linear regression of conversion on B, D and their product gives the same number as the product's coefficient, and its heteroskedasticity-robust (HC0) standard error equals the cell formula exactly.

```python
cells = {(0, 0): (100, 1_000), (0, 1): (120, 1_000), (1, 0): (110, 1_000), (1, 1): (160, 1_000)}
rate = {k: c / m for k, (c, m) in cells.items()}
interaction = (rate[1, 1] - rate[0, 1]) - (rate[1, 0] - rate[0, 0])
se = math.sqrt(sum(r * (1 - r) / cells[k][1] for k, r in rate.items()))
z = interaction / se
assert math.isclose(interaction, 0.03) and round(se, 4) == 0.0207
main_b = ((rate[1, 0] - rate[0, 0]) + (rate[1, 1] - rate[0, 1])) / 2       # B's effect averaged over C/D
se_main = 0.5 * math.sqrt(sum(r * (1 - r) / cells[k][1] for k, r in rate.items()))
assert math.isclose(main_b, 0.025) and math.isclose(se / se_main, 2)       # 2x the SE, so 4x the data
assert round(z, 2) == 1.45 and round(2 * stats.norm.sf(z), 2) == 0.15
assert round(interaction - Z95 * se, 3) == -0.011 and round(interaction + Z95 * se, 3) == 0.071

rows = [(b, d, 1.0 if i < conv else 0.0) for (b, d), (conv, m) in cells.items() for i in range(m)]
b, d, y = map(np.array, zip(*rows))
X = np.column_stack([np.ones_like(y), b, d, b * d])
coef = np.linalg.lstsq(X, y, rcond=None)[0]
assert np.allclose(coef, [0.10, 0.01, 0.02, 0.03])
bread = np.linalg.inv(X.T @ X)
hc0 = bread @ (X.T * (y - X @ coef) ** 2) @ X @ bread
assert math.isclose(math.sqrt(hc0[3, 3]), se)
```

If an interaction is real and harmful, turn off that one combination; neither experiment is wrong, the two changes conflict. Interactions between unrelated changes are usually small, which is why layers work; changes to the same surface go in one layer with mutually exclusive buckets.

## When users affect each other
Everything above assumes one user's outcome does not depend on another's assignment. Social features and marketplaces break this: a treatment that helps riders book faster also takes drivers from control riders, so the measured gain overstates a full launch. The fixes randomize larger units: **clusters** (friend groups, cities) or **time windows**, switching a whole market between arms in alternating periods (switchback tests). Power drops sharply, but the estimate means what you want.

## Pitfalls
- **No MDE, no plan.** Fix the minimum effect worth shipping and the sample size before launch.
- **Novelty and primacy effects.** Users click new things because they are new, or avoid them because they are unfamiliar. Plot the effect by day since first exposure, and cover at least a full week.
- **Guardrail metrics.** Check metrics you must not hurt, such as latency, crashes and revenue, in every test. A click-through gain with a revenue loss is often clickbait.
- **Many metrics, many chances.** Twenty independent metrics at \(\alpha = 0.05\) give one false win per null test on average, and a 64% chance of at least one. Name one primary metric in advance; correct for multiplicity or treat the rest as exploratory.
- **Segments can reverse the total.** When the segment mix differs between arms, a change can win in every segment and lose overall (Simpson's paradox). Randomization keeps the mix equal only for segments fixed before exposure and a split held constant; a traffic split changed mid-test, or segments defined by behaviour after exposure, break it, and post-hoc slices also multiply the chances of a false win.
- **Heavy tails.** A mean can rise while the median falls. Cap extreme values with a rule fixed in advance, and look at the distribution.
- **Statistically significant is not important.** With a million users per arm, a 0.1-point change on a 5% rate is significant. Read the confidence interval against the MDE, as in [Is this coin fair?](../is-this-coin-fair/).

When you cannot randomize at all, see [causal inference](../causal-inference/).

This post draws on R. Kohavi, D. Tang and Y. Xu, Trustworthy Online Controlled Experiments (Cambridge University Press, 2020) and Evan Miller's essay "How Not To Run an A/B Test". CUPED is from A. Deng, Y. Xu, R. Kohavi and T. Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (WSDM 2013); overlapping layers from D. Tang, A. Agarwal, D. O'Brien and M. Meyer, "Overlapping Experiment Infrastructure" (KDD 2010); always-valid p-values from R. Johari, P. Koomen, L. Pekelis and D. Walsh, "Peeking at A/B Tests" (KDD 2017); the group sequential designs from S. J. Pocock (1977) and P. C. O'Brien and T. R. Fleming (1979).
