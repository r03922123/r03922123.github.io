# Simulation-based evaluation: testing a change before it meets real users

In a toy shop with three offers, replaying 5,000 logged rows through a reward model that is too simple underrates a new discount policy by 0.24, with a spread of only 0.02 over 300 repeats. A doubly robust estimator built on the same model lands on the truth. Simulation-based evaluation estimates the effect of a change before (or instead of) randomizing it, using a model of how the population behaves. An A/B test is the most trustworthy way to learn what a change does, but it is slow, it exposes real people to a change that may be bad, and some changes cannot be randomized at all. Simulation is a filter in front of live experiments: it narrows the candidates and flags obvious harm, and the survivors still go to a live test.

Each method below is only as good as one specific assumption. To see which, I run every estimator inside a toy world where the true answer is known, so I can tell when it is right and when it is quietly wrong.

## Three tools, three assumptions
| method | needs | answers | rests on |
|---|---|---|---|
| off-policy evaluation | logs with known action probabilities | value of a new policy, from old data | the old policy sometimes took the new policy's actions |
| synthetic control | one treated unit, many untreated ones, a long history | effect of a change that has already happened somewhere | a weighted mix of untreated units tracks the treated one |
| population simulation | a behavioral model | outcomes for any change, sliced any way | the behavioral model holds where you use it |

They are not rungs of one ladder. Off-policy evaluation is the cheapest and most principled when its condition holds. Synthetic control is retrospective: it measures a change already made in one region or market. A full simulation can answer anything, and can be confidently wrong.

## Replaying the logs: off-policy evaluation
Suppose a system chose an action \(a\) for each user with context \(x\), with a known probability \(\mu(a \mid x)\), and logged the reward \(r\). The goal is the average reward \(V(\pi)\) that a different policy \(\pi\) would have earned.

The **direct method (DM)** fits a reward model \(\hat q(x, a)\) and averages \(\hat q(x, \pi(x))\). It has low variance but inherits every error of the model.

**Inverse propensity scoring (IPS)** keeps the logged rows where the old action matches the new policy's choice and weights each by \(1/\mu(a \mid x)\), so rarely chosen actions count for more:

\[ \hat V_{\text{IPS}} = \frac{1}{n} \sum_{i=1}^{n} \frac{\mathbb{1}[a_i = \pi(x_i)]}{\mu(a_i \mid x_i)}\, r_i . \]

It is unbiased whenever \(\mu\) is right and positive, but small propensities make it noisy. It is the same correction as in [position bias](/posts/position-bias/).

**Doubly robust (DR)** starts from the model's prediction and adds an IPS-weighted correction for the model's error:

\[ \hat V_{\text{DR}} = \frac{1}{n} \sum_{i=1}^{n} \Big[ \hat q(x_i, \pi(x_i)) + \frac{\mathbb{1}[a_i = \pi(x_i)]}{\mu(a_i \mid x_i)} \big( r_i - \hat q(x_i, a_i) \big) \Big] . \]

It stays unbiased if either the propensities or the reward model is correct (strictly, when \(\hat q\) is fitted on data separate from the rows it corrects; fitting in-sample, as below, adds a small bias that is negligible here), and when the model is roughly right the correction is small, so its variance is usually well below IPS.

The toy world is a shop with three offers: nothing, a small discount, a big discount. Each user has one feature \(x\), say how engaged they are. Discounts pay off only for less engaged users, with a step at \(x = 0\) and \(x = -1\) that a linear model cannot represent. The logging policy mostly offers nothing. The new policy gives big discounts below \(x = -1\) and small ones below 0.

```python
import numpy as np
from scipy import optimize, stats

rng = np.random.default_rng(0)


def mean_reward(x, a):                      # true expected profit per user, unknown to the estimators
    lift = np.select([a == 0, a == 1, a == 2],
                     [0.0, np.where(x < 0, 0.6, -0.2), np.where(x < -1, 1.5, -0.6)])
    return 2.0 + 0.5 * x + lift


def logging_probs(x):                       # mu(a | x) of the old policy, known and logged
    logits = np.stack([np.ones_like(x), 0.3 * x, np.full_like(x, -0.5)], axis=1)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def new_policy(x):
    return np.where(x < -1, 2, np.where(x < 0, 1, 0))


def log_data(n, probs=logging_probs):
    x = rng.normal(size=n)
    p = probs(x)
    a = (rng.random(n)[:, None] > p.cumsum(1)).sum(1)        # sample an action per row
    r = mean_reward(x, a) + rng.normal(0, 1.0, n)
    return x, a, r, p


x_pop = rng.normal(size=1_000_000)
truth = mean_reward(x_pop, new_policy(x_pop)).mean()         # what a perfect live test would find
old_value = (logging_probs(x_pop) * np.stack(
    [mean_reward(x_pop, np.full(x_pop.size, k)) for k in range(3)], 1)).sum(1).mean()
assert truth > old_value + 0.3                               # the new policy really is better
```

Each estimator sees 5,000 logged rows, repeated 300 times to measure bias and spread:

```python
def fit_q(x, a, r):                          # per-action linear reward model: deliberately too simple
    coefs = [np.polyfit(x[a == k], r[a == k], 1) for k in range(3)]
    return lambda xs, acts: np.select([acts == k for k in range(3)],
                                      [np.polyval(c, xs) for c in coefs])


def off_policy_estimates(x, a, r, p):
    pi = new_policy(x)
    q = fit_q(x, a, r)
    w = (a == pi) / p[np.arange(len(a)), a]
    dm = q(x, pi).mean()
    ips = (w * r).mean()
    dr = (q(x, pi) + w * (r - q(x, a))).mean()
    return dm, ips, dr


runs = np.array([off_policy_estimates(*log_data(5000)) for _ in range(300)])
bias, spread = runs.mean(0) - truth, runs.std(0)
dm_bias, ips_bias, dr_bias = bias
dm_sd, ips_sd, dr_sd = spread

assert abs(dm_bias) > 5 * dm_sd               # DM: precise, and precisely wrong
assert abs(ips_bias) < 3 * ips_sd / np.sqrt(300) and abs(dr_bias) < 3 * dr_sd / np.sqrt(300)
assert dr_sd < 0.6 * ips_sd                   # DR: unbiased like IPS, with about half the spread
```

| estimator | bias | standard deviation |
|---|---|---|
| direct method | −0.24 | 0.02 |
| IPS | ≈ 0 | 0.06 |
| doubly robust | ≈ 0 | 0.03 |

The direct method has the smallest spread and is the only wrong one, the dangerous combination. Its linear model cannot see the steps, so it underrates the discounts. Doubly robust uses the same bad model and still lands on the truth, because the weighted residuals repair the model's error.

### What the logs cannot tell you
IPS and DR need one thing from the old policy: every action the new policy takes must have had a nonzero chance of being logged in that context. This is called **support** (or overlap). Suppose the old policy never offered the big discount to engaged users (\(x > 0.5\)), and I ask what giving everyone the big discount would earn:

```python
def cautious_probs(x):                       # never offers the big discount when x > 0.5
    p = logging_probs(x)
    p[x > 0.5, 2] = 0
    return p / p.sum(1, keepdims=True)


x, a, r, p = log_data(200_000, cautious_probs)
everyone_big = mean_reward(x, np.full(x.size, 2)).mean()
w = (a == 2) / p[np.arange(x.size), a]
ips_all_big = (w * r).mean()
unsupported = (p[:, 2] == 0).mean()

assert 0.25 < unsupported < 0.35             # nearly a third of users were never eligible
assert abs(ips_all_big - everyone_big) > 0.4 # IPS is far off even with 200,000 rows
```

Nearly a third of users never had a chance of the big discount, and IPS reports 1.14 against a true 1.73. No amount of data fixes this: the logs hold no evidence about those users under that action, and IPS silently counts them as zero. Measure the fraction of the new policy's decisions outside the logged support before trusting an estimate. Keeping a little randomization in the old policy is what makes off-policy evaluation possible later.

## Synthetic control: when you cannot randomize
Some changes apply to a whole region at once, such as a new fee in one market, so there is no randomized control group. Synthetic control, introduced by Abadie and Gardeazabal (2003) and developed by Abadie, Diamond and Hainmueller (2010), builds one. It finds nonnegative weights, summing to 1, over untreated "donor" units so that their weighted average tracks the treated unit before the change. After the change, the gap between the treated unit and its synthetic twin estimates the effect.

The change must already have happened somewhere, so before a full rollout this means piloting in one region and measuring the pilot.

Below, 21 regions follow two shared trends with different sensitivities. Region 0 grows like the fastest-growing few, and is treated after period 30 with a true effect of +1.0. A plain before/after comparison absorbs the trend, and difference-in-differences assumes region 0 moves like the donors' average, which it does not.

```python
T, T0, J, effect = 40, 30, 20, 1.0
t = np.arange(T)
factors = np.stack([t / 10, np.sin(t / 3)])                 # shared trends
loadings = rng.uniform(0, 2, size=(J + 1, 2))
level = rng.normal(5, 1, J + 1)
fastest = 1 + np.argsort(loadings[1:, 0])[-4:]               # the four donors with the steepest trend
loadings[0], level[0] = loadings[fastest].mean(0), level[fastest].mean()   # region 0 grows like them
y = level[:, None] + loadings @ factors + rng.normal(0, 0.2, (J + 1, T))
y[0, T0:] += effect


def sc_weights(treated_pre, donors_pre):
    k = len(donors_pre)
    fit = optimize.minimize(lambda w: ((treated_pre - w @ donors_pre) ** 2).sum(), np.full(k, 1 / k),
                            method="SLSQP", bounds=[(0, 1)] * k,
                            constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return fit.x


def gap(i):                                   # unit i minus its synthetic twin built from the others
    others = [j for j in range(J + 1) if j != i]
    w = sc_weights(y[i, :T0], y[others, :T0])
    return y[i] - w @ y[others]


w_fast = sc_weights(y[0, :T0], y[1:, :T0])[fastest - 1].sum()
assert w_fast > 0.4
g = gap(0)
sc_est = g[T0:].mean()
before_after = y[0, T0:].mean() - y[0, :T0].mean()
did = before_after - (y[1:, T0:].mean() - y[1:, :T0].mean())

assert abs(sc_est - effect) < 0.25
assert abs(did - effect) > 2 * abs(sc_est - effect) and before_after > 2 * effect
assert np.sqrt((g[:T0] ** 2).mean()) < 0.35  # the twin tracks the pre-period to within the noise
```

The synthetic twin puts about half its weight on the four fast-growing donors and estimates the effect at 1.19, close to the true 1.0. Difference-in-differences says 2.87 and before/after says 3.69, because both credit region 0's own faster growth to the change.

With one treated unit there is no standard error in the usual sense. The 2010 paper's answer is a **placebo test**: pretend each donor was the treated one, rebuild its twin from the rest, and compare how much worse each fit gets after period 30 than before. If the real treated unit's post/pre error ratio is the largest of 21, the chance of that under "no effect" is \(1/21 \approx 0.048\):

```python
def rmspe_ratio(i):
    e = gap(i)
    return np.sqrt((e[T0:] ** 2).mean() / (e[:T0] ** 2).mean())


ratios = np.array([rmspe_ratio(i) for i in range(J + 1)])
placebo_p = (ratios >= ratios[0]).mean()
assert ratios.argmax() == 0 and np.isclose(placebo_p, 1 / (J + 1))
assert ratios[0] > 2 * np.sort(ratios)[-2]       # and by a wide margin
```

Region 0's ratio is the largest of the 21 (about 5.9, against 2.6 for the runner-up), so the placebo p-value is the smallest possible, 1/21.

Synthetic control fails when the treated unit lies outside what any convex mix of donors can reproduce (it is the largest market, or the first of its kind), when the pre-period is short or unstable, or when the change spills over into the donors.

## Simulating the population
The most flexible option is to simulate individual people, apply the change, and aggregate. The behavioral model can be a hand-built rule, a model learned from logs, or an LLM-driven agent; cost and the risk of being wrong both grow with its freedom.

Here, buyers have a willingness to pay. 85% are regular buyers with a wide spread, and 15% are budget buyers clustered around 10.5, just above the current price of 10. The question is whether to raise the price to 11.

```python
n = 400_000
budget = rng.random(n) < 0.15
wtp = np.where(budget, rng.lognormal(np.log(10.5), 0.15, n), rng.lognormal(np.log(20), 0.5, n))


def simulate(price):
    buy = wtp >= price
    return {"conv": buy.mean(), "revenue": price * buy.mean(),
            "budget_conv": buy[budget].mean(), "regular_conv": buy[~budget].mean()}


now, raised = simulate(10), simulate(11)
assert raised["revenue"] > now["revenue"]                              # the headline says ship it
assert raised["budget_conv"] < 0.65 * now["budget_conv"]               # budget buyers lose over a third
assert raised["regular_conv"] > 0.95 * now["regular_conv"]
```

Revenue per visitor rises by about 2%, while budget buyers' conversion falls from 63% to 38%. An average that looks good can hide a segment that is clearly harmed, so slice simulated results the same way you would slice a live test.

The second lesson is about range. Real behavioral models are fitted to history, and history covers a narrow range of conditions. Fit a straight demand line to past price tests between 9.50 and 10.50 and it is excellent there. At a price of 15 it predicts that revenue falls when, in this world, it rises:

```python
past_prices = np.linspace(9.5, 10.5, 5)
slope, intercept = np.polyfit(past_prices, [simulate(p)["conv"] for p in past_prices], 1)
fitted_conv = lambda price: intercept + slope * price

assert abs(fitted_conv(10.3) - simulate(10.3)["conv"]) < 0.002        # inside the range: excellent
assert 15 * fitted_conv(15) < now["revenue"] < simulate(15)["revenue"] # outside: the decision flips
```

Check a simulator against past changes it did not see, and trust it only near the conditions it was calibrated on. Strategic responses are the worst case: sellers, advertisers or fraudsters who learn the new rule change their behavior, and no model fitted to the old rule contains that.

## Sizing the live test by simulation
A simulation also says how big the follow-up experiment must be, and simulating that experiment checks the formula. Taking the simulator's predicted conversions at prices 10 and 11, the normal-approximation sample size for 80% power at a two-sided 5% level, on revenue per visitor, is checked against 4,000 simulated A/B tests, plus 4,000 A/A tests for the false-alarm rate:

```python
p0, p1 = now["conv"], raised["conv"]
diff = 11 * p1 - 10 * p0
var = 11 ** 2 * p1 * (1 - p1) + 10 ** 2 * p0 * (1 - p0)
n_arm = int(np.ceil((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) ** 2 * var / diff ** 2))


def z_stat(price_b, cb, ca):
    se = np.sqrt((price_b ** 2 * cb * (1 - cb) + 100 * ca * (1 - ca)) / n_arm)
    return (price_b * cb - 10 * ca) / se


ctrl = rng.binomial(n_arm, p0, 4000) / n_arm
power = (np.abs(z_stat(11, rng.binomial(n_arm, p1, 4000) / n_arm, ctrl)) > 1.96).mean()
false_alarm = (np.abs(z_stat(10, rng.binomial(n_arm, p0, 4000) / n_arm, ctrl)) > 1.96).mean()

assert 8_000 < n_arm < 11_000
assert abs(power - 0.80) < 0.03 and abs(false_alarm - 0.05) < 0.015
```

About 10,000 visitors per arm suffice, but only if the simulator's effect size is right. If it overstates the lift, the live test is underpowered. Size for a somewhat smaller effect than the simulator predicts. More on the live side in [A/B testing](../ab-testing/).

## When to use it, and pitfalls
- **Off-policy evaluation** ranks many candidate policies cheaply when logs record action probabilities and candidates stay within logged support. Prefer doubly robust and report the spread.
- **Synthetic control** fits a change that cannot be randomized and has already happened in one unit, with a long, stable history and comparable donors. Always run the placebo test.
- **Population simulation** answers questions about mechanisms (who gains, who loses, what breaks at the extremes) and stress-tests rare segments a live split would barely reach.
- **Support is not optional.** If the old policy never did it, the logs cannot evaluate it.
- **A low-variance estimate can be the wrong one**, like the direct method above.
- **Validate the simulator on held-out history**, stay near its calibration range, and slice before trusting an average.
- **Confirm live.** A simulation raises confidence in a change; a smaller, well-powered experiment confirms it. For the causal groundwork see [causal inference](../causal-inference/), and for how offline and online evaluation fit together see [evaluating ML systems](/posts/evaluating-ml-systems/).

Every world here is a toy I built so the true answer is known, so the numbers show how each method behaves when its assumption holds or breaks, not how often it holds on real logs or markets. Two questions stay open before a live test: whether the new policy stays inside logged support, which the logs can measure, and whether the behavioral model holds outside its calibration range, which only held-out history or the live test can check.

IPS goes back to Horvitz and Thompson (1952); the doubly robust estimator for policy evaluation follows Dudík, Langford and Li (2011); synthetic control and its placebo test are from Abadie and Gardeazabal (2003) and Abadie, Diamond and Hainmueller (2010). Athey and Imbens (2019), "Machine Learning Methods That Economists Should Know About", is a good bridge between these methods and ML.
