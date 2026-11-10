# The exponential distribution: waiting for the next event

Rides are requested at an average of 6 per hour, independently of each other. How long until the next one? The [Poisson distribution](../poisson-distribution/) answers a counting question, how many events land in a fixed window. The exponential distribution answers the matching timing question, how long until the first event. Same process, other side of the coin.

One correction up front, because it is a common slip: the exponential is a **continuous** distribution, so it has a probability density function (PDF), not a probability mass function (PMF). A waiting time can be 3.7 minutes or 3.7001 minutes, and the chance of hitting any single exact value is zero. Probabilities come from areas under the density, and the density itself can be larger than 1.

## From counts to waiting times
In a Poisson process with rate \(\lambda\) events per unit time, the number of events in any window of length \(t\) is Poisson with mean \(\lambda t\), and counts in disjoint windows are independent. Let \(T\) be the time until the first event. "The first event comes after \(t\)" is the same statement as "no events in \([0, t]\)", so

\[ P(T > t) = P(N(t) = 0) = \frac{(\lambda t)^0 e^{-\lambda t}}{0!} = e^{-\lambda t}, \qquad t \ge 0 . \]

A tempting shortcut is to multiply \(e^{-\lambda}\) (no event in one unit) by itself \(t\) times. That is right for whole numbers of units, by independence, but says nothing about \(t = 2.5\); the window-of-length-\(t\) argument covers every \(t\) at once.

The survival function \(S(t) = P(T > t)\) gives the cumulative distribution function and, by differentiating, the density:

\[ F(t) = P(T \le t) = 1 - e^{-\lambda t}, \qquad f(t) = \frac{d}{dt} F(t) = \lambda e^{-\lambda t}, \qquad t \ge 0 . \]

With \(\lambda = 6\) per hour, \(f(0) = 6\) per hour: a density above 1 is fine, because what must equal 1 is the area, \(\int_0^\infty \lambda e^{-\lambda t}\,dt = 1\).

A useful way to see that this is the right continuous limit: chop time into slots of width \(\Delta\), and let each slot hold an event with probability \(\lambda \Delta\). The slot of the first event is geometric, and as \(\Delta \to 0\) the waiting time becomes exponential. The code builds exactly that and compares it with the formulas.

```python
import numpy as np
from scipy import integrate, stats

rng = np.random.default_rng(0)
lam = 6.0                                   # events per hour
dt = 1e-4                                   # slot width in hours

# Discrete approximation: one Bernoulli(lam * dt) trial per slot, wait for the first success.
slots = rng.geometric(lam * dt, size=200_000)
T_discrete = slots * dt

for t in [0.05, 0.1, 0.25, 0.5]:
    empirical = (T_discrete > t).mean()
    assert abs(empirical - np.exp(-lam * t)) < 0.005, (t, empirical)

# The density integrates to 1 even when it exceeds 1 somewhere.
f = lambda t, lam=lam: lam * np.exp(-lam * t)
area, _ = integrate.quad(f, 0, np.inf)
assert f(0) == 6.0 and abs(area - 1) < 1e-10

# scipy agrees, but note its parametrization: scale = 1 / rate.
dist = stats.expon(scale=1 / lam)
for t in [0.05, 0.3, 1.0]:
    assert np.isclose(dist.pdf(t), f(t)) and np.isclose(dist.sf(t), np.exp(-lam * t))

# The worked example: P(wait > 15 minutes) at 6 rides per hour.
p_wait = np.exp(-lam * 0.25)
assert np.isclose(p_wait, 0.2231, atol=1e-4)
```

So at 6 rides per hour, about 22% of the time you wait more than 15 minutes.

## Mean, variance and median
For a non-negative variable the mean is the area under the survival curve, which makes the exponential's mean a one-liner:

\[ \mathbb{E}[T] = \int_0^\infty S(t)\,dt = \int_0^\infty e^{-\lambda t}\,dt = \frac{1}{\lambda} . \]

A similar integral gives \(\mathbb{E}[T^2] = 2/\lambda^2\), so \(\operatorname{Var}(T) = 1/\lambda^2\) and the standard deviation equals the mean. Setting \(S(t) = 1/2\) gives the median, \(\ln 2 / \lambda \approx 0.69/\lambda\).

| Quantity | Formula | 6 per hour |
|---|---|---|
| Mean wait | \(1/\lambda\) | 10 min |
| Standard deviation | \(1/\lambda\) | 10 min |
| Median wait | \(\ln 2/\lambda\) | 6.9 min |
| \(P(T > 15\text{ min})\) | \(e^{-\lambda t}\) | 0.223 |

The median sits well below the mean: most waits are short, and a long right tail of slow ones pulls the average up. If you report one number for a skewed waiting time, say which one it is.

## Sampling by inverse transform
Since \(F\) has a closed-form inverse, sampling is one line. If \(U\) is uniform on \((0, 1)\), then \(F^{-1}(U) = -\ln(1 - U)/\lambda\) is exponential, and because \(1 - U\) is also uniform, \(-\ln U / \lambda\) works in exact arithmetic too. In code, prefer the \(1 - U\) form: generators such as numpy's `random()` return values in \([0, 1)\), so \(U = 0\) is possible and \(-\ln 0\) is infinite, while \(1 - U\) is never 0. Production libraries often use faster methods (numpy's generator uses a ziggurat algorithm), but this is the standard first example of the inverse-transform method.

```python
lam = 6.0
U = rng.random(500_000)
T = -np.log(1.0 - U) / lam                  # inverse transform; 1 - U lies in (0, 1]

assert abs(T.mean() - 1 / lam) < 0.002            # mean 1/lambda (10 minutes)
assert abs(T.std() - 1 / lam) < 0.002             # sd equals the mean
assert abs(np.median(T) - np.log(2) / lam) < 0.002
assert stats.kstest(T, "expon", args=(0, 1 / lam)).pvalue > 0.01

# numpy's generator also takes the SCALE (mean), not the rate.
T_np = rng.exponential(scale=1 / lam, size=500_000)
assert abs(T_np.mean() - 1 / lam) < 0.002
```

## Memorylessness and constant hazard
The defining property: having already waited \(s\) tells you nothing about the remaining wait.

\[ P(T > s + t \mid T > s) = \frac{e^{-\lambda (s + t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(T > t) . \]

Ten minutes into a wait for a ride, the expected remaining wait is still 10 minutes. Among continuous distributions on \([0, \infty)\), the exponential is the only one with this property (the geometric plays the same role among discrete ones), because \(S(s + t) = S(s)\,S(t)\) for all \(s, t\) forces \(S\) to be an exponential function.

An equivalent statement uses the **hazard rate**, the instantaneous event rate among those still waiting:

\[ h(t) = \frac{f(t)}{S(t)} = \frac{\lambda e^{-\lambda t}}{e^{-\lambda t}} = \lambda . \]

A constant hazard is the modelling assumption to check. Light bulbs that burn out at random, radioactive decay and arrivals from a large population are close to it. Machines that wear out (rising hazard) and users who are less likely to churn the longer they stay (falling hazard) are not; the Weibull distribution adds a shape parameter for exactly those cases.

```python
T = rng.exponential(scale=1 / lam, size=1_000_000)
for s, t in [(1 / 6, 1 / 6), (0.5, 0.1), (0.2, 0.3)]:
    survivors = T[T > s]
    conditional = (survivors > s + t).mean()
    assert abs(conditional - np.exp(-lam * t)) < 0.01, (s, t, conditional)
    assert abs((survivors - s).mean() - 1 / lam) < 0.005    # remaining wait is still 1/lambda

# Contrast: a uniform wait on [0, 20] minutes is NOT memoryless.
W = rng.uniform(0, 20, size=1_000_000)
assert abs((W[W > 10] - 10).mean() - 5) < 0.05           # already waited 10: only ~5 left, not 10
```

## Racing clocks: the minimum of exponentials
Suppose three independent event types race: ride requests at rate \(\lambda_1\), cancellations at \(\lambda_2\), support tickets at \(\lambda_3\). The time until the first of any of them is the minimum of three exponentials, and

\[ P(\min_i T_i > t) = \prod_i e^{-\lambda_i t} = e^{-(\sum_i \lambda_i) t}, \]

so the minimum is exponential with the summed rate. The chance that type \(i\) wins the race is \(\lambda_i / \sum_j \lambda_j\), and which one wins is independent of when the race ends. Together with memorylessness, this is why merging independent Poisson streams gives a Poisson stream, and it is the core of simulating continuous-time Markov chains: draw the time to the next event from the total rate, then pick which event by the rates.

Going the other way, the gaps between successive events of a Poisson process are independent exponentials, so the time of the \(n\)-th event is a sum of \(n\) of them, a Gamma (Erlang) distribution with shape \(n\) and rate \(\lambda\). The code checks both facts, and closes the loop with the Poisson count: summing exponential gaps and counting how many fit in one hour reproduces Poisson(\(\lambda\)).

```python
rates = np.array([4.0, 1.5, 0.5])
clocks = rng.exponential(scale=1 / rates, size=(400_000, 3))
first_time, winner = clocks.min(axis=1), clocks.argmin(axis=1)

assert abs(first_time.mean() - 1 / rates.sum()) < 0.001
assert stats.kstest(first_time, "expon", args=(0, 1 / rates.sum())).pvalue > 0.01
assert np.allclose(np.bincount(winner) / len(winner), rates / rates.sum(), atol=0.003)
for k in range(3):                                   # the winner does not depend on when the race ends
    assert abs(first_time[winner == k].mean() - 1 / rates.sum()) < 0.002

# n-th arrival time is Gamma(n, rate); counts in one hour are Poisson(lambda).
lam = 6.0
gaps = rng.exponential(scale=1 / lam, size=(100_000, 40))
arrivals = gaps.cumsum(axis=1)
fifth = arrivals[:, 4]
assert stats.kstest(fifth, "gamma", args=(5, 0, 1 / lam)).pvalue > 0.01
counts = (arrivals <= 1.0).sum(axis=1)               # 40 gaps is far more than ever fit in one hour
observed = np.bincount(counts, minlength=15)[:15] / len(counts)
assert np.allclose(observed, stats.poisson.pmf(np.arange(15), lam), atol=0.005)
```

## Estimating the rate, with and without censoring
Given complete waiting times \(t_1, \dots, t_n\), the log-likelihood is \(n \ln \lambda - \lambda \sum_i t_i\), and setting its derivative to zero gives

\[ \hat\lambda = \frac{n}{\sum_i t_i} = \frac{1}{\bar t}, \]

events divided by total time watched. (The general recipe is in [MLE and MAP](../statistical-estimation-mle-map/).)

Real data are rarely complete. A study ends, a user is still subscribed, a machine is still running: for those units we only know \(T > c_i\). That is **right censoring**, and each censored unit contributes \(S(c_i) = e^{-\lambda c_i}\) to the likelihood instead of \(f(t_i)\). The estimate keeps the same shape:

\[ \hat\lambda = \frac{\text{number of observed events}}{\text{total time at risk, censored units included}} . \]

Two naive fixes both go wrong. Dropping the censored units keeps only the quick events, so \(\hat\lambda\) comes out too high. Treating the censoring time as if the event happened then counts events that never occurred, so it also comes out too high. The simulation shows the size of each error and checks the closed form against a numerical optimizer.

```python
from scipy.optimize import minimize_scalar

true_rate, n, horizon = 0.5, 50_000, 3.0             # e.g. churn events per year, 3-year study window
T = rng.exponential(scale=1 / true_rate, size=n)
observed_event = T <= horizon
time_at_risk = np.minimum(T, horizon)

mle = observed_event.sum() / time_at_risk.sum()
drop_censored = 1 / T[observed_event].mean()
censor_as_event = 1 / time_at_risk.mean()

def neg_log_lik(r):
    return -(observed_event.sum() * np.log(r) - r * time_at_risk.sum())

numeric = minimize_scalar(neg_log_lik, bounds=(1e-6, 10), method="bounded").x
assert abs(numeric - mle) < 1e-4
assert abs(mle - true_rate) < 0.01
assert drop_censored > 1.4 * true_rate                # about 0.88: badly too high
assert censor_as_event > 1.2 * true_rate              # about 0.64: still too high
```

About 22% of units here are censored (\(e^{-1.5}\)), and dropping them inflates the rate by about 75%. The same "events over exposure" estimator is the backbone of survival analysis, where the Kaplan-Meier curve drops the constant-hazard assumption altogether.

## The waiting-time paradox
Buses arrive as a Poisson process with a 10-minute mean gap. You show up at a random moment. By memorylessness your expected wait is the full 10 minutes, not 5. Yet the gap between the bus you just missed and the one you catch averages 20 minutes, twice the typical gap.

There is no contradiction. A random arrival time is more likely to fall inside a long gap than a short one, simply because long gaps cover more of the clock. This length-biased sampling is the **inspection paradox**, and it turns up whenever you sample by time rather than by event: average class size seen by students versus by the registrar, or session length measured from users who are online right now.

```python
lam = 6.0                                            # buses per hour
bus_times = np.cumsum(rng.exponential(scale=1 / lam, size=2_000_000))
you = rng.uniform(bus_times[0], bus_times[-1], size=300_000)   # always between two buses
idx = np.searchsorted(bus_times, you)                # next bus after you arrive
wait = bus_times[idx] - you
gap_you_landed_in = bus_times[idx] - bus_times[idx - 1]

assert abs(wait.mean() - 1 / lam) < 0.002            # about 10 minutes, not 5
assert abs(gap_you_landed_in.mean() - 2 / lam) < 0.003      # about 20 minutes
assert abs(np.diff(bus_times).mean() - 1 / lam) < 0.001     # while the typical gap is 10
```

## When to use it, and pitfalls
Reach for the exponential when you model time to an event that has a roughly constant rate: arrivals to a queue, time between requests to a server, failures of components without wear, decay. It is also the building block for Poisson processes, M/M/1 queues, continuous-time Markov chains and the baseline model in survival analysis.

- **Rate versus scale.** \(\lambda\) is a rate; \(1/\lambda\) is the mean. `numpy.random.Generator.exponential` and `scipy.stats.expon` both take the scale \(1/\lambda\). Passing the rate instead makes the mean \(\lambda\) rather than \(1/\lambda\), off by a factor of \(\lambda^2\): at 6 per hour, waits come out 36 times too long.
- **A density is not a probability.** \(f(t)\) can exceed 1; only integrals of it are probabilities.
- **Check the constant-hazard assumption.** A quick diagnostic: the coefficient of variation (sd over mean) should be near 1, and \(\ln S(t)\) estimated from data should be a straight line in \(t\). Bursty traffic gives a CV well above 1; scheduled events give one well below.
- **Censoring.** Survival-type data almost always contain units still waiting. Use events over total exposure, not the mean of finished waits.
- **Mean versus median.** For skewed waits, the median (\(0.69/\lambda\)) describes the typical experience; the mean drives capacity planning.
- **Sampling by time.** Waits and gaps observed from a random moment are length-biased, as in the paradox above.

For the discrete side of the same process see [Poisson distribution](../poisson-distribution/), and for a compact table of distributions see the [probability cheatsheet](../probability-cheatsheet/).

The derivation from the Poisson zero-count probability is the standard textbook argument; its whole-unit version is laid out in a [Mathematics Stack Exchange answer](https://math.stackexchange.com/a/1089139) and in Milefoot's [notes on the exponential distribution](https://www.milefoot.com/math/stat/pdfc-exponential.htm), which this post rewrites and extends.
