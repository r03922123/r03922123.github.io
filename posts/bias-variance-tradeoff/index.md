# The bias-variance tradeoff, measured

Fit polynomials to 300 training sets of 30 noisy points from a sine wave, and test error falls from 0.628 at degree 0 to 0.108 at degree 3, then climbs to 4.60 at degree 9, while training error falls at every degree. Almost all of that climb is variance: at degree 9 the squared bias is 0.004 and the variance 4.51. A model can be wrong in two different ways. It can be wrong on average: however many times you retrain it on fresh data, its predictions centre on the wrong value, because its form cannot express the truth. That is **bias**. Or it can be right on average but jumpy: each new training set gives quite different predictions, because the model reacts to the particular noise it happened to see. That is **variance**. Under squared error, the expected test error is exactly the squared bias plus the variance plus a noise floor that no model can remove.

The useful part is that the two errors respond to different fixes. More data cures variance and does little for bias; more capacity cures bias and usually adds variance. I derive the split, then measure every term by brute force: train the same model on hundreds of simulated training sets and look at how its predictions scatter.

## The decomposition
Assume the data come from \(y = f(x) + \varepsilon\), where \(f\) is the true regression function and the noise \(\varepsilon\) has mean 0 and variance \(\sigma^2\). A training set \(D\) is random, so the fitted model \(\hat f_D\) is random too. Write \(\bar f(x) = \mathbb{E}_D[\hat f_D(x)]\) for the average prediction over all training sets you might have drawn. At a fixed test point \(x\), with a fresh label \(y\) whose noise is independent of \(D\):

\[ \mathbb{E}_{D,\varepsilon}\big[(y - \hat f_D(x))^2\big] = \underbrace{\big(\bar f(x) - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D\big[(\hat f_D(x) - \bar f(x))^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}} \]

The proof is two lines. Write \(y - \hat f_D = \varepsilon + (f - \bar f) + (\bar f - \hat f_D)\) and square. Every cross term vanishes in expectation: \(\varepsilon\) has mean zero and is independent of \(D\), \(f - \bar f\) is a constant at this \(x\), and \(\bar f - \hat f_D\) has mean zero over \(D\) by the definition of \(\bar f\).

Three details are easy to get wrong:

- Bias is measured against \(f(x)\), the noise-free truth, not against an observed label \(y\). The gap between \(\bar f\) and a noisy label mixes bias with noise.
- Variance is spread across **training sets**, not across test points. It is the answer to "how much would this prediction move if I collected my data again?" Randomness in training (initialization, shuffling, bootstrap samples) adds to it as well.
- The clean additive split holds for squared error. For 0-1 classification loss there is no unique decomposition, and several competing definitions exist; the intuition carries over, the algebra does not.

## Measuring it by simulation
Because I choose \(f\), I can do what is impossible with real data: draw 300 independent training sets of 30 points each, fit the same model to every one, and compute \(\bar f\) and the spread directly. The truth is \(f(x) = \sin(2\pi x)\) on \([0, 1]\) with noise \(\sigma = 0.3\), and the model is a polynomial whose degree sets the capacity.

```python
import numpy as np

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)                  # the truth, which the learner never sees
sigma, n, reps = 0.3, 30, 300
x_test = np.linspace(0.02, 0.98, 200)


def features(x, degree):
    return np.polynomial.legendre.legvander(2 * x - 1, degree)    # Legendre basis: well conditioned


def fit_predict(x, y, degree, lam=0.0):
    A = features(x, degree)
    w = np.linalg.solve(A.T @ A + lam * np.eye(degree + 1), A.T @ y)
    return features(x_test, degree) @ w


def bias_variance(preds):
    """preds[r, j] is the model trained on dataset r, evaluated at test point j; averaged over j."""
    mean_pred = preds.mean(axis=0)                   # f-bar at each test point
    return np.mean((mean_pred - f(x_test)) ** 2), np.mean(preds.var(axis=0))


datasets = []
for _ in range(reps):
    x = rng.uniform(0, 1, n)
    datasets.append((x, f(x) + rng.normal(0, sigma, n)))
y_fresh = f(x_test) + rng.normal(0, sigma, (reps, x_test.size))   # new labels, never trained on

rows = {}
for d in range(10):
    preds = np.array([fit_predict(x, y, d) for x, y in datasets])
    b2, var = bias_variance(preds)
    test_mse = np.mean((y_fresh - preds) ** 2)
    rows[d] = (b2, var, test_mse)
    assert abs(test_mse - (b2 + var + sigma**2)) < 0.02 * test_mse   # the decomposition, by simulation

b2s, vars_, mses = (np.array([rows[d][k] for d in range(10)]) for k in range(3))
assert b2s[0] > b2s[1] > b2s[3] and (b2s[3:] < 0.01).all()           # bias falls, then stays near zero
assert all(vars_[d] < vars_[d + 1] for d in range(3, 9))               # variance keeps rising
assert mses.argmin() == 3                                              # the sweet spot in between

train_mse = [np.mean([np.mean((features(x, d) @ np.linalg.lstsq(features(x, d), y, rcond=None)[0] - y) ** 2)
                      for x, y in datasets]) for d in range(10)]
assert all(a > b for a, b in zip(train_mse, train_mse[1:]))           # training error only ever falls
```

| Degree | Bias² | Variance | Test MSE |
|---|---|---|---|
| 0 | 0.518 | 0.023 | 0.628 |
| 1 | 0.174 | 0.023 | 0.285 |
| 3 | 0.004 | 0.014 | 0.108 |
| 5 | 0.000 | 0.032 | 0.122 |
| 7 | 0.001 | 0.481 | 0.573 |
| 9 | 0.004 | 4.51 | 4.60 |

Each figure is averaged over the 200 test points; test MSE includes the noise floor \(\sigma^2 = 0.09\), and matches bias² + variance + 0.09 to within 2% in every row. The small nonzero bias² at degrees 7 and 9 is simulation noise, not bias: \(\bar f\) is itself estimated from 300 fits, and that estimate inherits a share of the variance.

Degree 1 is a straight line through a sine wave: its predictions barely move between datasets, but they are reliably wrong. Degree 3 is the first that can bend the right way, and bias drops almost to zero. After that, bias stays negligible and every extra degree only buys variance, until degree 9, with 10 coefficients and 30 noisy points, produces wild swings near the edges of the interval. Training error, meanwhile, falls with every degree (a bigger polynomial contains every smaller one, so least squares can only fit the training points better), so it cannot pick the sweet spot.

## Regularization is the same dial
Degree is not the only capacity control. Keep the degree-9 polynomial and add a ridge penalty \(\lambda \lVert w \rVert^2\) instead (see [L1 and L2 regularization](../regularization-l1-l2/)). Increasing \(\lambda\) pulls every fit toward zero, which makes fits on different datasets agree more (less variance) and also pulls the average fit away from the truth (more bias):

```python
sweep = {}
for lam in [1e-5, 1e-2, 1.0, 10.0]:
    sweep[lam] = bias_variance(np.array([fit_predict(x, y, 9, lam) for x, y in datasets]))
lams = sorted(sweep)
assert all(sweep[a][1] > sweep[b][1] for a, b in zip(lams, lams[1:]))     # variance only falls
assert sweep[10.0][0] > 10 * sweep[1e-2][0]                               # bias eventually climbs
best = min(lams, key=lambda l: sum(sweep[l]))
assert best == 1.0 and sum(sweep[best]) + sigma**2 < rows[3][2] + 0.03   # close to choosing degree 3
```

| Ridge \(\lambda\) | Bias² | Variance | Test MSE |
|---|---|---|---|
| 0.00001 | 0.002 | 2.46 | 2.55 |
| 0.01 | 0.000 | 0.079 | 0.170 |
| 1 | 0.016 | 0.020 | 0.126 |
| 10 | 0.191 | 0.010 | 0.291 |

A well-regularized flexible model ends up close to the right-sized one (test MSE 0.126 against 0.108). This is why "pick a big model and tune the regularization" is a workable strategy, and why [dropout](../dropout/), weight decay and early stopping are all, in this picture, ways to buy back variance at a small price in bias.

## Reading it from learning curves
On real data you get one training set and cannot compute \(\bar f\). What you can do is watch training and validation error as the training set grows, and the two failure modes leave different fingerprints.

```python
x_eval = np.linspace(0, 1, 401)       # the whole input range: test inputs come from the training distribution


def learning_curve(degree, sizes, reps=200):
    out = []
    for m in sizes:
        train, test = [], []
        for _ in range(reps):
            x = rng.uniform(0, 1, m)
            y = f(x) + rng.normal(0, sigma, m)
            w = np.linalg.lstsq(features(x, degree), y, rcond=None)[0]
            train.append(np.mean((features(x, degree) @ w - y) ** 2))
            test.append(np.mean((features(x_eval, degree) @ w - f(x_eval)) ** 2) + sigma**2)
        out.append((np.mean(train), np.median(test)))
    return np.array(out)


sizes = [20, 40, 80, 320]
simple, flexible = learning_curve(1, sizes), learning_curve(7, sizes)
gap_simple, gap_flexible = simple[:, 1] - simple[:, 0], flexible[:, 1] - flexible[:, 0]
plateau = 0.5 - 3 / np.pi**2 + sigma**2      # best straight line through sin(2 pi x) on [0, 1], plus noise

assert (simple > 0.24).all() and (gap_simple > 0).all()               # both far above the 0.09 floor, test above train
assert abs(simple[-1, 1] - plateau) < 0.01 and abs(simple[-1, 1] - simple[-2, 1]) < 0.01   # stuck at bias² + noise
assert gap_flexible[0] > 3 * gap_simple[0]                             # small data: a wide train/test gap
assert flexible[-1, 1] < 0.1 and gap_flexible[-1] < 0.01               # more data closes it
assert abs(flexible[0, 0] - sigma**2 * (1 - 8 / 20)) < 0.006           # 8 coefficients on 20 points: below the floor
```

| Training points | Degree 1: train | Degree 1: test | Degree 7: train | Degree 7: test |
|---|---|---|---|---|
| 20 | 0.263 | 0.312 | 0.056 | 0.238 |
| 40 | 0.270 | 0.299 | 0.070 | 0.115 |
| 80 | 0.286 | 0.294 | 0.082 | 0.100 |
| 320 | 0.284 | 0.289 | 0.088 | 0.092 |

Test error is measured over the whole interval \([0, 1]\) the training inputs come from, and is the median over 200 runs, because a degree-7 fit on 20 points occasionally explodes and would swamp a mean. The degree-1 curves meet at 0.286, which can be checked by hand: the best straight line through \(\sin(2\pi x)\) on \([0, 1]\) leaves a squared error of \(1/2 - 3/\pi^2 \approx 0.196\), and the noise adds 0.09.

- **High bias:** training error is already high and validation error sits just above it. Both plateau at bias² plus noise, and more data does not move them.
- **High variance:** training error is low, validation error is much higher, and the gap shrinks as data grows.

One correction to the usual rule of thumb, "low train accuracy means high bias": training error has to be judged against the noise floor, not against zero. Here no model can get below 0.09, so a training MSE of 0.09 would be perfect, and the degree-7 model reaching 0.056 on 20 points is memorizing noise. That number is no accident: least squares with \(p\) coefficients on \(n\) points has expected training error \(\sigma^2 (1 - p/n)\) when the model can express the truth, here \(0.09 \times (1 - 8/20) = 0.054\). On real problems the floor is unknown; a strong baseline, human agreement on the labels, or the best published result stands in for it.

| Symptom | Diagnosis | Fixes that help |
|---|---|---|
| Train and validation error both high and close | High bias | More capacity, better features, less regularization, boosting |
| Low train error, large gap to validation | High variance | More data, augmentation, stronger regularization, smaller model, bagging, early stopping |
| Both low and close, but above the target | Near the noise floor | Better labels or new information in the inputs; the model is not the bottleneck |

The [overfitting playbook](../overfitting-playbook/) ranks the variance fixes in more detail.

## Ensembles attack one term each
Averaging and boosting are the two classic ensemble strategies, and they sit on opposite sides of the decomposition (more in [from decision trees to boosting](../decision-trees-to-boosting/)).

**Bagging** trains \(B\) models on bootstrap resamples and averages them. If each has variance \(s^2\) and any two have correlation \(\rho\), the average has variance

\[ \operatorname{Var}\Big(\frac1B \sum_{b=1}^B \hat f_b\Big) = \rho\, s^2 + \frac{1 - \rho}{B}\, s^2 . \]

The second term vanishes as \(B\) grows; the first does not. Bootstrap resamples of one dataset overlap heavily, so the trees stay correlated, and \(\rho s^2\) is a floor that more trees cannot break. Random forests lower \(\rho\) by also restricting each split to a random subset of features. The average has the same expectation as a single bagged tree, so bias barely moves. That is why bagging pairs with deep, low-bias, high-variance trees.

**Boosting** goes the other way: it starts from a very simple model, such as a one-split tree (a stump), and adds stumps one at a time, each fit to the residuals of the sum so far. The rounds drive bias down; variance creeps up slowly, which is why the number of rounds and the learning rate need tuning.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

n, reps, B = 100, 60, 25
trees = np.empty((reps, B, x_test.size))
for r in range(reps):
    x = rng.uniform(0, 1, n)
    y = f(x) + rng.normal(0, sigma, n)
    for b in range(B):
        i = rng.integers(0, n, n)                                    # bootstrap resample
        trees[r, b] = DecisionTreeRegressor().fit(x[i, None], y[i]).predict(x_test[:, None])

single, bagged = bias_variance(trees[:, 0]), bias_variance(trees.mean(axis=1))
assert bagged[1] < 0.55 * single[1]                                    # variance roughly halves
assert bagged[0] < 0.01 and single[0] < 0.01                           # bias was never the problem

s2 = trees.var(axis=(0, 1))                                            # one tree's variance, per test point
dev = trees - trees.mean(axis=(0, 1))
pair_cov = ((dev.sum(axis=1) ** 2 - (dev**2).sum(axis=1)) / (B * (B - 1))).mean(axis=0)
rho = pair_cov / s2
assert np.isclose(np.mean(rho * s2 + (1 - rho) * s2 / B), bagged[1])  # the formula above, exact for sample moments
assert 0.3 < rho.mean() < 0.6 and bagged[1] < 1.1 * np.mean(rho * s2)  # already close to the floor

stages = {1: [], 10: [], 50: [], 200: []}
for r in range(30):
    x = rng.uniform(0, 1, n)
    y = f(x) + rng.normal(0, sigma, n)
    gb = GradientBoostingRegressor(max_depth=1, n_estimators=200, learning_rate=0.1).fit(x[:, None], y)
    for k, p in enumerate(gb.staged_predict(x_test[:, None]), 1):
        if k in stages:
            stages[k].append(p)
boost = {k: bias_variance(np.array(v)) for k, v in stages.items()}
assert boost[1][0] > boost[10][0] > boost[50][0] > boost[200][0]       # bias falls round after round
assert boost[200][1] > boost[10][1]                                    # variance slowly rises
```

| Model | Bias² | Variance |
|---|---|---|
| One deep tree (one bootstrap sample) | 0.001 | 0.085 |
| Bag of 25 deep trees | 0.001 | 0.042 |
| Floor \(\rho s^2\) (\(\rho \approx 0.46\)) | | 0.040 |
| Boosted stumps, 1 round | 0.438 | 0.006 |
| Boosted stumps, 10 rounds | 0.130 | 0.006 |
| Boosted stumps, 50 rounds | 0.028 | 0.010 |
| Boosted stumps, 200 rounds | 0.002 | 0.015 |

Bagging halved the variance and was already within 5% of its floor at 25 trees. Boosting cut bias by a factor of more than 200 while variance went from 0.006 to 0.015.

## Double descent: where the U breaks
The textbook picture says test error traces a U as capacity grows. Belkin, Hsu, Ma and Mandal (2019) showed that for models that can fit the training data exactly, the curve often has a second act: error peaks where the model has just enough parameters to interpolate, then falls again as capacity keeps growing. They called it **double descent**.

The simplest place to see it is least squares on Gaussian features. The truth uses 400 features with slowly decaying weights; the model sees only the first \(p\) of them and is fit on \(n = 40\) rows. Below \(p = n\) it is ordinary least squares. Above it there are infinitely many exact fits, and `lstsq` returns the one with the smallest norm, which is also what gradient descent from zero converges to. For \(p < n - 1\) the expected risk has a closed form. The omitted features act as extra noise, and for a Gaussian design the expected trace of \((X^\top X)^{-1}\) is \(p/(n - p - 1)\):

\[ \mathbb{E}[\text{test MSE}] = \big(\sigma^2 + \lVert w_{\text{omitted}} \rVert^2\big)\, \frac{n - 1}{n - p - 1} \]

Split the factor as \((n-1)/(n-p-1) = 1 + p/(n-p-1)\). The 1 is the noise plus the signal the model cannot see; the \(p/(n-p-1)\) part is variance, and it blows up as \(p\) approaches \(n\).

```python
rng = np.random.default_rng(3)
n, D, noise_var = 40, 400, 0.25
w_true = np.arange(1, D + 1) ** -0.7
w_true *= 2 / np.linalg.norm(w_true)
ps = [5, 10, 20, 30, 36, 40, 44, 50, 80, 200, 400]
ridgeless, ridge = {p: [] for p in ps}, {p: [] for p in ps}
for _ in range(100):
    X = rng.normal(size=(n, D))
    y = X @ w_true + rng.normal(0, np.sqrt(noise_var), n)
    for p in ps:
        Xp = X[:, :p]
        for store, w in [(ridgeless, np.linalg.lstsq(Xp, y, rcond=None)[0]),
                         (ridge, Xp.T @ np.linalg.solve(Xp @ Xp.T + 5.0 * np.eye(n), y))]:
            err = w_true.copy()
            err[:p] -= w
            store[p].append(err @ err + noise_var)                 # exact test MSE for Gaussian inputs

med = {p: np.median(ridgeless[p]) for p in ps}
for p in [5, 10, 20]:
    omitted = noise_var + w_true[p:] @ w_true[p:]
    assert abs(np.mean(ridgeless[p]) / (omitted * (n - 1) / (n - p - 1)) - 1) < 0.08   # the formula
assert max(med, key=med.get) == n                                      # the peak sits at p = n
assert med[n] > 10 * med[10] and med[80] < med[50] < med[44]           # and error descends again after it
ridge_med = [np.median(ridge[p]) for p in ps[1:]]
assert all(a < b for a, b in zip(ridge_med, ridge_med[1:]))           # ridge: no peak, just a slow rise
assert np.median(ridge[n]) < med[n] / 10
```

| Features \(p\) | 5 | 10 | 20 | 30 | 36 | 40 | 44 | 50 | 80 | 200 | 400 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Minimum-norm least squares | 1.88 | 1.68 | 1.90 | 3.45 | 7.70 | 57.8 | 7.74 | 4.06 | 2.92 | 3.52 | 3.87 |
| Ridge, \(\lambda = 5\) | 1.87 | 1.61 | 1.63 | 1.82 | 1.94 | 2.01 | 2.11 | 2.26 | 2.75 | 3.52 | 3.87 |

Median test MSE over 100 draws; the mean is infinite at \(p = n\). Predicting zero everywhere scores 4.25.

Three things are worth taking from this. The decomposition itself still holds everywhere; what fails is the assumption that variance rises steadily with parameter count. Past the interpolation threshold, adding features gives the minimum-norm solution more directions to spread its fit over, and variance falls. And whether the second descent ends below the classical sweet spot depends on the problem: here three quarters of the signal sits in the first 10 features and the small model still wins, while in Belkin et al.'s random-feature and neural-network experiments the large models came out ahead. A modest ridge penalty removes the spike altogether; Nakkiran, Venkat, Kakade and Ma (2021) proved that optimally tuned ridge does this for linear regression with isotropic Gaussian inputs, as here, and saw the same effect in neural networks.

## Pitfalls
- **Reading the tradeoff as a law.** Bias and variance often trade against each other, but not always. More data lowers variance for free; a better feature can lower both; and past the interpolation threshold, more parameters can lower variance.
- **Using training error to choose capacity.** It only falls as capacity grows: asserted for the polynomial sweep above, and equally true for a weaker ridge penalty, more boosting rounds or more features. Only held-out error shows the trade.
- **Expecting bagging to fix bias, or more trees to fix everything.** Bagging leaves bias alone, and its variance reduction stops at \(\rho s^2\).
- **Forgetting the noise floor.** When train and validation error meet at a level you are unhappy with, and that level is the noise in the labels, no model change helps. Spend the effort on the data.

These numbers come from synthetic problems whose truth I chose: a sine wave with noise \(\sigma = 0.3\) for the sweeps, and Gaussian features with \(n = 40\) rows for double descent. They show how each term moves, not how large it is on real data, where \(\bar f\) cannot be computed and learning curves read against an estimated noise floor are the working tool. Whether a model past the interpolation threshold beats the right-sized one stays open: it did not here, and it did in Belkin et al.'s experiments.

For where this fits among the other basics, see the [ML fundamentals quick reference](../ml-fundamentals-quick-reference/); for why squared error is the natural loss here, see [regression as maximum likelihood](../regression-as-maximum-likelihood/).

The decomposition and the bagging variance formula follow chapters 7 and 15 of Hastie, Tibshirani and Friedman's The Elements of Statistical Learning (2009) and section 3.2 of Bishop's Pattern Recognition and Machine Learning (2006); the bias/variance framing for learning machines was popularised by Geman, Bienenstock and Doursat (1992). Double descent is from Belkin, Hsu, Ma and Mandal (PNAS, 2019); the result on tuned ridge is from Nakkiran, Venkat, Kakade and Ma (ICLR, 2021).
