---
title: From decision trees to boosting: one tree, then many
summary: A decision tree grown from scratch and matched against scikit-learn, why one tree is fragile, how bagging and random forests average the fragility away, and how AdaBoost, gradient boosting and XGBoost build trees one after another, each fixing the last one's mistakes.
group: Statistics and experimentation
order: 8
tags: decision trees · random forests · gradient boosting · XGBoost · ensembles
---
When I grow 20 deep regression trees on bootstrap resamples of the same 600 synthetic rows, the root question flips between two features. At a typical test point their predictions have a standard deviation of about 1.0, twice the noise level. Averaging 60 trees cuts test MSE from 2.67 for one tree to 1.29. A decision tree is a stack of yes/no questions about the features: is `age <= 30`, then is `income <= 50k`, and so on, until you reach a leaf that holds a prediction. Each question cuts the data into two parts, and a good question leaves each part more uniform than the whole.

One tree is easy to read but unstable: a small change in the data can change the first question, and then everything below it changes too. **Bagging** and **random forests** grow many deep trees and average them, which cuts variance. **Boosting** grows many shallow trees one after another, each fixing what the sum so far gets wrong, which cuts bias. This post builds the tree and the boosting methods and tests each: the tree and gradient boosting match scikit-learn, AdaBoost's weights match its loss gradient, and XGBoost's leaf weight matches a numerical minimizer.

## Scoring a split
A split sends a node's samples \(S\) into a left part \(L\) and a right part \(R\). The tree picks the split with the lowest size-weighted **impurity** of the two parts:

\[ \frac{|L|}{|S|}\, C(L) + \frac{|R|}{|S|}\, C(R). \]

For classification, with \(p_c\) the share of class \(c\) in a node, the three usual choices of \(C\) are

\[ \text{error} = 1 - \max_c p_c, \qquad \text{entropy} = -\sum_c p_c \log_2 p_c, \qquad \text{Gini} = \sum_c p_c (1 - p_c). \]

For regression, \(C\) is the mean squared error around the node's own mean \(\bar y_S\), which is also the value the leaf predicts:

\[ C(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar y_S)^2. \]

Each term \(p_c(1 - p_c)\) of Gini is the variance of a Bernoulli variable, "is this sample in class \(c\)?". So Gini is the total variance of the one-hot label vector, and a classification tree with Gini is a regression tree on one-hot targets. The code below uses that.

Misclassification error is rarely used for growing trees, because it ignores progress that does not flip a majority. Start from 500 samples of each class. Splitting them into (375, 125) and (125, 375) misclassifies a quarter of the samples, and so does splitting into (250, 500) and (250, 0). By error the two splits tie, yet the second one produced a pure leaf that never needs splitting again. Gini and entropy are strictly concave, so they reward that purity. A split that keeps the same majority on both sides scores zero under error even when it helps.

```python
import numpy as np


def gini(p):
    p = np.asarray(p, float)
    return float(np.sum(p * (1 - p)))


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def misclass(p):
    return float(1 - np.max(p))


def gain(impurity, parent, left, right):
    """Parent impurity minus the size-weighted impurity of the children; arguments are class counts."""
    parent, left, right = (np.asarray(c, float) for c in (parent, left, right))
    n, nl, nr = parent.sum(), left.sum(), right.sum()
    return impurity(parent / n) - nl / n * impurity(left / nl) - nr / n * impurity(right / nr)


even = ([500, 500], [375, 125], [125, 375])
pure = ([500, 500], [250, 500], [250, 0])
assert np.isclose(gain(misclass, *even), 0.25) and np.isclose(gain(misclass, *pure), 0.25)   # a tie
assert gain(gini, *pure) > gain(gini, *even) and gain(entropy, *pure) > gain(entropy, *even)

same_majority = ([70, 30], [40, 20], [30, 10])
assert np.isclose(gain(misclass, *same_majority), 0) and gain(gini, *same_majority) > 0

rng = np.random.default_rng(0)
y = rng.integers(0, 3, 1000)
assert np.isclose(gini(np.bincount(y) / len(y)), np.eye(3)[y].var(axis=0).sum())   # Gini = one-hot variance
```

Entropy gain is the mutual information between the label and the side of the split; [entropy and information](../entropy-and-information/) works that through. Gini and entropy almost always pick the same split, and Gini avoids a logarithm.

## Growing a tree, greedily
Finding the best tree is NP-hard, so CART (Breiman, Friedman, Olshen and Stone, 1984) grows it greedily: at each node, try every feature and every threshold between consecutive sorted values, keep the split with the lowest child impurity, recurse on both sides, and stop at a depth limit, a minimum leaf size, or when no split helps.

Sorting a feature once makes every threshold cheap. The squared error of a group is \(\sum y^2 - (\sum y)^2 / n\), so prefix sums over the sorted samples give the error of every left/right split in one pass, \(O(n \log n)\) per feature per node. Leaves store the mean target; on one-hot targets that is the class proportions, which is exactly `predict_proba`.

```python
class Tree:
    """CART on a target matrix: real values for regression, one-hot rows for Gini classification."""

    def __init__(self, max_depth=3, min_leaf=1):
        self.max_depth, self.min_leaf = max_depth, min_leaf

    def fit(self, X, Y):
        self.root = self._grow(X, Y.reshape(len(Y), -1).astype(float), 0)
        return self

    def _best_split(self, X, Y):
        n = len(Y)
        best = (np.sum((Y - Y.mean(0)) ** 2), None, None)            # (children's squared error, feature, threshold)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="stable")
            x, Ys = X[order, j], Y[order]
            cs, cs2 = np.cumsum(Ys, 0), np.cumsum(Ys**2, 0)
            nl = np.arange(1, n)                                      # left size for each cut position
            sse = ((cs2[:-1] - cs[:-1] ** 2 / nl[:, None]).sum(1)
                   + (cs2[-1] - cs2[:-1] - (cs[-1] - cs[:-1]) ** 2 / (n - nl)[:, None]).sum(1))
            ok = (x[1:] > x[:-1]) & (nl >= self.min_leaf) & (n - nl >= self.min_leaf)
            if ok.any():
                k = np.flatnonzero(ok)[np.argmin(sse[ok])]
                if sse[k] < best[0] - 1e-12:
                    best = (sse[k], j, (x[k] + x[k + 1]) / 2)         # midpoint threshold, as scikit-learn uses
        return best[1:]

    def _grow(self, X, Y, depth):
        stop = depth == self.max_depth or len(Y) < 2 * self.min_leaf
        j, t = (None, None) if stop else self._best_split(X, Y)
        if j is None:
            return Y.mean(0)                                          # leaf: mean target or class proportions
        left = X[:, j] <= t
        return (j, t, self._grow(X[left], Y[left], depth + 1), self._grow(X[~left], Y[~left], depth + 1))

    def predict(self, X):
        out = []
        for x in X:
            node = self.root
            while isinstance(node, tuple):
                j, t, left, right = node
                node = left if x[j] <= t else right
            out.append(node)
        return np.array(out)


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

rng = np.random.default_rng(1)
X = rng.normal(size=(400, 5))
y_reg = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(0, 0.3, 400)
ours = Tree(max_depth=5, min_leaf=3).fit(X, y_reg)
sk = DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=0).fit(X, y_reg)
assert np.allclose(ours.predict(X)[:, 0], sk.predict(X))

y_cls = (X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.5, 400) > 1).astype(int) + (X[:, 2] > 1)
ours = Tree(max_depth=3).fit(X, np.eye(3)[y_cls])                   # Gini tree = regression on one-hot labels
sk = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=0).fit(X, y_cls)
assert np.allclose(ours.predict(X), sk.predict_proba(X))
```

The checks use training points: when two features give the same partition of a small node, the libraries may pick different ones, which only matters for new points.

## Why one tree is fragile
Trees have real strengths. A root-to-leaf path reads as a rule. Only the order of a feature's values matters, so an increasing transform (log, exp, rank) gives the same partition and outliers in the features do no harm. Interactions come for free. Missing values and categorical features are handled natively by some implementations, not all.

The weakness is variance. A deep tree's leaves average a handful of noisy labels, and the greedy search hardens early choices: if two features are almost equally good at the root, a few resampled rows decide which wins, and the trees share nothing below it. Outliers in the target, unlike in the features, do hurt: with squared error one extreme label drags its leaf's mean.

```python
def make_data(n, rng):
    X = rng.normal(size=(n, 10))                                      # features 4..9 are pure noise
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.5 * X[:, 3] + rng.normal(0, 0.5, n)
    return X, y


rng = np.random.default_rng(2)
X, y = make_data(600, rng)
X_test, y_test = make_data(2000, rng)

roots, preds = [], []
for _ in range(20):
    i = rng.integers(0, len(y), len(y))                                # bootstrap resample of the rows
    t = DecisionTreeRegressor(random_state=0).fit(X[i], y[i])
    roots.append(t.tree_.feature[0])
    preds.append(t.predict(X_test))
assert set(roots) == {0, 3}                                            # the root question flips between two features
assert np.array(preds).std(axis=0).mean() > 0.9                        # about twice the noise sd of 0.5

Xt = X.copy()
Xt[:, 0] = np.exp(Xt[:, 0])                                           # an increasing transform of one feature
a = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y).predict(X)
b = DecisionTreeRegressor(max_depth=4, random_state=0).fit(Xt, y).predict(Xt)
assert np.allclose(a, b)
```

Across 20 bootstrap samples, the root split alternated between two features, and at a typical test point the deep trees' predictions had a standard deviation of about 1.0, twice the noise level of 0.5.

## Bagging and random forests
**Bagging** (bootstrap aggregating, Breiman 1996) trains \(B\) trees, each on \(n\) rows drawn with replacement from the \(n\) training rows, and averages their predictions (or takes a vote). Each bootstrap sample leaves out a fraction \((1 - 1/n)^n \approx 1/e \approx 0.368\) of the rows, and those **out-of-bag** rows give each tree a free validation set.

Averaging does not remove variance completely. If each tree's prediction at a point \(x\) has variance \(\sigma^2\) and any two have correlation \(\rho\), the average of \(B\) trees has variance \(\rho \sigma^2 + (1 - \rho)\sigma^2 / B\), so more trees only drive it down to the floor \(\rho \sigma^2\). [The bias-variance tradeoff](../bias-variance-tradeoff/) derives and measures this. **Random forests** (Breiman 2001) attack \(\rho\): each split may only choose from a random subset of the features, so trees cannot all lean on the same strong feature.

Here \(\rho\) is a correlation over draws of the training set: two trees grown on the same data share it, and that shared part is what averaging cannot remove. It is not the correlation between two trees' predictions across test points, which mostly measures that both trees learned the signal. The code estimates it by growing a forest on each of 8 fresh training sets: the spread between trees of one forest is \((1 - \rho)\sigma^2\), and the spread of the forest average from one training set to the next is \(\rho\sigma^2\) plus a small \((1 - \rho)\sigma^2 / B\).

```python
from sklearn.ensemble import RandomForestRegressor

assert abs((1 - 1 / len(y)) ** len(y) - np.exp(-1)) < 1e-3           # share of rows a bootstrap leaves out

single = np.mean((DecisionTreeRegressor(random_state=0).fit(X, y).predict(X_test) - y_test) ** 2)


def tree_correlation(max_features, n_sets=8, B=60):
    """Correlation of two trees' predictions at a fixed point, over training sets; averaged over points."""
    within, means = [], []
    for k in range(n_sets):
        Xk, yk = make_data(600, np.random.default_rng(100 + k))           # a fresh training set
        f = RandomForestRegressor(n_estimators=B, max_features=max_features, random_state=k, n_jobs=1).fit(Xk, yk)
        P = np.array([t.predict(X_test[:500]) for t in f.estimators_])
        within.append(P.var(axis=0, ddof=1))                            # (1 - rho) sigma^2
        means.append(P.mean(axis=0))
    unshared = np.mean(within)
    shared = np.var(means, axis=0, ddof=1).mean() - unshared / B         # rho sigma^2
    return shared / (shared + unshared)


results = {}
for name, max_features in [("bagging", 1.0), ("random forest", 0.33)]:
    forest = RandomForestRegressor(n_estimators=60, max_features=max_features, oob_score=True,
                                   random_state=0, n_jobs=1).fit(X, y)
    rho = tree_correlation(max_features)
    test_mse = np.mean((forest.predict(X_test) - y_test) ** 2)
    oob_mse = np.mean((forest.oob_prediction_ - y) ** 2)
    results[name] = rho, test_mse, oob_mse
    assert test_mse < 0.5 * single                                     # averaging more than halves the error
    assert abs(oob_mse - test_mse) < 0.15 * test_mse                   # out-of-bag error tracks test error

assert results["random forest"][0] < 0.7 * results["bagging"][0]       # feature subsampling decorrelates trees
```

| Model | Tree-to-tree correlation \(\rho\) | Test MSE | Out-of-bag MSE |
|---|---|---|---|
| One deep tree | | 2.67 | |
| Bagging, 60 trees | 0.10 | 1.29 | 1.31 |
| Random forest, a third of the features per split | 0.05 | 1.29 | 1.34 |

Feature subsampling halved \(\rho\), as intended, but test error barely moved: the floor \(\rho\sigma^2\) was already small next to the bias, and each tree got weaker. Six of the ten features are noise, so a split that may only look at three features sees just one or two of the four useful ones most of the time, and none of them one time in six. `max_features` trades correlation between trees against the strength of each one, and it is worth tuning. These numbers come from one synthetic task with 600 training rows, so they show how averaging and feature subsampling act, not how large the gain will be on other data.

## Boosting: fit what is still wrong
Boosting grows weak trees, often stumps (one split) or depth-3 trees, in sequence, each chosen to fix the current ensemble. A stump alone is badly biased; hundreds of small corrections remove the bias.

### AdaBoost: reweight the mistakes
AdaBoost (Freund and Schapire, 1997) keeps a weight on every training example. Each round fits a stump to the weighted data, gives it a vote \(\alpha = \tfrac12 \log\frac{1 - \text{err}}{\text{err}}\), and multiplies the weight of every misclassified example by \(e^{\alpha}\) and every correct one by \(e^{-\alpha}\). Unrolled, the weight of example \(i\) after \(m\) rounds is proportional to \(e^{-y_i F_m(x_i)}\), where \(F_m\) is the weighted vote so far and \(y_i \in \{-1, +1\}\). That is the size of the gradient of the exponential loss \(e^{-yF}\). Friedman, Hastie and Tibshirani (2000) used this to show that AdaBoost fits an additive model stagewise under exponential loss: the reweighting is a gradient method in disguise.

```python
rng = np.random.default_rng(3)
Xc = rng.normal(size=(500, 2))
yc = np.where(Xc[:, 0] ** 2 + Xc[:, 1] ** 2 > 1.4, 1, -1)            # a circle: no single stump fits it

F = np.zeros(len(yc))
w = np.full(len(yc), 1 / len(yc))
train_err = []
for m in range(100):
    stump = DecisionTreeClassifier(max_depth=1).fit(Xc, yc, sample_weight=w)
    h = stump.predict(Xc)
    err = w[h != yc].sum()
    alpha = 0.5 * np.log((1 - err) / err)
    F += alpha * h
    w = w * np.exp(-alpha * yc * h)
    w /= w.sum()
    assert np.allclose(w, np.exp(-yc * F) / np.exp(-yc * F).sum())  # weights = exponential-loss gradient
    train_err.append(np.mean(np.sign(F) != yc))

assert train_err[0] > 0.3 and train_err[-1] < 0.05
```

One stump misclassifies 36% of the circle data; after 10 rounds the vote misclassifies 5%, after 100 rounds 1%.

### Gradient boosting: fit the negative gradient
Friedman (2001) made the gradient view the algorithm itself. For any differentiable loss \(\ell(y, F)\), start from a constant \(F_0\) and repeat:

\[ r_i = -\frac{\partial \ell(y_i, F)}{\partial F}\Big|_{F = F_{m-1}(x_i)}, \qquad F_m = F_{m-1} + \nu\, h_m, \]

where \(h_m\) is a small regression tree fit to the \(r_i\) and \(\nu\), the learning rate, shrinks each step. For squared loss \(\tfrac12 (y - F)^2\), \(r_i\) is the plain residual \(y_i - F\), which is where "fit the residuals" comes from. For other losses it is not: under log loss it is \(y_i - p_i\), and under absolute loss it is the sign of the residual.

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

lr = 0.1
F = np.full(len(y), y.mean())
for m in range(100):
    r = y - F                                                          # negative gradient of (y - F)^2 / 2
    F += lr * DecisionTreeRegressor(max_depth=3, random_state=m).fit(X, r).predict(X)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, max_depth=3,
                                criterion="squared_error", random_state=0).fit(X, y)
assert np.allclose(gbr.predict(X), F)
```

For log loss, a tree fit to the gradients picks good splits, but its leaf values (mean gradients) are not good step sizes. Friedman replaces each leaf value with one Newton step, the sum of gradients over the sum of second derivatives in that leaf. With \(p = \sigma(F)\), the gradient is \(g = p - y\) and the second derivative is \(h = p(1 - p)\). scikit-learn does the same, and the from-scratch version matches it to rounding error:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


yb = (yc > 0).astype(float)
F = np.full(len(yb), np.log(yb.mean() / (1 - yb.mean())))            # start at the log-odds of the base rate
for m in range(50):
    p = sigmoid(F)
    g, hess = p - yb, p * (1 - p)
    leaf = DecisionTreeRegressor(max_depth=2, random_state=m).fit(Xc, -g).apply(Xc)
    for L in np.unique(leaf):
        in_leaf = leaf == L
        F[in_leaf] += lr * (-g[in_leaf].sum() / hess[in_leaf].sum())  # one Newton step per leaf

gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=lr, max_depth=2,
                                 criterion="squared_error", random_state=0).fit(Xc, yb)
assert np.allclose(gbc.decision_function(Xc), F)
```

### XGBoost: second order and a penalty on the leaves
XGBoost (Chen and Guestrin, 2016) puts the second derivative into the tree growing itself, not only the leaf values. With \(g_i\) and \(h_i\) the first and second derivatives of the loss at the current prediction, adding a tree \(f\) changes the loss by about \(\sum_i [g_i f(x_i) + \tfrac12 h_i f(x_i)^2]\). It adds a penalty \(\gamma T + \tfrac12 \lambda \sum_j w_j^2\) on the number of leaves \(T\) and the leaf values \(w_j\) (an L1 term on the \(w_j\) is also available). For a leaf holding gradient sum \(G\) and second-derivative sum \(H\), the objective is a parabola in \(w\), minimised at

\[ w^* = -\frac{G}{H + \lambda}, \qquad \text{with value } -\frac12 \frac{G^2}{H + \lambda}. \]

A split is scored by how much it lowers that value, \(\tfrac12 \big[\tfrac{G_L^2}{H_L + \lambda} + \tfrac{G_R^2}{H_R + \lambda} - \tfrac{G^2}{H + \lambda}\big] - \gamma\), and is not made if the score is negative. So \(\gamma\) prunes, and \(\lambda\) shrinks every leaf toward zero.

```python
from scipy.optimize import minimize_scalar


def leaf_weight(g, h, lam):
    return -g.sum() / (h.sum() + lam)


def leaf_objective(w, g, h, lam):
    return np.sum(g * w + 0.5 * h * w**2) + 0.5 * lam * w**2


def split_score(g, h, left, lam, gamma):
    s = lambda gg, hh: gg.sum() ** 2 / (hh.sum() + lam)
    return 0.5 * (s(g[left], h[left]) + s(g[~left], h[~left]) - s(g, h)) - gamma


rng = np.random.default_rng(4)
g, h, lam = rng.normal(size=50), rng.uniform(0.1, 0.25, 50), 1.0
w_star = leaf_weight(g, h, lam)
assert np.isclose(minimize_scalar(lambda w: leaf_objective(w, g, h, lam)).x, w_star)       # brute force agrees
assert np.isclose(leaf_objective(w_star, g, h, lam), -0.5 * g.sum() ** 2 / (h.sum() + lam))

left = rng.random(50) < 0.5                                            # score = drop in the objective
best = lambda mask: leaf_objective(leaf_weight(g[mask], h[mask], lam), g[mask], h[mask], lam)
assert np.isclose(split_score(g, h, left, lam, 0.0), best(np.ones(50, bool)) - best(left) - best(~left))

resid = rng.normal(size=50)                                            # squared loss: g = -residual, h = 1
assert np.isclose(leaf_weight(-resid, np.ones(50), 0.0), resid.mean())            # plain gradient boosting
assert abs(leaf_weight(-resid, np.ones(50), 10.0)) < abs(resid.mean())             # lambda shrinks the leaf
```

With squared loss and \(\lambda = 0\), \(w^*\) is the mean residual, so this is ordinary gradient boosting. The rest of XGBoost is engineering: shrinkage, row subsampling (from Friedman's stochastic gradient boosting) and column subsampling (from random forests), approximate quantile-based split finding, a default direction for missing values, and parallel, cache-aware training. LightGBM and scikit-learn's `HistGradientBoostingRegressor` use the same leaf formula.

## Bagging or boosting?
| | Bagging and random forests | Gradient boosting |
|---|---|---|
| Trees | Deep, independent, trained in parallel | Shallow, sequential |
| Main effect | Lowers variance | Lowers bias |
| More trees | Never hurts, only costs time | Eventually overfits: use early stopping |
| Key knobs | Number of features per split, leaf size | Learning rate, number of rounds, depth, subsampling |
| Tuning effort | Works well with defaults | Needs a validation set |

On tabular data, gradient-boosted trees are a strong default and often beat neural networks; Grinsztajn, Oyallon and Varoquaux (2022) found tree ensembles ahead on a benchmark of medium-sized tabular datasets. That is a prior, not a law: a random forest and a linear baseline are still worth running.

## Pitfalls
- **Trees do not extrapolate.** A prediction is an average of training labels, so outside the training range it stays flat. A forest fit to \(y = 3x\) on \(x \in [0, 10]\) predicts about 30 at both \(x = 20\) and \(x = 100\). For trends, model the trend first or predict differences.
- **Impurity-based feature importance is biased.** It credits every split a feature wins on the training data, including splits fitted to noise, and features with many distinct values win more of those (Strobl, Boulesteix, Zeileis and Hothorn, 2007). Permutation importance on held-out data is the safer check.
- **Boosting needs early stopping.** Each round lowers training loss, so pick the number of rounds on a validation set; a smaller learning rate needs more rounds but usually generalizes better.

```python
from sklearn.inspection import permutation_importance

forest = RandomForestRegressor(n_estimators=60, random_state=0, n_jobs=1).fit(X, y)
mdi = forest.feature_importances_
perm = permutation_importance(forest, X_test, y_test, n_repeats=3, random_state=0).importances_mean
assert mdi[4:].sum() > 0.1                                             # six noise features share real impurity credit
assert np.all(np.abs(perm[4:]) < 0.005) and np.all(perm[:4] > 0.05)   # permutation sees through it

x_line = np.linspace(0, 10, 200)[:, None]
line = RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=1).fit(x_line, 3 * x_line[:, 0])
far = line.predict([[20.0], [100.0]])
assert np.isclose(far[0], far[1]) and abs(far[0] - 30) < 0.5          # flat beyond the data
```

In the forest above, the six pure-noise features took about 18% of the impurity importance between them; their permutation importance was zero to two decimal places.

Every result the code produces comes from small synthetic datasets with fixed seeds. No test compares a forest's test error with gradient boosting's, and early stopping and the learning rate are left untested. On real data, which family wins and how to set its knobs are questions for a validation set.

The outline of this post follows the CS229 notes on decision trees and Yann Dubois's machine-learning glossary entry on trees, rewritten with new examples and code; the bagging and boosting material follows chapters 8, 10 and 15 of Hastie, Tibshirani and Friedman's The Elements of Statistical Learning.
