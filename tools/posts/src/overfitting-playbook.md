---
title: The overfitting playbook: diagnose, then pick the lever
summary: How to tell overfitting from underfitting, distribution shift and leakage using learning curves, then the levers that fix it — more data, augmentation, weight decay, dropout, label smoothing, early stopping and capacity — each checked with a small experiment, and the order in which to reach for them.
group: Machine learning fundamentals
order: 9
tags: overfitting · regularization · augmentation · early stopping
---
Fit 20 noisy points from a sine curve with a degree-12 polynomial, and its training error is lower than the best degree's while its validation error is more than twice as high. That model overfits: it learns patterns that are true of its training set and nowhere else, the noise in the labels and the quirks of the particular rows it saw. Training error keeps falling, and error on new data stops falling and then rises.

The fixes are not interchangeable, and the fixes for underfitting point the opposite way, so I start with diagnosis. Then come five levers, most with a small experiment that shows what it does, and a table that orders them by cost.

## Diagnose first
The basic instrument is a pair of numbers, training error and validation error, measured on the same metric with the model in evaluation mode. Watch them as one thing varies: model capacity, training set size or training steps.

| Training error | Validation error | Reading | First move |
|---|---|---|---|
| Low | Much higher, and rising as capacity or training grows | Overfitting (high variance) | The levers in this post |
| High | Close to training error, also high | Underfitting (high bias) | Bigger model, better features, train longer, regularize less |
| Low | Higher, but the validation rows come from a different population | Distribution shift, not overfitting | Fix the split, or train on data that looks like the target |
| Low | Close to training and surprisingly good | Possibly leakage | Audit the features and the split |

A gap between the two is not by itself a problem. What matters is validation error: the best model is the one where it is lowest, and that model usually still has a gap.

The classic picture: fit 20 noisy points from a sine curve with polynomials of rising degree. Training error can only go down, since each larger model contains the smaller ones. Validation error falls while the model gains the shape it needs, then climbs as the extra freedom goes into the noise.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(0)


def sine_data(n, r):
    x = r.uniform(-1, 1, size=(n, 1))
    return x, np.sin(3 * x[:, 0]) + r.normal(scale=0.3, size=n)     # noise variance 0.09


def mse(model, X, y):
    return np.mean((model.predict(X) - y) ** 2)


def poly(degree):
    return make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())


X_tr, y_tr = sine_data(20, rng)
X_va, y_va = sine_data(2000, rng)
train_err, val_err = [], []
for d in range(1, 13):
    m = poly(d).fit(X_tr, y_tr)
    train_err.append(mse(m, X_tr, y_tr))
    val_err.append(mse(m, X_va, y_va))
train_err, val_err = np.array(train_err), np.array(val_err)

assert (np.diff(train_err) <= 1e-9).all()                  # more capacity never hurts the training fit
best = val_err.argmin() + 1
assert 3 <= best <= 6                                      # the sweet spot
assert val_err[0] > 2 * val_err.min()                      # degree 1: underfit, both errors high
assert train_err[0] > 0.09
assert val_err[-1] > 2 * val_err.min() and train_err[-1] < train_err[best - 1]   # degree 12: fits train better, validates worse
```

It is tempting to read a training error below the noise level (0.09 here) as proof of memorization, but that sign is weaker than it looks. Every fitted model absorbs some noise: for least squares with \(p\) coefficients on \(n\) rows and noise variance \(\sigma^2\), the expected training error of a model that contains (or nearly contains) the true function is \(\sigma^2 (1 - p/n)\). Averaged over many samples of 20 points, degree 3 sits near \(0.09 \times 16/20 = 0.072\) and degree 12 near \(0.09 \times 7/20 \approx 0.032\), and on any single small sample the numbers wander: the best model above already fits its training set to about 0.024. The reliable signal is the pair moving apart, training error still falling while validation error climbs.

```python
avg_train = {3: [], 12: []}
for seed in range(200):
    X, y = sine_data(20, np.random.default_rng(1000 + seed))
    for d in avg_train:
        avg_train[d].append(mse(poly(d).fit(X, y), X, y))
for d, errs in avg_train.items():
    expected = 0.09 * (1 - (d + 1) / 20)                    # p = d + 1 coefficients, intercept included
    assert abs(np.mean(errs) - expected) < 0.1 * expected
assert train_err[best - 1] < 0.09                           # even the best model sits below the noise level
```

### Is the gap really overfitting?
Check first that both sets come from the same distribution. In **adversarial validation**, label training rows 0 and validation rows 1 and see whether a classifier can tell them apart. An AUC near 0.5 means it cannot; well above that, the gap may come from the split itself (a time cut-off, a new region), and no regularizer will close it.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def adversarial_auc(train, val):
    X = np.vstack([train, val])
    is_val = np.r_[np.zeros(len(train)), np.ones(len(val))]
    p = cross_val_predict(LogisticRegression(), X, is_val, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(is_val, p)


A, B = rng.normal(size=(500, 5)), rng.normal(size=(500, 5))
B_shifted = B + np.array([0.8, 0, 0, 0, 0])                 # one feature drifted
assert abs(adversarial_auc(A, B) - 0.5) < 0.05
assert adversarial_auc(A, B_shifted) > 0.65
```

## Lever 1: more data
With a fixed model, more data shrinks variance and leaves bias alone. A **learning curve**, error against training set size, shows how much more data would buy. Below, the degree-12 polynomial that overfit above is trained on more and more points. Its validation error falls from wildly unstable to the noise floor, and the gap closes from both sides: training error rises as memorizing gets harder.

```python
curve = {}
for n in [20, 50, 200, 1000]:
    runs = []
    for seed in range(10):
        X, y = sine_data(n, np.random.default_rng(100 + seed))
        m = poly(12).fit(X, y)
        runs.append((mse(m, X, y), mse(m, X_va, y_va)))
    curve[n] = np.median(runs, axis=0)

assert curve[20][1] > 10 * curve[20][0]                     # tiny data: huge gap
assert curve[1000][1] - curve[1000][0] < 0.01               # gap nearly closed
assert curve[1000][1] < 0.1                                 # close to the noise floor
assert curve[1000][0] > curve[20][0]                        # training error rises with n
```

If the validation curve is still falling at your current size, more data will pay. If it has flattened next to an equally high training error, the problem is bias and more data will not help. And data is not free: labelling costs money, and rows from a different population than the one you serve can move the model the wrong way.

## Lever 2: augmentation
Augmentation manufactures new training rows by transforming existing ones in ways that should not change the answer: crops, flips, colour jitter and Cutout (masking a random square) for images; time stretch, added noise and masked time or frequency bands (SpecAugment) for audio; back-translation or word dropout for text. Three rules apply:

- The transform must keep the label. A horizontal flip is fine for cats and wrong for the digit 7 or for text in an image.
- Augment the training data only. Validation and test sets measure the real distribution.
- Draw fresh transforms each epoch, so the model never sees the same copy twice.

The simplest augmentation shows what augmentation does in general. Adding Gaussian noise to the inputs of a linear model is, in expectation, exactly ridge regression: averaged over the noise, \(\mathbb{E}\lVert y - (X + E) w \rVert^2 = \lVert y - Xw \rVert^2 + n \sigma^2 \lVert w \rVert^2\) for \(n\) rows and noise standard deviation \(\sigma\) (Bishop made this connection in 1995). Augmentation is a penalty written in data instead of in the loss.

```python
from sklearn.linear_model import Ridge

r = np.random.default_rng(1)
n, p, sigma = 30, 20, 0.5


def correlated_data(m):
    shared = r.normal(size=(m, 1))
    X = 0.8 * shared + 0.6 * r.normal(size=(m, p))
    return X, X @ np.full(p, 0.3) + r.normal(size=m)


X, y = correlated_data(n)
X_test, y_test = correlated_data(5000)
ols = LinearRegression(fit_intercept=False).fit(X, y)
copies = 500
X_aug = np.concatenate([X + sigma * r.normal(size=X.shape) for _ in range(copies)])
noisy = LinearRegression(fit_intercept=False).fit(X_aug, np.tile(y, copies))
ridge = Ridge(alpha=n * sigma**2, fit_intercept=False).fit(X, y)   # Ridge's loss is a sum over rows

assert np.abs(noisy.coef_ - ridge.coef_).max() < 0.05      # noisy copies = an L2 penalty
assert mse(noisy, X_test, y_test) < 0.6 * mse(ols, X_test, y_test)
```

**Mixup** (Zhang et al., 2018) goes further and is not label-preserving at all: it blends two examples and their labels with a weight \(\lambda \sim \text{Beta}(\alpha, \alpha)\), which pushes the model to behave linearly between training points; Thulasidasan et al. (2019) later found that it also makes networks less overconfident. Because cross-entropy is linear in the target, mixing the labels gives the same loss as mixing the two losses, which is how it is often implemented.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)


def mixup(x, y, num_classes, alpha=0.4):
    lam = float(np.random.default_rng(0).beta(alpha, alpha))
    perm = torch.randperm(len(x))
    targets = F.one_hot(y, num_classes).double()
    return lam * x + (1 - lam) * x[perm], lam * targets + (1 - lam) * targets[perm], perm, lam


x, y = torch.randn(16, 10, dtype=torch.float64), torch.randint(0, 3, (16,))
x_mix, y_mix, perm, lam = mixup(x, y, 3)
logits = torch.nn.Linear(10, 3).double()(x_mix)
soft_loss = -(y_mix * logits.log_softmax(-1)).sum(-1).mean()
mixed_losses = lam * F.cross_entropy(logits, y) + (1 - lam) * F.cross_entropy(logits, y[perm])
assert torch.allclose(y_mix.sum(-1), torch.ones(16, dtype=torch.float64))
assert torch.allclose(soft_loss, mixed_losses)
```

## Lever 3: penalties in the loss and the network
These change what the optimizer is rewarded for, and most of them are a single argument.

- **Weight decay** pulls every weight toward zero at each step. With SGD it is the same as an L2 penalty; with Adam it is not, and `torch.optim.AdamW` is the version that decays all weights alike. [L1 and L2 regularization](../regularization-l1-l2/) works through both, and why biases and normalization gains are usually left undecayed.
- **Dropout** zeroes a random fraction of activations during training and scales up the rest, so no unit can rely on a particular partner (Srivastava et al., 2014; more in [dropout](../dropout/)).
- **Label smoothing** (Szegedy et al., 2016) trains against a softened target, so the model is never asked to push one logit to infinity. With \(K\) classes and strength \(\varepsilon\), the target is \(1 - \varepsilon + \varepsilon / K\) for the true class and \(\varepsilon / K\) for every other class, not a flat 0.9 and 0.1.

```python
eps, K = 0.1, 4
logits4, target = torch.randn(5, K), torch.tensor([0, 1, 2, 3, 0])
smooth = torch.full((5, K), eps / K)
smooth[torch.arange(5), target] += 1 - eps
by_hand = -(smooth * logits4.log_softmax(-1)).sum(-1).mean()
assert torch.allclose(F.cross_entropy(logits4, target, label_smoothing=eps), by_hand)
assert torch.allclose(smooth[0], torch.tensor([0.925, 0.025, 0.025, 0.025]))
logits2, target2 = torch.randn(5, 2), torch.tensor([0, 1, 1, 0, 1])   # two classes: targets 0.95 and 0.05
soft2 = torch.where(F.one_hot(target2, 2).bool(), torch.tensor(0.95), torch.tensor(0.05))
assert torch.allclose(F.cross_entropy(logits2, target2, label_smoothing=eps),
                      -(soft2 * logits2.log_softmax(-1)).sum(-1).mean())
```

### Dropout and BatchNorm
BatchNorm also regularizes a little, because each example is normalized with the statistics of whichever batch it lands in, and Ioffe and Szegedy (2015) reported that it reduced, and in some cases removed, the need for dropout. LayerNorm has no such noise, since it uses only the example's own statistics (see [batch norm vs layer norm](../batchnorm-vs-layernorm/)), and Transformers, which use it, rely on explicit dropout and weight decay, though large language models pretrained on a single pass over their data usually turn dropout off.

Putting dropout directly before a BatchNorm layer causes a specific bug, described by Li et al. (2019) as variance shift. During training, dropout inflates the variance of what BatchNorm sees, and BatchNorm's running statistics record the inflated value. At evaluation dropout switches off, the variance drops, and BatchNorm now scales every activation by the wrong amount. The usual fix is ordering: put dropout after the last normalization layer, for example just before the classifier head.

```python
x = torch.randn(4096, 16)                                     # unit-variance activations
drop, bn = torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(16, momentum=None)   # running stats = plain average
drop.train(), bn.train()
with torch.no_grad():
    for _ in range(20):
        bn(drop(x))
assert torch.allclose(bn.running_var.mean(), torch.tensor(2.0), atol=0.1)   # dropout(p=0.5) doubles the variance
drop.eval(), bn.eval()
with torch.no_grad():
    out_var = bn(drop(x)).var(dim=0).mean()
assert 0.4 < out_var < 0.6                                    # at test time the layer outputs half the variance it was built for
```

## Lever 4: early stopping
Gradient descent fits the strong, consistent directions in the data first and the noise later, and stopping early keeps the model in the first phase. Evaluate on validation data every epoch, keep a copy of the best weights, and stop once validation has not improved for `patience` epochs. The stopping point then comes from this run's own validation curve rather than from a guess made before training.

For linear least squares, early stopping acts much like an L2 penalty (Goodfellow, Bengio and Courville work this out in chapter 7): starting from zero, the weights grow along each direction over time, and stopping caps how far they get. The run below has more features than rows, so gradient descent eventually interpolates the training set exactly.

```python
r = np.random.default_rng(1)
n, p = 40, 60
w_true = r.normal(scale=0.3, size=p)


def wide_data(m):
    X = r.normal(size=(m, p))
    return X, X @ w_true + r.normal(scale=1.5, size=m)


X, y = wide_data(n)
X_val, y_val = wide_data(500)
X_test, y_test = wide_data(500)
lr = 0.1 / np.linalg.eigvalsh(X.T @ X / n).max()

w, best_val, best_w, best_epoch, wait, patience = np.zeros(p), np.inf, None, 0, 0, 100
for epoch in range(5000):
    w -= lr * X.T @ (X @ w - y) / n
    val = np.mean((X_val @ w - y_val) ** 2)
    if val < best_val:
        best_val, best_w, best_epoch, wait = val, w.copy(), epoch, 0   # checkpoint
    else:
        wait += 1
        if wait >= patience:
            break

w_end = np.linalg.lstsq(X, y, rcond=None)[0]                 # where training would end up: an exact fit
test_mse = lambda w: np.mean((X_test @ w - y_test) ** 2)
assert np.mean((X @ w_end - y) ** 2) < 1e-12
assert epoch < 5000 - 1 and best_epoch == epoch - patience   # stopped on its own
assert test_mse(best_w) < 0.7 * test_mse(w_end)
assert np.linalg.norm(best_w) < 0.6 * np.linalg.norm(w_end)  # stopping early keeps the weights small
```

Since the stopping point is chosen on the validation set, the validation score at that point is optimistic. Report the final number on a separate test set, as the asserts above do.

## Lever 5: capacity
Fewer layers, fewer units, shallower trees, more neighbours in k-NN. For classical models this is the first knob, not the last: an unrestricted decision tree memorizes its training set, and limiting its depth or leaf size is the main regularizer it has.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

X, y = sine_data(300, np.random.default_rng(5))
deep = DecisionTreeRegressor(random_state=0).fit(X, y)
tuned = GridSearchCV(DecisionTreeRegressor(random_state=0),
                     {"max_depth": range(1, 12), "min_samples_leaf": [1, 5, 10, 20]}, cv=5).fit(X, y)
assert mse(deep, X, y) == 0                                  # one leaf per point
assert mse(tuned, X_va, y_va) < 0.8 * mse(deep, X_va, y_va)
assert tuned.best_params_["max_depth"] < deep.get_depth()
```

On 300 sine points the unrestricted tree fits its training set exactly, and the tree tuned by 5-fold cross-validation has less than 0.8 times its validation error.

Deep networks often behave the other way: a bigger, well-regularized network frequently beats a smaller one, and test error, after peaking near the size at which the model can just fit the training set exactly, can fall again as the model grows past it, the "double descent" of Belkin et al. (2019) and Nakkiran et al. (2019). Shrinking a network saves inference cost but is rarely the best way to buy accuracy.

## Which lever, in what order
| Lever | Cost | Changes | Reach for it when |
|---|---|---|---|
| Early stopping | Almost none | Training length | Always; keep the best checkpoint |
| Weight decay (AdamW) | One argument | Loss / optimizer | By default for neural networks; tune the strength |
| Augmentation | Low to medium | Data | Known invariances (images, audio, text) |
| Dropout | Low | Network | Large fully connected layers and Transformers |
| Label smoothing | One argument | Targets | Classification with overconfident outputs |
| Capacity limits | Low | Model | First knob for trees, k-NN and small tabular models |
| More data | Often high | Data | The learning curve is still falling |

Cheapest first: early stopping and weight decay, then the augmentations your domain supports, then the learning curve to decide whether collecting data is worth it. Every lever except more data trades a little bias for less variance ([bias–variance trade-off](../bias-variance-tradeoff/)); more data lowers variance at no cost in bias, which is why it is worth paying for when the learning curve says so.

## Pitfalls
- **Reading training loss from the training loop.** It is averaged with dropout and augmentation on and can sit above validation loss for a healthy model. Re-evaluate the training set in evaluation mode before diagnosing.
- **Overfitting the validation set.** Every choice made by looking at validation (epochs, \(\lambda\), architecture) leaks a little of it into the model. Keep a test set you look at rarely.
- **A small gap that hides leakage.** Preprocessing fitted on all the data, duplicates across the split, or a feature derived from the label make validation look as good as training.
- **Adam with `weight_decay`.** In `torch.optim.Adam` that argument is an L2 penalty that adaptive scaling weakens for some weights and strengthens for others. Use AdamW.

Every experiment here is small: polynomials, linear models and trees on synthetic data whose noise I set, so they show what each lever does, not how much it buys on a given model. Weight decay has no experiment of its own; its L2 form appears only as the ridge fit that noisy inputs reproduce. For dropout, label smoothing and mixup I check only the mechanics (the targets, the losses, the variance shift) without training a network to watch the gap close. Double descent is cited, not reproduced. On a real network, its own learning curves and validation set decide which lever pays.

Chapter 7 of Goodfellow, Bengio and Courville's Deep Learning (2016) covers most of these levers, including early stopping as an L2 penalty; the noise-as-ridge result is from Bishop (Neural Computation, 1995). AdamW is from Loshchilov and Hutter (ICLR 2019), batch normalization from Ioffe and Szegedy (ICML 2015), and the mixup calibration result from Thulasidasan et al. (NeurIPS 2019). Mixup is from Zhang, Cisse, Dauphin and Lopez-Paz (ICLR 2018), Cutout from DeVries and Taylor (2017), SpecAugment from Park et al. (2019), dropout from Srivastava et al. (JMLR 2014), label smoothing from Szegedy et al. (CVPR 2016), and the dropout and BatchNorm variance shift from Li, Chen, Hu and Yang (CVPR 2019).
