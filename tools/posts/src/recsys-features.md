---
title: Features for recommendation: user, item, context and the match between them
summary: The four kinds of feature a recommender ranks with, built on a simulated click log: counts over time windows, item rates smoothed with a fitted Beta prior, trend features that do not reward size, cross features that make the ranking personal, and the point-in-time rule that keeps the offline numbers honest.
group: Recommender systems
order: 3
tags: recommendation · feature engineering · ranking · data leakage
---
Compute an item's click rate over the whole of a simulated 30-day log and a ranker's offline AUC rises to 0.725; served with the rates it could actually have known, it scores 0.690, below the honest model's 0.693. A recommender's ranking model answers one question many times a second: how likely is this user to engage with this item, right now? The model can only be as good as what it is told about the three things in that sentence (the user, the item, the moment) and, above all, about how well the user and the item fit each other.

"Use the user's history" is not a feature. A feature has a definition precise enough to compute twice and get the same answer: what is counted, over which time window, as of which moment, and smoothed how. This post sorts features into four kinds, builds one or two of each on a simulated click log, and then shows how breaking point-in-time correctness makes a feature look better offline than it is. On that log, a logistic regression goes from a test AUC of 0.603 with user features alone to 0.693 with all four kinds.

## Four kinds of feature
| kind | question it answers | examples | how fast it changes |
|---|---|---|---|
| user | who is looking? | clicks in the last hour, day and week; long-run CTR; recent item ids; a learned user embedding | minutes to months |
| item | what is being shown? | category, price, creator; text or image embedding; smoothed CTR; trend | static to minutes |
| context | in what situation? | hour of day, day of week, device, surface, location | every request |
| cross | how well do these two fit? | user's share of clicks in this item's category; user–item embedding similarity; item price relative to what the user usually pays | every request |

User, item and context features each describe one side. A cross feature is the only kind that says something about **this pair**. That matters more than it first sounds: in a model that adds up its inputs, the user and context terms shift every candidate's score for a request by the same amount, so without a cross, every user sees the items in the same order. "This user likes this kind of thing" is what personalisation means, and only a cross (or a model that learns one) can express it.

## A click log to work with
The simulation has 2,000 users, 1,500 items in 12 categories and 100,000 impressions over 30 days. Each user has a taste over categories (most users like a few), each item a hidden quality, and clicks are a bit more likely in the evening. Exposure is heavy-tailed, as it is in real systems: a few items get most impressions and many get almost none. The first 20 days train a model and the last 10 test it.

```python
import numpy as np

rng = np.random.default_rng(7)
n_users, n_items, n_cats, n_events, days = 2000, 1500, 12, 100_000, 30

item_cat = rng.integers(0, n_cats, n_items)
item_quality = rng.normal(0, 0.6, n_items)
user_taste = rng.dirichlet(np.full(n_cats, 0.3), n_users)   # most users like a few categories
user_bias = rng.normal(0, 0.5, n_users)                     # some users click everything

user_w, item_w = rng.pareto(1.5, n_users) + 1, rng.pareto(1.0, n_items) + 1   # heavy-tailed exposure
u = rng.choice(n_users, n_events, p=user_w / user_w.sum())
i = rng.choice(n_items, n_events, p=item_w / item_w.sum())
t = np.sort(rng.uniform(0, days * 86_400, n_events))          # seconds, in time order
evening = (t // 3600 % 24 >= 19).astype(float)

logit = -2.6 + 4.0 * user_taste[u, item_cat[i]] + item_quality[i] + user_bias[u] + 0.4 * evening
p_click = 1 / (1 + np.exp(-logit))
y = (rng.random(n_events) < p_click).astype(int)
train = t < 20 * 86_400                                      # first 20 days train, last 10 test

imps = np.bincount(i[train], minlength=n_items)
print(f"CTR {y.mean():.3f}; median item has {np.median(imps):.0f} training impressions, "
      f"{(imps < 10).mean():.0%} have fewer than 10")
```

The overall CTR is about 11%, and more than a third of the items have fewer than ten training impressions. That second number drives the item section below.

## User features: counts over several windows
The workhorse user features are counts and rates over several windows: the last hour, day, week and month. Short windows capture intent, long ones habit, and together they tell a user on a binge right now from one who has been steady for months.

With each user's past click times in a sorted list, a window ending now is one binary search, checked here against a brute-force count.

```python
from bisect import bisect_left


def window_count(times, now, width):
    """Events in [now - width, now), given a sorted list of past event times."""
    return len(times) - bisect_left(times, now - width)


past = sorted(rng.uniform(0, 10 * 86_400, 300).tolist())
for now in (2 * 86_400, 9.5 * 86_400, 10 * 86_400):
    for width in (3600, 86_400, 7 * 86_400):
        brute = sum(now - width <= s < now for s in past)
        assert window_count([s for s in past if s < now], now, width) == brute
```

Beyond counts there are two richer user features. The **sequence** of recently viewed item and category ids feeds models that attend over the history, such as the Deep Interest Network of Zhou et al. (2018); see [sequential user modeling](../sequential-user-modeling/). And a **learned user embedding** comes from matrix factorization, a graph model, or simply an average of the embeddings of items the user engaged with.

Demographics such as age, gender and location are often restricted by law or policy, and they need review before use. Replacing them with behavioural features does not make the fairness question go away: behaviour can encode the same attributes, so the model's outputs still need checking across groups.

## Item features: rates need a prior
**Content** features (category, tags, price, creator, text and image embeddings) exist the moment an item is created, which makes them the main tool for [cold start](../cold-start/). **Engagement statistics** (CTR, completion rate, likes per view) measure quality, but only once there is enough data.

A raw rate on little data is mostly noise. An item clicked once in one impression has a CTR of 100%, and a model that trusts it will push that item to the top. The fix is the Beta prior from [uncertainty in a rate](../beta-uncertainty/): add \(K\) pseudo-impressions at the global rate \(M\),

\[ \widehat{\text{CTR}} = \frac{\text{clicks} + K M}{\text{impressions} + K}. \]

Items with little data are pulled toward \(M\) and items with a lot keep their own rate. \(K\) need not be guessed: the spread of raw CTRs across items is the spread of true CTRs plus binomial noise, so subtracting the expected noise variance (about \(M(1 - M)/n\) for an item with \(n\) impressions) gives the true spread, and a Beta with mean \(M\) and that variance has \(K = M(1 - M)/\sigma^2 - 1\). The prior is fitted on the training period and then frozen.

```python
c_tr = np.bincount(i[train], y[train], n_items)
n_tr = np.bincount(i[train], minlength=n_items)
seen = n_tr > 0
raw = c_tr[seen] / n_tr[seen]

# method of moments: spread of raw CTRs = spread of true CTRs + binomial noise
M = c_tr.sum() / n_tr.sum()
var_true = raw.var() - np.mean(M * (1 - M) / n_tr[seen])
K = M * (1 - M) / var_true - 1                               # prior Beta(K*M, K*(1-M))
smoothed = (c_tr[seen] + K * M) / (n_tr[seen] + K)
print(f"prior mean {M:.3f}, prior strength {K:.1f} pseudo-impressions")

# how close is each estimate to the item's true average click probability?
true_ctr = np.bincount(i[train], p_click[train], n_items)[seen] / n_tr[seen]
few = n_tr[seen] < 10
mse_raw = np.mean((raw[few] - true_ctr[few]) ** 2)
mse_smooth = np.mean((smoothed[few] - true_ctr[few]) ** 2)
print(f"items with < 10 impressions: MSE raw {mse_raw:.4f}, smoothed {mse_smooth:.4f}")
assert mse_smooth < mse_raw / 2
```

The fitted prior is worth about 24 impressions, and on items with fewer than ten impressions it cuts the squared error by about a factor of three.

The **Wilson score lower bound** is often suggested for the same problem. It is the lower end of a confidence interval for the rate, so it is deliberately pessimistic: a sensible way to sort a "top rated" list, but not an estimate of the rate. As a model input, the smoothed rate plus the impression count is the better choice.

```python
from scipy import optimize


def wilson_lower(clicks, n, z=1.96):
    p = clicks / n
    centre = p + z * z / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / (1 + z * z / n)


# the Wilson interval is every p that a z-test would not reject; find its lower end by root finding
for clicks, n in [(1, 1), (3, 4), (300, 400)]:
    f = lambda p: abs(clicks / n - p) - 1.96 * np.sqrt(p * (1 - p) / n)
    assert np.isclose(wilson_lower(clicks, n), optimize.brentq(f, 1e-9, clicks / n - 1e-12))

print({f"{c}/{n}": round(wilson_lower(c, n), 3) for c, n in [(1, 1), (3, 4), (300, 400)]})
assert wilson_lower(1, 1) < wilson_lower(300, 400)

# as an estimate of the rate it is biased low: averaged over items it is far below the overall CTR
wilson_items = wilson_lower(c_tr[seen], n_tr[seen])
print(f"mean Wilson bound {wilson_items.mean():.3f} vs overall CTR {M:.3f}")
assert wilson_items.mean() < M / 2 and np.all(wilson_items <= raw + 1e-12)
```

One click in one impression scores 0.207 and 300 in 400 scores 0.705. Averaged over the simulated items the bound is 0.047, less than half the overall training CTR of 0.115, which is exactly what a lower bound on thin data should do and exactly why it is a poor stand-in for the rate itself.

### Trend
To surface items that are taking off, compare recent activity with the period before. The raw difference, clicks this hour minus clicks last hour, mostly measures size: a big item's ordinary noise beats a small item's genuine breakout. A smoothed log ratio measures growth instead.

```python
def trend(now, before, k=20):
    """Log growth of a count, with k pseudo-counts so that 0 -> 3 is not infinite growth."""
    return np.log((now + k) / (before + k))


hits = {"steady hit": (100_400, 100_000), "breaking out": (400, 20), "noise": (3, 0)}
assert 100_400 - 100_000 < np.sqrt(100_400 + 100_000)        # the hit's change is within one Poisson sd
by_diff = max(hits, key=lambda k: hits[k][0] - hits[k][1])
by_trend = max(hits, key=lambda k: trend(*hits[k]))
assert by_diff == "steady hit" and by_trend == "breaking out"
```

## Context features
Context describes the request: hour, day of week, holidays, device, the surface (home feed, search, a detail page) and location for anything local. Hour and day are cyclic, so encode them as buckets or sine and cosine pairs, not as a number where 23 is far from 0. Context matters most through interactions: what someone wants on a phone at 8 a.m. differs from what they want on a TV on Friday night.

## Cross features: the match
The loop below builds all the features for every impression **as of the moment it happened**: it reads the running counts first and only then adds the event to them. The cross feature is the share of the user's past clicks that fell in this item's category, with add-one smoothing.

```python
u_imp, u_clk = [0] * n_users, [0] * n_users
i_imp, i_clk = [0] * n_items, [0] * n_items
u_cat_clk = [[0] * n_cats for _ in range(n_users)]
click_times = [[] for _ in range(n_users)]
cat_of, rows = item_cat.tolist(), []
for uu, ii, now, ev, clicked in zip(u.tolist(), i.tolist(), t.tolist(), evening.tolist(), y.tolist()):
    c, ct = cat_of[ii], click_times[uu]
    rows.append((
        (u_clk[uu] + K * M) / (u_imp[uu] + K),             # user: smoothed CTR (reusing the item prior)
        window_count(ct, now, 86_400),                      # user: clicks in the last day
        window_count(ct, now, 7 * 86_400),                  # user: clicks in the last week
        (i_clk[ii] + K * M) / (i_imp[ii] + K),              # item: smoothed CTR
        np.log1p(i_imp[ii]),                                # item: how much evidence there is
        ev,                                                 # context: evening
        (u_cat_clk[uu][c] + 1) / (u_clk[uu] + n_cats),      # cross: user's share of clicks in this category
    ))
    u_imp[uu] += 1                                          # only now does the event join the history
    i_imp[ii] += 1
    if clicked:
        u_clk[uu] += 1; i_clk[ii] += 1; u_cat_clk[uu][c] += 1; ct.append(now)
X = np.array(rows)
```

A logistic regression trained on the first 20 days and tested on the last 10 shows what each kind adds.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

test = ~train


def evaluate(cols, X_fit=X, X_eval=X):
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_fit[train][:, cols], y[train])
    p = model.predict_proba(X_eval[test][:, cols])[:, 1]
    return roc_auc_score(y[test], p), log_loss(y[test], p)


ladder = {"user": [0, 1, 2], "+ item": [0, 1, 2, 3, 4], "+ context": [0, 1, 2, 3, 4, 5],
          "+ cross": [0, 1, 2, 3, 4, 5, 6]}
results = {name: evaluate(cols) for name, cols in ladder.items()}
for name, (auc, ll) in results.items():
    print(f"{name:10s} AUC {auc:.3f}  log loss {ll:.4f}")
aucs = [auc for auc, _ in results.values()]
assert aucs == sorted(aucs) and aucs[-1] - aucs[-2] > 0.005
```

| features | test AUC | test log loss |
|---|---|---|
| user | 0.603 | 0.3480 |
| + item | 0.679 | 0.3347 |
| + context | 0.682 | 0.3336 |
| + cross | 0.693 | 0.3304 |

A linear model cannot invent "this user likes this category" out of a user column and an item column; it has to be handed the cross. Context helps only a little here because the simulation gives it a small, purely additive effect.

Two honest caveats about these numbers. First, the cross adds less than the item features did. Taste is a strong signal in the simulation, but the median user has only a couple of clicks in the training window, so their category shares are mostly smoothing; the estimated cross scores far below the true taste it is trying to measure. Crosses earn more as histories grow longer. Second, the absolute AUCs look modest, but the ceiling is low: even a model that knew every impression's true click probability would score only about 0.75, because a click with probability 0.3 still fails most of the time.

```python
oracle = roc_auc_score(y[test], p_click[test])
cross_alone = evaluate([6])[0]
taste_alone = roc_auc_score(y[test], user_taste[u, item_cat[i]][test])
print(f"oracle AUC {oracle:.3f} | estimated cross alone {cross_alone:.3f} | true taste alone {taste_alone:.3f}")
assert 0.72 < oracle < 0.78 and oracle > results["+ cross"][0]
assert cross_alone < taste_alone - 0.05
```

### Where crosses live
Not every model can use a cross feature. A **two-tower** retrieval model computes the user vector and the item vector separately and scores the pair by their dot product, which is what lets it search millions of items with a nearest-neighbour index. It has no input slot for a feature that needs both sides, so user–item interactions reach it only through what the dot product of two fixed-size vectors can express. Hand-built crosses belong in the [ranking](../recommendation-ranking/) stage, where the model sees each user–item pair together; see [two-tower retrieval](../two-tower-retrieval/) for the other half of that split.

Ranking models can also learn crosses themselves. Tree ensembles find interactions through their splits ([decision trees to boosting](../decision-trees-to-boosting/)), and the Deep & Cross Network of Wang et al. (2017) adds layers that build explicit polynomial crosses of the inputs. Even then, a few hand-made crosses that encode known structure, such as category affinity or price relative to the user's usual spend, tend to make training faster and give a strong baseline to compare against.

## Point-in-time correctness
A costly feature bug is computing a feature with information that would not exist when the prediction is made. The common version is an item CTR computed over the whole log and joined onto every row. For a row in the training set, that number includes clicks from the future and, for items with few impressions, a large share of the row's own label.

```python
X_leaky = X.copy()
c_all, n_all = np.bincount(i, y, n_items), np.bincount(i, minlength=n_items)
X_leaky[:, 3] = c_all[i] / n_all[i]          # whole-log item CTR: includes the future and the row's own label
X_leaky[:, 4] = np.log1p(n_all[i])

all_cols = ladder["+ cross"]
offline = evaluate(all_cols, X_leaky, X_leaky)   # what the offline report says
served = evaluate(all_cols, X_leaky, X)          # what production sees: point-in-time features
honest = results["+ cross"]
print(f"leaky, offline {offline[0]:.3f} | leaky, served {served[0]:.3f} | point-in-time {honest[0]:.3f}")
assert offline[0] > honest[0] + 0.02 and served[0] < honest[0]
assert served[1] > honest[1]
```

The leaky model reports an AUC of 0.725 offline. Served with the features production can actually compute, it scores 0.690, below the honest model's 0.693, with a worse log loss. The offline gain was the label leaking through the feature, and the model learned to lean on a feature that is weaker in production than in training.

The rule is to compute every training feature **as of the timestamp of the event it describes**, using only data that had arrived by then. Feature stores call this a point-in-time join. Two refinements matter in practice. Labels arrive late (a purchase may follow a click by days), so a count "as of" time \(t\) should only include outcomes that were known at \(t\). And the code that computes features for training should be the same code that serves them, or the two drift apart, which is called training–serving skew.

## Pitfalls
- **A feature without a window.** "Number of clicks" grows with account age and means different things for different users. Fix the window, or use a rate.
- **Raw rates on small counts.** Smooth them with a prior and also pass the count, so the model can judge how much the rate is worth.
- **Feedback loops.** Engagement features are measured on what the system chose to show, and at the positions it chose; see [position bias](/posts/position-bias/). An item the system never shows never earns a good CTR.
- **Leakage.** Any aggregate over a period that overlaps or follows its event. Evaluate on a time-based split, never a random one.
- **Two code paths.** Where possible, log the features actually served and train on those.

Every number here comes from one simulated log, seven features and a logistic regression, so the ladder shows what each kind of feature can add, not how much it adds on real data. Two questions stay open: how much the cross earns once histories grow longer, and whether crosses learned by trees or a Deep & Cross Network make the hand-made ones redundant.

For how these features fit into a full retrieval and ranking system, see Eugene Yan's "System Design for Recommendations and Search" and Google's Recommendation Systems course for developers; the explanations, simulation and code here are original.
