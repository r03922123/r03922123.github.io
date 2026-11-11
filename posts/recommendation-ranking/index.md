# Recommendation ranking: retrieve, score, re-rank

A recommender has to pick ten items out of millions for one person in a few tens of milliseconds. No single model can afford to look carefully at every item, so almost every production system is a funnel: a fast stage that throws away nearly everything, a slow stage that orders what is left, and a last pass that applies the rules a sorted list of scores cannot express.

I run that funnel on a simulated catalogue of 240 users and 2,000 items where the true preferences are known, so I can check each stage: what retrieval keeps, what the ranker adds, whether the ranking loss matters, and what diversity costs. The ranker lifts NDCG@10 from 0.44 to 0.67, mostly through one cross feature, and pointwise, RankNet and LambdaRank training tie.

## Why two stages
Suppose a ranking model needs 20 microseconds to score one (user, item) pair with all its features. Scoring a catalogue of 10 million items would take 200 seconds per request. Scoring 500 candidates takes 10 milliseconds.

| Stage | In, out | Model | What it optimizes |
|---|---|---|---|
| Retrieval | millions to hundreds | dot product of a user and an item embedding, served by an approximate nearest neighbour index | recall: do not lose the good items |
| Ranking | hundreds to tens | gradient-boosted trees or a neural network over rich features, including user-item crosses | the order at the top of the list |
| Re-ranking | tens to the final page | rules and light models | diversity, freshness, business constraints |

Retrieval is cheap because the user and the item never meet inside the model. A [two-tower model](../two-tower-retrieval/) computes item embeddings offline; at request time it computes one user vector and asks a [vector index](../vector-databases/) for the nearest items. The price is that the score must be a dot product, so anything that depends on the pair, such as "this item is in the category this user binges", has to be squeezed into the embeddings. The ranker sees only a few hundred items, so it can afford explicit pair features. Covington, Adams and Sargin (2016) give a well-known description of this design for YouTube.

## A small world to rank
The simulation has 240 users and 2,000 items. Each user's true affinity for an item is a taste match (a dot product of hidden vectors), plus the item's general quality, plus a bonus when the item is in the user's favourite category, plus noise. Affinities are cut into grades 0 to 3, so about 10 items per user are graded 2 or above: those are the relevant ones.

```python
import numpy as np
from scipy.special import expit
from sklearn.metrics import ndcg_score

rng = np.random.default_rng(7)
n_users, n_items, dim, n_cats = 240, 2000, 16, 12

U = rng.normal(size=(n_users, dim)) / dim**0.25     # hidden user tastes
V = rng.normal(size=(n_items, dim)) / dim**0.25     # hidden item traits
quality = rng.normal(size=n_items)                  # some items are simply better
cat = rng.integers(n_cats, size=n_items)            # item category
fav = rng.integers(n_cats, size=n_users)            # each user's favourite category
match = (cat == fav[:, None]).astype(float)         # users x items

affinity = U @ V.T + 0.7 * quality + 1.5 * match + rng.normal(0, 0.5, (n_users, n_items))
grade = np.digitize(affinity, np.quantile(affinity, [0.97, 0.995, 0.999]))   # 0 ignored ... 3 loved
relevant = grade >= 2
assert 8 <= relevant.sum(1).mean() <= 12
```

## Stage 1: retrieval
A trained two-tower model learns embeddings close to the hidden ones, plus an item bias; here that is simulated with noise. The simulated retrieval model does not see the category bonus, a property of the pair: a real two-tower model could learn it only by spending embedding dimensions on it, and this one does not. It keeps the top 60 of 2,000 items per user, found by `argpartition` in linear time; a full sort gives the same set.

```python
U_emb = U + rng.normal(0, 0.3, U.shape)
V_emb = V + rng.normal(0, 0.3, V.shape)
retrieval = U_emb @ V_emb.T + 0.5 * quality         # dot product plus an item bias

K = 60
cand = np.argpartition(-retrieval, K, axis=1)[:, :K]
rows = np.arange(n_users)[:, None]
full_sort = np.argsort(-retrieval, axis=1)[:, :K]
assert all(set(a) == set(b) for a, b in zip(cand, full_sort))

has_rel = relevant.sum(1) > 0
recall = (relevant[rows, cand].sum(1)[has_rel] / relevant.sum(1)[has_rel]).mean()
assert recall > 10 * K / n_items                    # random picks would keep 3% of relevant items
print(f"recall@{K}: {recall:.2f}")
```

Recall at 60 comes out near 0.43, against 0.03 for random picks. Every relevant item lost here is lost for good, since the ranker only reorders what it is given. That is why retrieval is judged by recall rather than precision, and why real systems merge several retrieval sources (embeddings, popularity, recent items, follows) before ranking.

## Measuring a ranked list
The ranker's job is the order at the top, so its metric should care about position. **Discounted cumulative gain** adds each item's graded relevance \(g_r\), discounted by its rank \(r\); dividing by the best possible value gives NDCG, between 0 and 1. Järvelin and Kekäläinen (2002) introduced the measure; the \(\log_2(r + 1)\) discount below is the form most libraries use:

\[ \mathrm{DCG@}k = \sum_{r=1}^{k} \frac{g_r}{\log_2(r + 1)}, \qquad \mathrm{NDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k} \]

where IDCG is the DCG of the list sorted by true relevance. Some libraries use the gain \(2^{g_r} - 1\), which favours top grades more; scikit-learn uses \(g_r\), and so does this post. **MRR**, the mean of \(1/\text{rank}\) of the first relevant item, suits tasks with one right answer; **recall@k** ignores order inside the cut.

```python
def dcg(gains, k):
    gains = gains[..., :k]
    return (gains / np.log2(np.arange(2, gains.shape[-1] + 2))).sum(-1)


def ndcg_at_k(rel, score, k=10):
    """NDCG@k per row: rel are graded labels, score is what the model sorts by."""
    order = np.argsort(-score, axis=-1)
    return dcg(np.take_along_axis(rel, order, -1), k) / dcg(-np.sort(-rel, axis=-1), k)


def reciprocal_rank(rel, score, threshold=2):
    hits = np.flatnonzero(rel[np.argsort(-score)] >= threshold)
    return 1 / (hits[0] + 1) if len(hits) else 0.0


# a ranked list with grades 3, 2, 0, 1; the ideal order is 3, 2, 1, 0
by_hand = (3 + 2 / np.log2(3) + 0 + 1 / np.log2(5)) / (3 + 2 / np.log2(3) + 1 / np.log2(4))
rel, score = np.array([3., 2., 0., 1.]), np.array([4., 3., 2., 1.])
assert np.isclose(ndcg_at_k(rel, score, 4), by_hand) and round(by_hand, 3) == 0.985
assert reciprocal_rank(np.array([0., 1., 3.]), np.array([3., 2., 1.])) == 1 / 3
```

## Features the ranker can use
Because the ranker sees each (user, item) pair explicitly, it can use four kinds of feature:

| Kind | Examples |
|---|---|
| User | activity level, historical click and purchase rates, long-term interest embeddings |
| Item | popularity, age and freshness, price, content embeddings |
| Cross | user's past engagement with this item's category or creator, similarity of the item to the user's recent history |
| Context | time of day, device, where the slot is on the page, what else is on the page |

Cross features often carry the most signal, because they describe the pair directly, while a retrieval dot product can only approximate them through whatever its embeddings happened to learn. The simulated ranker gets three: the retrieval score, the item's observed popularity (a noisy view of its quality) and the cross feature "item is in the user's favourite category". Two thirds of the users train it, the rest test it. Every feature must be computed as of the moment of the impression, or it leaks the future into training; see [recsys features](../recsys-features/).

```python
y = grade[rows, cand].astype(float)                   # users x candidates
X = np.stack([retrieval[rows, cand],                  # from stage 1
              (quality + rng.normal(0, 0.3, n_items))[cand],   # item: observed popularity
              match[rows, cand]], -1)                 # cross: favourite category
keep = y.max(1) > 0                                   # NDCG needs at least one nonzero grade
X, y, cand, user = X[keep], y[keep], cand[keep], np.flatnonzero(keep)
n = len(y)
train, test = slice(0, 2 * n // 3), slice(2 * n // 3, None)
mu, sd = X[train].mean((0, 1)), X[train].std((0, 1))  # scale with training statistics only
X = (X - mu) / sd

ours = ndcg_at_k(y[test], X[test][..., 0])
assert np.isclose(ours.mean(), ndcg_score(y[test], X[test][..., 0], k=10))
```

## Three ways to train a ranker
**Pointwise.** Treat each (user, item) pair as a separate example and regress its label, or classify click versus no click. The list never appears in the loss. With a linear scorer and squared error this is ordinary least squares:

\[ \mathcal{L}_{\text{point}} = \sum_{u,i} \left(s_{ui} - y_{ui}\right)^2 \]

**Pairwise.** Ranking only needs the right order, so penalize pairs in the wrong order. RankNet (Burges et al., 2005) applies a logistic loss to the score difference of every pair in which item \(i\) should beat item \(j\):

\[ \mathcal{L}_{\text{pair}} = \sum_{(i,j):\,y_i > y_j} \log\left(1 + e^{-(s_i - s_j)}\right) \]

**LambdaRank** (Burges, Ragno and Le, 2006). RankNet treats a mix-up at ranks 50 and 51 like one at ranks 1 and 2, but NDCG only cares about the top. LambdaRank keeps the RankNet gradient for each pair and multiplies it by how much NDCG would change if the two items swapped places. With linear gains that change is

\[ |\Delta \mathrm{NDCG}_{ij}| = \frac{|y_i - y_j|\;|d_i - d_j|}{\mathrm{IDCG}}, \qquad d_r = \frac{1}{\log_2(r + 1)} \text{ for } r \le k, \text{ else } 0 \]

so pairs that both sit below rank \(k\) get no weight at all. The pair's push on the scores, the "lambda", is

\[ \lambda_{ij} = -\frac{|\Delta \mathrm{NDCG}_{ij}|}{1 + e^{\,s_i - s_j}} \]

and each item's gradient is the sum of the lambdas of the pairs it should win minus those of the pairs it should lose. The gradients are defined directly; there is no smooth loss whose derivative they are exactly, which is why the method is described by its lambdas. **LambdaMART** is the same lambdas used as the targets for [gradient-boosted trees](../decision-trees-to-boosting/); LightGBM's `lambdarank` objective and XGBoost's `rank:ndcg` implement it.

The code computes both pairwise losses and their gradients with respect to the scores, checks the gradient against a finite difference, then fits the three linear rankers.

```python
def pair_loss(s, y, k=None):
    """RankNet loss and d loss / d scores, per list; with k, the LambdaRank weighting."""
    diff = s[:, :, None] - s[:, None, :]               # s_i - s_j
    better = y[:, :, None] > y[:, None, :]             # i should rank above j
    w = better.astype(float)
    if k is not None:                                  # weight each pair by |delta NDCG@k| of a swap
        rank = np.argsort(np.argsort(-s, 1), 1)
        disc = np.where(rank < k, 1 / np.log2(rank + 2), 0.0)
        ideal = dcg(-np.sort(-y, 1), k)
        w = w * np.abs(y[:, :, None] - y[:, None, :]) * np.abs(disc[:, :, None] - disc[:, None, :]) \
            / ideal[:, None, None]
    n_pairs = better.sum()
    loss = (w * np.logaddexp(0, -diff)).sum() / n_pairs
    lam = -w * expit(-diff) / n_pairs                  # the lambdas
    return loss, lam.sum(2) - lam.sum(1)               # pairs i wins, minus pairs i loses


s0 = X[train] @ np.array([0.3, -0.2, 0.5])
for k in (None, 10):
    _, grad = pair_loss(s0, y[train], k)
    for u, i in [(3, 5), (0, np.argmax(s0[0])), (7, np.argmin(s0[7]))]:
        bump = np.zeros_like(s0)
        bump[u, i] = 1e-6
        numeric = (pair_loss(s0 + bump, y[train], k)[0] - pair_loss(s0 - bump, y[train], k)[0]) / 2e-6
        assert grad[u, i] != 0 and np.isclose(numeric, grad[u, i], rtol=1e-4)

# the |delta NDCG| weight equals the real change from swapping two items in the ranked list
s_u, y_u = s0[0], y[train][0]
order = np.argsort(-s_u)
for a, b in [(0, 1), (0, 7), (2, 30), (12, 40)]:          # positions in the ranked list
    i, j = order[a], order[b]
    swapped = s_u.copy()
    swapped[[i, j]] = swapped[[j, i]]
    real = abs(ndcg_at_k(y_u, swapped) - ndcg_at_k(y_u, s_u))
    d = lambda r: 1 / np.log2(r + 2) if r < 10 else 0.0
    formula = abs(y_u[i] - y_u[j]) * abs(d(a) - d(b)) / dcg(-np.sort(-y_u), 10)
    assert np.isclose(real, formula)


def fit_pairwise(X, y, k=None, steps=60, lr=0.3):
    w = np.zeros(X.shape[-1])
    for _ in range(steps):                             # fixed-length steps: lambdas are tiny numbers
        _, g = pair_loss(X @ w, y, k)
        gw = np.einsum("uif,ui->f", X, g)
        w -= lr * gw / np.linalg.norm(gw)
    return w


def fit_pointwise(X, y):
    A = np.c_[X.reshape(-1, X.shape[-1]), np.ones(y.size)]
    return np.linalg.lstsq(A, y.ravel(), rcond=None)[0]


w_point = fit_pointwise(X[train], y[train])
w_rank = fit_pairwise(X[train], y[train])
w_lambda = fit_pairwise(X[train], y[train], k=10)
w_nocross = fit_pointwise(X[train][..., :2], y[train])

result = {
    "retrieval score only": ndcg_at_k(y[test], X[test][..., 0]).mean(),
    "pointwise, no cross feature": ndcg_at_k(y[test], X[test][..., :2] @ w_nocross[:2]).mean(),
    "pointwise": ndcg_at_k(y[test], X[test] @ w_point[:3]).mean(),
    "RankNet": ndcg_at_k(y[test], X[test] @ w_rank).mean(),
    "LambdaRank": ndcg_at_k(y[test], X[test] @ w_lambda).mean(),
}
for name, v in result.items():
    print(f"{name:28s} NDCG@10 = {v:.3f}")

losses = [result["pointwise"], result["RankNet"], result["LambdaRank"]]
assert max(losses) - min(losses) < 0.01                       # the loss barely matters here
assert result["pointwise"] - result["pointwise, no cross feature"] > 0.1   # the cross feature does
assert min(losses) - result["retrieval score only"] > 0.15
```

| Ranker on the test users | NDCG@10 |
|---|---|
| retrieval score only | 0.438 |
| pointwise, no cross feature | 0.489 |
| pointwise | 0.666 |
| RankNet | 0.665 |
| LambdaRank | 0.665 |

Two results. The ranker lifts NDCG@10 from 0.44 to 0.67, and most of that comes from the cross feature: without it the same candidates reach only 0.49. And the three losses tie. With a few well-behaved features and a linear ranker, the three losses find about the same order here. The NDCG weighting earns its keep when the model cannot fit everything and must choose where to be wrong; LambdaRank chooses to be wrong at the bottom of the list. LambdaMART remains a strong baseline on learning-to-rank benchmarks, but the priorities stay the same: candidates and features first, loss second.

## Scores versus probabilities
Pairwise and lambda losses depend only on score differences. Adding a constant to every score in a list leaves them unchanged, so the scores have no absolute meaning. A pointwise model trained with squared error or log loss and an intercept matches the average label instead.

```python
s = X[train] @ w_rank
assert np.isclose(pair_loss(s, y[train])[0], pair_loss(s + 5.0, y[train])[0])
fitted = np.c_[X[train].reshape(-1, 3), np.ones(y[train].size)] @ w_point
assert np.isclose(fitted.mean(), y[train].mean())     # least squares with an intercept is calibrated on average

from sklearn.linear_model import LogisticRegression
clicked = (y[train] >= 2).ravel()                     # a binary label, like a click
logit = LogisticRegression(C=1e6, max_iter=1000).fit(X[train].reshape(-1, 3), clicked)
p = logit.predict_proba(X[train].reshape(-1, 3))[:, 1]
assert abs(p.mean() - clicked.mean()) < 1e-4          # log loss with an intercept matches the click rate
```

That difference often decides the design. A feed that ranks by \(P(\text{click}) \cdot \text{value} + \alpha\, P(\text{like})\), or an ad auction that multiplies a bid by predicted click-through rate, needs calibrated probabilities. So many large rankers are pointwise multi-task networks, one head per engagement type, combined by a tuned formula, rather than pure listwise models. Log loss is the usual pointwise choice for clicks; see [cross-entropy vs MSE](../cross-entropy-vs-mse/).

## Re-ranking for diversity
Sorting by score repeats itself. Here the category bonus pushes nearly half of a user's top 10 into their favourite category: right by the per-item labels, which cannot see redundancy, and tiresome for a person scrolling a feed.

Maximal marginal relevance (Carbonell and Goldstein, 1998) builds the list greedily, each time picking the item with the best trade-off between its score and its similarity to what is already chosen:

\[ i^{*} = \arg\max_{i \notin S} \Big[\, \lambda\, \tilde{s}_i - (1 - \lambda) \max_{j \in S} \mathrm{sim}(i, j) \Big] \]

where \(\tilde{s}\) is the score rescaled to \([0, 1]\). Here similarity is 1 for the same category and 0 otherwise.

```python
def mmr(score, cats, k=10, lam=0.8):
    score = (score - score.min()) / (score.max() - score.min())
    chosen, left = [], list(range(len(score)))
    while len(chosen) < k:
        penalty = np.array([max((cats[i] == cats[j] for j in chosen), default=0) for i in left])
        best = left[int(np.argmax(lam * score[left] - (1 - lam) * penalty))]
        chosen.append(best)
        left.remove(best)
    return np.array(chosen)


def list_ndcg(rel, picked, k=10):
    return dcg(rel[picked], k) / dcg(-np.sort(-rel), k)


S, C, Y = X[test] @ w_lambda, cat[cand[test]], y[test]
top10 = np.argsort(-S, 1)[:, :10]
base_div = np.mean([len(set(C[u, top10[u]])) for u in range(len(S))])
base_ndcg = np.mean([list_ndcg(Y[u], top10[u]) for u in range(len(S))])
fav_share = np.mean(C[np.arange(len(S))[:, None], top10] == fav[user[test]][:, None])
assert 0.4 < fav_share < 0.5
for lam in (0.9, 0.8, 0.7):
    picks = [mmr(S[u], C[u], lam=lam) for u in range(len(S))]
    div = np.mean([len(set(C[u, p])) for u, p in enumerate(picks)])
    cost = base_ndcg - np.mean([list_ndcg(Y[u], p) for u, p in enumerate(picks)])
    print(f"lambda {lam}: {div:.1f} categories in the top 10 (was {base_div:.1f}), NDCG cost {cost:+.3f}")
    if lam == 0.9:
        assert div > base_div + 0.8 and cost < 0.01    # diversity almost for free
    if lam == 0.7:
        assert div > base_div + 3 and cost > 0.05      # pushed harder, it starts to hurt relevance
```

At \(\lambda = 0.9\) the top 10 goes from about 5.2 to 6.3 categories at no NDCG cost: a penalty of 0.1 against a weight of 0.9 can only reorder items whose rescaled scores are within about 0.11 of each other. At 0.7 it reaches 8.6 categories and gives up 0.07 NDCG. Offline NDCG can only show the cost; whether users value the variety more shows up only online.

## Where the labels come from
The simulation had true grades. Real rankers learn from logged behaviour, which is biased by the system that produced it.

- **Position bias.** Items shown higher get clicked more regardless of quality, so raw clicks teach the model to copy its own past ranking. [Position bias](/posts/position-bias/) covers the correction with inverse propensity weighting, randomization and position-as-a-feature.
- **Exposure bias.** An item that was never shown is not a negative; it is missing. Reserve a little traffic for exploration.
- **Popularity bias.** Popular items dominate the positives and, with in-batch negatives in retrieval, the negatives too, which over-penalizes them. Retrieval models subtract the log of each item's sampling probability from its logit to correct this (Yi et al., 2019).
- **New items** have no history, so every engagement feature is empty. See [cold start](../cold-start/).

## Offline and online
Offline NDCG, recall and MRR are cheap, so they choose which experiments to run. But they are measured on logs collected under the old ranker, so they favour models that resemble it and cannot reward a good item it never showed. The decision belongs to an online [A/B test](../ab-testing/) on engagement, retention or revenue, with guardrails on latency. [Evaluating ML systems](/posts/evaluating-ml-systems/) follows these metrics from model to business.

## Pitfalls
- **Leaky features.** A "user clicked this category" count computed over the whole log, including the day being predicted, inflates offline metrics and vanishes in production.
- **Averaging NDCG over lists with no relevant item.** IDCG is zero; drop or count those lists explicitly and report which.
- **Training on different candidates than serving.** A ranker trained only on shown items has never seen the odd candidates a new retrieval source sends it.
- **Optimizing the loss before the inputs.** As the table shows, a missing cross feature costs far more than the choice between pointwise, pairwise and LambdaRank training.

These numbers come from one simulated catalogue, linear rankers over three features and true grades in place of logged clicks, so they show how the stages divide the work, not the size of each gain on real data. Two questions stay open: whether the losses still tie with more features or a nonlinear ranker, and whether users value the variety that re-ranking adds, which only an online test can show.

The overall framing of the funnel follows Eugene Yan's [system design for recommendations and search](https://eugeneyan.com/writing/system-design-for-discovery/); the prose, simulation and code here are original. RankNet is due to Burges et al. (ICML 2005), LambdaRank to Burges, Ragno and Le (NIPS 2006), and both with LambdaMART are surveyed in Burges, "From RankNet to LambdaRank to LambdaMART: An Overview" (Microsoft Research, 2010).
