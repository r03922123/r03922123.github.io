# Two-tower retrieval: from millions of items to a few hundred candidates

A recommender with ten million items cannot look carefully at all of them for every request, so it works as a funnel. A cheap first stage, retrieval, pulls a few hundred plausible items from the whole catalogue; a more expensive [ranking model](../recommendation-ranking/) orders those with every feature it has; a last pass applies business rules. Each stage sees fewer items and can spend more on each.

The two-tower model is the standard first stage. One network turns the user into a vector, another turns an item into a vector, and the score is their dot product. That restriction looks severe, and it is exactly what makes retrieval fast. This post builds one, trains it, serves it from an index, and checks most claims in code. On a simulated catalogue, the towers I train find, on average, 0.516 of each test user's next five items among 50 retrieved out of 2,700, against 0.272 for the 50 most popular items and 0.388 without the logQ correction.

## Why retrieval needs a different model
A ranking model is free to cross user and item features: "watched three cooking videos today" times "is a cooking video" is what makes it accurate. The price is one model evaluation per (user, item) pair. At an optimistic microsecond per pair, ten million items cost ten seconds per request.

The two-tower model removes the pairing. If the score has the form

\[ s(u, i) = f(u)^\top g(i), \]

with a user tower \(f\) and an item tower \(g\) that never see each other's inputs, then \(g(i)\) does not depend on the user and can be computed for every item ahead of time. A request runs \(f\) once and then searches for the item vectors with the largest dot product, a nearest-neighbour problem with fast approximate solutions.

The idea goes back to DSSM, which matched search queries to documents this way, and reached recommendation at scale with YouTube's candidate generator (Covington, Adams and Sargin, 2016) and its later two-tower model with item features (Yi et al., 2019).

## A small world to learn
The example invents a ground truth. Items have hidden taste factors and a hidden popularity, users have hidden tastes, and a user picks an item with probability proportional to \(\exp(\text{user} \cdot \text{taste} + \text{popularity})\). The model sees none of that: only item IDs, a noisy 16-number content description of each item (think category, text and image embeddings), and each user's last ten items. Three hundred items are held back as launched after training, never appearing in any history or label.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(0)
torch.manual_seed(0)
torch.set_num_threads(1)

n_items, d_true, d_feat = 3000, 8, 16
taste = rng.normal(size=(n_items, d_true))            # hidden item factors
appeal = rng.normal(size=n_items)                     # hidden popularity
mix = rng.normal(size=(d_true, d_feat)) / np.sqrt(d_true)
content = (taste @ mix + 0.7 * rng.normal(size=(n_items, d_feat))).astype(np.float32)  # what we observe

is_new = np.zeros(n_items, bool)
is_new[rng.choice(n_items, 300, replace=False)] = True  # launched after training: no interactions yet
old_ids, new_ids = np.flatnonzero(~is_new), np.flatnonzero(is_new)


def draw(user, pool, k):
    """k items per user, drawn with probability proportional to exp(user . taste + appeal)."""
    logits = user @ taste[pool].T + appeal[pool]
    p = np.exp(logits - logits.max(1, keepdims=True))
    cdf = np.cumsum(p / p.sum(1, keepdims=True), 1)
    u = rng.random((len(user), k))
    return pool[np.minimum([np.searchsorted(c, x) for c, x in zip(cdf, u)], len(pool) - 1)]


def make_users(n):
    u = rng.normal(size=(n, d_true))
    return draw(u, old_ids, 10), draw(u, old_ids, 5), draw(u, new_ids, 5)  # history, next items, next new items


hist_train, next_train, _ = make_users(3000)
hist_test, next_test, next_new_test = make_users(1000)
```

## The two towers
Each tower can be as deep as it likes and use any features from its own side. The item tower combines a learned ID embedding with the content features. The user tower averages the ID embeddings and content of the user's last ten items and passes that through its own MLP. Both outputs are scaled to unit length, so the score is a cosine, divided by a temperature \(\tau\) in training.

The towers do not have to share nothing, as is sometimes said; here they share the ID embedding table, since a history is made of items. The real rule is narrower: **no input from one side reaches the other side's tower**. They meet only in the final dot product.

```python
class TwoTower(nn.Module):
    def __init__(self, dim=32, tau=0.1):
        super().__init__()
        self.ids = nn.Embedding(n_items + 1, dim)            # row 0: "unknown item"; shared by both towers
        self.content = torch.tensor(content)
        self.item_mlp = nn.Sequential(nn.Linear(dim + d_feat, 64), nn.ReLU(), nn.Linear(64, dim))
        self.user_mlp = nn.Sequential(nn.Linear(dim + d_feat, 64), nn.ReLU(), nn.Linear(64, dim))
        self.tau = tau

    def item_tower(self, items, id_known=True):
        items = torch.as_tensor(items)
        ids = torch.where(torch.as_tensor(id_known), items + 1, 0)
        x = torch.cat([self.ids(ids), self.content[items]], -1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def user_tower(self, history):
        h = torch.as_tensor(history)
        x = torch.cat([self.ids(h + 1).mean(1), self.content[h].mean(1)], -1)   # pooled history
        return F.normalize(self.user_mlp(x), dim=-1)


# The only place the towers meet is a dot product, so a precomputed item matrix gives the same scores.
m0 = TwoTower()
with torch.no_grad():
    V = m0.item_tower(old_ids)                                 # offline, once
    q = m0.user_tower(hist_test[:1])                           # online, per request
    pair = torch.stack([(m0.user_tower(hist_test[:1]) * m0.item_tower(old_ids[j:j + 1])).sum() for j in range(5)])
assert torch.allclose((q @ V.T)[0, :5], pair, atol=1e-6)
```

Real towers carry much more, such as query text, device, time and a [sequence model](../sequential-user-modeling/) over recent behaviour on the user side, and text and image encoders, category and age on the item side, but the shape of the model stays the same.

## Training with in-batch negatives
The data is (user, item they engaged with) pairs, and each pair is a classification over items. A softmax over ten million classes is too expensive, so the batch supplies them: compute all \(B \times B\) scores for a batch of \(B\) pairs, and treat row \(b\) as a softmax whose answer is item \(b\), with the other \(B - 1\) items as negatives. This is the InfoNCE loss from [contrastive loss](../contrastive-loss/):

\[ \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp(s_{bb})}{\sum_{j=1}^{B} \exp(s_{bj})}, \qquad s_{bj} = \frac{f(u_b)^\top g(i_j)}{\tau} - \log q_{i_j} . \]

- **The logQ correction.** In-batch negatives are other users' positives, so popular items turn up as negatives far more often than niche ones, and the model learns to push them down. Subtracting the log of each item's sampling probability \(q_i\) from its logit, in training only, removes that bias. Yi et al. (2019) applied this correction to in-batch softmax, estimating \(q_i\) from streaming counts.
- **Duplicates.** An item that appears twice in a batch is a false negative for the other copy's user, so those entries are masked.

The test is 1,000 new users, scored by recall at 50: the share of each user's next five items found among the 50 retrieved out of 2,700.

```python
rows = np.repeat(np.arange(len(hist_train)), next_train.shape[1])
pos = next_train.ravel()
freq = np.bincount(pos, minlength=n_items) / len(pos)     # how often each item lands in a batch as a positive


def train(log_q=True, epochs=3, batch=512, id_dropout=0.1, seed=0):
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    model = TwoTower()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        order = g.permutation(len(pos))
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            items = pos[b]
            q = model.user_tower(hist_train[rows[b]])
            v = model.item_tower(items, id_known=g.random(len(b)) > id_dropout)
            logits = q @ v.T / model.tau                      # row i: user i against every item in the batch
            if log_q:
                logits = logits - torch.tensor(np.log(freq[items]), dtype=torch.float32)
            dup = torch.as_tensor(items[:, None] == items[None, :]) & ~torch.eye(len(b), dtype=torch.bool)
            loss = F.cross_entropy(logits.masked_fill(dup, -1e9), torch.arange(len(b)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def recall_at(top, truth):
    """Share of each user's held-out items that appear in their retrieved list, averaged."""
    return (top[:, :, None] == truth[:, None, :]).any(1).mean()


def retrieve(model, history, pool, k, id_known=True):
    with torch.no_grad():
        V = model.item_tower(pool, id_known).numpy()
        Q = model.user_tower(history).numpy()
    return pool[np.argpartition(-(Q @ V.T), k, axis=1)[:, :k]], Q, V


K = 50
plain, model = train(log_q=False), train(log_q=True)
r_plain = recall_at(retrieve(plain, hist_test, old_ids, K)[0], next_test)
top, Q, V = retrieve(model, hist_test, old_ids, K)
r_model = recall_at(top, next_test)
r_pop = recall_at(np.tile(old_ids[np.argsort(-freq[old_ids])[:K]], (len(hist_test), 1)), next_test)
r_rand = K / len(old_ids)
print(f"random {r_rand:.3f}  popular {r_pop:.3f}  no logQ {r_plain:.3f}  logQ {r_model:.3f}")
assert r_model > 1.2 * r_plain and r_plain > r_rand and r_model > 1.5 * r_pop

top_plain = retrieve(plain, hist_test, old_ids, K)[0]
print(f"mean label frequency of retrieved items: no logQ {freq[top_plain].mean():.5f}  logQ {freq[top].mean():.5f}")
assert freq[top_plain].mean() < freq[top].mean() / 3       # uncorrected: the popular items go missing
```

| Retriever | Recall at 50 |
|---|---|
| Random 50 items | 0.019 |
| Most popular 50 items | 0.272 |
| Two-tower, no logQ correction | 0.388 |
| Two-tower, logQ correction | 0.516 |

The towers nearly double the popularity list, and the one-line correction adds a third more recall. Without it, the items retrieved appear less than a third as often in the training labels as those the corrected model retrieves: the popular items, which are the likeliest clicks, go missing.

## Serving: an offline index and an online query
- **Offline**, a batch job runs the item tower over the catalogue and loads the vectors into a nearest-neighbour index, again whenever the model is retrained. New items are added as they arrive.
- **Online**, each request runs the user tower once, searches the index for the top few hundred items, and passes them to ranking.

Exact search over ten million 128-dimensional vectors is over a billion multiply-adds per request, so real systems search approximately. The simplest index is the **inverted file** (IVF): cluster the item vectors with k-means, keep a list of items per cluster, and at query time scan only the clusters whose centroids are closest to the query. The number probed, `n_probe`, trades recall for speed.

```python
def kmeans(X, k, iters=20, seed=0):
    """Spherical k-means: centroids are kept at unit length, assignment by largest dot product."""
    C = X[np.random.default_rng(seed).choice(len(X), k, replace=False)]
    for _ in range(iters):
        assign = np.argmax(X @ C.T, 1)
        C = np.stack([X[assign == c].sum(0) if (assign == c).any() else C[c] for c in range(k)])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C, np.argmax(X @ C.T, 1)


class IVF:
    def __init__(self, X, n_lists):
        self.X = X
        self.C, assign = kmeans(X, n_lists)
        self.lists = [np.flatnonzero(assign == c) for c in range(n_lists)]

    def search(self, q, k, n_probe):
        probe = np.argsort(-(self.C * q).sum(1))[:n_probe]           # nearest clusters first
        cand = np.concatenate([self.lists[c] for c in probe])
        best = cand[np.argsort(-(self.X[cand] * q).sum(1))[:k]]
        return best, len(cand)


index = IVF(V, n_lists=32)
exact = np.argsort(-(Q @ V.T), axis=1)[:, :10]                # brute force over every item
results = []
for n_probe in [1, 2, 4, 8, 16, 32]:
    found, scanned = zip(*(index.search(q, 10, n_probe) for q in Q[:300]))
    rec = np.mean([np.isin(e, f).mean() for e, f in zip(exact[:300], found)])
    results.append((n_probe, np.mean(scanned) / len(V), rec))
    print(f"n_probe {n_probe:2d}: scans {results[-1][1]:.0%} of items, recall@10 vs exact {rec:.3f}")
recalls = [r for _, _, r in results]
assert all(a <= b for a, b in zip(recalls, recalls[1:])) and recalls[-1] == 1.0
```

| Clusters probed (of 32) | Items scanned | Recall at 10 vs exact |
|---|---|---|
| 1 | 3% | 0.357 |
| 2 | 6% | 0.538 |
| 4 | 13% | 0.740 |
| 8 | 25% | 0.905 |
| 16 | 50% | 0.987 |
| 32 | 100% | 1.000 |

Recall never falls as clusters are added, because the probed lists only grow and any exact neighbour among the candidates stays in the top 10. With every cluster probed, IVF is exact. How fast the curve rises depends on how clustered the vectors are and on the number of lists, so a real index needs this same check on its own queries. The main index families are all available in FAISS; ScaNN takes a similar partition-then-quantize approach:

| Index | How it works | Trade-off |
|---|---|---|
| Flat | Compares with every vector | Exact; cost grows with the catalogue |
| IVF | Scans only nearby clusters | Fast and simple; stores full vectors, so no memory saved |
| HNSW | Walks a layered neighbour graph toward the query | High recall at low latency; graph links add memory |
| PQ | Stores a short code per chunk of each vector | Large memory savings, some accuracy lost; usually paired with IVF |

Memory often decides. A hundred million 128-dimensional float32 vectors take 51.2 GB; 16-byte product-quantization codes take 1.6 GB. [Vector databases](../vector-databases/) covers storage and updates.

## New items come through the content features
Matrix factorization has nothing to say about an item nobody has touched: its embedding was never trained. A two-tower model with content in the item tower can embed a new item from its description. At launch, a new item goes in with the shared "unknown" ID and its real content. The `id_dropout` above makes that input a familiar one: in training, an item's ID is swapped for the unknown ID 10% of the time, so the tower has seen this input before instead of meeting an untrained row for the first time. The code checks how much each part contributes by also training a model without ID dropout.

```python
def cold_recall(m, k=20):
    return recall_at(retrieve(m, hist_test, new_ids, k, id_known=False)[0], next_new_test)


r_cold, r_cold_nodrop = cold_recall(model), cold_recall(train(log_q=True, id_dropout=0.0))
r_cold_rand = 20 / len(new_ids)
print(f"new items, recall@20: ID dropout {r_cold:.3f}  no ID dropout {r_cold_nodrop:.3f}  random {r_cold_rand:.3f}")
assert min(r_cold, r_cold_nodrop) > 4 * r_cold_rand          # the content features carry the new items
assert abs(r_cold - r_cold_nodrop) < 0.1                     # dropout is a small refinement here
```

For the 300 items never seen in training, recall at 20 is 0.514, against 0.067 for a random pick. Most of that comes from the content features: the model trained without ID dropout still reaches 0.504. In this small world the content description carries the item well, so dropout adds little. It is cheap insurance for items whose content says less than their ID, but measure what it buys before relying on it. Once the item collects interactions, the next retraining learns its ID embedding. [Cold start](../cold-start/) covers the wider problem, including new users.

## Hard negatives
In-batch and random negatives are mostly easy, and once the model separates them they teach little. **Hard negatives** are items the current model scores highly but the user did not engage with, mined by running retrieval and sampling from its results. They force finer distinctions, with a risk: the top-scored items without an interaction are often ones the user would have liked but never saw. For Facebook's embedding-based search, Huang et al. (2020) found that negatives taken from the very top of the results did worse than ones sampled further down the list, and that hard negatives worked best blended with random ones rather than on their own. Yang et al. (2020) mix in-batch negatives with items drawn uniformly from the catalogue, which also lets rarely-clicked long-tail items be pushed down.

## How it compares

| Approach | Captures | Falls short on |
|---|---|---|
| Lexical match (BM25) | Exact words in a query | Synonyms, personalization, requests with no text |
| Matrix factorization | Co-engagement patterns | New items and users; no content or context |
| Cross-encoder or ranker | Full user-item interaction | One evaluation per pair, too slow for a catalogue |
| Two-tower | Content, context and engagement at catalogue scale | No feature crosses; bounded by the index's recall |

Production systems usually merge several retrievers, such as a two-tower model, co-visitation lists and lexical match, before ranking.

## Pitfalls
- **Version skew.** User and item vectors share a space only if they come from the same model. Ship both towers and the index as one versioned unit.
- **Pair features in a tower.** "Has this user seen this item before?" belongs to the pair, so neither tower can compute it at serving time. Smuggled into training anyway (say, computed for the training user and fed to the item tower), it makes offline metrics look excellent and then vanishes in production. It belongs in ranking.
- **Judging by the loss.** In-batch loss depends on batch size and temperature. Measure recall at the number of candidates actually passed on, against the full catalogue, on users held out by time; see [evaluating ML systems](/posts/evaluating-ml-systems/).
- **Silent index loss.** Approximate search drops candidates without error. Check index recall against exact search on real queries whenever the index changes.
- **Feedback loops.** Retrieval learns from interactions with items earlier retrieval surfaced, the same loop described in [position bias](/posts/position-bias/). Some exploration traffic keeps the data from narrowing.

Every recall here comes from one seeded run on an invented ground truth, where choices depend only on taste and popularity, so the numbers show what each piece does, not what a real catalogue will give. Two pieces are described but not tested: the code never mines hard negatives, and it counts \(q_i\) exactly from the labels where Yi et al. estimate it from a stream, so both remain to be tried on this data.

Sources: Po-Sen Huang et al., Learning Deep Structured Semantic Models for Web Search using Clickthrough Data (CIKM 2013); Paul Covington, Jay Adams and Emre Sargin, [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190) (RecSys 2016); Xinyang Yi et al., [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996) (RecSys 2019); Jui-Ting Huang et al., [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632) (KDD 2020); Ji Yang et al., Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations (WWW 2020); Jeff Johnson, Matthijs Douze and Hervé Jégou, [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734) (2017), the paper behind [FAISS](https://github.com/facebookresearch/faiss); Ruiqi Guo et al., Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ICML 2020, ScaNN); Yu. A. Malkov and D. A. Yashunin, [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (2016); Hervé Jégou, Matthijs Douze and Cordelia Schmid, Product Quantization for Nearest Neighbor Search (TPAMI 2011). The serving and system-design framing also draws on Eugene Yan's [System Design for Recommendations and Search](https://eugeneyan.com/writing/system-design-for-discovery/) (2021).
