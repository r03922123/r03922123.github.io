# Sequential user modeling: from averaged history to target attention

On a synthetic task where each user has between one and six interests, averaging the history scores a test AUC of 0.79 for users with one or two interests and 0.70 for those with five or six; target attention holds 0.81 and 0.77. What a user did is usually more informative than who they are, and it arrives as a sequence: clicks, plays, purchases, searches, each with a timestamp. Sequential user modeling turns that sequence into something a ranking model can use.

The simplest approach averages the embeddings of everything the user touched. That works for a user with one interest, but most have several. Someone who buys running shoes, cookbooks and phone chargers gets a vector somewhere between the three, close to none of them, and when the model asks about a pair of trail shoes the shoe purchases are diluted by everything else. I build up the standard fix, target attention, measure how much it helps, and follow the work that made it affordable for histories of tens of thousands of events. On a history of 20,000 behaviours, the 50 chosen by the attention model's own score reproduce full attention to within about 5%.

## The ranking setup
Throughout, the model scores one candidate item \(t\) (the target) for a user whose history is a list of behaviour embeddings \(e_1, \dots, e_N\), most recent last. A pooling step reduces the history to a user vector \(u\), and a small network turns \([u, e_t]\) plus other features into a click probability. This is the ranking stage; retrieval has different constraints (see [two-tower retrieval](../two-tower-retrieval/) and [recommendation ranking](../recommendation-ranking/)).

Mean pooling is

\[ u_{\text{mean}} = \frac{1}{N}\sum_{i=1}^{N} e_i , \]

and it is the same for every candidate. That is the root of the problem: the user vector is computed before the model knows what it is being asked about.

## Target attention: DIN
The Deep Interest Network (Zhou et al., 2018, from Alibaba's display advertising team) makes the user vector depend on the candidate. A small network called the activation unit looks at each past behaviour next to the target and produces a weight, and the user vector is the weighted sum:

\[ u(t) = \sum_{i=1}^{N} a(e_i, e_t)\, e_i . \]

The activation unit is an MLP over the two embeddings plus interaction terms (the paper's diagram feeds in their outer product; the authors' released code and most implementations use the element-wise product and the difference instead, as below). For a trail-shoe target, past shoe purchases get large weights and cookbooks small ones, so \(u(t)\) looks like the user's shoe interest; a cookbook target gets a different vector from the same history.

Two details of DIN are easy to get wrong.

- **The weights are not normalized.** The DIN paper deliberately drops the softmax a transformer would apply. A user with twenty relevant behaviours should look more interested than a user with one, and forcing the weights to sum to 1 would erase that difference.
- **Padding has to be masked.** Histories are padded to a fixed length, and the activation unit's biases give padded positions a nonzero weight. With an all-zero padding vector that weight multiplies zero and does no harm, but a learned padding embedding (an ID-0 row in the embedding table) leaks into the user vector, and any softmax variant hands the padding probability mass. Masking the weights to zero covers every case.

To see the effect, here is a synthetic task built to have the property that matters. Each user has between one and six interests, each a random direction in embedding space, and their history is drawn from those interests. A candidate is drawn either from one of the user's interests or at random, and the true click probability rises with the similarity between the candidate and the user's single closest past behaviour. Both models get the same MLP on \([u, e_t, u \odot e_t]\); the only difference is how \(u\) is pooled.

```python
import numpy as np
import torch
import torch.nn as nn

rng = np.random.default_rng(0)
torch.manual_seed(0)
torch.set_num_threads(1)                                  # deterministic and fast enough on a CPU
d, L = 16, 30                                             # embedding size, padded history length


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def auc(y, s):                                            # Mann-Whitney form of ROC AUC
    r = np.empty(len(s))
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
    pos = y == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


_r = np.random.default_rng(7)
_y = _r.random(500) < 0.3                              # cross-check against the pairwise definition
_s = _r.standard_normal(500) + _y
assert np.isclose(auc(_y.astype(int), _s), (_s[_y][:, None] > _s[~_y][None, :]).mean())


def make_data(n):
    m = rng.integers(1, 7, n)                             # number of interests per user, 1..6
    centers = unit(rng.standard_normal((n, 6, d)))
    pick = rng.integers(0, 6, (n, L)) % m[:, None]       # which interest each behaviour comes from
    hist = unit(centers[np.arange(n)[:, None], pick] + 0.3 * rng.standard_normal((n, L, d)))
    mask = np.arange(L) < rng.integers(10, L + 1, n)[:, None]
    hist[~mask] = 0.0                                     # zero padding
    on_interest = rng.random(n) < 0.5
    tgt = np.where(on_interest[:, None], centers[np.arange(n), rng.integers(0, 6, n) % m],
                   unit(rng.standard_normal((n, d))))
    tgt = unit(tgt + 0.3 * rng.standard_normal((n, d)))
    closest = np.where(mask, np.einsum("nld,nd->nl", hist, tgt), -1).max(1)
    p = 1 / (1 + np.exp(-(12 * closest - 9)))             # true click probability
    y = (rng.random(n) < p).astype(np.float32)
    return dict(hist=hist.astype(np.float32), mask=mask, tgt=tgt.astype(np.float32), y=y, p=p, m=m)


train, test = make_data(20_000), make_data(10_000)


class Head(nn.Module):                                    # shared by both models
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, u, t):
        return self.mlp(torch.cat([u, t, u * t], -1)).squeeze(-1)


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Head()

    def forward(self, hist, mask, t):
        m = mask.float().unsqueeze(-1)
        return self.head((hist * m).sum(1) / m.sum(1), t)


class DIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit = nn.Sequential(nn.Linear(4 * d, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head = Head()

    def weights(self, hist, mask, t):
        tt = t.unsqueeze(1).expand_as(hist)
        a = self.unit(torch.cat([hist, tt, hist - tt, hist * tt], -1)).squeeze(-1)
        return a * mask.float()                           # no softmax; padding masked to 0

    def forward(self, hist, mask, t):
        u = (self.weights(hist, mask, t).unsqueeze(-1) * hist).sum(1)
        return self.head(u, t)


def tensors(D):
    return torch.tensor(D["hist"]), torch.tensor(D["mask"]), torch.tensor(D["tgt"])


def fit_and_score(model, epochs=5, batch=256):
    X, y = tensors(train), torch.tensor(train["y"])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            j = perm[i:i + batch]
            loss = nn.functional.binary_cross_entropy_with_logits(model(*(x[j] for x in X)), y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return model(*tensors(test)).numpy()


scores = {}
for M in (MeanPool, DIN):
    torch.manual_seed(0)
    scores[M.__name__] = fit_and_score(M())
scores["oracle"] = test["p"]                              # the true probability: the best any model can do

few, many = test["m"] <= 2, test["m"] >= 5
for name, s in scores.items():
    print(f"{name:9s} AUC all {auc(test['y'], s):.3f}   1-2 interests {auc(test['y'][few], s[few]):.3f}"
          f"   5-6 interests {auc(test['y'][many], s[many]):.3f}")

a = {k: auc(test["y"], s) for k, s in scores.items()}
a_many = {k: auc(test["y"][many], s[many]) for k, s in scores.items()}
a_few = {k: auc(test["y"][few], s[few]) for k, s in scores.items()}
assert a["DIN"] > a["MeanPool"] + 0.03                    # target attention wins overall
assert a["oracle"] > a["DIN"]                             # and neither beats the truth
assert a_few["MeanPool"] - a_many["MeanPool"] > a_few["DIN"] - a_many["DIN"]   # averaging suffers more as interests multiply
```

The printed table, for this seed:

| Model | AUC, all users | 1–2 interests | 5–6 interests |
|---|---|---|---|
| Mean pooling | 0.76 | 0.79 | 0.70 |
| DIN-style target attention | 0.80 | 0.81 | 0.77 |
| True probability | 0.87 | 0.86 | 0.86 |

With one or two interests the average is still a decent summary and the two models are close. As interests multiply, the average drifts towards the middle of several unrelated directions and mean pooling falls off, while target attention degrades much less. (The DIN paper also introduced the Dice activation and a mini-batch-aware regularizer for training on sparse ID features; neither is needed here.)

## Why no softmax, and why masking matters
Both warnings are easy to check. Given one history with a single relevant behaviour and another with fifteen, softmax pooling returns a vector of about the same length, pointing roughly the same way, for both; unnormalized pooling keeps the count. Softmax over a padded history hands probability mass to the padding unless those scores are set to \(-\infty\) first. Unnormalized pooling is safe with all-zero padding but not with a nonzero padding embedding.

```python
t = unit(rng.standard_normal(d))
related = unit(t + 0.2 * rng.standard_normal((15, d)))
unrelated = unit(rng.standard_normal((29, d)))
one = np.vstack([related[:1], unrelated])                 # 1 relevant behaviour + 29 others
fifteen = np.vstack([related, unrelated[:15]])            # 15 relevant + 15 others


def softmax_pool(h, temp=0.1):
    w = np.exp((h @ t - (h @ t).max()) / temp)
    return (w / w.sum()) @ h


def din_pool(h):
    return np.maximum(h @ t, 0) @ h                       # a stand-in activation unit: clipped similarity


ratio_soft = np.linalg.norm(softmax_pool(fifteen)) / np.linalg.norm(softmax_pool(one))
ratio_din = np.linalg.norm(din_pool(fifteen)) / np.linalg.norm(din_pool(one))
assert 0.8 < ratio_soft < 1.2 and ratio_din > 4           # only unnormalized pooling sees "15 vs 1"
assert softmax_pool(one) @ softmax_pool(fifteen) > 0.85 * np.linalg.norm(softmax_pool(one)) * np.linalg.norm(softmax_pool(fifteen))

short = related[:3]                                       # a 3-item history padded to length 30
padded = np.vstack([short, np.zeros((27, d))])
scores_pad = padded @ t + 0.5                             # an activation unit with a bias of 0.5
w_unmasked = np.exp(scores_pad) / np.exp(scores_pad).sum()
masked = np.where(np.arange(30) < 3, scores_pad, -np.inf)
w_masked = np.exp(masked) / np.exp(masked).sum()
assert w_unmasked[3:].sum() > 0.5                         # padding steals most of the attention
assert np.isclose(w_masked[3:].sum(), 0.0)

w_raw = padded @ t + 0.5                                  # unnormalized weights, padding not masked
assert np.allclose(w_raw @ padded, w_raw[:3] @ short)     # zero padding: weight x 0 adds nothing
pad_emb = 0.3 * unit(rng.standard_normal(d))              # a learned padding embedding is not zero
padded_learned = np.vstack([short, np.tile(pad_emb, (27, 1))])
w_learned = padded_learned @ t + 0.5
leak = w_learned @ padded_learned - w_learned[:3] @ short
assert np.linalg.norm(leak) > 0.5 * np.linalg.norm(w_learned[:3] @ short)   # 27 padded slots leak in
```

## Adding order: DIEN
DIN treats the history as a bag: shuffle it and nothing changes. The Deep Interest Evolution Network (Zhou et al., 2019) adds the order back with two recurrent layers. An interest extractor GRU runs over the behaviours, trained with an auxiliary loss that asks each hidden state to distinguish the next real behaviour from a sampled negative, so the states learn to describe interest rather than just compress clicks. An interest evolving layer then runs a second GRU in which the attention score against the target scales the update gate (AUGRU), so behaviours unrelated to the candidate barely change the state. The final state replaces DIN's weighted sum.

The cost is sequential computation per candidate, which does not stretch to histories of thousands of events; the lifelong models below keep DIN-style target attention and instead focus on choosing which behaviours to attend over.

## Lifelong histories: search first, then attend
Target attention costs at least \(O(N d)\) per candidate (more with an MLP activation unit), and a request scores hundreds or thousands of candidates. That is fine for the last few hundred behaviours, not for a full history of tens of thousands of events, which is where stable, slow-changing interests live.

The Search-based Interest Model (SIM; Pi et al., 2020) split the work into two stages:

1. A **General Search Unit (GSU)** cheaply selects the top \(K\) behaviours relevant to the target, with \(K\) in the tens or low hundreds. SIM proposed a hard search (keep behaviours in the target's category) and a soft search (top \(K\) by inner product of embeddings, served with an approximate nearest-neighbour index; see [vector databases](../vector-databases/)).
2. An **Exact Search Unit (ESU)** runs full multi-head target attention, with time information, over only those \(K\).

Two later papers refined the first stage. ETA (Chen et al., 2021) replaced the inner product with SimHash: each embedding is projected onto a fixed set of random hyperplanes and only the signs are kept, giving a short binary code (64 bits below). The fraction of differing bits between two codes estimates the angle between the embeddings divided by \(\pi\), and it is computed with an XOR and a popcount. The search is still linear in \(N\), not constant time, but each comparison is one machine word instead of a \(d\)-dimensional float dot product, cheap enough to run on every request and to train end to end with the model.

TWIN (Chang et al., 2023, from Kuaishou) pointed out a consistency problem: a GSU that ranks by a different score than the ESU throws away behaviours the ESU would have weighted heavily. TWIN makes the GSU use the ESU's own attention score, made affordable by splitting behaviour features into item-inherent ones, whose key projections are computed and cached ahead of time, and user–item cross features such as time since the behaviour, compressed to a scalar bias. TWIN-V2 later compressed very long histories further by clustering behaviours offline.

The code below builds a history of 20,000 behaviours, gives the ESU a learned-looking relevance score \((W_q e_t) \cdot (W_k e_i)\), and compares GSUs by how close the pooled vector over their top 50 gets to attention over the full history.

```python
n_cat, d2, N, K = 200, 32, 20_000, 50
rng = np.random.default_rng(1)
centers = unit(rng.standard_normal((n_cat, d2)))
favourites = rng.choice(n_cat, 40, replace=False)
cats = np.where(rng.random(N) < 0.8, rng.choice(favourites, N), rng.integers(0, n_cat, N))
E = unit(centers[cats] + 0.12 * rng.standard_normal((N, d2)))     # lifelong behaviour embeddings
t = unit(centers[favourites[0]] + 0.12 * rng.standard_normal(d2))
Wq = np.eye(d2) + 0.6 * rng.standard_normal((d2, d2)) / np.sqrt(d2)
Wk = np.eye(d2) + 0.6 * rng.standard_normal((d2, d2)) / np.sqrt(d2)

keys = E @ Wk                                             # item-side projections: computed once, cached
esu = keys @ (Wq.T @ t)                                   # the ESU's relevance score for every behaviour
assert np.allclose(esu[:100], [(Wq.T @ t) @ (Wk.T @ e) for e in E[:100]])   # same as projecting both sides per request


def attend(idx, temp=0.05):
    w = np.exp((esu[idx] - esu[idx].max()) / temp)
    return (w / w.sum()) @ E[idx]


def topk(x, k):
    return np.argpartition(-x, k)[:k]


full = attend(np.arange(N))                               # attention over all 20,000: the reference
truth = set(topk(esu, K))


def report(idx):
    return len(truth & set(idx)) / K, np.linalg.norm(attend(idx) - full) / np.linalg.norm(full)


recall_same, err_same = report(topk(esu, K))              # GSU uses the ESU's own score (TWIN)
recall_raw, err_raw = report(topk(E @ t, K))              # GSU uses raw embedding similarity
assert recall_same == 1.0 and err_same < 0.06             # 50 of 20,000 behaviours recover full attention
assert recall_raw < 0.5 and err_raw > 2 * err_same        # a mismatched GSU discards what the ESU wanted

bits = 64
planes = rng.standard_normal((d2, bits))


def simhash(x):
    return (x @ planes) > 0


hamming = (simhash(E) != simhash(t)).sum(1)               # one XOR + popcount per behaviour in practice
angle = np.arccos(np.clip(E @ t, -1, 1))
near = angle < 0.5
assert abs((hamming[near] / bits).mean() - (angle[near] / np.pi).mean()) < 0.01   # bits differ at rate angle / pi
raw_top = set(topk(E @ t, K))
order = np.argsort(hamming, kind="stable")
kept = {m: len(raw_top & set(order[:m * K])) / K for m in (1, 4, 8)}
assert kept[1] < 0.5 and 0.6 < kept[4] < kept[8] and kept[8] > 0.9   # codes are coarse: shortlist wider
```

Fifty behaviours out of twenty thousand, chosen by the ESU's own score, reproduce full attention to within about 5%, because attention over a long history concentrates on a few relevant events. A GSU ranking by plain embedding similarity keeps only about a third of the behaviours the ESU wanted, and its pooled vector is off by more than twice as much. And 64 random sign bits estimate the angle between embeddings well on average, but they are coarse: the 50 nearest codes contain only about a third of the 50 nearest embeddings, a shortlist four times larger about two thirds, and one eight times larger nearly all of them. Hash-based first stages pay for their speed with a wider shortlist or more bits.

## Time and the leakage trap
Absolute timestamps generalize poorly. What matters is how long before the request each behaviour happened, bucketed on a log scale so that "5 minutes ago" and "3 months ago" both get resolution. Building that feature is also where the most common bug hides: a training example's history must contain only behaviours strictly before the impression being labelled. Build histories from a snapshot taken after the fact and they contain the click being predicted; offline AUC looks superb and production does not.

```python
def history_features(events, request_ts, max_len=50):
    """events: (timestamp, item) pairs. Keeps only the past and returns log2-bucketed time gaps."""
    past = sorted((ts, item) for ts, item in events if ts < request_ts)[-max_len:]
    gaps = np.array([request_ts - ts for ts, _ in past], dtype=float)
    buckets = np.floor(np.log2(np.maximum(gaps, 1))).astype(int)
    return [item for _, item in past], buckets


events = [(1_000, "a"), (1_060, "b"), (90_000, "c"), (100_000, "clicked target"), (100_500, "later")]
items, buckets = history_features(events, request_ts=100_000)
assert items == ["a", "b", "c"]                           # the labelled event and anything after it are excluded
assert list(buckets) == [16, 16, 13]                      # about 27 hours ago (twice) and 2.8 hours ago
```

## Target attention or self-attention
Transformer-based sequential recommenders such as SASRec (Kang and McAuley, 2018) and BERT4Rec (Sun et al., 2019) use self-attention: behaviours attend to each other, and the model predicts the next item. The two families answer different questions and usually sit in different stages.

| | Target attention (DIN, SIM, TWIN) | Self-attention (SASRec, BERT4Rec) |
|---|---|---|
| Needs the candidate? | Yes, the user vector depends on it | No, one user vector per request |
| Typical stage | Ranking a few hundred candidates | Retrieval over millions, via nearest neighbours |
| Cost per request | Grows with candidates × history length | Grows with history length only |
| Strength | Picks the relevant slice of a diverse history | Models order and transitions between items |

Many production systems use both: a self-attentive or two-tower model to retrieve, and target attention in the ranker. See [Transformers4Rec](../transformers4rec/) for the self-attention side.

## Pitfalls
- **Leaky histories.** Cut every training sequence at the impression time. This is the most common reason offline gains vanish online.
- **Unmasked padding.** With softmax attention, padding takes probability mass; with a learned padding embedding, it adds a vector too. Mask before normalizing and zero the padded contributions.
- **Mismatched two-stage relevance.** A cheap first stage that disagrees with the attention model silently caps what the model can learn. Measure the first stage's recall against the second stage's own top \(K\).
- **Serving skew.** Production histories must be built from the same event types, with the same deduplication, as training histories; real-time pipelines and batch tables drift apart easily (see [recsys features](../recsys-features/)).
- **New users.** An empty history makes every pooling method return nothing useful. Fall back to profile and context features, and consider randomly truncating or dropping histories during training so the model learns to rank sensibly without them; the [cold start](../cold-start/) post covers the rest.

The DIN and lifelong experiments run on synthetic data built to have the property that matters, with one seed each, so they show why averaging fails and why the first stage should share the attention score, not how much either change gains on real logs. Two things stay open here: whether DIEN's order modelling adds to target attention on this task, since it is described but not run, and what the time-gap buckets do for ranking, since the code builds them but trains no model on them.

Papers: Zhou et al., "Deep Interest Network for Click-Through Rate Prediction" (KDD 2018, [arXiv:1706.06978](https://arxiv.org/abs/1706.06978)); Zhou et al., "Deep Interest Evolution Network for Click-Through Rate Prediction" (AAAI 2019, [arXiv:1809.03672](https://arxiv.org/abs/1809.03672)); Pi et al., "Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction" (CIKM 2020, [arXiv:2006.05639](https://arxiv.org/abs/2006.05639)); Chen et al., "End-to-End User Behavior Retrieval in Click-Through Rate Prediction Model" (ETA, 2021); Chang et al., "TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou" (KDD 2023). Eugene Yan's [patterns for personalization](https://eugeneyan.com/writing/patterns-for-personalization/) is good further reading on how these ideas are applied in practice.
