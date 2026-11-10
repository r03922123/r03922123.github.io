# Transformers4Rec: session-based recommendation as language modeling

Many visitors to a shop or a travel site never log in. All the system knows about them is what they did in the last few minutes: a pair of running shoes, then another pair, then a sports watch. Session-based recommendation predicts the next item from that short trail alone. Even when the user is known, the current session often says more about what they want right now than their history from last year.

A session is an ordered list of item ids, and predicting the next id given the earlier ones is the same problem as predicting the next word in a sentence. Transformers4Rec is a library from the NVIDIA Merlin team (Moreira, Rabhi, Lee, Ak and Oldridge, RecSys 2021) that takes that analogy literally. It connects the Transformer architectures in Hugging Face Transformers to recommendation inputs, losses and metrics. This post builds the core idea from scratch in PyTorch, measures it properly, and then shows how the library packages the same pieces.

## Sessions as sentences
Write a session as \(s = (i_1, i_2, \dots, i_T)\), where each \(i_t\) is an item from a catalog of \(N\) items. A next-item model learns

\[ P(i_{t+1} \mid i_1, \dots, i_t) = \operatorname{softmax}(h_t W^\top)_{i_{t+1}} \]

where \(h_t\) is a \(d\)-dimensional summary of the session so far and \(W\) is an \(N \times d\) matrix with one output vector per item. Training minimizes cross-entropy on every position of every session, exactly like a language model (see [cross-entropy vs MSE](../cross-entropy-vs-mse/)).

The summary \(h_t\) is where models differ. GRU4Rec (Hidasi et al., 2016) used a recurrent network. SASRec (Kang and McAuley, 2018) replaced it with causal self-attention, the GPT recipe. BERT4Rec (Sun et al., 2019) trained a bidirectional encoder by masking items, the BERT recipe. Transformers4Rec makes these choices configurable rather than inventing a new architecture.

## From an event log to sequences
Raw data is a log of events: a session id, an item id, a timestamp, often a price and a category. Turning it into training sequences takes four steps.

1. Sort by session and time. Logs arrive out of order, and every later step assumes they are in order.
1. Drop consecutive repeats. Refreshing a page logs the same item twice, and a model that learns "the next item is the current one" scores well offline and helps nobody.
1. Group into one list per session, and drop sessions with a single event, because they have no target.
1. Keep the most recent \(L\) items, since attention cost grows with length and old clicks matter less.

```python
import numpy as np
import pandas as pd

events = pd.DataFrame({
    "session": ["a", "b", "a", "a", "b", "a", "c", "b", "a"],
    "item":    [ 7,   3,   7,   9,   3,   9,   5,   4,   2 ],
    "ts":      [ 1,   2,   3,   4,   5,   6,   7,   8,   9 ],
}).sample(frac=1, random_state=0)                          # logs rarely arrive in order


def to_sequences(df, max_len):
    df = df.sort_values(["session", "ts"], kind="stable")
    prev = df.groupby("session")["item"].shift()           # previous item in the same session, NaN at the start
    df = df[df["item"] != prev]                            # drop consecutive repeats
    seqs = df.groupby("session")["item"].agg(list)
    seqs = seqs[seqs.map(len) >= 2]                        # an input and at least one target
    return seqs.map(lambda s: s[-max_len:])                # the most recent max_len items


seqs = to_sequences(events, max_len=20)
assert seqs.to_dict() == {"a": [7, 9, 2], "b": [3, 4]}     # c had one event; the repeats in a and b are gone

unsorted = events[events["item"] != events["item"].shift()]   # comparing rows of the raw, shuffled log
assert ((unsorted["session"] == "a") & (unsorted["item"] == 9)).sum() == 2   # a repeat survives
```

The last two lines show the usual mistake. Shifting the whole log, without sorting and without grouping by session, compares each row with an unrelated neighbour, so real repeats survive and the first event of one session gets compared with the last event of another.

Time features need care too. Day of week as the integers 0 to 6 puts Sunday as far from Monday as it can be. The usual fix maps it onto a circle with \(\sin(2\pi x / P)\) and \(\cos(2\pi x / P)\), and the period \(P\) must be the number of distinct values, 7, not the largest value, 6:

```python
def cyclical(x, period):
    angle = 2 * np.pi * np.asarray(x) / period
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1)


days = cyclical(np.arange(7), period=7)                    # Monday = 0 ... Sunday = 6
steps = np.linalg.norm(np.diff(days, axis=0), axis=1)
assert np.allclose(steps, np.linalg.norm(days[6] - days[0]))   # Sunday to Monday is one step like any other
assert np.allclose(cyclical(6, period=6), cyclical(0, period=6))   # with period 6, Sunday becomes Monday
```

## A session Transformer from scratch
To see whether attention over the whole session helps, the data needs structure that the last item alone cannot reveal. In this simulation each session has a hidden intent: one of 10 categories of 10 items, walked through in order. With probability 0.65 the next event is the next item of that walk. Otherwise it is a detour to a random item anywhere in the catalog, after which the walk resumes. After a detour the last item says nothing about the intent, but the earlier items do.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(2)
rng = np.random.default_rng(0)
N_CAT, PER_CAT, L = 10, 10, 20
N_ITEMS = N_CAT * PER_CAT                                  # item ids 1..100; 0 is padding


def make_session(length):
    c, p, s = rng.integers(N_CAT), rng.integers(PER_CAT), []
    for _ in range(length):
        if rng.random() < 0.65:
            s.append(1 + c * PER_CAT + p)                  # next step of the hidden walk
            p = (p + 1) % PER_CAT
        else:
            s.append(1 + rng.integers(N_ITEMS))            # a detour anywhere in the catalog
    return s, c


data = [make_session(rng.integers(4, L + 1)) for _ in range(4000)]
train = [s for s, _ in data[:3000]]
test, test_intent = [s for s, _ in data[3000:]], np.array([c for _, c in data[3000:]])


def pad(batch):
    x = torch.zeros(len(batch), L, dtype=torch.long)
    for i, s in enumerate(batch):
        x[i, :len(s)] = torch.tensor(s)
    return x
```

The model is a small GPT: item and position embeddings, two pre-norm Transformer layers with a causal mask so position \(t\) only sees positions up to \(t\), and a padding mask so padded slots are ignored. The output layer reuses the input embedding table, **weight tying** (Press and Wolf, 2017; Inan et al., 2017). An item gets one vector that serves both as input and as output, which removes \(N \times d\) parameters. That matters when the catalog has millions of items, and items with few training examples learn from both directions.

```python
class SessionTransformer(nn.Module):
    def __init__(self, n_items, d=48, heads=4, layers=2, max_len=L, tie=True):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_items + 1, bias=False)
        if tie:
            self.out.weight = self.item.weight             # one vector per item, used on both ends
        nn.init.normal_(self.item.weight, std=0.02)        # small init: tied logits start near zero
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x):
        T = x.shape[1]
        h = self.item(x) + self.pos(torch.arange(T))
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # True = may not attend
        h = self.encoder(h, mask=causal, src_key_padding_mask=(x == 0))
        return self.out(self.norm(h))                       # (batch, T, n_items + 1) logits


model, untied = SessionTransformer(N_ITEMS), SessionTransformer(N_ITEMS, tie=False)
count = lambda m: sum(p.numel() for p in m.parameters())
assert count(untied) - count(model) == (N_ITEMS + 1) * 48
assert model.out.weight is model.item.weight

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
X = pad(train)
for epoch in range(6):
    model.train()
    for idx in torch.randperm(len(X)).split(200):
        x = X[idx]
        logits = model(x[:, :-1])                          # predict position t + 1 from positions up to t
        loss = F.cross_entropy(logits.reshape(-1, N_ITEMS + 1), x[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad()
        loss.backward()
        opt.step()
assert loss.item() < np.log(N_ITEMS) - 1                   # well below a uniform guess over the catalog
```

The small initialization is not decoration. With tied weights, PyTorch's default embedding init (standard deviation 1) makes the initial logits large, and with the default init this model never got below the uniform loss of \(\log 101 \approx 4.6\) in the same six epochs.

## Measuring it: Recall@k and NDCG@k
Evaluation hides the last item of each held-out session, scores the whole catalog from the prefix, and finds the rank \(r\) of the hidden item. With one relevant item per session the two standard metrics are

\[ \text{Recall@}k = \mathbb{1}[r \le k], \qquad \text{NDCG@}k = \frac{\mathbb{1}[r \le k]}{\log_2(1 + r)} \]

averaged over sessions. With a single target, Recall@k is the same as hit rate. The logarithm is base 2, so a hit at rank 1 scores exactly 1. With a natural logarithm, rank 1 would score \(1 / \ln 2 \approx 1.44\) and the metric would not be normalized.

Two baselines keep the model honest. **Popularity** recommends the most frequent items to everyone. **Last item** recommends whatever most often followed the current item in training, a first-order Markov chain, which is hard to beat on many real datasets.

```python
from sklearn.metrics import ndcg_score


def ranks_of(scores, targets):
    """1-based rank of each target item in its row of scores."""
    target_score = scores[np.arange(len(targets)), targets][:, None]
    return 1 + (scores > target_score).sum(axis=1)


def recall_ndcg(ranks, k=10):
    hit = ranks <= k
    return hit.mean(), np.where(hit, 1 / np.log2(1 + ranks), 0.0).mean()


assert recall_ndcg(np.array([1]))[1] == 1.0 and recall_ndcg(np.array([3]))[1] == 0.5


@torch.no_grad()
def transformer_scores(prefixes):
    model.eval()
    logits = model(pad(prefixes))
    last = torch.tensor([len(p) - 1 for p in prefixes])
    return logits[torch.arange(len(prefixes)), last].numpy()   # the output at each session's last real item


prefixes, targets = [s[:-1] for s in test], np.array([s[-1] for s in test])
jitter = 1e-6 * rng.random((len(test), N_ITEMS + 1))          # break ties at random
pop = np.bincount([i for s in train for i in s], minlength=N_ITEMS + 1).astype(float)
follows = np.zeros((N_ITEMS + 1, N_ITEMS + 1))
for s in train:
    for a, b in zip(s, s[1:]):
        follows[a, b] += 1

scores = {"popularity": np.tile(pop, (len(test), 1)) + jitter,
          "last item": follows[[p[-1] for p in prefixes]] + jitter,
          "transformer": transformer_scores(prefixes)}
for s in scores.values():
    s[:, 0] = -np.inf                                         # never recommend padding

detour = np.array([(p[-1] - 1) // PER_CAT != c for p, c in zip(prefixes, test_intent)])
results = {}
for name, s in scores.items():
    r = ranks_of(s, targets)
    results[name] = recall_ndcg(r) + (recall_ndcg(r[detour])[0],)
    one_hot = np.eye(N_ITEMS + 1)[targets][:, 1:]
    assert np.isclose(results[name][1], ndcg_score(one_hot, s[:, 1:], k=10))   # matches scikit-learn

pop_r, last_r, tf_r = results["popularity"], results["last item"], results["transformer"]
assert tf_r[0] > last_r[0] + 0.1 > pop_r[0] + 0.1             # Recall@10: transformer > last item > popularity
assert tf_r[2] > last_r[2] + 0.4                              # the gap is widest right after a detour
on_track = {name: recall_ndcg(ranks_of(s[~detour], targets[~detour]))[0] for name, s in scores.items()}
assert abs(on_track["transformer"] - on_track["last item"]) < 0.05   # on track, the two are level
```

| Model | Recall@10 | NDCG@10 | Recall@10 after a detour |
|---|---|---|---|
| Popularity | 0.10 | 0.05 | 0.09 |
| Last item | 0.49 | 0.46 | 0.08 |
| Session Transformer | 0.65 | 0.56 | 0.60 |

The table shows one run; about a third of the test prefixes end on a detour. Popularity knows nothing about the session. Last item is strong while the walk is on track and collapses after a detour to the level of popularity, because a random item says nothing about where the walk was going. The Transformer attends back past the detour to the items that reveal the intent. On prefixes that end on the walk the two are level on Recall@10 (about 0.68 each), so the Transformer's whole lead comes from the detours. It is also exactly the gap a Transformer should close and a bigram cannot.

## Four ways to train the same model
The model above trains with **causal language modeling (CLM)**: predict each item from the ones before it. Transformers4Rec supports three more objectives borrowed from NLP.

- **Masked language modeling (MLM)**, as in BERT4Rec: hide random items and predict them from both sides. Each session gives fewer targets per pass but richer context. At inference, a mask is placed after the last item. During training at least one item per session must be masked, or that session contributes no target and no loss.
- **Permutation language modeling (PLM)**, from XLNet: predict items in a random factorization order, which gives bidirectional context without a mask token.
- **Replacement token detection (RTD)**, from ELECTRA: a small generator replaces some items, and the main model learns to spot which ones were replaced, which turns every position into a training signal.

The Transformers4Rec paper compared these objectives on several e-commerce and news datasets. The useful takeaway is that the objective is a hyperparameter worth tuning, not a fixed property of the architecture.

## Side information
Item ids alone ignore what the log knows about each event. Transformers4Rec builds an input vector per position from several feature sequences.

- **Categorical** features such as category or brand get their own embedding tables.
- **Continuous** features such as price or time since the previous event are projected to \(d\) dimensions by a linear layer. An alternative is a soft one-hot embedding: a linear layer and a softmax produce weights over a small embedding table, and the feature becomes a weighted mix of its rows.
- The per-feature vectors are concatenated and passed through an MLP, or summed, before the Transformer.

Prices are long-tailed, so they are usually log-transformed and standardized, and a price relative to its category's average often carries more signal than the raw number. For regularization on these tables, **stochastic swap noise** replaces a feature value, with a small probability, by the same feature taken from another random row of the batch. Every value it inserts is a real value from the data, so it works for categorical features, and it creates new feature combinations the model has to be robust to. [Dropout](../dropout/) and weight tying do the rest.

## The same pipeline with Merlin
In the Merlin stack, NVTabular does the ETL and Transformers4Rec builds the model. NVTabular runs on cuDF on a GPU, with a slower pandas-based CPU mode. The two blocks below are **not executed** here. They sketch the API as it appears in the project's README and examples, so check them against the release you install.

```plaintext
# ETL with NVTabular (not executed: needs the Merlin stack, ideally on a GPU)
import nvtabular as nvt

cats = ["item_id", "category"] >> nvt.ops.Categorify()     # contiguous ids; low ids reserved for null/unknown
features = ["session_id", "timestamp"] + cats
sessions = features >> nvt.ops.Groupby(
    groupby_cols=["session_id"],
    sort_cols=["timestamp"],                               # keep each session's list in time order
    aggs={"item_id": ["list", "count"], "category": ["list"], "timestamp": ["first"]},
    name_sep="-",
)
seq = sessions["item_id-list", "category-list"] >> nvt.ops.ListSlice(-20)   # the last 20 items
workflow = nvt.Workflow(seq + sessions["item_id-count"])
workflow.fit_transform(nvt.Dataset("events.parquet")).to_parquet("sessions/")
```

```plaintext
# Model with Transformers4Rec (not executed)
from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt

# schema: the Merlin schema of the parquet files written by the NVTabular workflow
inputs = tr.TabularSequenceFeatures.from_schema(
    schema, max_sequence_length=20, continuous_projection=64,
    aggregation="concat", masking="mlm",                   # or "clm", "plm", "rtd"
)
config = tr.XLNetConfig.build(d_model=64, n_head=4, n_layer=2, total_seq_length=20)
body = tr.SequentialBlock(inputs, tr.MLPBlock([64]), tr.TransformerBlock(config, masking=inputs.masking))
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True,
                              metrics=[NDCGAt(top_ks=[10, 20], labels_onehot=True),
                                       RecallAt(top_ks=[10, 20], labels_onehot=True)]),
    inputs=inputs,
)
model = tr.Model(head)
```

Each piece maps onto the from-scratch version: `TabularSequenceFeatures` is the embedding layer plus side features, `masking` chooses the training objective, `TransformerBlock` wraps a Hugging Face model in place of the hand-built encoder, and `NextItemPredictionTask(weight_tying=True)` is the tied output layer with its cross-entropy loss. The library also provides a Hugging Face style `Trainer`, and its examples train and evaluate incrementally over consecutive time windows.

## Pitfalls
- **Sort before deduplicating.** Shifting an unsorted log, or shifting across session boundaries, removes the wrong rows without any error.
- **Split by time.** Train on earlier sessions and test on later ones. A random split leaks future popularity into training.
- **Always report the baselines.** A last-item Markov chain is often competitive, and on some datasets the Transformer only wins on long sessions.
- **Watch the special ids.** Padding and the null and out-of-vocabulary ids that `Categorify` reserves must never be scored as recommendations.
- **The full softmax gets expensive.** With millions of items, scoring the whole catalog per position dominates training. Sampled softmax or in-batch negatives, as in [contrastive losses](../contrastive-loss/), are the usual fix, and serving moves to approximate nearest-neighbour search (see [two-tower retrieval](../two-tower-retrieval/) and [vector databases](../vector-databases/)).
- **Sessions are not users.** For logged-in users with long histories, see [sequential user modeling](../sequential-user-modeling/), and for brand-new items and users, [cold start](../cold-start/).

The library and its training objectives are described in "Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation" (Moreira et al., RecSys 2021), the NVIDIA-Merlin/Transformers4Rec repository and the NVIDIA developer blog post on session-based recommendation. The prose and code here are written from scratch.
