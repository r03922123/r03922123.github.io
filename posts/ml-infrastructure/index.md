# ML infrastructure: the plumbing around the model

Join a daily feature to training rows on the calendar day and it leaks the label it predicts: on 20,000 simulated click events, a 7-day click count reaches AUC about 0.75 that way and 0.62 with the point-in-time join I build below. A model is a function. Everything that feeds it the right inputs, trains it without losing a day's work to a dead machine, and answers a request in tens of milliseconds is infrastructure. Designing it is a different job from designing the model: the questions are where data moves, when it is correct, and what happens when a part fails.

This post walks through the three layers (data, training and serving) and the distributed-systems parts they are built from, with a small simulation for each idea. The modeling side of the same problem is in [a framework for designing ML systems](../ml-system-design-framework/).

## The building blocks
Almost every ML platform is assembled from the same handful of parts:

| Part | What it does | Where ML uses it |
|---|---|---|
| Message queue or log | Durable, ordered stream of events that many consumers read at their own pace | Impressions, clicks and purchases; input to streaming features |
| Key-value store | Single-key reads and writes in about a millisecond | Online feature store, cached embeddings, model outputs |
| Object store or data lake | Cheap, very large, append-mostly files | Raw logs, training sets, checkpoints, model artefacts |
| Columnar warehouse | Scans and aggregations over billions of rows | Batch features, labels, offline analysis |
| Time-series database | Values indexed by time, queried over windows | Metrics, monitoring and drift dashboards |
| Stream processor | Runs a computation on each event as it arrives | Real-time counts and windows written to the online store |

## Data: events, features and training rows
Everything starts with logging. Each request is logged with an id, the items shown and the context; each engagement event (click, purchase, watch) carries the same id so the two can be joined. Events go onto a queue rather than straight into a database: the queue absorbs bursts, lets several consumers read the same stream, and lets a consumer replay from an offset after a crash. Replay means an event can arrive twice, so consumers must be **idempotent**, for example by deduplicating on the event id.

Features are then computed in one of two ways:

- **Batch**: a scheduled job over the warehouse. Cheap and easy to backfill, but as stale as the last run. Fine for long windows ("purchases in 90 days") and slow-moving attributes.
- **Streaming**: a stream processor updates the value per event. Fresh within seconds, but harder to operate and to recompute for the past. Needed for session intent ("items viewed in the last 10 minutes").

A **feature store** keeps both behind one definition. Its **offline store** holds every historical value with a timestamp, for training sets; its **online store** is a key-value store with only the latest value, for serving. One definition feeding both prevents training-serving skew, where two implementations of "clicks in the last 7 days" quietly disagree. [Features for recommendation](../recsys-features/) covers what goes into those definitions.

### Point-in-time joins
Building a training row means attaching to each labelled event the feature values that **serving would have seen at that moment**. The common bug is to join on the calendar day. A daily job publishes each user's 7-day click count at the end of the day, and that value already counts every click made during the day, including the one the row is trying to predict.

The fix is an as-of join (`merge_asof` in pandas): for each event, take the latest snapshot published at or before its time. Below, the two joins are compared by how well the feature alone predicts the click.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(3)
n_users, n_events, day = 500, 20_000, 86_400
p_user = rng.beta(2, 6, n_users)                              # each user's true click rate
ev = pd.DataFrame({"user": rng.integers(0, n_users, n_events),
                   "ts": np.sort(rng.uniform(0, 30 * day, n_events))})
ev["click"] = (rng.random(n_events) < p_user[ev.user]).astype(int)
ev["day"] = (ev.ts // day).astype(int)

# A daily batch job: at the end of day d, each user's clicks over days d-6 .. d.
daily = ev.groupby(["user", "day"]).click.sum().unstack(fill_value=0).reindex(
    index=range(n_users), columns=range(30), fill_value=0)
clicks_7d = daily.T.rolling(7, min_periods=1).sum().T
snap = clicks_7d.stack().rename("clicks_7d").reset_index()
snap["snap_ts"] = (snap.day + 1) * float(day)                 # the value exists only once the day is over

# Wrong: join on the calendar day, so a row sees an end-of-day value that already counts its own click.
naive = ev.merge(snap[["user", "day", "clicks_7d"]], on=["user", "day"])
# Right: the latest snapshot published at or before the event.
pit = pd.merge_asof(ev.sort_values("ts"), snap.sort_values("snap_ts")[["user", "snap_ts", "clicks_7d"]],
                    left_on="ts", right_on="snap_ts", by="user", direction="backward")
pit["clicks_7d"] = pit.clicks_7d.fillna(0)                    # first day: nothing published yet

for _, r in pit.sample(300, random_state=0).iterrows():       # brute-force check of the as-of join
    s = snap[(snap.user == r.user) & (snap.snap_ts <= r.ts)]
    assert r.clicks_7d == (s.loc[s.snap_ts.idxmax(), "clicks_7d"] if len(s) else 0)

auc_naive = roc_auc_score(naive.click, naive.clicks_7d)
auc_pit = roc_auc_score(pit.click, pit.clicks_7d)
print(f"feature AUC: joined on the day {auc_naive:.3f}, point in time {auc_pit:.3f}")
assert auc_naive > auc_pit + 0.05
```

The day join makes the feature look far stronger (AUC about 0.75 against 0.62) because part of its signal is the label itself. In production the online store holds only yesterday's value, so that signal vanishes and the model trained on it is miscalibrated.

Two ways get training rows right. **Backfill** recomputes historical features with point-in-time joins, so a new feature can be tried on months of history. **Forward logging** records the exact feature vector the server used, keyed by request id, and joins labels later: correct by construction, but a new feature has no history until it has been logged for a while. Mature systems do both.

## Training: spreading the work and surviving failures
### Data parallelism and all-reduce
When the model fits on one device but training takes too long, copy it onto \(N\) workers, give each a slice of the batch, and average the gradients before every update. With a mean loss and equal slices, that average is exactly the full-batch gradient.

The averaging is an **all-reduce**. Sending every gradient to one server makes its network link the bottleneck. A **ring all-reduce** splits each gradient into \(N\) chunks and passes them around a ring: \(N-1\) steps in which each worker adds the chunk it receives (afterwards each holds the full sum of one chunk), then \(N-1\) steps that pass the finished chunks around. Each worker sends \(2(N-1)/N\) times the gradient size, which barely grows with \(N\).

```python
import torch

torch.manual_seed(0)
X, y = torch.randn(512, 8), torch.randn(512)
w = torch.zeros(8, requires_grad=True)


def grad(xb, yb):
    w.grad = None
    ((xb @ w - yb) ** 2).mean().backward()
    return w.grad.clone()


full = grad(X, y)
per_worker = [grad(xb, yb) for xb, yb in zip(X.chunk(4), y.chunk(4))]
assert torch.allclose(torch.stack(per_worker).mean(0), full, atol=1e-6)


def ring_allreduce(arrays):
    """Sum equal-length arrays held by N workers; each worker only ever sends to its right neighbour."""
    n = len(arrays)
    chunks = [np.array_split(a.astype(float), n) for a in arrays]
    sent = [0] * n
    for s in range(n - 1):                                        # reduce-scatter
        msgs = [(i, (i - s) % n, chunks[i][(i - s) % n].copy()) for i in range(n)]
        for i, c, data in msgs:
            chunks[(i + 1) % n][c] += data
            sent[i] += data.size
    for s in range(n - 1):                                        # all-gather
        msgs = [(i, (i + 1 - s) % n, chunks[i][(i + 1 - s) % n].copy()) for i in range(n)]
        for i, c, data in msgs:
            chunks[(i + 1) % n][c] = data
            sent[i] += data.size
    return [np.concatenate(c) for c in chunks], sent


n_workers, size = 8, 1_000_000
grads = [rng.normal(size=size) for _ in range(n_workers)]
out, sent = ring_allreduce(grads)
assert all(np.allclose(o, np.sum(grads, axis=0)) for o in out)
assert max(sent) == 2 * (n_workers - 1) * size // n_workers      # 1.75x the gradient for 8 workers
```

When the model itself does not fit, it is split: **tensor parallelism** splits each large matrix across devices, **pipeline parallelism** puts consecutive layers on different devices, and sharded data parallelism (ZeRO, FSDP) spreads optimizer state, gradients and weights over the workers. Recommendation models hit a different wall, embedding tables with billions of rows, which are sharded by row across machines while the small dense part runs data parallel.

### Checkpoints that actually resume
With hundreds of machines, something fails every few hours. A checkpoint must capture everything that determines the next step: the weights, the optimizer state (Adam's moment estimates), the step counter behind the learning-rate schedule, and the position in the data stream. Below, a 200-step run is repeated with a crash at step 130 and a resume from the step-100 checkpoint.

```python
import io


Xd, yd = torch.randn(4096, 8), torch.randn(4096, 1)


def start():
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    return model, torch.optim.Adam(model.parameters(), lr=1e-2), torch.Generator().manual_seed(2)


def train(model, opt, gen, first, last, store=None, every=50):
    for step in range(first, last):
        idx = torch.randint(0, len(Xd), (64,), generator=gen)       # which rows come next is state too
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(Xd[idx]), yd[idx]).backward()
        opt.step()
        if store is not None and (step + 1) % every == 0:
            store.seek(0)
            torch.save({"step": step + 1, "model": model.state_dict(), "opt": opt.state_dict(),
                        "data_rng": gen.get_state()}, store)


def weights(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


model, opt, gen = start()
train(model, opt, gen, 0, 200)
reference = weights(model)

disk = io.BytesIO()
model, opt, gen = start()
train(model, opt, gen, 0, 130, store=disk)                          # dies after step 130; last save at 100

for restore_opt, restore_data in ((True, True), (False, True), (True, False)):
    disk.seek(0)
    ck = torch.load(disk)
    model, opt, gen = start()
    model.load_state_dict(ck["model"])
    if restore_data:
        gen.set_state(ck["data_rng"])
    if restore_opt:
        opt.load_state_dict(ck["opt"])
    train(model, opt, gen, ck["step"], 200)
    identical = torch.equal(weights(model), reference)
    print(f"resumed at step {ck['step']}, optimizer restored: {restore_opt}, data order restored: "
          f"{restore_data}, identical: {identical}")
    assert identical == (restore_opt and restore_data)
```

With the full state the resumed run matches bit for bit. Without the optimizer state, Adam restarts its moments and its bias-correction step count from zero; without the data generator's state, the resumed run replays batches from the start of the stream instead of the ones that came next. Either way the run drifts. On GPUs bitwise equality is too strict (parallel reductions add in varying order), but every piece of state still has to be saved.

### How often to checkpoint
Checkpointing costs time \(C\) on every save; failing costs the work done since the last save. With failures arriving at random with mean time between failures \(M\), and a save every \(T\) hours of work, the expected wall-clock time per hour of useful work is
\[ \frac{M\left(e^{(T+C)/M} - 1\right)}{T} , \]
ignoring restart time. For \(C \ll T \ll M\) its minimum is close to **Young's approximation** \(T^* \approx \sqrt{2CM}\). Below, a cluster that fails every 10 hours and takes 6 minutes to save:

```python
M, C = 10.0, 0.1                                       # hours between failures, hours to write a checkpoint
T = np.linspace(0.2, 5, 481)
overhead = M * (np.exp((T + C) / M) - 1) / T            # wall-clock hours per hour of useful work
t_best, t_young = T[np.argmin(overhead)], np.sqrt(2 * C * M)

sim_T, segments = 1.4, 200_000                          # Monte Carlo check of the formula at one interval
wall, left = np.zeros(segments), np.ones(segments, bool)
while left.any():                                       # retry each segment until it runs without a failure
    fail = rng.exponential(M, left.sum())
    ok = fail >= sim_T + C
    wall[left] += np.where(ok, sim_T + C, fail)
    left[np.flatnonzero(left)[ok]] = False
formula = M * (np.exp((sim_T + C) / M) - 1) / sim_T
print(f"best interval {t_best:.2f} h, Young {t_young:.2f} h, overhead {overhead.min() - 1:.1%}; "
      f"simulated {wall.mean() / sim_T:.4f} vs formula {formula:.4f}")
assert abs(t_best - t_young) / t_young < 0.1
assert abs(wall.mean() / sim_T - formula) < 0.005
assert min(overhead[0], overhead[-1]) - 1 > 2 * (overhead.min() - 1)   # too often and too rarely both hurt
```

The best interval is about 1.35 hours, close to Young's 1.41, and the overhead is still about 15%. Since it scales roughly as \(\sqrt{2C/M}\), the real lever is cheaper saves: write asynchronously from a copy in host memory, or have each worker save only its own shard.

### Retraining, validation and rollout
A model decays as users and items change, so training becomes a scheduled pipeline: build the newest window, train (often warm-starting from the last model), then pass gates. The gates compare the candidate with the live model on a recent held-out window, check calibration and slices, and replay real requests to confirm output format and latency. A passing model goes to a **registry** as an immutable version with the feature definitions it expects. Serving moves a pointer to it: shadow traffic, then a canary share, then everyone. Rollback moves the pointer back, which works only if the old version and its features are still available.

## Serving: a request in tens of milliseconds
A ranking request carries one user and, after retrieval, a few hundred candidates. **Request features** (user and context) are one key-value read. **Item features** are one batched multi-get for all candidates, not hundreds of round trips. **Cross features** are computed in-process from the two or read from a precomputed table. The model scores the batch, business rules apply, and the response is logged with its request id. [Recommendation ranking](../recommendation-ranking/) runs the retrieve and rank funnel this feeds into.

### Fan-out makes the tail the median
A key-value store with many shards, or a retrieval index split across machines, answers a request only when **every** shard has replied. If each call has a 1% chance of stalling, a request that touches 100 shards hits at least one stall \(1 - 0.99^{100} \approx 63\%\) of the time. One mitigation from Dean and Barroso's "The Tail at Scale" is the **hedged request**: if a shard has not answered by its 95th-percentile latency, send the same call to a replica and take whichever finishes first.

```python
reqs, shards = 20_000, 100


def shard_latency(shape):
    base = rng.lognormal(np.log(5), 0.3, shape)                # about 5 ms normally
    stall = rng.random(shape) < 0.01                           # 1% of calls stall (a pause, a busy neighbour)
    return base + stall * rng.uniform(40, 120, shape)


one = shard_latency(reqs)
fan = shard_latency((reqs, shards)).max(1)                     # wait for the slowest shard
slow_share = (fan > 30).mean()
print(f"one shard: p50 {np.median(one):.1f} ms, p99 {np.percentile(one, 99):.1f} ms; "
      f"{shards} shards: p50 {np.median(fan):.1f} ms, {slow_share:.0%} of requests over 30 ms")
assert abs(slow_share - (1 - 0.99 ** shards)) < 0.02
assert np.median(fan) > np.percentile(one, 99)                # the typical fanned-out request is slower than one shard's p99

hedge_after = np.percentile(one, 95)
first, backup = shard_latency((reqs, shards)), shard_latency((reqs, shards))
hedged = np.minimum(first, np.where(first > hedge_after, hedge_after + backup, np.inf)).max(1)
extra = (first > hedge_after).mean()
print(f"hedged after {hedge_after:.1f} ms: p99 {np.percentile(hedged, 99):.0f} ms "
      f"instead of {np.percentile(fan, 99):.0f} ms, for {extra:.1%} extra calls")
assert np.percentile(hedged, 99) < 0.5 * np.percentile(fan, 99) and extra < 0.06
```

With fan-out, the median request is slower than one shard's 99th percentile. Hedging cuts the p99 several-fold for about 5% more load. This is why latency budgets are set on each dependency's p99, not its average.

### Caching under skewed traffic
A few items, queries and users account for much of the traffic, so caching features, embeddings or whole results pays off. Under Zipf-distributed requests, a least-recently-used cache holding 1% of a million items serves most lookups:

```python
from collections import OrderedDict


def lru_hit_rate(keys, capacity):
    cache, hits = OrderedDict(), 0
    for k in keys:
        if k in cache:
            hits += 1
            cache.move_to_end(k)
        else:
            cache[k] = True
            if len(cache) > capacity:
                cache.popitem(last=False)                       # evict the least recently used
    return hits / len(keys)


def lru_brute(keys, capacity):                                 # list scan: slow but obviously right
    cache, hits = [], 0
    for k in keys:
        if k in cache:
            hits += 1
            cache.remove(k)
        elif len(cache) == capacity:
            cache.pop(0)
        cache.append(k)
    return hits / len(keys)


small = rng.integers(0, 50, 5_000).tolist()
assert all(lru_hit_rate(small, c) == lru_brute(small, c) for c in (1, 5, 20))

n_items = 1_000_000
popularity = 1.0 / np.arange(1, n_items + 1)
traffic = rng.choice(n_items, 300_000, p=popularity / popularity.sum()).tolist()
rates = {frac: lru_hit_rate(traffic, int(frac * n_items)) for frac in (0.001, 0.01)}
print({f"{f:.1%} cached": f"{r:.0%} hits" for f, r in rates.items()})
top_k = popularity[:int(0.01 * n_items)].sum() / popularity.sum()  # pin the 1% most popular items forever
print(f"a static cache of the true top 1% would hit {top_k:.0%}")
assert 0.5 < rates[0.01] < top_k
```

LRU gets most of the way to a cache that knows the true popularity in advance, without being told anything; the gap is items that are requested once, evict a popular one, and are never seen again. The price of any cache is staleness, so the time-to-live must be shorter than the feature's useful freshness: hours for item metadata, seconds or no cache for session features.

### Sharding with consistent hashing
Hashing a key modulo the node count is simple, but adding one node reassigns almost every key, emptying every cache at once. **Consistent hashing** places each node on a ring as many virtual points and gives each key to the next point clockwise, so adding an \(N\)-th node moves only about \(1/N\) of the keys, and every one of them moves to the new node.

```python
import bisect
import hashlib


def h(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big")


def build_ring(nodes, vnodes=100):
    points = sorted((h(f"{n}#{v}"), n) for n in nodes for v in range(vnodes))
    return [p for p, _ in points], [n for _, n in points]


def owner(ring, key):
    pos, names = ring
    return names[bisect.bisect(pos, h(key)) % len(pos)]


keys = [f"user:{k}" for k in range(50_000)]
moved_mod = np.mean([h(k) % 10 != h(k) % 11 for k in keys])
ring10 = build_ring([f"node{i}" for i in range(10)])
ring11 = build_ring([f"node{i}" for i in range(11)])
moved_ring = np.mean([owner(ring10, k) != owner(ring11, k) for k in keys])
print(f"going from 10 to 11 nodes moves {moved_mod:.0%} of keys with modulo, {moved_ring:.0%} with a ring")
assert moved_mod > 0.85 and moved_ring < 0.12
assert all(owner(ring11, k) == "node10" for k in keys if owner(ring10, k) != owner(ring11, k))  # only to the new node
```

Going from 10 to 11 nodes, the modulo rule moves almost every key and the ring about a tenth, all of them to the new node.

### Batching and the rest
Accelerators are efficient only on batches, so model servers use **dynamic batching**: wait a few milliseconds to collect requests, then run them together. If a forward pass costs a fixed 2 ms plus 0.05 ms per example, one example at a time gives about 490 predictions a second and batches of 64 about 12,000, at the cost of up to the wait window in added latency.

```python
fixed_ms, per_ex_ms = 2.0, 0.05
throughput = {b: b / (fixed_ms + per_ex_ms * b) * 1000 for b in (1, 64)}
assert round(throughput[1]) == 488 and throughput[64] > 25 * throughput[1]
```

Three more pieces complete the path. **Real-time features** reach the online store through the stream processor, so a click shows up in the next request within seconds. **Pagination** should serve page two from the list ranked for page one, cached under the session id, or items repeat and skip as scores shift. And every response is **logged** with its features, closing the loop back to training data.

## Pitfalls
- **Joining features by date instead of by time.** The model learns from the future and the offline metrics lie. Use as-of joins or forward logging.
- **Two code paths for one feature.** Training and serving compute it differently and the model is served inputs it never saw. Define features once.
- **Checkpoints that save only the weights.** The resumed run silently differs; the learning-rate schedule and data order restart too.
- **Budgeting latency on averages.** With fan-out, the tail of each dependency becomes the typical request.
- **Caching without an expiry tied to freshness.** Fast and wrong.
- **No rollback path.** Keep the previous model version and its feature definitions deployable until the new one has proven itself.
- **Non-idempotent consumers.** Replays after a crash double-count events, and counts feed straight into features.

The numbers here come from small simulations on rates the code assumes: a 1% stall per shard call, a failure every 10 hours, Zipf traffic over a million items. They show each mechanism, not the sizes a real system will see, and the bitwise resume holds on a CPU, not on GPUs. What stays open is measuring those inputs on a real system: its failure rate and save cost set the checkpoint interval, and how skewed its traffic is sets what a cache can buy.

Sources: the overall structure (data, training and serving infrastructure and the distributed-systems parts beneath them) follows a write-up on [Yuan Meng's blog](https://www.yuan-meng.com/) on what this area of ML engineering covers. Feature stores, batch and streaming features and continual retraining draw on Chip Huyen, Designing Machine Learning Systems (O'Reilly, 2022); queues, replication and partitioning on Martin Kleppmann, Designing Data-Intensive Applications (O'Reilly, 2017); and real-time feature serving on Eugene Yan's [Real-time Machine Learning For Recommendations](https://eugeneyan.com/writing/real-time-recommendations/) (2021). Hedged requests are from Jeffrey Dean and Luiz André Barroso, "The Tail at Scale" (Communications of the ACM, 2013); consistent hashing from Karger et al. (STOC 1997); the checkpoint interval from J. W. Young, "A first order approximation to the optimum checkpoint interval" (Communications of the ACM, 1974); ring all-reduce as brought to deep learning by Baidu (2017) and the Horovod library (Sergeev and Del Balso, 2018). All are rewritten here.
