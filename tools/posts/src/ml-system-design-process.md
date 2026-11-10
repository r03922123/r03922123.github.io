---
title: The ML system design process, step by step
summary: How to take a one-line request ("recommend products on the home page") to a design that can be reviewed: clarify and size the problem, write the objective in one sentence, sketch the pipeline around a baseline, specify labels and features, choose a model on explicit axes, plan the evaluation and the serving path, then work through the edge cases and the monitoring. Small calculations size the traffic, check label agreement, size an A/B test, estimate a cache hit rate and compare exploration strategies for new items.
group: ML systems and design
order: 2
tags: system design · design review · evaluation · serving
---
A design request usually arrives as one sentence: "we want personalised recommendations on the home page". Between that sentence and a model in production sit a dozen decisions, and bad designs usually make them in the wrong order or never write them down.

The [framework post](../ml-system-design-framework/) lists the questions a design has to answer. This one is the working process: the steps for writing or reviewing a design, what each should leave in the design document, and the small calculations that turn opinions into numbers. One running example is used throughout: an online shop with 5 million daily users wants a "recommended for you" row on its home page.

## Step 1: clarify the problem and size it
Before any modelling, pin down four things.

- **Who and what.** Which users see the row, on which surface, and what they are supposed to do next (open a product, add it to the cart, buy).
- **The current system.** If a bestseller list fills the row today, what is wrong with it? "Same items for everyone" and "stale by Friday" lead to different designs.
- **Scale.** Daily active users, peak requests per second, catalogue size, how fast the catalogue changes.
- **Constraints.** Real-time or batch, the latency budget for the whole request, privacy rules on which data may be used.

Scale deserves a back-of-envelope estimate, because it decides what fits in memory and what can be precomputed.

```python
import numpy as np
from scipy import stats

dau, visits_per_user, peak_factor = 5_000_000, 3, 3
avg_qps = dau * visits_per_user / 86_400
peak_qps = peak_factor * avg_qps                         # assume the busiest hour runs 3x the daily average
n_items, n_users, dim = 2_000_000, 20_000_000, 64
item_index_gb = n_items * dim * 4 / 1e9                  # float32 item embeddings
user_table_gb = n_users * dim * 4 / 1e9
impressions_per_day = dau * visits_per_user * 20         # 20 items in the row
print(f"peak {peak_qps:.0f} QPS, item index {item_index_gb:.2f} GB, user table {user_table_gb:.1f} GB, "
      f"{impressions_per_day / 1e6:.0f}M impression rows per day")
assert 500 < peak_qps < 600
assert item_index_gb < 1 < user_table_gb                 # the item index fits in RAM on every serving replica; the user table goes to a shared key-value store
```

A few hundred requests per second is modest; 300 million logged impressions a day is not.

## Step 2: write the objective in one sentence
Translate the business goal into a prediction, in a sentence of this shape: *build a [classification, regression or ranking] model that predicts [a probability, a value or an order] from [these inputs]*. For the example: "rank candidate products for a user by the probability that they add the product to the cart within the session, given their history, the product and the context."

Then pair the offline metric that judges the model with the online metric that judges the product.

| Task | Offline metric | Online metric |
|---|---|---|
| Ranking a list | NDCG@k, recall@k | click-through and add-to-cart rate, revenue per session |
| Rare-event classification | PR-AUC, recall at a fixed precision | caught events, false alarms handled per reviewer hour |
| Regression (delivery time) | MAE, calibration by bucket | complaints about lateness, cancellations |

The rest of the design is checked against this sentence: a choice that cannot be traced back to it is suspect.

## Step 3: sketch the pipeline around a baseline
Draw the whole loop before any box gets detailed: how data is logged, how training data is built, how the model reaches serving, and how serving produces the next logs.

```plaintext
offline:  event logs -> labelled examples -> features -> train -> validate -> model registry
                                                                       \-> item embeddings -> ANN index
online:   request -> candidates (ANN + popular) -> fetch features -> rank -> business rules -> response
                                                                                         \-> impression log
```

Put a **baseline** in the diagram from the start: for recommendations, the most popular products in the user's region, refreshed hourly. It is cheap, it exercises logging and serving end to end, it is the fallback when the model times out, and it sets the number every later model must beat.

## Step 4: specify the labels and the features
**Training data.** Say where examples come from (existing logs, or new collection), how many are needed, and what a positive is. For the shop, implicit labels such as clicks and add-to-carts are plentiful but biased by what was shown and where; see [position bias](/posts/position-bias/). A smaller set of human judgements ("is this a sensible recommendation for this user?") is still useful for evaluation.

Human labels can come from domain experts, a dedicated labelling team, written rules or an LLM; they trade cost, speed and consistency. Whoever labels, have a sample labelled twice and measure agreement. Raw agreement is misleading when one class dominates, which is why **Cohen's kappa** corrects it by the agreement expected by chance from each labeller's marginal rates:
\[ \kappa = \frac{p_o - p_e}{1 - p_e}, \qquad p_e = \sum_c p_{A}(c)\, p_{B}(c) . \]

```python
from sklearn.metrics import cohen_kappa_score

rng = np.random.default_rng(0)


def kappa(a, b):
    p_o = np.mean(a == b)
    p_e = sum(np.mean(a == c) * np.mean(b == c) for c in np.union1d(a, b))
    return (p_o - p_e) / (1 - p_e)


truth = rng.random(5000) < 0.10                           # 10% of judged pairs are truly relevant
careless = [rng.random(5000) < 0.10 for _ in range(2)]    # say "relevant" 10% of the time, at random
careful = [truth ^ (rng.random(5000) < 0.05) for _ in range(2)]   # right 95% of the time
for name, (a, b) in {"careless": careless, "careful": careful}.items():
    k = kappa(a, b)
    assert abs(k - cohen_kappa_score(a, b)) < 1e-9
    print(f"{name:9s} raw agreement {np.mean(a == b):.2f}, kappa {k:.2f}")
assert np.mean(careless[0] == careless[1]) > 0.8 and abs(kappa(*careless)) < 0.1
assert kappa(*careful) > 0.5
```

The careless pair agree on over 80% of items while ignoring the content; kappa shows they do no better than chance. A low kappa often means an ambiguous guideline, and the fix is then to rewrite it with examples.

**Features.** List each feature with how and when it is computed, how it is represented, and what happens when it is missing. [Features for recommendation](../recsys-features/) covers how to build them; the design document only needs the spec.

| Feature | Computed | Representation | When missing |
|---|---|---|---|
| Purchases by category, last 90 days | offline, daily | log counts per category | zeros plus a "new user" flag |
| Product price | offline, on change | log price relative to the category median | category median |
| Product title and description | offline, at publish | text embedding | never missing: required at publish |
| Products viewed this session | online, at request | last 20 product IDs as a sequence | empty sequence |
| Device, hour of day | online, at request | categorical; hour as sine and cosine | "unknown" category |

Online features need a low-latency store, and every feature must be computed by the same code for training and serving, or the model is trained on one definition and served another.

## Step 5: choose the model on explicit axes
Climb from the baseline only as far as the constraints allow: popularity, then collaborative filtering (matrix factorisation) or content similarity, then gradient-boosted trees on the feature table, then neural models. Judge each rung on the same axes.

| Axis | Question |
|---|---|
| Quality | How much does it beat the previous rung offline, and is that worth the rest of this table? |
| Latency | Does it fit its share of the request budget (often a few tens of milliseconds, not the whole budget)? |
| Cost | Training time, serving hardware, how often it must be retrained |
| Interpretability | Must individual decisions be explained to users, auditors or the on-call engineer? |
| Maturity | Has the approach been run in production by others, with known failure modes? |

For the shop, a sensible design is a two-tower model for retrieval ([two-tower retrieval](../two-tower-retrieval/)) and boosted trees or a small neural ranker on top ([recommendation ranking](../recommendation-ranking/)). If the ranker needs more capacity, well-studied architectures exist: Wide & Deep (Cheng et al., 2016) combines memorised feature crosses with a deep network, DCN-V2 (Wang et al., 2021) learns explicit feature crosses, and MMoE (Ma et al., 2018) shares experts across several objectives such as click and purchase.

## Step 6: plan the evaluation
[Evaluating ML systems](/posts/evaluating-ml-systems/) covers the metrics in depth; the design needs a plan. Offline: a time-based split, recall@k for retrieval and NDCG@k for ranking, each compared with the baseline. Online: an A/B test randomised by user, with one primary metric, a few guardrails (page latency, returns, the share of the catalogue ever shown), and a **minimum detectable effect** that fixes the sample size before the test starts.

For a click-through rate of \(p_0 = 4\%\) and a relative lift worth detecting of 5% (\(p_1 = 4.2\%\)), at 5% significance and 80% power, each arm needs
\[ n = \frac{(z_{0.975} + z_{0.8})^2 \,[\,p_0(1-p_0) + p_1(1-p_1)\,]}{(p_1 - p_0)^2} \]
observations. The check below computes it and confirms by simulation that it delivers the promised power.

```python
p0, p1, alpha, power = 0.040, 0.042, 0.05, 0.80
z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
n = int(np.ceil(z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2))


def rejects(pa, pb, sims=4000):
    a, b = rng.binomial(n, pa, sims) / n, rng.binomial(n, pb, sims) / n
    pool = (a + b) / 2
    zstat = (b - a) / np.sqrt(2 * pool * (1 - pool) / n)
    return np.mean(np.abs(zstat) > stats.norm.ppf(1 - alpha / 2))


print(f"{n:,} per arm; simulated power {rejects(p0, p1):.2f}, false positive rate {rejects(p0, p0):.3f}")
assert 150_000 < n < 160_000
assert abs(rejects(p0, p1) - power) < 0.03 and abs(rejects(p0, p0) - alpha) < 0.015
```

If the unit is a user, about 155,000 per arm, or 310,000 across both arms, is roughly 6% of one day's 5 million users, so sample size is not the bottleneck; the test should still run for at least a full week to cover weekday patterns. The simulation treats each observation as independent; in practice impressions from the same user are correlated, so analyse per user (or with the delta method) as [A/B testing](../ab-testing/) explains.

## Step 7: plan the serving path
Decide what is computed per request and what is precomputed. The usual levers:

- **Precompute and cache.** Item embeddings and the ANN index are built offline; user embeddings can be cached with a time-to-live; popular results can be cached outright.
- **Shrink the model.** Distillation into a smaller model, INT8 quantization and pruning cut latency and cost, each with an offline quality check.
- **Infrastructure.** A feature store with online and offline sides, a model registry, and a **shadow** deployment that scores live traffic without serving it, to catch latency problems and skew early.

Caching pays because traffic is skewed. If product lookups follow a Zipf law, caching the top 1% of products serves most requests:

```python
ranks = np.arange(1, n_items + 1)
weights = 1.0 / ranks                                     # Zipf with exponent 1
share = weights / weights.sum()
cached = n_items // 100
analytic = share[:cached].sum()
sample = rng.choice(n_items, size=1_000_000, p=share)
simulated = np.mean(sample < cached)
print(f"caching 1% of products serves {analytic:.0%} of lookups (simulated {simulated:.0%})")
harmonic = (np.log(cached) + np.euler_gamma) / (np.log(n_items) + np.euler_gamma)   # H_n ~ ln n + gamma
assert abs(analytic - simulated) < 0.005 and abs(analytic - harmonic) < 1e-3 and analytic > 0.65
```

The exponent is an assumption to measure on real logs; a flatter popularity curve makes the cache much less effective.

## Step 8: the deep dive
With the main path agreed, go through what will break it.

- **Cold start.** A new product has no interactions, so retrieval leans on its content embedding, and it needs some exposure to collect its own data. A new user has no history, so use popularity, context (region, device, referral) and whatever they do in their first session. [Cold start](../cold-start/) treats both.
- **Sparsity and seasonality.** Share information through content and category-level features; add time-aware features and retrain as fast as tastes move.
- **Selection bias.** The model only sees outcomes for what it chose to show. Correct training with inverse propensity weighting, and compare rankers online with interleaving, which is an evaluation method rather than a correction.
- **Scale.** Approximate nearest neighbour search (libraries such as FAISS or ScaNN; see [vector databases](../vector-databases/)) for retrieval; data-parallel training with all-reduce when one machine is too slow, and sharded embedding tables when they no longer fit on one device.

Exposure for new products is an exploration problem. A greedy policy that always shows the product with the best observed click rate locks onto whichever got lucky first. **Thompson sampling** draws a click rate for each product from its [Beta posterior](../beta-uncertainty/) and shows the highest draw, so uncertain products still get shown until their data says otherwise.

```python
runs, k, steps = 200, 20, 5000
true_ctr = rng.beta(2, 40, (runs, k))                     # 20 new products per run, CTRs around 5%
best = true_ctr.max(1)


def regret(policy):
    clicks, shown, lost = np.zeros((runs, k)), np.zeros((runs, k)), 0.0
    for t in range(steps):
        if policy == "greedy":
            score = np.where(shown > 0, clicks / np.maximum(shown, 1), np.inf)   # try each once, then exploit
        else:
            score = rng.beta(clicks + 1, shown - clicks + 1)
        pick = score.argmax(1)
        p = true_ctr[np.arange(runs), pick]
        clicks[np.arange(runs), pick] += rng.random(runs) < p
        shown[np.arange(runs), pick] += 1
        lost += (best - p).sum()
    return lost / runs                                    # expected clicks lost per run vs always showing the best


greedy, thompson = regret("greedy"), regret("thompson")
print(f"clicks lost over {steps} impressions: greedy {greedy:.0f}, Thompson sampling {thompson:.0f}")
assert thompson < 0.6 * greedy
```

**Monitoring and retraining.** Track model metrics (NDCG, calibration), product metrics (click-through, conversion, diversity of what is shown) and drift in the inputs and the scores; the [framework post](../ml-system-design-framework/) computes a population stability index for the inputs. Alert when a metric falls below its floor or a drift score rises above its ceiling. An alert should start a diagnosis, not an automatic retrain: retraining fixes a model that has gone stale, but not a broken upstream pipeline or a label whose meaning has changed.

## Pitfalls
- **Skipping the sizing.** Whether an index fits in memory follows from five minutes of arithmetic.
- **No baseline in the first diagram.** Without it there is no fallback and no number to beat.
- **An A/B test with no minimum detectable effect.** Tests that are stopped when they look good produce false wins.
- **Giving the model the whole latency budget.** Feature fetching, candidate generation and business rules need their share.
- **Treating a drift alarm as a retraining trigger.** Find out what changed first.

Sources: the order of the steps broadly tracks Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022), and the split into offline and online paths draws on Eugene Yan's [System Design for Recommendations and Search](https://eugeneyan.com/writing/system-design-for-discovery/) (2021), both rewritten here. Architectures cited: Cheng et al., "Wide & Deep Learning for Recommender Systems" (2016); Wang et al., "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems" (2021); Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" (KDD 2018).
