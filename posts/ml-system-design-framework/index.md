# A framework for designing ML systems

Rank 2,000 simulated videos, a tenth of them clickbait, by click probability and a session gets 15.7 clicks and 15.5 minutes of watching; by expected minutes, 8.7 clicks and 79 minutes. When an ML project disappoints, the cause often sits upstream of the model: it optimized the wrong thing, a rule would have done the job, the labels quietly meant something else, or nobody noticed the inputs had changed after launch. The architecture is a late decision in a chain, and every earlier link can break it.

This post lays out that chain as four questions, answered in order, with small simulations of what goes wrong when one is skipped. I simulate a proxy rewarding clickbait, a rule matching boosted trees on one task only, a threshold set by business costs, and a drift alarm. Sibling posts go deeper on each step; this one is the map.

## Four questions, in order

1. **What does the business want, and what ML objective stands in for it?** Name the goal, the prediction that serves it, the metric that shows it worked, and the assumptions linking them.
2. **What are the data, the labels and the features?** Where each training example comes from, what counts as a positive, and what is known at prediction time.
3. **What model, and why?** A baseline first, then the cheapest model that meets the constraints on quality, latency, cost and explainability.
4. **How is it evaluated, shipped and kept healthy?** Offline metrics, an online experiment, a rollout plan, monitoring and retraining.

Each answer constrains the next. "Use a two-tower model" answers question 3; without answers to 1 and 2 nobody can say whether it is a good answer. The loop runs many times, since monitoring often sends you back to step 1, but each pass goes through the questions in this order.

## Step 1: from business goal to ML objective
A business goal is not something a model can optimize. "Increase engagement" has to become a prediction with a label, and a measured outcome that tells you whether the prediction helped.

| Business goal | Example system | ML objective | Success metric |
|---|---|---|---|
| Engagement | Video feed | P(watch past 50% or like) for each candidate | Watch time per user, return rate |
| Lower operating cost | Support ticket routing | Classify a ticket to the right team | Share of tickets resolved with no manual triage, misroute rate |
| Satisfaction | Search autocomplete | P(next query given prefix and history) | Searches ending in a click with no reformulation |
| Risk | Fake account detection | P(account is fake) from behaviour | Fake accounts removed per reviewer hour, false removals |
| Revenue | Ad ranking | P(click) and P(conversion), calibrated | Revenue per thousand impressions, advertiser return |

Every row hides an assumption: that the ML objective moves the success metric. It is a **proxy**, and proxies can be gamed by the very model trained on them. The standard defence is to pick the objective carefully and to watch **guardrail metrics** that should not get worse, such as complaints, hides and unsubscribes.

The classic failure is optimizing clicks when the business wants time well spent. In the simulation below every item has an appeal (how tempting its thumbnail is) and a quality (how long people watch once they click). A slice of the catalogue is clickbait: very appealing, short watches. Two rankers use the true probabilities, so the only difference between them is the objective.

```python
import numpy as np
from scipy.special import expit

rng = np.random.default_rng(0)
n_items = 2000
appeal = rng.normal(0, 1, n_items)
quality = rng.gamma(2.0, 2.0, n_items)                  # mean minutes watched after a click
bait = rng.random(n_items) < 0.1
appeal[bait] += 1.5                                     # tempting thumbnails ...
quality[bait] *= 0.15                                   # ... that people abandon quickly

p_click = expit(appeal - 2.0)
exp_minutes = p_click * quality                         # expected minutes per impression


def serve(score, k=20, sessions=5000):
    """Show the top-k items by score, simulate sessions, return clicks and minutes per session."""
    top = np.argsort(-score)[:k]
    clicks = rng.random((sessions, k)) < p_click[top]
    minutes = clicks * rng.exponential(quality[top], (sessions, k))
    return clicks.sum(1).mean(), minutes.sum(1).mean(), bait[top].mean()


ctr_clicks, ctr_minutes, ctr_bait = serve(p_click)
wt_clicks, wt_minutes, wt_bait = serve(exp_minutes)
print(f"rank by P(click):        {ctr_clicks:.2f} clicks, {ctr_minutes:5.1f} min, {ctr_bait:.0%} bait")
print(f"rank by expected minutes: {wt_clicks:.2f} clicks, {wt_minutes:5.1f} min, {wt_bait:.0%} bait")
assert ctr_clicks > 1.2 * wt_clicks                     # the click ranker wins its own metric ...
assert wt_minutes > 1.5 * ctr_minutes                   # ... and loses the one the business cares about
assert ctr_bait > 0.5 > wt_bait
```

The click ranker gets 15.7 clicks per session against 8.7, and 15.5 minutes against 79, because 85% of its slots go to clickbait. An offline evaluation that only reported AUC on clicks would have approved it. The fix is in the objective (predict P(click) and the watch time given a click, and rank by their product, or train on a "meaningful watch" label) and in the success metric, which must be the business outcome rather than the model's own target. [Evaluating ML systems](/posts/evaluating-ml-systems/) follows this chain from business value down to model metrics for several kinds of system.

## Is ML the right tool?
Before designing a model, check that one is needed. ML earns its cost when:

- something observable correlates with the outcome you want to predict, whether a true label or a stand-in that moves with it;
- the mapping from inputs to outcome has too many cases, or shifts too often, for hand-written rules to keep up;
- examples are already logged or can be gathered, and new ones keep coming so the model can be refreshed;
- the decision is made often enough that a small gain per decision adds up to more than the cost of building, serving and maintaining a model.

ML is the wrong tool when a short rule already captures the pattern, when there is no data and no way to get labels, or when every decision must be traced to a written policy. That last case does not rule out learning entirely: a small tree or a scorecard can be read line by line, but a black box usually cannot be defended to a regulator.

The cheapest check is to compare a model against the best simple rule. Below, the "rule" is the best single-feature threshold (a depth-one tree) and the model is gradient boosting. On a task whose label really is a threshold, the model adds nothing. On a task driven by interactions between features, no single rule comes close.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

threadpool_limits(1)                                    # small models: extra threads only add overhead

X = rng.normal(size=(20000, 6))
tasks = {
    "threshold": (X[:, 0] > 1.0) ^ (rng.random(20000) < 0.05),           # one cut, 5% label noise
    "interaction": (X[:, 0] * X[:, 1] + np.sin(2 * X[:, 2]) * X[:, 3] > 0.3) ^ (rng.random(20000) < 0.05),
}
gap = {}
for name, y in tasks.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rule = DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr)
    model = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    auc_rule = roc_auc_score(yte, rule.predict_proba(Xte)[:, 1])
    auc_model = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    gap[name] = auc_model - auc_rule
    print(f"{name:12s} best rule AUC {auc_rule:.3f}   boosted trees AUC {auc_model:.3f}")
assert gap["threshold"] < 0.01
assert gap["interaction"] > 0.2
```

The AUC gap stays under 0.01 on the threshold task and exceeds 0.2 on the interaction task. The rule is also the **baseline** for question 3: whatever model you propose has to beat it by enough to pay for itself.

## Step 2: data, labels and features
This is where most of the effort goes and most of the silent bugs live.

- **Labels.** Decide what a positive is and write it down. Explicit labels (human review, reported spam) are clean but scarce and slow; implicit ones (clicks, purchases, watch time) are plentiful but biased by what the system chose to show, as [position bias](/posts/position-bias/) shows for clicks. Many labels arrive late: a purchase may follow a click by days, so the newest examples look negative until they mature.
- **Features.** Group them by what they describe (the user, the item, the context, and the match between them) and fix each one's definition: what is counted, over which window, as of which moment. [Features for recommendation](../recsys-features/) works through these on a click log.
- **Point in time.** A training row may only use what was known when the prediction would have been made. Joining today's feature values onto last month's labels leaks the future, and the offline score becomes a fiction.
- **Training-serving skew.** Compute a feature once, in one place, for both training and serving. Two implementations of "clicks in the last 7 days" will disagree, and the model learns from one and is served the other.

## A high-level design
Most ML systems split into an **offline path** and an **online path**. Offline: collect logs, generate labelled training data, compute features, train, validate, and publish a model and any precomputed artefacts such as item embeddings or an index. Online: a request arrives, features are fetched, the model scores, business rules apply, the result is returned, and the request and outcome are logged, which feeds the offline path again.

When the item set is large, the online path becomes a funnel, and its sizes come from arithmetic, not taste. A ranker that costs 20 µs of compute per item cannot score a 50 million item catalogue within a 100 ms request, so a cheap retrieval stage has to cut the list first:

```python
catalogue, budget_ms, ranker_us, overhead_ms = 50_000_000, 100, 20, 40
max_ranked = int((budget_ms - overhead_ms) * 1000 / ranker_us)   # items one request can afford to score
assert max_ranked == 3000 and catalogue / max_ranked > 10_000      # retrieval must discard > 99.99%

qps, candidates = 20_000, 500
cores = qps * candidates * ranker_us / 1e6                        # core-seconds of ranking per second
assert cores == 200
```

The same arithmetic sizes the serving fleet: 20,000 requests a second of 500 candidates each need 200 cores of ranking. [Recommendation ranking](../recommendation-ranking/) runs the retrieve, rank and re-rank funnel end to end, and [two-tower retrieval](../two-tower-retrieval/) covers the first stage.

## Step 3: model design
Start with the baseline: a rule, a popularity list, or logistic regression on a few strong features. It is fast to build, exercises the whole pipeline, and sets the number every later model must beat. Then climb only as far as the constraints allow.

| Choice | Strengths | Costs |
|---|---|---|
| Rules or heuristics | Transparent, no training data | Brittle; someone maintains them forever |
| Linear or logistic model | Fast, easy to debug; logistic outputs are usually well calibrated | Misses interactions unless you build them by hand |
| Gradient-boosted trees | Strong on tabular features, little tuning | Weak on raw text, images and IDs with millions of values |
| Neural networks | Learn from raw inputs and embeddings | Need more data, compute and care to serve |
| Pretrained or foundation model | Strong with few labels | Latency, cost per call, less control |

The deciding questions are usually not about accuracy: how many milliseconds the model gets per request, how often it must be retrained, whether its decisions must be explained, and how many labelled examples exist. [Decision trees to boosting](../decision-trees-to-boosting/) and [foundation models](../foundation-models/) cover two of these rungs.

## Step 4: evaluate, ship, monitor, repeat
Offline evaluation on a held-out set split by time decides whether a model is worth testing. An online A/B test decides whether it ships, measured on the success metric from step 1 plus the guardrails; see [A/B testing](../ab-testing/). Between the two, a **shadow** deployment scores live traffic without acting on it (catching latency problems and training-serving skew), and a **canary** sends a small share of traffic to the new model before the rest.

Business costs often set the last knob directly. A ticket router can either route a ticket automatically or send it to a person for triage. Say triage costs \(c_h = 4\) and a wrong automatic route costs \(c_m = 16\) once the bounce and the delay are counted. Routing a ticket whose top class has probability \(p\) is worth it when its expected cost is lower:
\[ (1 - p)\, c_m < c_h \quad\Longleftrightarrow\quad p > 1 - \frac{c_h}{c_m} = 0.75 . \]
That holds only if \(p\) is calibrated, so the check below sweeps thresholds on held-out tickets and confirms the best one sits near 0.75.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

Xt, yt = make_classification(n_samples=40000, n_features=20, n_informative=8, n_classes=5,
                             n_clusters_per_class=1, class_sep=1.0, random_state=1)
Xa, Xb, ya, yb = train_test_split(Xt, yt, test_size=0.5, random_state=1)
router = LogisticRegression(max_iter=2000).fit(Xa, ya)
proba = router.predict_proba(Xb)
p_top, correct = proba.max(1), proba.argmax(1) == yb

c_h, c_m = 4.0, 16.0
thresholds = np.linspace(0, 1, 101)
cost = [np.where(p_top >= t, np.where(correct, 0.0, c_m), c_h).mean() for t in thresholds]
best = thresholds[int(np.argmin(cost))]
auto_share = (p_top >= best).mean()
print(f"accuracy {correct.mean():.2f}; best threshold {best:.2f}; routes {auto_share:.0%} automatically; "
      f"cost per ticket {min(cost):.2f} vs {cost[-1]:.2f} all manual, {cost[0]:.2f} all automatic")
assert abs(best - (1 - c_h / c_m)) < 0.1
assert 0.5 < auto_share < 0.7
assert min(cost) < 0.8 * min(cost[0], cost[-1])
```

The sweep lands on 0.72, routes about 60% of tickets automatically, and costs 2.60 per ticket against 4.00 for all manual and 3.48 for all automatic. The threshold is a business decision expressed in probability, and it moves when the costs do.

After launch the main risk is change. Users, items and upstream data drift, and the model degrades without raising an error. Monitor the input features and the distribution of predictions, not only accuracy, because the labels needed to measure accuracy often arrive late. One common input check is the **population stability index**, which compares a feature's binned distribution in training (\(e_i\)) with what serving sees now (\(a_i\)):
\[ \text{PSI} = \sum_i (a_i - e_i) \ln \frac{a_i}{e_i} . \]
A common rule of thumb reads below 0.1 as stable and above 0.25 as a shift worth investigating.

```python
def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))[1:-1]       # decile cuts from training
    e = np.bincount(np.searchsorted(edges, expected), minlength=bins) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual), minlength=bins) / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


train_feature = rng.lognormal(3.0, 0.8, 50000)            # e.g. basket value at training time
same = psi(train_feature, rng.lognormal(3.0, 0.8, 20000))
shifted = psi(train_feature, rng.lognormal(3.5, 0.9, 20000))
print(f"PSI same distribution {same:.4f}, after a shift {shifted:.3f}")
assert same < 0.01 and shifted > 0.25
```

A fresh sample from the training distribution scores below 0.01; one with its log-mean moved from 3.0 to 3.5 and a wider spread scores above 0.25. An alarm sends you back around the loop: retrain on recent data if the relationship still holds, or return to steps 1 and 2 if the product, the users or the meaning of a label has changed.

## Pitfalls
- **Starting at the model.** An architecture chosen before the objective and the labels are fixed is a guess.
- **Optimizing the proxy into the ground.** Watch the business metric and the guardrails, not only the model's own target.
- **No baseline.** Without a rule or a simple model to beat, nobody can say whether the complex one was worth it.
- **Random splits on time-ordered data.** They leak the future and inflate offline scores. Split by time.
- **Offline wins that do not survive online.** Feedback loops, position bias and skew open a gap that only an experiment closes.
- **No owner after launch.** Drift, retraining and label pipelines are part of the system, not an afterthought.

Every simulation here makes its own data, with the clickbait share, label noise, ticket costs and drift set in the code, so each shows how a step can fail, not how often it fails on real data. On a real system, the gap between offline and online results stays open until an experiment closes it.

Sources: the overall loop from objective to monitoring follows Chip Huyen, Designing Machine Learning Systems (O'Reilly, 2022); the split into online and offline paths and the retrieval and ranking funnel also draw on Eugene Yan's [System Design for Recommendations and Search](https://eugeneyan.com/writing/system-design-for-discovery/) (2021) and on Yuan Meng's blog writing on ML model design for recommendation and search. The checklist for when ML is the right tool is adapted from chapter 1 of Huyen's book. All of it is rewritten here.
