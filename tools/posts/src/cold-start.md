---
title: Cold start: recommending with no history
summary: New items and new users have no interactions, so a recommender trained on interactions has nothing to say about them and, left alone, never collects any. A content-based prior for new items, simulations of greedy ranking, impression budgets and Thompson sampling, fast updates and an onboarding question for new users, graduation to the main model, and the fallback chain that keeps a feed from ever being empty.
group: Recommender systems
order: 6
tags: recommendation · cold start · exploration · Thompson sampling · Bayesian updating
---
In a simulated feed with ten slots and 4,000 requests, greedy ranking never shows 99.5% of 400 new items and finds none of the 11 stars, new items that belong in the top ten. A recommender learns from interactions, so anything without interactions is invisible to it. A video uploaded a minute ago has no click-through rate; an account created a minute ago has no history. That is the **cold-start problem**, and on a busy platform it is not a corner case: new items arrive every second, and part of every day's traffic comes from people the system has never seen.

I build the standard tools in order: a prior from content for new items, exploration to collect their first evidence, fast updates and one good question for new users, and a fallback chain for when everything else fails. I check the first three in small simulations and describe the fallback chain without one. In them, Thompson sampling with a content prior finds six of the 11 stars and earns the most clicks, and new users click most when every click and skip updates their posterior, more still when half of them answer one onboarding question.

## Two problems and one loop
- **New items** have none of the engagement features (CTR, completion rate, likes per view) the ranker leans on, and their collaborative embeddings were never trained.
- **New users** have no history for the user tower to pool. What remains is the request itself (locale, device, time, how they arrived) and whatever the user is willing to say.

A feedback loop turns the gap into a trap. A new item scored low is not shown; an item not shown collects no clicks; an item with no clicks keeps its low score. The only data that could correct the model is data its own choices prevent. The same loop drives [position bias](/posts/position-bias/), and the cure is similar: deliberately spend some traffic where the model is unsure.

## New items: start from a prior, not from zero
Treat each item's click rate as unknown and describe the belief about it with a Beta distribution, as in [uncertainty in a rate](../beta-uncertainty/). A prior with mean \(m\) worth \(K\) pseudo-impressions, updated with \(c\) clicks in \(n\) impressions, gives the posterior mean

\[ \hat p = \frac{c + K m}{n + K} = \frac{n}{n + K}\cdot\frac{c}{n} + \frac{K}{n + K}\cdot m . \]

With no data the estimate is \(m\); as impressions arrive the weight moves to the observed rate, and the prior's share \(K/(n + K)\) is a half at \(n = K\) and a tenth at \(n = 9K\).

The [features post](../recsys-features/) fits one global prior for all items. A new item does have something better: its **content** (category, creator, duration, embeddings of its title and images). Regress the click rate of mature items on their content and use the prediction as each new item's own prior mean \(m_i\), with \(K\) fitted from how far mature items stray from their prediction once binomial noise is removed. In the simulation, eight content numbers explain part of each item's quality and a hidden term the rest.

```python
import numpy as np

rng = np.random.default_rng(0)
d = 8
w_true = rng.normal(size=d) * 0.35


def logistic(z):
    return 1 / (1 + np.exp(-z))


def make_items(n):
    """Content features and a true click rate; part of the quality is invisible in the content."""
    x = rng.normal(size=(n, d))
    return x, logistic(-3.0 + x @ w_true + 0.5 * rng.normal(size=n))


n_old = 300                                                   # mature items with plenty of history
x_old, ctr_old = make_items(n_old)
imps_old = rng.integers(200, 5000, size=n_old)
clicks_old = rng.binomial(imps_old, ctr_old)
raw = clicks_old / imps_old

# global prior: one mean for everyone, strength from the spread of rates net of binomial noise
M = clicks_old.sum() / imps_old.sum()
K_glob = M * (1 - M) / (raw.var() - np.mean(M * (1 - M) / imps_old)) - 1

# content prior: ridge regression of the (smoothed) logit rate on content
smooth = (clicks_old + K_glob * M) / (imps_old + K_glob)
X1 = np.c_[x_old, np.ones(n_old)]
coef = np.linalg.solve(X1.T @ X1 + np.eye(d + 1), X1.T @ np.log(smooth / (1 - smooth)))


def content_prior(x):
    return logistic(np.c_[x, np.ones(len(x))] @ coef)


m_old = content_prior(x_old)
K_cont = np.mean(m_old * (1 - m_old)) / (np.mean((raw - m_old) ** 2) - np.mean(m_old * (1 - m_old) / imps_old)) - 1
print(f"global prior: mean {M:.3f}, K {K_glob:.0f};  content prior: K {K_cont:.0f}")

# new items: how close is each estimate to the true rate after n impressions?
x_new, ctr_new = make_items(1000)
m_new = content_prior(x_new)
rmse = {}
for n in [0, 10, 100, 1000]:
    c = rng.binomial(n, ctr_new)
    rmse[n] = [np.sqrt(np.mean(((c + K * m) / (n + K) - ctr_new) ** 2)) for K, m in [(K_glob, M), (K_cont, m_new)]]
    print(f"n={n:5d}  RMSE global prior {rmse[n][0]:.4f}  content prior {rmse[n][1]:.4f}")

assert rmse[0][1] < 0.75 * rmse[0][0]                         # content says a lot before any impression
assert rmse[10][1] < rmse[10][0]
assert abs(rmse[1000][1] - rmse[1000][0]) < 0.001             # with enough data the prior stops mattering
assert K_cont > K_glob                                       # less unexplained spread, so a stronger prior
```

| Impressions | RMSE, global prior | RMSE, content prior |
|---|---|---|
| 0 | 0.0479 | 0.0316 |
| 10 | 0.0428 | 0.0302 |
| 100 | 0.0210 | 0.0186 |
| 1000 | 0.0071 | 0.0070 |

Before a single impression, content cuts the error by a third. The advantage is gone by a thousand impressions, which is the point: the prior stands in for evidence until evidence arrives. It is also stronger (\(K\) of 53 against 20), because content has already explained part of the spread.

Two variants are common: averaging the rates of the \(k\) mature items nearest in content-embedding space, and embedding the new item from its content inside a [two-tower model](../two-tower-retrieval/) trained with item IDs randomly hidden. All three fail on an item unlike anything seen before.

## A prior is not enough: someone has to look
Ranking by the posterior mean is greedy. A new item with an average or even good prior will not displace incumbents that have proven themselves, so it is never shown and its estimate never moves.

The simulation runs a feed of ten slots for 4,000 requests, starting from the 300 mature items, with a new item arriving every ten requests. Eleven of the 400 new items are "stars", better than the tenth-best incumbent; a star counts as found once it has 200 impressions. Efficiency is expected clicks over the second half of the run as a share of what an oracle knowing every true rate would earn. Four policies:

- **Greedy:** the posterior mean with the content prior.
- **Impression budget:** greedy, but one slot goes to the new item with the best content prior among those with fewer than 10 impressions, so every item gets a trial.
- **Thompson sampling:** draw a rate from each item's posterior and show the ten highest draws. An item with little data has a wide posterior and sometimes draws high; as evidence accumulates its draws narrow. Run once with the global prior and once with the content prior.

```python
n_arrivals = 400
x_arr, ctr_arr = make_items(n_arrivals)
ctr_all = np.r_[ctr_old, ctr_arr]
m_all = content_prior(np.r_[x_old, x_arr])
star = ctr_arr > np.sort(ctr_old)[-10]                        # new items that belong in the top ten


def run_feed(policy, rounds=4000, slots=10, seed=1):
    g = np.random.default_rng(seed)
    n = np.r_[imps_old, np.zeros(n_arrivals)].astype(float)
    c = np.r_[clicks_old, np.zeros(n_arrivals)].astype(float)
    earned = possible = 0.0
    for t in range(rounds):
        live = n_old + min(n_arrivals, t // 10 + 1)           # one new item every ten requests
        cl, nl, ml = c[:live], n[:live], m_all[:live]
        if policy == "thompson, global prior":
            score = g.beta(cl + K_glob * M, nl - cl + K_glob * (1 - M))
        elif policy == "thompson, content prior":
            score = g.beta(cl + K_cont * ml, nl - cl + K_cont * (1 - ml))
        else:
            score = (cl + K_cont * ml) / (nl + K_cont)        # greedy posterior mean
        shown = np.argpartition(-score, slots)[:slots]
        if policy == "impression budget":
            fresh = np.setdiff1d(np.flatnonzero(nl < 10), shown)
            if len(fresh):
                shown[np.argmin(score[shown])] = fresh[np.argmax(ml[fresh])]
        n[shown] += 1
        c[shown] += g.random(slots) < ctr_all[shown]
        if t >= rounds // 2:
            earned += ctr_all[shown].sum()
            possible += np.sort(ctr_all[:live])[-slots:].sum()
    imps_new = n[n_old:]
    return {"efficiency": earned / possible, "never shown": np.mean(imps_new == 0),
            "stars found": np.sum(imps_new[star] >= 200)}       # a star counts as found once it has 200 impressions


policies = ["greedy", "impression budget", "thompson, global prior", "thompson, content prior"]
feed = {p: run_feed(p) for p in policies}
print(f"{star.sum()} stars among {n_arrivals} new items")
for p, r in feed.items():
    print(f"{p:24s}" + "  ".join(f"{k} {v:.3g}" for k, v in r.items()))

greedy, budget, ts_global, ts_content = feed.values()
assert greedy["never shown"] > 0.98 and greedy["stars found"] == 0  # the loop: new items are almost never tried
assert budget["never shown"] == 0
assert budget["efficiency"] == min(r["efficiency"] for r in feed.values())  # the least efficient policy
assert ts_global["never shown"] < 0.05 and ts_global["efficiency"] < greedy["efficiency"]
assert min(ts_global["stars found"], ts_content["stars found"]) > max(greedy["stars found"], budget["stars found"])
assert ts_content["efficiency"] > max(greedy["efficiency"], budget["efficiency"], ts_global["efficiency"]) + 0.02
assert ts_content["never shown"] > 0.5                          # it skips items that look weak on paper
```

| Policy | Efficiency | New items never shown | Stars found |
|---|---|---|---|
| Greedy | 0.923 | 99.5% | 0 of 11 |
| Impression budget | 0.890 | 0% | 1 of 11 |
| Thompson, global prior | 0.905 | 3% | 4 of 11 |
| Thompson, content prior | 0.950 | 61% | 6 of 11 |

Greedy tries only the two new items whose content prior beats the incumbents and finds no star, while the feed keeps showing worse mature items. The budget fixes visibility and little else: ten impressions at a 6% click rate is under one click, far too little to move a posterior worth 50 pseudo-impressions, so every item gets its trial, proves nothing and drops back. It is the least efficient policy.

Thompson sampling spends exploration where it can change a decision. With the global prior it tries almost every new item and finds four stars, but exploring that broadly still costs more than it earns over this horizon. With the content prior it skips most items that look weak on paper, finds six stars and earns the most clicks. Across other seeds the two variants find similar numbers of stars; the content prior finds them cheaply. The flip side is that a star whose appeal is invisible in its content starts behind. Thompson sampling goes back to Thompson (1933); Chapelle and Li (2011) showed how well it works for click rates, and contextual bandits such as LinUCB (Li et al., 2010) share evidence across items through their features.

A common design stages the budget: a small trial, a larger one for items whose early signal clears a bar, then the main ranker. That is a hand-tuned approximation of what Thompson sampling does automatically. Any budget must be large enough to separate a star from its prior, and new items per day times budget per item must fit inside the traffic set aside, or the queue of untested items grows without limit.

## New users: learn from every impression
A new user is the mirror image: many unknown preferences and a short horizon, since someone whose first screens are dull may not return. The signal available is popularity (per region or time of day), context carried by the request (device, language, referrer, campaign), an onboarding question, and the session itself, if each click and skip reaches the next screen within seconds.

In the simulation each of 3,000 new users has one favourite category out of twenty, drawn by popularity, so about 40% have a favourite outside the top four. They click it at 40% and everything else at 4%, over ten screens of four items. The policies keep a Beta posterior per user and category, starting from that category's population rate.

```python
n_cat, n_users, screens, per_screen = 20, 3000, 10, 4
pop = 1 / np.arange(1, n_cat + 1)
pop /= pop.sum()                                              # Zipf-like popularity of categories
fav = rng.choice(n_cat, n_users, p=pop)
rows = np.arange(n_users)[:, None]
p_click = np.full((n_users, n_cat), 0.04)
p_click[rows[:, 0], fav] = 0.40
mu = 0.04 + 0.36 * pop                                        # population click rate per category
K_cat = mu * (1 - mu) / (0.36 ** 2 * pop * (1 - pop)) - 1    # Beta prior matching the population spread


def run_users(policy, answered=0.0, seed=3, n_screens=screens):
    g = np.random.default_rng(seed)
    a, b = np.tile(K_cat * mu, (n_users, 1)), np.tile(K_cat * (1 - mu), (n_users, 1))
    told = g.random(n_users) < answered                       # these users named their favourite at signup
    a[told, fav[told]] += 4
    clicks = np.zeros((n_users, n_screens))
    for t in range(n_screens):
        if policy == "popularity":
            score = np.tile(mu, (n_users, 1))
        elif policy == "thompson":
            score = g.beta(a, b)
        else:
            score = a / (a + b)                               # greedy on the per-user posterior mean
        show = np.argsort(-score, axis=1)[:, :per_screen]
        y = g.random(show.shape) < p_click[rows, show]
        a[rows, show] += y                                    # clicks raise a category ...
        b[rows, show] += ~y                                   # ... and skips lower it
        clicks[:, t] = y.sum(1)
    return clicks


niche = fav >= per_screen
users = {"popularity": run_users("popularity"), "thompson": run_users("thompson"),
         "posterior mean": run_users("mean"), "posterior mean + question": run_users("mean", answered=0.5)}
for name, c in users.items():
    print(f"{name:26s} clicks per user {c.sum(1).mean():.2f}  screens 1-3 {c[:, :3].mean():.2f}  "
          f"screens 8-10, niche users {c[niche, -3:].mean():.2f}")

pop_c, ts_c, mean_c, q_c = users.values()
assert mean_c.sum(1).mean() > pop_c.sum(1).mean()
assert mean_c[niche, -3:].mean() > 2 * pop_c[niche, -3:].mean()   # niche users are who popularity fails
assert ts_c[:, :3].mean() < pop_c[:, :3].mean()                     # exploration costs early screens
assert ts_c.sum(1).mean() < min(pop_c.sum(1).mean(), mean_c.sum(1).mean())  # ... and does not repay them in ten screens
assert q_c.sum(1).mean() > mean_c.sum(1).mean()
assert q_c[:, :3].mean() > max(pop_c[:, :3].mean(), mean_c[:, :3].mean())  # the answer helps from the first screen

# a longer session: 60 screens instead of 10
long = {p: run_users(p, n_screens=60).sum(1).mean() for p in ["popularity", "thompson", "mean"]}
print("60 screens, clicks per user:", {p: round(v, 1) for p, v in long.items()})
assert long["thompson"] > long["popularity"] and long["thompson"] > 0.95 * long["mean"]
```

| Policy | Clicks per user, 10 screens | Clicks per screen, screens 1-3 | Clicks per screen, screens 8-10, niche users |
|---|---|---|---|
| Popularity | 3.76 | 0.38 | 0.16 |
| Thompson sampling | 3.22 | 0.28 | 0.30 |
| Posterior mean | 4.03 | 0.39 | 0.32 |
| Posterior mean + question (half answer) | 4.64 | 0.46 | 0.42 |

Popularity leaves niche users at the 4% background rate forever. Updating on every impression fixes that with no explicit exploration: the first screen is the popular one, each skip lowers that category in the user's posterior, and on the second screen over nine users in ten already see at least one category the popular screen left out. Skips are information. Thompson sampling, which won for items, loses here: with twenty categories and ten screens its randomness costs more early clicks than it repays. Over a 60-screen session the gap all but closes: Thompson sampling then beats popularity and comes within 5% of the posterior mean. Exploration has to pay back within the time the user stays.

The question beats everything from the first screen although only half the users answer it. If there is room for one question, ask the one whose answer is hardest to predict and would most change the first screen; a question everyone answers alike carries no [information](../entropy-and-information/).

## Graduation: from cold treatment to the main model
- **A separate cold-start model** on content and context, until the item or user has enough evidence. A hard switch makes scores jump, so blending by evidence, like \(n/(n + K)\) above, is smoother.
- **One model for both**, trained with the interaction features or IDs of warm examples randomly hidden so it learns to fall back on content. DropoutNet (Volkovs, Yu and Poutanen, 2017) does this for collaborative embeddings.
- **Freshness as a feature.** Covington, Adams and Sargin (2016) fed the age of each training example into YouTube's candidate generator and set it to zero, or slightly negative, at serving time, so the model could represent the preference for fresh uploads instead of averaging it away.

## The fallback chain
Whatever the models do, the feed must never come back empty:

```plaintext
1. personalized ranking           enough history; main retrieval and ranker
2. lightly personalized           a few interactions; session items, item-to-item neighbours
3. content and context            onboarding answers, locale, device, item content
4. popularity                     trending or editorial, per region; always available, cached
```

Each tier fills the slots the one above left, without duplicates, and has a timeout, so a slow feature store drops the request a tier instead of failing it. The last tier depends on no model.

## Measuring it
Averages hide cold start, because warm traffic dominates. Report the cold slices separately: for items, time to the first \(N\) impressions and how fast eventual stars are found; for users, engagement in the first sessions and return rate after a day and a week. Offline, split by time so test items and users really are new. The net cost of exploration shows only in an [A/B test](../ab-testing/) against a version without it (see also [evaluating ML systems](/posts/evaluating-ml-systems/)), and logging each exploratory item's display probability makes that traffic usable for off-policy evaluation later. Thompson sampling has no closed-form display probability, so estimate it by repeating the draw a few hundred times per request.

## Pitfalls
- **Zero as the estimate.** No clicks means an unknown rate, not a zero one. Use a prior, and give the model the impression count too.
- **Trials that decide nothing.** A budget too small to move the estimate past the bar only costs clicks.
- **Exploring a new user as if they will stay.** Exploration has to pay back within the session.
- **Trusting content absolutely.** Keep some exploration that does not depend on the content prior.
- **Losing the exploration data.** It is the only evidence on new items that the greedy ranker did not filter out; log it with its propensities so it can be reweighted later.

Each policy here runs once, on one simulated catalogue and on users who each have one favourite category, so the tables show how the policies differ, not how much they would gain on real traffic. Two questions stay open: how to find a star whose appeal is invisible in its content, and what exploration costs net, which shows only in an A/B test against a version without it.

Sources: Andrew Schein, Alexandrin Popescul, Lyle Ungar and David Pennock, [Methods and Metrics for Cold-Start Recommendations](https://dl.acm.org/doi/10.1145/564376.564421) (SIGIR 2002); William R. Thompson, On the likelihood that one unknown probability exceeds another in view of the evidence of two samples (Biometrika, 1933); Olivier Chapelle and Lihong Li, An Empirical Evaluation of Thompson Sampling (NIPS 2011); Lihong Li, Wei Chu, John Langford and Robert Schapire, A Contextual-Bandit Approach to Personalized News Article Recommendation (WWW 2010), for exploration with features; Maksims Volkovs, Guangwei Yu and Tomi Poutanen, DropoutNet: Addressing Cold Start in Recommender Systems (NIPS 2017); Paul Covington, Jay Adams and Emre Sargin, [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190) (RecSys 2016). For a wider tour of bandit algorithms, Lilian Weng's [The Multi-Armed Bandit Problem and Its Solutions](https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/) (2018).
