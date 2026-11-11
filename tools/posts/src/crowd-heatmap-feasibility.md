---
title: Is a city crowd heatmap feasible? A design exploration
summary: Working through whether one developer on a laptop could build an app that shows, and forecasts, how crowded each part of a city is. Which public data can stand in for crowd counts, how to keep it private, a forecasting baseline ladder that a simulation tests, and a batch-first system sketch that carries over from Taipei to other cities.
group: ML systems and design
order: 5
tags: system design · forecasting · geospatial · privacy
---
The idea is simple to state: open a map, see which parts of the city are packed right now and which will be in an hour, and go somewhere quieter. Taipei is the starting city, with the hope that the same design carries over to Tokyo or New York.

This post is a feasibility study, not a launch report. Nothing here has been built. I walk the idea through the questions in [A framework for designing ML systems](../ml-system-design-framework/) to find where it is easy, where it is hard, and where it might quietly not work at all. In a simulation of 30 cells over ten weeks, I find the event calendar matters more than the model: it cuts forecast error on event evenings from about two thirds to about a quarter, while boosting on the clock alone ties a per-cell weekday and hour profile. The constraints are deliberate: one developer, one laptop (an Apple-silicon MacBook Air), public data only, and a model cheap enough to retrain on that laptop.

## What the product has to answer
A user asks one of two questions: "is this place busy right now?" and "when should I go?". Both reduce to a number per place per hour, so the ML objective is a **forecast of a crowd index for each map cell, for each of the next few hours**.

| Piece | Choice |
|---|---|
| Unit of space | A grid cell a few hundred metres across |
| Unit of time | One hour, forecast 1 to 12 hours ahead |
| Prediction | Expected crowd index per cell and hour |
| What the user sees | Quiet, normal or busy, relative to that place's usual level |
| Success metric | Users who check the map report fewer unpleasant crowds; offline, forecast error against baselines |

The relative scale in the last-but-one row matters more than it looks. A side street at its busiest is still emptier than a main station on a quiet morning, so absolute counts would paint the same few hubs red forever. People want to know whether a place is **busier than usual**.

## The hard part: there is no label
Nobody publishes "people per square metre at Ximending, 3 pm". Every usable signal is a proxy for crowding, and each has a blind spot.

| Source | What it measures | Blind spot |
|---|---|---|
| Metro ridership by station and hour | Entries and exits at stations | Published with a lag, not live; says nothing about people who walk or drive |
| Shared-bike dock availability, live | Bikes taken and returned per station | Small numbers; driven by rebalancing trucks as much as by people |
| Bus and rail real-time feeds | Vehicle positions, sometimes occupancy | Movement, not where people stop |
| Weather observations and forecasts | Rain, temperature | Context only, not a count |
| Event and holiday calendars | Concerts, games, markets, long weekends | Must be collected and cleaned by hand |
| Map points of interest (OpenStreetMap) | What each cell contains: malls, markets, offices | Static |

For Taipei, the city's open data portal [data.taipei](https://data.taipei/) and the national [TDX](https://tdx.transportdata.tw/) transport platform cover the first three rows, and the Central Weather Administration publishes weather. Other cities publish different things, so each city needs its own small set of source adapters. New York publishes hourly subway ridership, for instance, while another city may offer only timetables. That is the main cost of "adapt to other cities", and it is a data cost, not a model cost.

Two tempting sources are off the table. The busy-hours bars in commercial map apps are not exposed through an official API, and scraping them breaks the terms of service. Population counts derived from phone networks exist in some countries, but they are usually sold commercially, and where an aggregated version is released openly it tends to be averages over weeks or months rather than a live feed. That makes it useful for checking the crowd index, not for driving a live map. The design has to work without either source.

The crowd index is therefore defined, not observed: a weighted blend of the proxies in each cell, rescaled to that cell's own history. This is the biggest risk in the project. A night market fed mostly by people walking in shows up only faintly in station counts, so the map will be most wrong exactly where crowds are a matter of taste rather than transit.

## From stations to a heatmap
Proxies arrive at points (stations, docks) but the map shows cells. A Gaussian kernel spreads each station's count over nearby cells, with weights normalized so every station hands out exactly its own count. The open-source [H3](https://h3geo.org/) hexagon grid is a common choice for real maps, because all six neighbours of a hexagon share an edge with it and sit at the same distance, which makes smoothing and neighbour features uniform. A square grid keeps the sketch dependency-free.

```python
import numpy as np

rng = np.random.default_rng(0)
cell_m = 500                                            # cell side in metres
nx, ny = 20, 16                                         # a 10 km x 8 km area
cx, cy = np.meshgrid((np.arange(nx) + 0.5) * cell_m, (np.arange(ny) + 0.5) * cell_m)
centres = np.c_[cx.ravel(), cy.ravel()]

stations = rng.uniform([0, 0], [nx * cell_m, ny * cell_m], (40, 2))
counts = rng.poisson(rng.uniform(50, 3000, 40))         # entries + exits in one hour


def to_grid(points, values, centres, bandwidth=400.0):
    d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    w = np.exp(-0.5 * d2 / bandwidth ** 2)
    w /= w.sum(1, keepdims=True)                        # each station distributes exactly its own count
    return values @ w


heat = to_grid(stations, counts, centres)
assert np.isclose(heat.sum(), counts.sum())             # nothing created or lost
assert heat.shape == (nx * ny,)
```

The bandwidth is a product decision as much as a statistical one. Too narrow and the map is a scatter of dots on stations; too wide and every district blurs into one warm smear.

## Privacy
With only aggregated public data, the app never holds anyone's location, and privacy risk is low. The risk arrives with the first feature that collects data from users: "share my location to improve the map", or a "was it crowded?" button that comes with GPS attached. Three rules keep that part safe.

1. **Never store raw traces.** Turn a location into a cell and an hour on the device, or as soon as it reaches the server, and drop the coordinates.
2. **Suppress small counts.** A cell-hour seen by three users can identify them; publish nothing below a threshold.
3. **Add calibrated noise.** The Laplace mechanism of Dwork, McSherry, Nissim and Smith (2006) adds noise with scale \(1/\varepsilon\) to a count where each person contributes at most one, which gives \(\varepsilon\)-differential privacy for that release.

The noise is the same size whatever the count, so it is harmless for busy cells and badly distorts tiny ones. That is why rules 2 and 3 go together. One detail matters: decide what to suppress from the noisy count, not the true one, otherwise the decision to publish or hide a cell itself leaks the true count.

```python
eps = 1.0
true = rng.integers(1, 2000, 5000)
noisy = true + rng.laplace(0, 1 / eps, true.size)
rel_err = np.abs(noisy - true) / true

assert np.median(rel_err[true >= 200]) < 0.01           # busy cells: under 1% error
assert np.median(rel_err[true < 5]) > 0.25              # tiny cells: errors of 25% and more
k_min = 20
published = np.where(noisy >= k_min, np.round(noisy), np.nan)   # threshold the noisy count, never the true one
assert np.isnan(published[true < 10]).mean() > 0.9
```

One caveat: the privacy budget adds up. Publishing the same cell every hour for a month spends \(\varepsilon\) on every release, so a real deployment needs per-user contribution limits over time, not just per release. Coarse cells and hourly buckets are themselves a privacy feature.

## Forecasting: climb a ladder of baselines
Crowds are dominated by the clock: weekday commuting peaks, weekend afternoons, and the same pattern week after week. So the model should start with baselines that only know the clock and earn every step of complexity above them.

1. **Seasonal naive:** same cell, same hour, one week ago.
2. **Profile:** the average for this cell, weekday and hour over the training weeks.
3. **Gradient-boosted trees** with lagged counts, calendar features, rain and an event flag, trained with a Poisson loss because the target is a count (see [Poisson distribution](../poisson-distribution/) and [from decision trees to boosting](../decision-trees-to-boosting/)).

The simulation below builds ten weeks of hourly counts for 30 cells. Each cell mixes a commuter pattern (weekday peaks at 8 am and 6 pm) with a leisure pattern (weekend afternoons, halved by rain), and 40 announced events triple a cell's crowd for an evening. The forecast horizon is three hours. The first week only supplies lag features, weeks two to eight train, and the last two weeks test.

```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

threadpool_limits(1)                                    # data this small trains faster on one thread

hours = pd.date_range("2026-03-02", periods=10 * 7 * 24, freq="h")   # ten weeks, starting on a Monday
hod, dow = hours.hour.to_numpy(), hours.dayofweek.to_numpy()
n_cells, weekend = 30, dow >= 5


def bump(h, centre, width):
    return np.exp(-0.5 * ((h - centre) / width) ** 2)


commute = np.where(weekend, 0.3, 1.0) * (bump(hod, 8, 1.0) + bump(hod, 18, 1.5))
leisure = np.where(weekend, 1.6, 0.7) * bump(hod, 16, 3.5)

rain = np.zeros(len(hours), bool)                       # rainy spells from a two-state Markov chain
for t in range(1, len(hours)):
    rain[t] = rng.random() < (0.85 if rain[t - 1] else 0.03)

event = np.zeros((len(hours), n_cells), bool)
for _ in range(40):                                     # concerts, games, markets: known in advance
    c, d = rng.integers(n_cells), rng.integers(70)
    event[d * 24 + 18: d * 24 + 22, c] = True

base, mix = rng.uniform(20, 400, n_cells), rng.uniform(0, 1, n_cells)   # mix 1 = commuter hub
shape = mix * commute[:, None] + (1 - mix) * leisure[:, None] * np.where(rain, 0.5, 1.0)[:, None]
y = rng.poisson(base * (0.1 + shape) * np.where(event, 3.0, 1.0))      # hours x cells

df = pd.DataFrame({"t": np.repeat(np.arange(len(hours)), n_cells), "cell": np.tile(np.arange(n_cells), len(hours)),
                   "hour": np.repeat(hod, n_cells), "dow": np.repeat(dow, n_cells),
                   "rain": np.repeat(rain, n_cells), "event": event.ravel(), "y": y.ravel()})
H = 3                                                   # forecast horizon in hours
for lag in (H, 24, 168):                                # never a lag shorter than the horizon
    df[f"lag{lag}"] = df.groupby("cell")["y"].shift(lag)
df["rain_now"] = df.groupby("cell")["rain"].shift(H)   # rain observed when the forecast is issued
df = df.dropna()
train, test = df[df.t < 8 * 168], df[df.t >= 8 * 168]


def wape(pred, true):                                   # weighted absolute percentage error
    return np.abs(np.asarray(pred) - np.asarray(true)).sum() / np.asarray(true).sum()


profile = train.groupby(["cell", "dow", "hour"]).y.mean().rename("prof")
prof_pred = test.join(profile, on=["cell", "dow", "hour"]).prof


def fit(cols):
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=0,
                                      categorical_features=[c == "cell" for c in cols])
    return m.fit(train[cols], train.y)


clock_cols = ["cell", "hour", "dow", "lag3", "lag24", "lag168"]
all_cols = clock_cols + ["rain_now", "event"]
oracle_cols = clock_cols + ["rain", "event"]           # rain at the target hour: a perfect weather forecast
gbm_clock, gbm, gbm_oracle = fit(clock_cols), fit(all_cols), fit(oracle_cols)

preds = {"seasonal naive": test.lag168.to_numpy(), "profile": prof_pred.to_numpy(),
         "boosting, clock only": gbm_clock.predict(test[clock_cols]),
         "boosting + rain now + events": gbm.predict(test[all_cols]),
         "boosting + perfect rain forecast + events": gbm_oracle.predict(test[oracle_cols])}
ev = test.event.to_numpy()
scores = {k: wape(p, test.y) for k, p in preds.items()}
on_events = {k: wape(p[ev], test.y[ev]) for k, p in preds.items()}

assert all(int(c[3:]) >= H for c in all_cols if c.startswith("lag"))
assert scores["profile"] < scores["seasonal naive"]
assert abs(scores["boosting, clock only"] - scores["profile"]) < 0.02        # same information, same score
assert scores["boosting + rain now + events"] < 0.95 * scores["profile"]
assert scores["boosting + perfect rain forecast + events"] < scores["boosting + rain now + events"] - 0.01
assert min(on_events[k] for k in ["seasonal naive", "profile", "boosting, clock only"]) > 0.6
assert on_events["boosting + rain now + events"] < 0.5 * on_events["profile"]
```

| Model (3 hours ahead) | WAPE, all test hours | WAPE, event hours |
|---|---|---|
| Seasonal naive | 0.18 | 0.69 |
| Profile by cell, weekday, hour | 0.14 | 0.67 |
| Boosting, clock and lags only | 0.14 | 0.64 |
| Boosting + rain now + event flag | 0.13 | 0.26 |
| Same, with a perfect rain forecast | 0.11 | 0.24 |

Averaging seven weeks beats copying one, because a single week carries that week's noise. Boosting on the same clock information adds nothing: the profile already captures it. The event calendar is where most of the gain is. On event evenings, the hours a crowd-avoidance app exists for, every clock-only method is off by about two thirds, and the event flag cuts that error to about a quarter. Rain helps less than it might, because a shower that starts after the forecast is issued is invisible to "is it raining now?". The last row feeds the model the actual rain at the target hour, which no real system has; it is an upper bound on what a good weather-forecast feed could add. The lesson for the real project: spend effort on the event calendar, then on weather forecasts, before spending it on the model.

The first `assert` guards the easiest mistake to make here. A three-hour forecast cannot use the count from one hour ago, because at prediction time it does not exist yet. Build every feature from what is known at the moment the forecast is issued, which is why the honest model sees rain at issue time and the perfect-forecast row is labelled as such.

## From a forecast to a colour
The model predicts counts; the map shows quiet, normal or busy relative to the cell's own history. Per-cell quantiles from the training weeks do it, and the forecast goes through the same function as the actual count, so the two can be compared:

```python
q = train.groupby("cell").y.quantile([0.4, 0.8]).unstack()
pred = pd.Series(gbm.predict(test[all_cols]), index=test.index)


def level(cell, value):
    lo, hi = q.loc[cell]
    return "quiet" if value < lo else "busy" if value > hi else "normal"


small, big = base.argmin(), base.argmax()
peak_small = test[test.cell == small].y.max()
assert level(small, peak_small) == "busy"               # a small cell's peak still shows as busy ...
assert peak_small < q.loc[big, 0.4]                     # ... even though a big hub's quiet hours reach higher

busy_on_events = {name: np.mean([level(c, v) == "busy" for c, v in zip(test.cell[ev], p[ev])])
                  for name, p in [("actual", test.y.to_numpy()), ("profile", prof_pred.to_numpy()), ("boosting", pred.to_numpy())]}
assert busy_on_events["actual"] > 0.9                   # event hours really are busy for their cell
assert busy_on_events["profile"] < 0.4 < 0.8 < busy_on_events["boosting"]
```

The relative scale is what makes a small cell's peak show as busy. The last check is the product-level version of the event result above: over 90% of event hours are actually busy for their cell, yet the clock-only profile labels fewer than 40% of them busy, while the model with the event flag gets over 80%.

## A system that fits on a laptop
The scale settles most of the architecture. Taipei city is about 270 km²; with cells of roughly 0.25 km² that is about 1,100 cells, times 12 horizons, about 13,000 numbers per hour. That is far too small to need online inference, so everything is precomputed in batch.

```python
cells = round(270 / 0.25)
per_hour = cells * 12
assert 1000 < cells < 1200 and per_hour < 15_000
```

```plaintext
pollers (every 5-10 min)        hourly job                      nightly job
  metro, bikes, weather,  -->   aggregate to cell x hour  -->   retrain, backtest against
  events                        build features                  the baselines, keep the
        |                       predict next 12 hours            model only if it wins
        v                               |
  raw store (Parquet / SQLite)          v
                                predictions table  -->  small read-only API  -->  web map
                                (cell, hour, level)     (cached JSON)             (open-source map library,
                                                                                  OpenStreetMap data)
```

Each piece has a cheap fallback. If a feed is down, features are filled from the profile and the map says the data is stale; if the nightly model loses to the profile in backtesting, the profile serves. Map tiles need care: OpenStreetMap's own tile servers are not meant for app traffic, so use a tile provider or self-hosted tiles built from OpenStreetMap data. A web app beats a native one for a first version, because the main use ("should I go now?") is a quick look, often from a shared link.

## Pitfalls
- **The proxy is the product.** If station counts miss where crowds form, no model fixes that. Validate the crowd index against ground truth first: count people at a handful of places by hand for a few hours and check that the index ranks them correctly.
- **The forecast changes what it forecasts.** If many people avoid the place marked busy, it empties and the next quiet place fills up. At small scale this does not matter. At large scale, spread recommendations over several quiet options instead of naming one.
- **Regime changes.** Holidays, typhoons, new stations and road closures break weekly seasonality. Holiday flags help, and a monitor on forecast error per cell catches the rest (see [simulation-based evaluation](../simulation-based-evaluation/) for building such checks before launch).
- **Feed fragility.** Open-data endpoints change format or go quiet without notice; validate schemas on ingest and alert on missing hours.
- **Terms of use.** Read every data source's licence and attribution requirement before building on it.

## Verdict
Feasible, with one big if. The data plumbing, the forecasting and the serving all fit comfortably on a laptop, and a profile baseline is already useful for "when should I go?". The forecasting results come from a simulation in which 40 events are announced in advance and triple a cell's crowd, so they show that an event flag can work, not how much a real calendar will catch. The open question is whether transit and bike proxies track the crowds people actually feel. That is cheap to check before writing any model code, and it is the first thing to do.
