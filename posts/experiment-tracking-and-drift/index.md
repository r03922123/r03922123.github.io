# Experiment tracking, model registries and drift monitoring

Shift one input feature by a full standard deviation and its population stability index jumps to 0.95, yet the model's AUC barely moves, from 0.865 to 0.864. Change how that feature relates to the label instead and every input looks untouched (all PSIs under 0.01) while AUC falls to 0.63. Both results come from the simulation at the end of this post, and they set its theme: monitoring tells you that something changed, not whether the model got worse. Before you can even ask that, you need to know exactly which model is running and how it was made.

This post covers the three tools behind the "MLflow", "model registry" and "monitoring" lines in ML engineering job ads. **Experiment tracking** stores, for every training run, what went in and what came out. A **model registry** names the models worth keeping, numbers their versions and says which one serves. **Drift monitoring** compares what the deployed model sees now with what it was trained on. I build a tiny version of each in plain Python so the rules show up as asserts, then show the MLflow calls that do the same job.

## What a run must record
A run is reproducible when someone else can rebuild it from its record alone and get the same metric. That takes six things:

| Field | Example | Why it is needed |
|---|---|---|
| Parameters | `C=1.0`, `split_seed=0` | Hyperparameters and every seed; a seed left out is a hidden parameter |
| Metrics | validation AUC | The result you compare runs by |
| Artifacts | the fitted model, plots | What you deploy or inspect later |
| Code version | git commit | The same parameters mean nothing under different code |
| Data version | a hash of the training table | Tables get refreshed in place under the same name |
| Environment | Python and library versions | Library changes alter numbers, as [Docker for ML](../docker-for-ml/) shows |

The tracker below keys each run on a hash of everything that goes in, so two runs with the same inputs get the same id. The data is a synthetic churn-style table: five features and a label drawn from a logistic model.

```python
import hashlib
import json
import platform

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def data_hash(X, y):
    return hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()[:12]


W = np.array([1.5, -1.0, 0.5, 0.0, 0.8])                 # true weights of the label model


def make_table(n, seed, w=W, shift=0.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    X[:, 0] += shift
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w - 1.0)))).astype(np.int8)
    return X, y


CODE_COMMIT = "3f9c2e1"                                  # stand-in for `git rev-parse HEAD`


def train_eval(params, X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=params["split_seed"])
    model = LogisticRegression(C=params["C"]).fit(Xtr, ytr)
    return model, roc_auc_score(yte, model.predict_proba(Xte)[:, 1])


runs = {}


def log_run(params, X, y):
    rec = {"params": params, "data": data_hash(X, y), "code": CODE_COMMIT,
           "env": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}}
    rec["run_id"] = digest(rec)                          # id = hash of the inputs, before the result exists
    _, rec["auc"] = train_eval(params, X, y)
    runs[rec["run_id"]] = rec
    return rec


def diff(r1, r2):
    return sorted(k for k in r1 if r1[k] != r2[k])


X, y = make_table(5000, seed=1)
first = log_run({"C": 1.0, "split_seed": 0}, X, y)
again = log_run({"C": 1.0, "split_seed": 0}, *make_table(5000, seed=1))    # rebuilt from the record
assert again["run_id"] == first["run_id"] and again["auc"] == first["auc"]  # same inputs, same metric, bit for bit

refreshed = log_run({"C": 1.0, "split_seed": 0}, *make_table(5000, seed=2))  # same table name, new rows
assert diff(first, refreshed) == ["auc", "data", "run_id"]                 # the data hash names the cause
print(f"AUC on the original table {first['auc']:.3f}, after the refresh {refreshed['auc']:.3f}")
assert round(first["auc"], 3) == 0.858 and round(refreshed["auc"], 3) == 0.874

split_aucs = [train_eval({"C": 1.0, "split_seed": s}, X, y)[1] for s in range(20)]
print(f"AUC range over 20 split seeds: {max(split_aucs) - min(split_aucs):.3f}")
assert 0.035 < max(split_aucs) - min(split_aucs) < 0.045
```

Rebuilding the run from its record gives the same AUC exactly, and a refreshed table moves AUC from 0.858 to 0.874 with identical parameters and code; without the data hash, that gain would look like progress from a change you made. The last assert is the case for logging seeds: on one table and one model, the train/test split seed alone moves AUC by about 0.04, more than many tuning gains.

Hashing `X.tobytes()` works here because the array is in memory. A real pipeline hashes the files or a table snapshot, and treats a missing hash as a failed run rather than a blank field.

## The same thing in MLflow
MLflow Tracking stores runs grouped into experiments. A run holds parameters, metrics, tags and artifacts, and MLflow sets some tags itself: the source file always, and the git commit in `mlflow.source.git.commit` when the code runs from a git repository. A dataset logged with `mlflow.log_input` carries a digest computed from its contents, which plays the role of my data hash. The calls below follow the MLflow 3 documentation; they are not run here, since MLflow is not installed in the build.

```plaintext
import mlflow
from mlflow import MlflowClient

mlflow.set_experiment("churn")
with mlflow.start_run():
    mlflow.log_params({"C": 1.0, "split_seed": 0})
    mlflow.log_input(mlflow.data.from_pandas(train_df, name="churn-2026-10", targets="churned"),
                     context="training")
    model = LogisticRegression(C=1.0).fit(X_train, y_train)
    mlflow.log_metric("val_auc", auc)
    mlflow.sklearn.log_model(model, name="model", registered_model_name="churn")

client = MlflowClient()
client.set_registered_model_alias("churn", "champion", 3)       # serve version 3
model = mlflow.pyfunc.load_model("models:/churn@champion")
client.set_registered_model_alias("churn", "champion", 2)       # roll back: move the pointer
```

One difference from the toy: MLflow does not key a run on its inputs, so two identical runs are two records, and spotting that they match is a query you run over their logged fields.

## What a registry adds
Tracking keeps every run, including the hundreds you will never deploy. A **registry** keeps the few that matter under a name. Each registration under the same name adds a new **version**, numbered from 1 upward. Each version links back to the run that produced it, which is its **lineage**: from the serving model you can walk to its parameters, data hash, commit and environment. An **alias** such as `champion` is a movable pointer to one version, and serving loads the alias, not a file path. MLflow's older fixed **stages** (Staging, Production, Archived) are deprecated since MLflow 2.9 in favour of aliases and tags.

```python
registry = {"versions": [], "aliases": {}}


def register(run):
    registry["versions"].append(run["run_id"])
    return len(registry["versions"])                     # versions count up from 1


def load(alias):
    return runs[registry["versions"][registry["aliases"][alias] - 1]]


v1 = register(first)
registry["aliases"]["champion"] = v1
v2 = register(refreshed)
registry["aliases"]["champion"] = v2                     # promote the model trained on the refreshed table
assert load("champion")["data"] == refreshed["data"]     # lineage: alias -> version -> run -> data hash

registry["aliases"]["champion"] = v1                     # rollback is one pointer move, no retraining
assert load("champion") == first and registry["versions"] == [first["run_id"], refreshed["run_id"]]
```

The rollback works only because version 1 was never overwritten, which is the whole point of numbering versions instead of copying a file over `model_prod.pkl`. In a pipeline, the step that trains a model and the step that moves the alias are separate, and gates run between them; [ML infrastructure](../ml-infrastructure/) covers that path. Orchestrators such as Kubeflow Pipelines run those steps as containers on Kubernetes; the tracker and registry are what the steps write to.

## Measuring drift: PSI and the KS test
Once a model serves, its inputs change. Two common numbers compare a feature's training distribution (the reference) with a recent window of serving data.

The **population stability index** bins the reference into deciles and compares the share of each window in those bins, \(e_i\) for the reference and \(a_i\) for the window:
\[ \text{PSI} = \sum_i (a_i - e_i) \ln \frac{a_i}{e_i} . \]
It is zero for identical shares and grows with the gap. The usual reading, below 0.1 stable and above 0.25 a real shift, is a rule of thumb from credit-scoring practice; no test or theorem stands behind those numbers. The [system design framework](../ml-system-design-framework/) post computes the same index.

The **Kolmogorov–Smirnov test** compares two samples of a continuous variable through their empirical CDFs. Its statistic \(D\) is the largest vertical gap between the two CDFs, and `scipy.stats.ks_2samp` returns \(D\) and a p-value for "both samples come from one distribution".

```python
from scipy.stats import ks_2samp


def psi(ref, cur, bins=10, floor=1e-4):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1]
    e = np.bincount(np.searchsorted(edges, ref), minlength=bins) / len(ref)
    a = np.bincount(np.searchsorted(edges, cur), minlength=bins) / len(cur)
    e, a = np.clip(e, floor, None), np.clip(a, floor, None)   # an empty bin would make ln(0)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(ref, cur, floor=1e-4):
    cats = np.union1d(ref, cur)
    e = np.clip([(ref == c).mean() for c in cats], floor, None)
    a = np.clip([(cur == c).mean() for c in cats], floor, None)
    return float(np.sum((a - e) * np.log(a / e)))


rng = np.random.default_rng(7)
ref = rng.normal(0, 1, 50_000)
shifts = {"no change": rng.normal(0, 1, 50_000), "mean +0.1 sd": rng.normal(0.1, 1, 50_000),
          "mean +0.5 sd": rng.normal(0.5, 1, 50_000), "sd x1.5": rng.normal(0, 1.5, 50_000)}
drift = {name: (psi(ref, cur), ks_2samp(ref, cur).statistic) for name, cur in shifts.items()}
for name, (p, d) in drift.items():
    print(f"{name:>13}: PSI {p:.3f}, KS D {d:.3f}")
assert drift["no change"][0] < 0.001 and round(drift["mean +0.1 sd"][0], 3) == 0.011
assert round(drift["mean +0.5 sd"][0], 2) == 0.25 and round(drift["sd x1.5"][0], 2) == 0.20
assert [round(d, 3) for _, d in drift.values()][:2] == [0.005, 0.045]
assert round(drift["mean +0.5 sd"][1], 2) == 0.20 and round(drift["sd x1.5"][1], 2) == 0.10

cat_ref = rng.choice(5, 50_000, p=[0.4, 0.25, 0.15, 0.1, 0.1])
cat_cur = np.where(rng.random(50_000) < 0.1, 5, rng.choice(5, 50_000, p=[0.4, 0.25, 0.15, 0.1, 0.1]))
new_cat = {fl: psi_categorical(cat_ref, cat_cur, floor=fl) for fl in (1e-4, 1e-3)}
assert round(new_cat[1e-4], 2) == 0.69 and round(new_cat[1e-3], 2) == 0.46
```

| 50,000 rows each | PSI | KS \(D\) |
|---|---|---|
| no change | < 0.001 | 0.005 |
| mean +0.1 sd | 0.011 | 0.045 |
| mean +0.5 sd | 0.25 | 0.20 |
| sd ×1.5 | 0.20 | 0.10 |

A half-standard-deviation shift lands right at the 0.25 line, and a 50% wider spread just below it, so the conventional cut-offs are about this coarse. The last assert is a warning about categories. When serving starts sending a category the reference never had (here 10% of rows), PSI depends on the floor you put under the empty reference share: 0.69 with a floor of 0.0001, 0.46 with 0.001. The number is an artefact of that choice, so check for unseen categories directly and report their share.

## Statistical vs practical significance
A p-value answers "could this gap be sampling noise?", and with enough rows almost any gap cannot. Here the window's mean is off by 0.02 standard deviations, a shift too small to matter for most models. For two unit normals that far apart, the largest CDF gap is \(2\Phi(0.01) - 1 \approx 0.008\), so that is the true \(D\). I draw five pairs of samples at each of four sizes and report the medians:

```python
from scipy.stats import norm

sizes = {}
for n in (1_000, 10_000, 100_000, 1_000_000):
    g = np.random.default_rng(n)
    reps = []
    for _ in range(5):
        r, c = g.normal(0, 1, n), g.normal(0.02, 1, n)
        test = ks_2samp(r, c)
        reps.append((psi(r, c), test.statistic, test.pvalue))
    reps = np.array(reps)
    sizes[n] = (np.median(reps, axis=0), reps)
    print(f"n={n:>9,}: median PSI {sizes[n][0][0]:.4f}, KS D {sizes[n][0][1]:.4f}, p {sizes[n][0][2]:.1e}")
med = {n: v[0] for n, v in sizes.items()}
assert round(med[1_000][2], 2) == 0.54 and round(med[10_000][2], 2) == 0.05   # small samples: noise or borderline
assert med[100_000][2] < 1e-4 and med[1_000_000][2] < 1e-25                    # same shift, "highly significant"
assert all(v[1][:, 0].max() < 0.03 for v in sizes.values())                    # PSI far below 0.1 in every draw
assert abs(med[1_000_000][1] - (2 * norm.cdf(0.01) - 1)) < 0.001               # D settles at the true 0.008
assert [round(med[n][1], 3) for n in sizes] == [0.036, 0.019, 0.011, 0.008]
assert round(med[1_000][0], 3) == 0.012 and round(med[10_000][0], 3) == 0.002
assert med[100_000][0] < 0.001 and med[1_000_000][0] < 0.001
```

| rows per sample | median PSI | median KS \(D\) | median KS p-value |
|---|---|---|---|
| 1,000 | 0.012 | 0.036 | 0.54 |
| 10,000 | 0.002 | 0.019 | 0.05 |
| 100,000 | < 0.001 | 0.011 | < 0.0001 |
| 1,000,000 | < 0.001 | 0.008 | < 1e-25 |

The shift is the same in every row. \(D\) settles at its true value of 0.008 while the median p-value falls from 0.54 to below 1e-25; the PSI at 1,000 rows is mostly sampling noise, as the next section shows. A serving log with a million rows a day will flag every feature this way every day. Read the effect size (\(D\) or PSI) and alert on that; use the p-value at most to ask whether a small window is too noisy to read. [p-values](../p-values/) and [A/B testing](../ab-testing/) make the same point for experiments.

## How often a fixed PSI threshold cries wolf
The opposite problem hits small windows. With no drift at all, sampling noise alone gives PSI a positive value. For small gaps \((a-e)\ln(a/e) \approx (a-e)^2/e\), which makes PSI a scaled chi-square statistic with mean about \((B-1)(1/n + 1/m)\) for \(B\) bins, \(n\) window rows and \(m\) reference rows. I draw 2,000 windows from the reference distribution itself at four window sizes:

```python
g = np.random.default_rng(42)
ref10k = g.normal(0, 1, 10_000)
null = {m: np.array([psi(ref10k, g.normal(0, 1, m)) for _ in range(2000)]) for m in (100, 200, 500, 5000)}
for m, vals in null.items():
    print(f"window {m:>5}: mean PSI {vals.mean():.4f} (formula {9 * (1 / m + 1 / 10_000):.4f}), "
          f"P(PSI > 0.1) {(vals > 0.1).mean():.3f}")
assert all(abs(v.mean() / (9 * (1 / m + 1 / 10_000)) - 1) < 0.1 for m, v in null.items())
assert (null[100] > 0.1).sum() == 785 and (null[200] > 0.1).sum() == 51     # 39% and 2.6% of 2,000
assert [round(v.mean(), 3) for v in null.values()] == [0.096, 0.047, 0.019, 0.003]
assert (null[500] > 0.1).mean() == 0 and (null[5000] > 0.1).mean() == 0
assert round(24 * (null[100] > 0.1).mean()) == 9                             # false alarms a day, hourly checks
```

| window rows | mean PSI with no drift | share of windows over 0.1 |
|---|---|---|
| 100 | 0.096 | 39% |
| 200 | 0.047 | 2.6% |
| 500 | 0.019 | 0% |
| 5,000 | 0.003 | 0% |

At 100 rows per window, two in five perfectly healthy windows cross 0.1, so hourly checks on a low-traffic feature would raise about 9 false alarms a day. The formula matches the simulated means within 10%, which gives a quick fix: pick the window so the noise floor sits well below the threshold, or set the threshold from the null distribution at the window size you actually have.

## Label delay: watch inputs and scores first
The metric you care about, such as AUC, needs labels, and labels often arrive late: whether a customer churned is known a month later, whether a loan defaults much later still. Inputs and the model's own scores are available the moment a request is served, so they are what you can watch now. The next simulation shows both what they catch and what they miss. The model from the first section scores a reference window and three serving windows: no change, a covariate shift (feature 0 moves by one standard deviation, the label rule stays) and concept drift (the inputs stay, but feature 0's true weight goes from 1.5 to −0.5).

```python
model, _ = train_eval({"C": 1.0, "split_seed": 0}, X, y)
X_ref, y_ref = make_table(20_000, seed=10)
s_ref = model.predict_proba(X_ref)[:, 1]
W_new = W.copy()
W_new[0] = -0.5
cases = {"no change": make_table(20_000, seed=11), "covariate shift": make_table(20_000, seed=12, shift=1.0),
         "concept drift": make_table(20_000, seed=13, w=W_new)}
watch = {}
for name, (Xc, yc) in cases.items():
    s = model.predict_proba(Xc)[:, 1]
    watch[name] = (max(psi(X_ref[:, j], Xc[:, j]) for j in range(5)), psi(s_ref, s), roc_auc_score(yc, s))
    print(f"{name:>15}: max feature PSI {watch[name][0]:.3f}, score PSI {watch[name][1]:.3f}, AUC {watch[name][2]:.3f}")
auc_ref = roc_auc_score(y_ref, s_ref)
assert round(auc_ref, 3) == 0.865
assert watch["no change"][0] < 0.01 and watch["no change"][1] < 0.01 and round(watch["no change"][2], 3) == 0.871
assert round(watch["covariate shift"][0], 2) == 0.95 and round(watch["covariate shift"][1], 2) == 0.50
assert round(watch["covariate shift"][2], 3) == 0.864                    # loud alarm, model still fine
assert watch["concept drift"][0] < 0.01 and watch["concept drift"][1] < 0.01
assert round(watch["concept drift"][2], 2) == 0.63                       # silent inputs, broken model
```

| serving window | max feature PSI | score PSI | AUC once labels arrive |
|---|---|---|---|
| reference | | | 0.865 |
| no change | < 0.01 | < 0.01 | 0.871 |
| covariate shift | 0.95 | 0.50 | 0.864 |
| concept drift | < 0.01 | < 0.01 | 0.63 |

Input and score drift alarm loudly on a shift that costs no ranking quality, and stay silent on the change that breaks the model. So drift on inputs and scores is an early warning that something upstream changed, often a broken pipeline or a new user segment, and a reason to look; only labelled metrics, when they land, say whether the model got worse. Monitoring needs both, on two clocks.

## Pitfalls and fixes

| Problem | Naive fix | How it breaks | Real fix | How it is measured |
|---|---|---|---|---|
| Last month's best model cannot be rebuilt | Keep the notebook and a dated file name | The table was refreshed in place and the seed was never saved | Log parameters with all seeds, a data hash, the commit and the environment for every run | A rerun from the record gives the same metric; the split seed alone moved AUC by 0.04 |
| Nobody knows which model serves | Copy the file to `model_prod.pkl` | The previous model is overwritten; there is no path back to its run | Registry versions, an alias for serving, lineage to the run | The alias resolves to a run and data hash; rollback is one pointer move |
| Drift tests on a large serving log | Alert when a KS p-value is below 0.05 | A 0.02-sd shift gives p below 1e-25 at a million rows | Alert on an effect size (PSI or KS \(D\)) | PSI under 0.03 for that shift in every draw, true \(D\) 0.008 |
| Drift checks on small windows | A fixed PSI threshold of 0.1 | 39% of healthy 100-row windows cross it | Larger windows, or a threshold from the null at that window size | Mean null PSI ≈ \((B-1)(1/n+1/m)\), checked within 10% |
| Labels arrive weeks late | Watch AUC only | Problems surface a month after they start | Watch inputs and scores now, labelled metrics when they land | Covariate shift: PSI 0.95, AUC 0.864 vs 0.865; concept drift: PSI < 0.01, AUC 0.63 |

## What this does not show
Everything here runs on synthetic data from a logistic model, the same family as the fitted model, which is why a covariate shift costs it nothing. A misspecified model, the usual case, can lose accuracy under covariate shift too, so the "loud alarm, model fine" row is one end of a range, not a promise. The tracker hashes in-memory arrays and uses a fake commit string; it shows what to record, not how to store it for a team. The PSI results depend on decile bins from the reference and on the floor under empty bins, and other binning choices give other numbers. I check one feature at a time; with many features and many checks a day, false alarms add up, which this post does not model.

What stays open is choosing thresholds from cost rather than convention: how much a missed drift costs against an hour spent on a false alarm, and how that trade changes with each feature's traffic. Detecting concept drift before labels arrive, from proxy signals such as early user behaviour, is the harder open problem.

The MLflow details (tracked fields, system tags, dataset digests, registry versions, aliases and the deprecation of stages in 2.9) follow the [MLflow documentation](https://mlflow.org/docs/latest/ml/); the KS test is `scipy.stats.ks_2samp` as documented in SciPy.
