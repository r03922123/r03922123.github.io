# Terraform and CI/CD for ML: plans, shared state and a model-quality gate

Two engineers run `apply` against the same state file without a lock, and in 4 of the 6 possible orderings of their reads and writes, one of them erases the other's record of an endpoint that still exists and still costs money. A CI gate that blocks a new model whenever its accuracy drops at all blocks about half of retrains that are exactly as good as the old one. Both failures come from the plumbing around the code, and both have standard fixes. **Infrastructure as code** (IaC) means describing servers, buckets and endpoints in text files that are reviewed and versioned like code; Terraform is the most common tool for it. **CI/CD** (continuous integration and continuous delivery) means every change runs automated checks before it merges, and a change that passes is shipped through staged environments by a pipeline rather than by hand.

This post explains how a Terraform plan is computed and why the state file has to be shared and locked, then what a pipeline should check for ML code and how a model reaches production. I build a tiny plan engine, replay the two-writer race over every ordering, and test three model-quality gates on simulated retrains. The container side is in [Docker for ML](../docker-for-ml/) and the systems a model runs on are in [ML infrastructure](../ml-infrastructure/).

## Describe the end state, not the steps
A shell script that creates a bucket is imperative: run it twice and it fails, or makes two buckets. Terraform is **declarative**: the files say what should exist, and Terraform works out the steps. The language is HCL; a sketch (not run):

```plaintext
terraform {
  backend "s3" {
    bucket       = "example-tf-state"
    key          = "serving/dev.tfstate"
    region       = "eu-west-1"
    use_lockfile = true          # S3-native state locking; off by default
  }
}

resource "aws_s3_bucket" "models" {
  bucket = "example-ml-models-dev"
  lifecycle {
    prevent_destroy = true       # reject any plan that would delete the model store
  }
}
```

The workflow is two commands. `terraform plan` computes and prints the changes without making them; `terraform apply` makes them. Saving the plan (`plan -out=tfplan`, then `apply tfplan`) applies exactly what a reviewer read.

## A plan compares three versions of the truth
Terraform's documentation describes `plan` in three steps: read the current state of every existing remote object so the recorded state is up to date, compare the configuration with that state, and propose the actions that would make the remote objects match the configuration. So three things are in play: the **desired** config in the files, the **state** (Terraform's record of what it created and the IDs that tie each resource to a real object), and the **actual** world. When the actual world differs from the state because someone changed it outside Terraform, that is **drift**.

Each resource gets one action. Some attributes can change in place; others, such as the machine image of a virtual machine, cannot, so changing them forces a **replace**: destroy and create (Terraform's default order; `create_before_destroy` reverses it). The engine below follows those rules on dicts.

```python
import copy


def plan(desired, state, actual, force_new, replace=(), protect=(), ignore=None):
    """Return (actions, drift). Each dict maps a resource address to its attributes."""
    ignore = ignore or {}
    refreshed, drift = {}, {}
    for addr, recorded in state.items():                     # refresh: read every object the state knows about
        if addr not in actual:
            drift[addr] = "deleted outside"
            continue
        refreshed[addr] = dict(actual[addr])
        changed = sorted(k for k in recorded if actual[addr].get(k) != recorded[k])
        if changed:
            drift[addr] = changed
    actions = {}
    for addr in sorted(desired.keys() | refreshed.keys()):
        if addr not in refreshed:
            actions[addr] = "create"
        elif addr not in desired:
            actions[addr] = "delete"
        else:
            changed = {k for k in desired[addr] if refreshed[addr].get(k) != desired[addr][k]}
            changed -= ignore.get(addr, set())                # lifecycle ignore_changes
            if addr in replace or changed & force_new.get(addr.split(".")[0], set()):
                actions[addr] = "replace"
            elif changed:
                actions[addr] = "update"
    for addr, act in actions.items():
        if addr in protect and act in ("replace", "delete"):  # lifecycle prevent_destroy
            raise RuntimeError(f"{addr} has prevent_destroy; plan rejected")
    return actions, drift


def summary(actions):
    acts = list(actions.values())
    add = acts.count("create") + acts.count("replace")       # a replace counts as one add and one destroy
    destroy = acts.count("delete") + acts.count("replace")
    return f"Plan: {add} to add, {acts.count('update')} to change, {destroy} to destroy."


def apply(actions, desired, state, actual):
    for addr, act in actions.items():
        if act == "delete":
            del actual[addr], state[addr]
        else:                                                # create, update and replace all end at the config
            actual[addr] = state[addr] = dict(desired[addr])


FORCE_NEW = {"instance": {"image"}, "bucket": {"name"}}      # attributes that cannot change in place
desired = {"bucket.models": {"name": "ml-models-dev", "versioning": True},
           "instance.trainer": {"size": "gpu.large", "image": "train-v2"},
           "endpoint.scorer": {"replicas": 2, "model": "ranker-v41"},
           "alarm.latency": {"p99_ms": 150}}
state = {"bucket.models": {"name": "ml-models-dev", "versioning": True},
         "instance.trainer": {"size": "gpu.large", "image": "train-v1"},
         "endpoint.scorer": {"replicas": 2, "model": "ranker-v40"},
         "queue.old": {"retention_h": 24}}
actual = copy.deepcopy(state)
actual["endpoint.scorer"]["replicas"] = 4                    # scaled up by hand in a web console
actual["bucket.scratch"] = {"name": "tmp-notebooks"}         # made by hand, never imported

actions, drift = plan(desired, state, actual, FORCE_NEW)
for addr, act in actions.items():
    print(f"{act:>8}  {addr}")
print("drift:", drift, "|", summary(actions))
assert actions == {"alarm.latency": "create", "endpoint.scorer": "update",
                   "instance.trainer": "replace", "queue.old": "delete"}
assert drift == {"endpoint.scorer": ["replicas"]}
assert summary(actions) == "Plan: 2 to add, 1 to change, 2 to destroy."
assert "bucket.scratch" not in actions                       # not in state, so invisible to the plan

apply(actions, desired, state, actual)
assert plan(desired, state, actual, FORCE_NEW) == ({}, {})   # a second plan is empty: apply converges
assert actual["bucket.scratch"] == {"name": "tmp-notebooks"}
```

Four resources change for four reasons: the alarm is new, the image change forces a replace, the queue left the config, and the endpoint is updated both for its new model and to undo the hand-made scale-up, which the refresh reported as drift. The plan reads "2 to add, 1 to change, 2 to destroy" because a replace counts once on each side. The bucket made by hand is never mentioned: Terraform only knows objects that are in its state, so an unmanaged object is neither tracked nor cleaned up until someone imports it. And once applied, a second plan is empty, which is what declarative buys: running it again is safe.

Three controls change what a plan may do, and the engine models each:

```python
assert plan(desired, state, actual, FORCE_NEW, replace={"instance.trainer"})[0] == {"instance.trainer": "replace"}

actual["endpoint.scorer"]["replicas"] = 6                    # an autoscaler, not a person, changed it
assert plan(desired, state, actual, FORCE_NEW)[0] == {"endpoint.scorer": "update"}
assert plan(desired, state, actual, FORCE_NEW, ignore={"endpoint.scorer": {"replicas"}})[0] == {}

renamed = {**desired, "bucket.models": {"name": "ml-models-v2", "versioning": True}}
try:
    plan(renamed, state, actual, FORCE_NEW, protect={"bucket.models"})
    raise AssertionError("renaming a protected bucket should be rejected")
except RuntimeError:
    pass
```

`terraform plan -replace=ADDRESS` forces a replace with no config change, for an object that has gone bad. `ignore_changes` stops planned updates to listed attributes, which is right when something else, such as an autoscaler, owns them; without it every plan fights the autoscaler. `prevent_destroy` turns a plan that would delete the resource into an error, which matters for the bucket of model artifacts, since renaming it forces a replace. One more mode exists for drift you meant to keep: refresh-only mode (`terraform apply -refresh-only`) updates the state to match the world and changes nothing else.

## The state file: shared, locked, and sensitive
The state exists because the config alone cannot say which real object `endpoint.scorer` is. Terraform's documentation gives three uses: mapping config to real resources, keeping metadata, and speed on large setups. Kept on one laptop, it blocks teamwork, so teams store it in a **remote backend** (an object store, or HCP Terraform). Shared state brings a new failure. Each `apply` reads the state, changes the world, and writes the whole state back. Below, two jobs each add one endpoint, and I run every ordering of their read and write steps, with and without a lock.

```python
from itertools import permutations


def run(schedule, use_lock):
    """Jobs A and B each add an endpoint. Each letter is that job's next step: read state and apply, then write."""
    stored = {"bucket.models": {"name": "ml-models-dev"}}    # the shared state file
    world = dict(stored)                                     # what really exists
    local, holder, blocked, queue = {}, None, 0, list(schedule)
    while queue:
        job = queue.pop(0)
        if job not in local:                                 # step 1: read state, plan, apply
            if use_lock and holder is not None:
                blocked += 1                                 # lock held: this run stops; rerun it afterwards
                queue.remove(job)
                queue += [job, job]
                continue
            holder, local[job] = job, dict(stored)
            new = f"endpoint.{job.lower()}"
            world[new] = local[job][new] = {"replicas": 1}
        else:                                                # step 2: write the whole state file back
            stored, holder = local.pop(job), None
    return stored, world, blocked


schedules = sorted(set(permutations("AABB")))
outcomes = {}
for use_lock in (False, True):
    runs = [run(s, use_lock) for s in schedules]
    lost = ["".join(s) for s, (stored, world, _) in zip(schedules, runs) if stored.keys() != world.keys()]
    outcomes[use_lock] = (lost, sum(b > 0 for *_, b in runs))
    print(f"lock={use_lock}: {len(lost)} of {len(schedules)} orderings lose a record {lost}; "
          f"{outcomes[use_lock][1]} runs stopped by the lock")
assert len(schedules) == 6
assert len(outcomes[False][0]) == 4 and outcomes[False][1] == 0
assert outcomes[True] == ([], 4)

stored, world, _ = run("ABBA", use_lock=False)               # A reads, B reads, B writes, A writes last
both = {"bucket.models": {"name": "ml-models-dev"}, "endpoint.a": {"replicas": 1}, "endpoint.b": {"replicas": 1}}
assert "endpoint.b" in world and "endpoint.b" not in stored
assert plan(both, stored, world, FORCE_NEW)[0] == {"endpoint.b": "create"}   # tries to create what already exists
```

Without a lock, every ordering where the two runs overlap loses an update: the last writer's copy, read before the other's write, silently drops the other's resource. The last assert shows the cost. The next plan wants to create `endpoint.b` again, since an object missing from the state is invisible to the refresh, so the apply either fails on a name clash or makes a duplicate, and the original runs on, unmanaged. With the lock, no ordering loses a record, and the 4 overlapping orderings end with one run stopped instead. That matches the documented behaviour: Terraform locks the state for every operation that could write it, and if it cannot take the lock it does not continue. `terraform force-unlock` exists for a lock left behind by a crashed run, and the documentation's advice is to use it only on your own lock.

Not every backend locks. The S3 backend's locking is opt-in (`use_lockfile = true`; its older DynamoDB-table locking is deprecated), so a team can share state for months without any lock and never notice until two applies overlap. The state can also hold secrets, such as a generated database password, so the backend needs access control and encryption, and the state file never goes into Git.

## Modules and variables: one definition, three environments
A **module** is a set of resources managed together; the directory you run Terraform in is the root module, and it calls child modules with `module` blocks. The usual layout writes the serving stack once and calls it per environment with different **variables**, each environment with its own state, so a plan for dev cannot touch prod:

```plaintext
# modules/serving/variables.tf
variable "env"            { type = string }
variable "instance_size"  { type = string }
variable "min_replicas"   { type = number }

# envs/prod/main.tf  (envs/dev and envs/staging differ only in these values and their backend key)
module "serving" {
  source        = "../../modules/serving"
  env           = "prod"
  instance_size = "gpu.large"
  min_replicas  = 3
}
```

The point is that staging differs from prod only in values someone chose, not in structure someone forgot. A model that passed staging then meets the same network rules, alarms and IAM roles in prod.

## What a pipeline checks for ML code
Unit tests catch a broken function. An ML change can also break the data, the model or the build while every function still works:

| Check | What it catches | Runs on |
|---|---|---|
| Unit tests | wrong logic in feature code or the serving handler | every PR |
| Data validation | schema changes, nulls, values out of range, a label rate far from the reference | every PR, and each new training window |
| Model-quality gate | a candidate worse than the live model on a fixed eval set | every PR that changes training code or data |
| Reproducible build | an image that differs from what was tested: lockfile installs, a pinned base image | every build |
| `terraform plan` | infrastructure changes the reviewer should read before they happen | every PR touching `.tf` files |

The gate is the hard one, because a model's score moves when only its random seed changes.

## A model-quality gate that does not cry wolf
I simulate a fixed eval set of 2,000 items, each with a difficulty shared by every model, and a trained model as right or wrong on each item, with seed-to-seed noise deciding the borderline ones. Each trial trains a baseline and a candidate and asks three gates whether to block. The naive gate blocks any drop. The unpaired gate runs a one-sided two-proportion z-test as if the two models had seen different items. The paired gate runs McNemar's test, which looks only at the items where the two models disagree and asks whether the candidate loses more of them than a coin flip would.

```python
import numpy as np
from scipy.stats import binomtest, norm

rng = np.random.default_rng(7)
n_items, trials = 2_000, 1_000
difficulty = rng.normal(0, 1.5, n_items)                    # the fixed eval set: some items are hard for any model


def correct(skill):
    """One trained model's right/wrong per item; the noise term is what a new seed changes."""
    return skill - difficulty + rng.normal(0, 0.5, n_items) > 0


def gates(base, cand):
    drop = base.mean() - cand.mean()
    worse, better = int(np.sum(base & ~cand)), int(np.sum(~base & cand))     # the items they disagree on
    pooled = (base.mean() + cand.mean()) / 2
    z = drop / np.sqrt(2 * pooled * (1 - pooled) / n_items)
    return {"any drop": drop > 0,
            "unpaired z-test": norm.sf(z) < 0.05,
            "paired McNemar": binomtest(worse, worse + better, 0.5, alternative="greater").pvalue < 0.05}


rates, mean_drop = {}, {}
for label, skill_loss in [("same quality", 0.0), ("small drop", 0.025), ("large drop", 0.1)]:
    runs = [(b, c) for b, c in ((correct(1.0), correct(1.0 - skill_loss)) for _ in range(trials))]
    mean_drop[label] = 100 * np.mean([b.mean() - c.mean() for b, c in runs])
    votes = [gates(b, c) for b, c in runs]
    rates[label] = {g: np.mean([v[g] for v in votes]) for g in votes[0]}
    print(f"{label:>12} (drop {mean_drop[label]:.2f} pts): " + ", ".join(f"{g} {r:.0%}" for g, r in rates[label].items()))

b, c = correct(1.0), correct(1.0)
se_paired = np.std(b.astype(float) - c, ddof=1) / np.sqrt(n_items)
se_unpaired = np.sqrt((b.var() + c.var()) / n_items)
print(f"accuracy {b.mean():.3f}, models disagree on {np.mean(b != c):.0%} of items, "
      f"standard error of the difference: paired {se_paired:.4f}, unpaired {se_unpaired:.4f}")

same, small, large = rates["same quality"], rates["small drop"], rates["large drop"]
assert 1.8 < mean_drop["large drop"] < 2.2 and 0.4 < mean_drop["small drop"] < 0.6
assert 0.45 < same["any drop"] < 0.55 and small["any drop"] > 0.7 and large["any drop"] > 0.95
assert same["unpaired z-test"] < 0.01 and small["unpaired z-test"] < 0.01 and 0.25 < large["unpaired z-test"] < 0.45
assert 0.03 < same["paired McNemar"] < 0.08 and small["paired McNemar"] < 0.2 and large["paired McNemar"] > 0.75
assert 0.08 < np.mean(b != c) < 0.12 and 0.45 < se_paired / se_unpaired < 0.6
```

| Gate | blocks a same-quality retrain | blocks a 0.5-point drop | blocks a 2-point drop |
|---|---|---|---|
| any drop | about half (49%) | about three in four (76%) | almost always (99%) |
| unpaired z-test | almost never (0%) | almost never (0%) | about a third (36%) |
| paired McNemar | about 1 in 17 (6%) | about 1 in 7 (15%) | about 4 in 5 (83%) |

The paired test is the only one that both lets equal retrains through and catches the large drop; the naive gate fails every second harmless retrain, which teaches a team to rerun or override it. Pairing wins because two models agree on most items, so the noise in their difference comes only from the items they disagree on, about one in ten here, and its standard error is about half the unpaired one. No gate catches half a point reliably on 2,000 items: the paired one blocks such a model about once in seven tries. To guard smaller drops, enlarge the eval set or accept a gate that also blocks some equal models; the same power trade-off appears in [A/B testing](../ab-testing/). The eval set must stay fixed across runs and should be pinned by a hash, or the gate compares two models on different questions.

## Shipping it: stages and a canary
CD moves an artifact that already passed CI through environments: build the image once, deploy it to staging, then promote the same image, by digest, to prod. A sketch of the jobs in GitHub Actions (not run):

```plaintext
on:
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4            # pin a full commit SHA in practice
      - run: uv sync --locked && uv run pytest
  eval-gate:
    needs: test
    runs-on: ubuntu-24.04
    steps:
      - run: uv run python -m train.gate --eval-set eval/fixed.parquet --baseline registry:prod
  deploy-prod:
    needs: eval-gate
    if: github.event_name == 'push'          # a separate workflow on push to main, in practice
    environment: production                  # waits for the environment's protection rules
    runs-on: ubuntu-24.04
    steps:
      - run: ./deploy.sh --canary
```

A job that names an **environment** must pass that environment's protection rules before it runs, such as required reviewers, a wait timer or a list of branches allowed to deploy, and it can read the environment's secrets only after those rules pass. That puts a human approval and the production credentials behind the same door.

A **canary** sends a small share of live traffic to the new model while the rest stays on the old one, compares error rate, latency and the business metric between the two, and then either widens the share step by step or rolls back. It is an online test with a decision rule, so the statistics of [A/B testing](../ab-testing/) apply: a small share means few requests, so it catches crashes and latency blow-ups quickly but a small quality drop slowly. Rollback is cheap only if the old model, its image and its feature definitions are still deployable.

## Problems and fixes

| Problem | Naive fix | How it breaks | Real fix | How it is measured |
|---|---|---|---|---|
| Infra changed by hand | a wiki page of console steps | staging and prod drift apart; nobody can rebuild either | Terraform config, applied only from a reviewed plan | a scheduled `plan` that should be empty |
| Two people apply at once | "ask in chat first" | an overlapping run erases a record; the next plan duplicates the object | remote backend with locking on | lock errors in CI logs; the race above loses nothing |
| Autoscaler fights Terraform | turn off the autoscaler | traffic spikes overload the service | `ignore_changes` on the replica count | plans stop proposing replica updates |
| Flaky model gate | block any accuracy drop | about half of equal retrains are blocked, so people override it | paired test on a fixed eval set | false-block rate on seed-only retrains |
| Risky model release | deploy to everyone at once | a bad model hits every user before anyone sees it | canary with automatic rollback | canary vs control on errors, latency and the business metric |

## A small, real pipeline: this site
This site runs one. Its workflow, `.github/workflows/gates.yml`, runs on every pull request to `main` and every push to `main`, with read-only repository permissions, and its single job runs `scripts/gates.sh`, the same script run locally before a push, so the two cannot drift apart. A pull request to `main` must pass that `gates` check before it merges. Three choices in it match the reproducible-build row above. Every action is pinned to a full commit SHA, since a moved tag would run new code. Every Python package is installed with `pip install --require-hashes --only-binary :all:`, so each wheel must match a recorded hash. And its final steps rebuild every generated page, then fail if the working tree differs from the commit or holds new files, so the committed HTML is exactly what the sources produce. A repository secret exists and is passed to one step only. The post build in that script runs every Python block in every post, so the asserts on this page are themselves a CI gate.

## Pitfalls
- **State in Git or on a laptop.** It goes stale the moment someone else applies, and it can hold secrets. Use a remote backend with locking and access control.
- **Applying an unsaved plan.** The world can change between review and apply. Apply the saved plan file that was reviewed.
- **Hand fixes in the console.** The next plan reverts them, or they linger as drift nobody reviewed. Change the config, or `import` the object.
- **Renaming a resource in config.** Terraform sees one deletion and one creation. A `moved` block, or `terraform state mv`, keeps the real object.
- **An eval set that changes between runs.** The gate then compares two models on different questions. Pin it.
- **Rebuilding the image per environment.** Staging tested a different artifact from the one prod runs. Promote one image by digest.

The simulations model only the logic: the plan engine knows four resource kinds and a hand-written list of attributes that force a replace, where real providers define those per attribute; the race has two jobs with one read and one write each; and the gate's models are drawn from a model of item difficulty I chose, in which the models agree on most items. Real models may agree less, which weakens the paired test's advantage, and the false-block and catch rates depend on the eval set's size and how far apart the models are. None of this ran against a cloud account or a real deployment.

What stays open is choosing the gate's margin on a real system: how large a drop matters to users, and how large an eval set it takes to see it, are product and data questions that the code above can only frame.

Terraform behaviour (the plan steps, `-replace`, `-refresh-only`, `lifecycle`, state and locking, the S3 backend) follows the [Terraform documentation](https://developer.hashicorp.com/terraform/docs); environment protection rules follow the [GitHub Actions documentation on deployment environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments). McNemar's test is from Q. McNemar, Psychometrika (1947).
