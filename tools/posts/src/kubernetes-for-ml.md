---
title: Kubernetes for ML: requests, probes and autoscaling for model servers
summary: What Kubernetes does with a model server's resource requests and limits, why a model pod gets OOMKilled or throttled, how the scheduler packs pods and GPUs, why a new pod can receive traffic before its weights are loaded, and how the Horizontal Pod Autoscaler picks a replica count. Small simulations give each rule a number.
group: ML systems and design
order: 6
tags: Kubernetes · deployment · autoscaling · serving · GPUs
---
Roll out a new version of a model server whose weights take 60 seconds to load, with no readiness probe, and Kubernetes declares all four replicas replaced after 6 seconds while none of them can answer yet: in the simulation below, 57 seconds of full outage and about 3,000 failed requests. Add a readiness probe and the same rollout fails no requests and never drops below three working pods. **Kubernetes** is a system that keeps a declared set of containers running on a pool of machines: you write down what should exist (four copies of this image, each with this much CPU and memory), and controllers keep adjusting the cluster until it matches.

Most of what goes wrong with ML workloads on Kubernetes comes from a few numbers it takes literally: the resources a container asks for, the moment a container says it is ready, and the metric the autoscaler divides by. This post simulates each one, so that the reasons a model pod is OOMKilled, gets traffic before its model is loaded, or flaps between replica counts can be explained with numbers. Building the image itself is covered in [Docker for ML](../docker-for-ml/); the wider serving and training stack is in [ML infrastructure](../ml-infrastructure/).

## Pod, Deployment, Service
A **Pod** is the smallest thing Kubernetes runs: one or more containers that share a network address and are scheduled onto one machine (a **node**) together. Pods are disposable. A **Deployment** declares how many identical pods should exist and replaces them, a few at a time, when the template changes. A **Service** gives those pods one stable address and spreads requests over the pods that are currently *ready*. A minimal model server:

```plaintext
apiVersion: apps/v1
kind: Deployment
metadata: {name: scorer}
spec:
  replicas: 4
  selector: {matchLabels: {app: scorer}}
  template:
    metadata: {labels: {app: scorer}}
    spec:
      containers:
      - name: scorer
        image: registry.example.com/scorer:0.3.1
        resources:
          requests: {cpu: "2", memory: 5Gi}
          limits:   {cpu: "2", memory: 5Gi}
        startupProbe:   {httpGet: {path: /ready, port: 8000}, periodSeconds: 10, failureThreshold: 30}
        readinessProbe: {httpGet: {path: /ready, port: 8000}, periodSeconds: 5}
        livenessProbe:  {httpGet: {path: /live,  port: 8000}, periodSeconds: 10, failureThreshold: 3}
---
apiVersion: v1
kind: Service
metadata: {name: scorer}
spec:
  selector: {app: scorer}
  ports: [{port: 80, targetPort: 8000}]
```

Every field in `resources` and in the three probes changes behaviour, and the sections below take them in turn.

## Requests and limits
A **request** is what the scheduler reserves for the container when it picks a node. A **limit** is a ceiling the Linux kernel enforces while the container runs. The two resources behave differently at the limit:

| | Request | Limit |
|---|---|---|
| Used by | the scheduler, to place the pod | the kernel, through cgroups |
| CPU | a share of CPU time is reserved | **throttled**: the container waits for the next accounting period |
| Memory | memory is reserved | **killed** (OOMKilled, exit code 137) once usage exceeds it and the kernel needs the memory back |

CPU is compressible, so going over costs latency. Memory is not, so going over costs the process. A pod whose requests equal its limits for every container is in the `Guaranteed` quality-of-service class, the last to be evicted when a node runs short.

### Why a model pod gets OOMKilled
The first trap fires before the server answers anything. The usual loading code builds the model, which allocates freshly initialised weights, then copies a loaded state dict into it, so for a moment two full copies of the weights exist. I count the distinct tensor storages alive at that moment, then repeat with the model built on PyTorch's `meta` device (shapes only, no memory) and loaded with `assign=True`, which keeps the loaded tensors instead of copying them:

```python
import heapq
import io
import math

import numpy as np
import torch

GiB = 2**30


def build():
    return torch.nn.Sequential(*[torch.nn.Linear(512, 512) for _ in range(6)])


def resident(*groups):
    """Bytes held by the distinct real (non-meta) storages behind these tensors."""
    seen = {}
    for group in groups:
        for t in group:
            if t.device.type != "meta":
                seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    return sum(seen.values())


disk = io.BytesIO()
torch.save(build().state_dict(), disk)
weight_bytes = sum(p.numel() * p.element_size() for p in build().parameters())

disk.seek(0)
sd = torch.load(disk)
naive = build()                                  # random weights allocated here...
naive.load_state_dict(sd)                        # ...then the loaded ones copied in
peak_naive = resident(naive.state_dict().values(), sd.values())

disk.seek(0)
sd = torch.load(disk)
with torch.device("meta"):
    lean = build()                               # no weights allocated
lean.load_state_dict(sd, assign=True)            # the module takes the loaded tensors as they are
peak_lean = resident(lean.state_dict().values(), sd.values())

x = torch.randn(3, 512)
assert torch.equal(naive(x), lean(x))
assert peak_naive == 2 * weight_bytes and peak_lean == weight_bytes

w_gib = 1.5e9 * 2 / GiB                          # a 1.5B-parameter model in bfloat16
limit = 5.0                                      # GiB, the memory limit in the manifest
print(f"weights {w_gib:.2f} GiB: naive load peaks at {2 * w_gib:.2f} GiB, lean load at {w_gib:.2f} GiB")
assert round(w_gib, 2) == 2.79 and round(2 * w_gib, 2) == 5.59 and w_gib < limit < 2 * w_gib
```

Both models compute the same outputs, but the naive load holds twice the weight bytes at its peak. For a 1.5B-parameter model in bfloat16 that is 5.59 GiB against 2.79 GiB, so under a 5 GiB limit the pod is killed during every start, restarts, and is killed again, which Kubernetes reports as `CrashLoopBackOff`.

The second trap fires under load. Each request in flight holds activations and buffers, so memory follows concurrency, which is random. Below, 20 requests a second arrive at random, each takes about 0.25 s and holds 0.15 GiB, on top of the weights and 0.5 GiB of runtime overhead. Then the same traffic goes through a server that admits at most as many concurrent requests as the limit has room for and queues the rest.

```python
rng = np.random.default_rng(0)
rate, hour = 20.0, 3600.0
arrive = np.cumsum(rng.exponential(1 / rate, int(rate * hour * 1.1)))
arrive = arrive[arrive < hour]
service = rng.lognormal(np.log(0.25) - 0.125, 0.5, arrive.size)    # mean 0.25 s
base, per_req = w_gib + 0.5, 0.15                                  # GiB


def in_flight(start, end):
    """Event times and the number of requests in flight just after each."""
    t = np.concatenate([start, end])
    d = np.concatenate([np.ones(start.size), -np.ones(end.size)])
    order = np.lexsort((d, t))
    return t[order], np.cumsum(d[order])


t, n = in_flight(arrive, arrive + service)
mem = base + per_req * n
dt = np.diff(t, append=hour)
mean_share = (mem * dt).sum() / hour / limit
over_share = (dt * (mem > limit)).sum() / hour
crossings = int(((mem[1:] > limit) & (mem[:-1] <= limit)).sum())

cap = int((limit - base) // per_req)             # concurrent requests the limit has room for
free_at = [0.0] * cap
start = np.empty_like(arrive)
for i, a in enumerate(arrive):                   # first come, first served over `cap` slots
    start[i] = max(a, heapq.heappop(free_at))
    heapq.heappush(free_at, start[i] + service[i])
_, n_cap = in_flight(start, start + service)
waited = np.mean(start > arrive)
p99_uncapped = np.percentile(service, 99)
p99_capped = np.percentile(start - arrive + service, 99)
print(f"uncapped: mean {mean_share:.0%} of the limit, over it {over_share:.2%} of the time, {crossings} crossings/hour")
print(f"capped at {cap}: peak {base + per_req * n_cap.max():.2f} GiB, {waited:.1%} of requests wait, "
      f"p99 {p99_capped:.3f} s vs {p99_uncapped:.3f} s")
assert 0.78 < mean_share < 0.84 and over_share > 0.004 and crossings > 600
assert cap == 11 and base + per_req * n_cap.max() <= limit
assert 0.014 < waited < 0.018 and p99_capped / p99_uncapped < 1.01
```

Average memory sits at about 81% of the limit, which looks safe on a dashboard, yet the pod goes over the limit more than 600 times an hour, and any one of those can end in a kill. Capping concurrency at 11 makes going over impossible, and costs 1.6% of requests a short wait with no visible change in p99 latency. Size a model pod's memory from weights plus the worst concurrency the server allows, never from the average.

### Why a CPU limit adds 100 ms
The kernel accounts CPU time in periods of 100 ms by default. A limit of 2 CPUs is a quota of 200 ms of CPU time per period; once the container has spent it, every thread waits for the next period. PyTorch on a node with 8 cores runs 8 threads by default, which spend 200 ms of quota in 25 ms of wall time:

```python
def cfs_latency(work_ms, threads, limit_cpus=None, period_ms=100.0):
    """Wall time for work_ms of CPU time spread over `threads`, starting at a period boundary."""
    if limit_cpus is None or threads <= limit_cpus:
        return work_ms / threads
    quota = limit_cpus * period_ms
    full, rest = divmod(work_ms, quota)
    if rest == 0:
        full, rest = full - 1, quota
    return full * period_ms + rest / threads


print("CPU ms | 8 threads, no limit | 8 threads, limit 2 | 2 threads, limit 2")
for work in (150, 190, 210, 400):
    print(f"{work:6d} | " + " | ".join(f"{cfs_latency(work, th, lim):6.1f} ms" for th, lim in ((8, None), (8, 2), (2, 2))))
assert cfs_latency(190, 8, 2) == cfs_latency(190, 8) == 23.75
assert cfs_latency(210, 8, 2) == 101.25 and cfs_latency(210, 8) == 26.25
assert cfs_latency(210, 2, 2) == 105.0 and cfs_latency(210, 8, 2) / cfs_latency(190, 8, 2) > 4
```

A request that needs 190 ms of CPU finishes in 24 ms; one that needs 210 ms, about 10% more, spills past the quota, waits out the period and finishes in 101 ms, more than four times as long. That cliff at every multiple of the quota shows up as an unexplained p99. Running only as many threads as the limit (2 threads, last column) removes the cliff and makes latency proportional to the work, at the price of slower small requests; raising the limit to the thread count removes it without that price.

## Scheduling is by requests, not usage
The scheduler filters out nodes where the pod does not fit, scores the rest and picks the best. "Fits" means that, for every resource, the requests already placed on the node plus the new pod's requests stay within what the node offers to pods. Actual usage plays no part:

```python
node_cpu = 8.0
placed = [{"request": 2.5, "using": 0.4}] * 3    # three pods that asked for far more than they use
new_request = 1.0
fits = sum(p["request"] for p in placed) + new_request <= node_cpu
busy = sum(p["using"] for p in placed) / node_cpu
assert not fits and round(busy, 2) == 0.15                 # rejected from a node that is 85% idle
```

The node is 85% idle and still refuses the pod. Inflated requests waste nodes; deflated ones pack pods onto nodes that then run out at the daily peak. Requests should come from measured usage.

GPUs are requested as an **extended resource** such as `nvidia.com/gpu`, which appears on a node once a vendor's device plugin is installed. Extended resources are whole numbers and cannot be overcommitted: a pod asks for 1 or 2 GPUs, never 0.5, and the scheduler never counts one GPU toward two pods. GPUs go in `limits`; the request defaults to the same value. Whole units make fragmentation a real cost. Below, a cluster of four 8-GPU nodes receives a stream of jobs asking for 1, 2, 4 or 8 GPUs, at a load that keeps it only about 30% busy on average. The only difference between the two runs is the scoring rule: **spread** places each job on the node with the most free GPUs, **pack** on the node with the fewest that still fit.

```python
def gpu_cluster(policy, n_nodes=4, per_node=8, n_jobs=20_000, seed=2):
    r = np.random.default_rng(seed)
    arrive = np.cumsum(r.exponential(1.0, n_jobs))
    size = r.choice([1, 2, 4, 8], n_jobs, p=[0.5, 0.25, 0.15, 0.10])
    duration = r.exponential(4.0, n_jobs)
    free, running, whole, blocked, stranded = [per_node] * n_nodes, [], 0, 0, 0
    for t, s, d in zip(arrive, size, duration):
        while running and running[0][0] <= t:
            _, node, used = heapq.heappop(running)
            free[node] += used
        fit = [i for i in range(n_nodes) if free[i] >= s]
        if s == per_node:
            whole += 1
            blocked += not fit
            stranded += not fit and sum(free) >= s       # enough GPUs in total, none on one node
        if fit:
            node = (max if policy == "spread" else min)(fit, key=lambda i: free[i])
            free[node] -= s
            heapq.heappush(running, (t + d, node, s))
    return blocked / whole, stranded / whole


offered = 4.0 * (0.5 * 1 + 0.25 * 2 + 0.15 * 4 + 0.10 * 8) / 32
spread, pack = gpu_cluster("spread"), gpu_cluster("pack")
print(f"average load {offered:.0%}; 8-GPU jobs refused: spread {spread[0]:.0%}, pack {pack[0]:.0%}")
assert round(offered, 2) == 0.3
assert 0.40 < spread[0] < 0.44 and 0.04 < pack[0] < 0.06
assert spread[1] > 0.95 * spread[0]           # refused while 8+ GPUs were free in pieces
```

Spreading refuses about 42% of the jobs that need a whole node, almost all of them while 8 or more GPUs sat free across the cluster in pieces; packing refuses about 5%. Kubernetes can score this way: the `NodeResourcesFit` plugin's `MostAllocated` strategy is the opt-in bin-packing rule. GPU clusters also commonly reserve node pools by job size, using taints and node selectors, for the same reason.

## Probes, model load time and rollouts
The kubelet on each node runs three kinds of **probe** against a container:

- **Readiness**: while it fails, the pod is removed from its Services' endpoints and gets no traffic. A container with no readiness probe counts as ready as soon as it starts.
- **Liveness**: after `failureThreshold` failures in a row (default 3, every `periodSeconds`, default 10), the container is restarted.
- **Startup**: while it has not yet succeeded, liveness and readiness probes do not run; if it fails `failureThreshold` times, the container is restarted.

A model server's process starts in seconds, but its model loads for much longer, so "started" and "able to answer" are far apart. A liveness probe alone kills the server before it can load:

```python
load_s = 60
liveness_budget = 3 * 10                         # failureThreshold x periodSeconds, the defaults
startup_budget = 30 * 10                         # the startupProbe in the manifest
assert liveness_budget < load_s < startup_budget # without a startup probe, killed about 30 s into every attempt
```

The readiness gap is worse during a rolling update. The Deployment adds new pods and removes old ones within two bounds: at most `maxSurge` pods above the desired count (25% by default, rounded up) and at most `maxUnavailable` pods below it (25%, rounded down), where "available" means *ready*. Without a readiness probe, a new pod counts as ready a few seconds after its container starts, so the Deployment removes the old pods long before any new one has loaded. The simulation steps one second at a time with four replicas, 50 requests a second spread evenly over ready pods, 3 s to start a container and 60 s to load the model; a request that lands on a pod still loading fails. The controller here is simplified to the two bounds.

```python
def rollout(probe, replicas=4, start_s=3, period_s=10, rps=50, horizon=200):
    surge, unavailable = math.ceil(replicas * 0.25), math.floor(replicas * 0.25)
    pods = [{"new": False, "born": -1_000} for _ in range(replicas)]

    def loaded(p, t):
        return t >= p["born"] + start_s + load_s

    def ready(p, t):                             # what Kubernetes believes
        if not probe:
            return t >= p["born"] + start_s
        first_pass = p["born"] + start_s + math.ceil(load_s / period_s) * period_s
        return t >= first_pass

    failed = outage = 0
    fewest_loaded, k8s_done = replicas, None
    for t in range(horizon):
        changed = True
        while changed:
            changed = False
            if len(pods) < replicas + surge and sum(p["new"] for p in pods) < replicas:
                pods.append({"new": True, "born": t})
                changed = True
            old = [p for p in pods if not p["new"]]
            if old and sum(ready(p, t) for p in pods) - 1 >= replicas - unavailable:
                pods.remove(old[0])
                changed = True
        endpoints = [p for p in pods if ready(p, t)]
        working = sum(loaded(p, t) for p in endpoints)
        failed += rps - (rps * working / len(endpoints) if endpoints else 0)
        outage += working == 0
        fewest_loaded = min(fewest_loaded, sum(loaded(p, t) for p in pods))
        if k8s_done is None and all(p["new"] and ready(p, t) for p in pods):
            k8s_done = t
    return round(failed), outage, fewest_loaded, k8s_done


for probe in (False, True):
    print("readiness probe" if probe else "no probe", rollout(probe))
no_probe, with_probe = rollout(False), rollout(True)
assert no_probe[1] == 57 and 2_900 < no_probe[0] < 3_100 and no_probe[2] == 0 and no_probe[3] == 6
assert with_probe[0] == 0 and with_probe[1] == 0 and with_probe[2] == 3 and 120 <= with_probe[3] <= 130
```

Without a probe, Kubernetes considers the rollout done after 6 seconds, while the model is still loading on every pod; the Service then has nothing that can answer for 57 seconds. With a readiness probe that passes only once the model is loaded, the rollout takes about two minutes but never has fewer than three working pods and fails no requests. The probe has to check the model, not the web server: an endpoint that returns 200 as soon as the framework is up is the same as no probe.

## The Horizontal Pod Autoscaler
The **Horizontal Pod Autoscaler** (HPA) checks a metric every 15 seconds and sets the replica count to

\[ \text{desiredReplicas} = \left\lceil \text{currentReplicas} \times \frac{\text{currentMetric}}{\text{targetMetric}} \right\rceil . \]

For CPU, the metric is **utilization**: usage as a fraction of the pod's CPU *request*, averaged over pods. Three rules soften the formula. If the ratio is within a **tolerance** of 1 (0.1 by default), nothing happens. When scaling up, pods that are not yet ready are counted as using nothing, which damps the step while new pods load. And scaling down uses the highest recommendation from the last 300 seconds, the **stabilization window**, while scaling up acts at once (at most 4 pods or 100% more per 15 s by default).

```python
SYNC_S, TOLERANCE, TARGET, POD_RPS, LOAD_S = 15, 0.1, 0.6, 10.0, 75


def recommend(n, ready, load_rps):
    """One HPA step on CPU: each ready pod is at 100% of its request when serving POD_RPS."""
    ratio = load_rps / (ready * POD_RPS) / TARGET
    if ratio > 1 and ready < n:                  # scaling up: not-ready pods count as idle
        ratio = ratio * ready / n
    return n if abs(ratio - 1) <= TOLERANCE else math.ceil(n * ratio)


assert recommend(4, 4, 4 * POD_RPS * 0.90) == 6  # 90% against a 60% target: ceil(4 x 1.5)
assert recommend(4, 4, 4 * POD_RPS * 0.65) == 4  # ratio 1.08 is inside the tolerance
```

Four pods at 90% CPU against a 60% target become six; at 65% the ratio is 1.08, inside the tolerance, so they stay at four. Because utilization is relative to the request, a request set too low makes every pod look busy and the HPA overscales. Below, six hours of traffic rise from 150 to 250 requests a second and fall to 50, with noise on every 15-second sample, and new pods take 75 s to load their model. The run with a window of 0 removes pods the moment a sample dips.

```python
def autoscale(window_s, seed=4, hours=6, lo=2, hi=60):
    r = np.random.default_rng(seed)
    t = np.arange(hours * 3600 // SYNC_S) * SYNC_S
    load = (150 + 100 * np.sin(2 * np.pi * t / (hours * 3600))) * r.lognormal(0, 0.15, t.size)
    born, history, created, unserved = [-10_000] * 25, [], 0, 0.0
    for now, rps in zip(t, load):
        ready = sum(now >= b + LOAD_S for b in born)
        unserved += max(0.0, rps - ready * POD_RPS)
        want = min(hi, max(lo, recommend(len(born), ready, rps)))
        history.append((now, want))
        if want < len(born) and window_s:        # scale down only to the highest recent recommendation
            want = min(len(born), max(w for s, w in history if s > now - window_s))
        want = min(want, len(born) + max(4, len(born)))
        if want > len(born):
            created += want - len(born)
            born += [now] * (want - len(born))
        born = sorted(born)[:want]               # scale-down removes the newest pods first
    return created, unserved / load.sum()


flappy, stable = autoscale(0), autoscale(300)
print(f"pods created in 6 h: window 0 s -> {flappy[0]}, window 300 s -> {stable[0]}; "
      f"unserved load {flappy[1]:.2%} vs {stable[1]:.2%}")
assert flappy[0] > 2_000 and stable[0] < 150 and flappy[0] > 10 * stable[0]
assert stable[1] == 0 and 0 < flappy[1] < 0.005
```

Without the window, the HPA creates over 2,000 pods in six hours, each loading a model for 75 seconds before it serves, and still leaves a little load unserved when a burst follows a scale-down; the default 300-second window creates fewer than 150 and serves everything. For model servers the stabilization window is not a detail: a pod is expensive to start, so removing one that will be needed a minute later is pure waste.

## The failure modes in one table
| Problem | Naive fix | How it breaks | Real fix | How it's measured |
|---|---|---|---|---|
| Pod OOMKilled at start | raise the memory limit until it stops | the limit pays for a load-time spike forever | build on `meta`, load with `assign=True` | peak resident bytes during load |
| Pod OOMKilled under load | size memory from the average | random concurrency crosses the limit hundreds of times an hour | cap concurrent requests to what the limit holds | peak memory and queue wait |
| Slow p99 with idle CPUs | more threads | threads burn the CPU quota early, then wait 100 ms | threads no more than the CPU limit, or a higher limit | latency against request size; throttled-period counters |
| Pod stays `Pending` | add nodes | requests, not usage, decide fit; GPUs fragment | requests from measured usage; bin-pack GPUs | free resources per node, not per cluster |
| Errors during a rollout | none: rollouts look instant | pods count as ready before the model loads | readiness probe on model loaded, startup probe for load time | failed requests during rollout |
| Replica count flaps | lower the target | pods are removed and recreated each dip | keep the stabilization window; requests set right | pods created per hour |

## What the simulations do not show
Every number above comes from a model of a mechanism, with rates and costs the code assumes: 60 to 75 seconds of model load, 0.15 GiB per request, independent noise on each traffic sample. The memory simulation counts crossings of the limit, not kills, since the kernel acts only under memory pressure. The CPU model gives one request the whole quota and starts it at a period boundary; concurrent requests share the quota and land at random phases. The rollout and autoscaling loops keep only the rules quoted here, not the real controllers (which also weigh pod age, readiness timing and metric windows), the GPU simulation drops refused jobs instead of queuing them, and nothing here runs on a real cluster.

What stays open is measuring these inputs for a given model: its load time sets the probes, its memory per request sets the concurrency cap, and its CPU per request sets both the limit and the HPA target. Past that, GPU sharing (time-slicing, MIG partitions) and the Dynamic Resource Allocation API change the whole-unit rule above, and are the next thing to read in the [Kubernetes documentation](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).

The rules follow the Kubernetes documentation on [resource management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/), [probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/), [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/), [GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/) and the [Horizontal Pod Autoscaler](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/).
