---
title: Spark for ML data: partitions, shuffles, skew and joins
summary: What a Spark job does with a feature pipeline, simulated in pandas: partitions and tasks, lazy plans cut into stages at each shuffle, why one hot key makes one task the straggler and how salting and partial aggregation fix it, the bytes a broadcast join saves over a shuffle join, and why collect() or toPandas() on a big table brings down the driver.
group: Coding and data
order: 5
tags: Spark · PySpark · data pipelines · shuffle · data skew
---
Hash 2,000,000 simulated click events by user id into 200 partitions, with user activity following a Zipf law, and one user owns 165,157 rows. The partition holding that user has 176,441 rows, 17.6 times the mean, and on 50 cores the stage finishes 4.6 times later than the same rows with evenly spread keys. Nothing in pandas or SQL prepares you for that, because on one machine a group is a group, whatever its size. Apache Spark runs a DataFrame program on many machines by splitting the table into **partitions**, and nearly every slow or failing Spark job comes from how rows move between them.

This post rebuilds the moving parts in plain pandas and numpy: a tiny lazy engine that cuts a plan into stages, the shuffle, skew and salting, the two join strategies and the bytes each moves, and the size of a table once it lands on the driver. The PySpark next to each piece is shown, not run. The pandas and SQL versions of the same operations are in the [pandas](../pandas-cheatsheet/) and [SQL](../sql-cheatsheet/) cheat sheets, and joining features without leaking the label is in [ML infrastructure](../ml-infrastructure/).

## The vocabulary
| Term | What it is | pandas analogue |
|---|---|---|
| Driver | The process that runs your Python, builds the plan and collects results | Your notebook |
| Executor | A worker process on a cluster machine, with a few cores and its own memory | None: pandas has one |
| Partition | A slice of the rows that one core processes at a time | A chunk from `pd.read_csv(..., chunksize=)` |
| Task | One partition pushed through one stage's chain of operations | One loop iteration over chunks |
| Stage | The tasks between two shuffles; all of them can run in parallel | None |
| Job | All the stages one action needs | One cell |

Reading a 10 GiB Parquet table with the default `spark.sql.files.maxPartitionBytes` of 128 MB gives at least 80 partitions, so at least 80 read tasks. The question that decides speed is never "how many rows" but "how many rows in the biggest task".

## Lazy plans and the stages a shuffle cuts
A transformation (`filter`, `withColumn`, `groupBy().agg`, `join`) returns a new DataFrame at once and reads nothing. It only extends a plan. An **action** (`count`, `collect`, `toPandas`, `write`) makes Spark optimise the whole plan, split it into stages and run it. A per-user feature job looks like this:

```plaintext
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("user-features").getOrCreate()
events = spark.read.parquet("data/events/")              # nothing is read yet

features = (events
    .withColumn("night", (F.hour("ts") < 6).cast("int"))  # narrow: row by row, inside each partition
    .groupBy("user_id")                                   # wide: all rows of a user must meet
    .agg(F.count("*").alias("n"), F.sum("label").alias("clicks"), F.sum("night").alias("night")))

features.explain()   # HashAggregate (partial_count, partial_sum), Exchange hashpartitioning(user_id, 200), HashAggregate
features.write.mode("overwrite").parquet("data/user_features/")   # the action: only now does the job run
```

A **narrow** transformation needs one input partition per output partition, so a chain of them runs in one pass, one task per partition. A **wide** one needs rows from every partition: Spark hashes the key, writes each row to a shuffle file for the partition its key belongs to, and the next stage's tasks fetch those files over the network. That **shuffle** is the stage boundary. Below, a short engine does the same: transformations only append to a plan, and `collect` cuts it at each shuffle and runs one task per partition per stage.

```python
import heapq
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
assert 10 * 1024**3 // (128 * 1024**2) == 80               # read tasks for 10 GiB at 128 MB each
n_rows, n_users, n_reads, n_parts = 2_000_000, 100_000, 16, 200
weight = 1 / np.arange(1, n_users + 1)                    # Zipf, exponent 1: the k-th most active user has weight 1/k
user_of_rank = rng.permutation(n_users)                   # so the hottest user's id is arbitrary
ev = pd.DataFrame({"user_id": user_of_rank[rng.choice(n_users, n_rows, p=weight / weight.sum())],
                   "item_id": rng.integers(0, 50_000, n_rows),
                   "ts": rng.uniform(0, 30 * 86_400, n_rows),
                   "label": (rng.random(n_rows) < 0.05).astype(np.int8)})
for j in range(8):
    ev[f"f{j}"] = rng.normal(size=n_rows).astype(np.float32)
reads = [ev.iloc[i * n_rows // n_reads:(i + 1) * n_rows // n_reads] for i in range(n_reads)]  # as if 16 files


def part_of(keys, n):
    """Hash partitioning: a key always lands in the same one of n partitions (Spark uses Murmur3; this is a mix)."""
    x = np.asarray(keys).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    return ((x ^ (x >> np.uint64(29))) % np.uint64(n)).astype(np.int64)


class Lazy:
    """A DataFrame as Spark holds it: input partitions plus a plan. Transformations only extend the plan."""

    def __init__(self, parts, plan=()):
        self.parts, self.plan = parts, plan

    def narrow(self, f):            # filter, select, withColumn: an output partition needs one input partition
        return Lazy(self.parts, self.plan + (("narrow", f),))

    def shuffle(self, key, n):      # groupBy, join, repartition: each row moves to the partition its key hashes to
        return Lazy(self.parts, self.plan + (("wide", (key, n)),))

    def collect(self, log):
        """The action: cut the plan at each shuffle; log the rows going into every task of every stage."""
        parts, chain = self.parts, []
        for kind, arg in self.plan + (("wide", None),):
            if kind == "narrow":
                chain.append(arg)
                continue
            log.append([len(p) for p in parts])
            out = []
            for p in parts:                                   # one task: one partition through the stage's chain
                for f in chain:
                    p = f(p)
                out.append(p)
            whole, chain = pd.concat(out, ignore_index=True), []
            if arg is None:
                return whole
            dest = part_of(whole[arg[0]], arg[1])             # the shuffle: every row is rewritten by destination
            whole, ends = whole.iloc[np.argsort(dest, kind="stable")], np.cumsum(np.bincount(dest, minlength=arg[1]))
            parts = [whole.iloc[a:b] for a, b in zip(np.r_[0, ends[:-1]], ends)]


calls = []


def add_night(p):
    calls.append(len(p))
    return p.assign(night=((p.ts // 3600) % 24 < 6).astype(np.int64))


def partial(p, by="user_id"):   # aggregate whatever rows this partition holds
    return p.groupby(by, as_index=False).agg(n=("label", "size"), clicks=("label", "sum"), night=("night", "sum"))


def final(p):                   # add up partial results that arrived for the same user
    return p.groupby("user_id", as_index=False)[["n", "clicks", "night"]].sum()


plan = Lazy(reads).narrow(add_night).narrow(partial).shuffle("user_id", n_parts).narrow(final)
assert calls == []                                            # building the plan touched no row
log_partial = []
feats = plan.collect(log_partial)
assert len(calls) == n_reads and sum(calls) == n_rows        # add_night ran once per read task, inside stage 1
assert [len(s) for s in log_partial] == [n_reads, n_parts]   # one shuffle, two stages: 16 tasks, then 200
truth = partial(add_night(ev)).set_index("user_id").sort_index()
assert feats.set_index("user_id").sort_index().equals(truth)  # same answer as one pandas groupby
assert sum(log_partial[1]) == 454_278                         # rows that crossed the shuffle, of 2,000,000
```

Nothing ran until `collect`. Then the plan became two stages, 16 tasks and 200 tasks, and the answer matched a single pandas `groupby`. Only 454,278 rows crossed the shuffle, not 2,000,000, because the plan aggregates inside each read partition before the shuffle. That pre-aggregation is what the `partial_count` in the PySpark plan above means: for built-in aggregates such as `count`, `sum` and `avg`, Spark does it on its own.

## Why the shuffle is the expensive step
Spark's own guide names the costs: the shuffle involves disk I/O, data serialization and network I/O. Every row entering it is serialized and written to local disk, then read back by a task on, most likely, another machine. A narrow transformation costs one pass over memory. So the useful question about any plan is how many shuffles it has and how many bytes each one moves, and `explain()` answers the first: count the `Exchange` lines.

| Transformation | Narrow or wide | Why |
|---|---|---|
| `filter`, `select`, `withColumn`, `union` | Narrow | Each output row depends on rows of one partition |
| `groupBy().agg`, `distinct` | Wide | Rows with the same key sit in every partition |
| `join` of two large tables | Wide | Both sides are hashed on the key |
| `join` with a broadcast side | Narrow on the big side | The small side is copied to each executor instead |
| `orderBy` | Wide | A global order needs range partitioning |
| `repartition(n)` | Wide; `coalesce(n)` to fewer partitions is not | `coalesce` merges neighbours without moving rows by key |

## Skew: the stage waits for its slowest task
A stage ends when its last task ends. With enough cores the tasks run side by side, so stage time is set by the **largest** partition, not the mean. Hash partitioning spreads distinct keys evenly, but it cannot split one key: every row of the hottest user lands in one partition. Below are the partition sizes when raw rows are shuffled by `user_id`, which is what a shuffle join or a `groupBy` with no pre-aggregation does, against the same row count with uniform user ids. Stage time is measured as the most rows any of 50 cores processes, with tasks handed to whichever core frees up first.

```python
def stage_time(task_rows, cores=50):
    free = [0] * cores
    for r in task_rows:
        heapq.heappush(free, heapq.heappop(free) + r)
    return max(free)


hot_rows = int(ev.user_id.value_counts().iloc[0])
zipf_parts = np.bincount(part_of(ev.user_id, n_parts), minlength=n_parts)
uniform_parts = np.bincount(part_of(rng.integers(0, n_users, n_rows), n_parts), minlength=n_parts)
more_parts = np.bincount(part_of(ev.user_id, 2_000), minlength=2_000)
print(f"hottest user {hot_rows} rows; largest partition {zipf_parts.max()} (Zipf), {uniform_parts.max()} (uniform); "
      f"stage time {stage_time(zipf_parts)} vs {stage_time(uniform_parts)}")
assert hot_rows == 165_157 and zipf_parts.max() == 176_441 and uniform_parts.max() == 11_057
assert round(zipf_parts.max() / zipf_parts.mean(), 1) == 17.6
assert stage_time(zipf_parts) == 193_747 and stage_time(uniform_parts) == 42_227
assert round(stage_time(zipf_parts) / stage_time(uniform_parts), 1) == 4.6
assert round(100 * n_rows / (50 * stage_time(zipf_parts))) == 21  # percent of core time spent working
assert round(100 * hot_rows / n_rows) == 8
assert more_parts.max() == 165_649 >= hot_rows               # ten times the partitions: the hot key stays whole
```

| Keys | Partitions | Largest partition | Mean | Stage time on 50 cores (rows) |
|---|---|---|---|---|
| Uniform | 200 | 11,057 | 10,000 | 42,227 |
| Zipf | 200 | 176,441 | 10,000 | 193,747 |
| Zipf | 2,000 | 165,649 | 1,000 | not run |

With the same total work, the skewed stage takes 4.6 times as long, and the 50 cores spend only 21% of it working. Ten times more partitions shrinks the mean tenfold and leaves the largest partition almost as big, because it is one user's 165,157 rows.

## Salting, and what it costs
**Salting** splits a hot key by hand. Add a random `salt` in 0..S−1, aggregate by `(user_id, salt)`, which spreads the hot user over S partitions, then aggregate the partial results again by `user_id`. The price is that second aggregation: one more shuffle and one more stage. Below, the raw-row plan, the salted plan with S = 16, and the partial-aggregation plan from earlier.

```python
S = 16


def add_salt(p):
    salt = rng.integers(0, S, len(p))
    return p.assign(salt=salt, k=p.user_id.to_numpy() * S + salt)


log_raw, log_salted = [], []
raw = Lazy(reads).narrow(add_night).shuffle("user_id", n_parts).narrow(partial).collect(log_raw)
salted = (Lazy(reads).narrow(add_night).narrow(add_salt).shuffle("k", n_parts)
          .narrow(lambda p: partial(p, ["user_id", "salt"])).shuffle("user_id", n_parts).narrow(final)
          .collect(log_salted))
for result in (raw, salted):
    assert result.set_index("user_id").sort_index().equals(truth)
after_shuffles = {name: [(sum(s), max(s), stage_time(s)) for s in log[1:]]
                  for name, log in (("raw rows", log_raw), ("salted", log_salted), ("partial agg", log_partial))}
for name, stages in after_shuffles.items():
    print(name, stages)
assert after_shuffles["raw rows"] == [(2_000_000, 176_441, 193_747)]
assert after_shuffles["salted"] == [(2_000_000, 23_782, 52_463), (453_988, 2_647, 9_674)]
assert round((52_463 + 9_674) / 193_747, 2) == 0.32                   # about a third of the straggler stage
assert after_shuffles["partial agg"] == [(454_278, 2_636, 9_694)]
```

| Plan | Shuffles | Rows shuffled | Largest task after each shuffle | Stage time after each shuffle (rows) |
|---|---|---|---|---|
| Raw rows by `user_id` | 1 | 2,000,000 | 176,441 | 193,747 |
| Salted, S = 16 | 2 | 2,000,000 + 453,988 | 23,782, then 2,647 | 52,463, then 9,674 |
| Partial aggregation | 1 | 454,278 | 2,636 | 9,694 |

Salting cuts the straggler's stage from 193,747 to 52,463 rows, and the second aggregation adds a stage of 9,674 on top: about a third of the original time, paid for with a second shuffle. Partial aggregation does better than both, because aggregating before the first shuffle is the same idea applied to every key at once. That is why salting a built-in `groupBy().agg(F.sum(...))` buys little in Spark: it already pre-aggregates. Salting pays where nothing can be combined before the shuffle: a join, an RDD `groupByKey`, or a grouped pandas UDF that needs all of a key's rows.

For a join, salt the large side and copy each row of the small side once per salt value, so every `(user_id, salt)` pair still meets its user:

```plaintext
S = 16
events_s = events.withColumn("salt", (F.rand(seed=0) * S).cast("int"))
users_s = users.withColumn("salt", F.explode(F.sequence(F.lit(0), F.lit(S - 1))))   # each user row S times
joined = events_s.join(users_s, on=["user_id", "salt"]).drop("salt")

# The same split, done by Spark at run time for sort-merge joins (adaptive query execution, on by default since 3.2):
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")   # a partition > 5x the median and > 256 MB is split
```

Here the cost is the copies: the users table grows S times, 16 times in this example. Salting only the hottest few keys keeps that small. Adaptive query execution (AQE) applies the same split automatically to a sort-merge join partition larger than both `skewedPartitionFactor` (5) times the median and `skewedPartitionThresholdInBytes` (256 MB); the [performance tuning guide](https://spark.apache.org/docs/latest/sql-performance-tuning.html) lists both settings and documents this handling for joins, not aggregations.

| Problem | Naive fix | How it breaks | Real fix | How it's measured |
|---|---|---|---|---|
| One task runs far longer than the rest | Raise `spark.sql.shuffle.partitions` | The hot key is still one partition: 165,649 rows at 2,000 partitions | Pre-aggregate; salt the hot keys; let AQE split join partitions; broadcast the small side | Max against median task input in the stage's task summary in the Spark UI |
| A join shuffles a big table | Add executors | The bytes moved stay the same; more machines share them | `F.broadcast()` the side that fits in memory | Shuffle write bytes per stage |
| The driver dies on `toPandas()` | Raise driver memory | Memory grows with rows: 57 GB at a billion rows | Aggregate or filter in Spark first; write Parquet | Rows times bytes per row before collecting |

## Broadcast join against shuffle join
A **shuffle join** (in Spark, usually a sort-merge join) hashes both tables on the key into the same partitions and joins partition by partition: both sides cross the network, and the big side's skew comes along. A **broadcast join** sends the whole small table to every executor, and each task of the big table joins its own partition locally: the big side never moves. Spark broadcasts on its own when it estimates a side below `spark.sql.autoBroadcastJoinThreshold`, 10 MB by default; `F.broadcast(users)` forces it. Below, both strategies on 20 executors, counting the bytes whose sending and receiving executor differ.

```python
users = pd.DataFrame({"user_id": np.arange(n_users), **{f"u{j}": rng.normal(size=n_users).astype(np.float32)
                                                       for j in range(4)}})
user_reads = [users.iloc[i * n_users // 4:(i + 1) * n_users // 4] for i in range(4)]
n_exec = 20


def nbytes(df):
    return int(df.memory_usage(index=False).sum())


def network_bytes(read_parts, first_exec):
    """Shuffle: read partition i sits on executor first_exec + i, output partition j on executor j % n_exec."""
    return sum(nbytes(p[part_of(p.user_id, n_parts) % n_exec != (first_exec + i) % n_exec])
               for i, p in enumerate(read_parts))


ev_groups, user_groups = dict(tuple(ev.groupby(part_of(ev.user_id, n_parts)))), dict(tuple(users.groupby(
    part_of(users.user_id, n_parts))))
shuffle_join = pd.concat([ev_groups[j].merge(user_groups[j], on="user_id") for j in ev_groups])
broadcast_join = pd.concat([p.merge(users, on="user_id") for p in reads])
reference = ev.merge(users, on="user_id")
for joined in (shuffle_join, broadcast_join):
    assert len(joined) == n_rows and (pd.util.hash_pandas_object(joined, index=False).sum()
                                      == pd.util.hash_pandas_object(reference, index=False).sum())

big, small = nbytes(ev), nbytes(users)
shuffle_net = network_bytes(reads, 0) + network_bytes(user_reads, n_reads)
broadcast_net = small * (n_exec + 1)                         # collected to the driver, then one copy per executor
crossover = big * (n_exec - 1) / (n_exec * (n_exec + 1) - (n_exec - 1))   # small-table size where they tie
print(f"events {big / 1e6:.0f} MB, users {small / 1e6:.1f} MB; network: shuffle join {shuffle_net / 1e6:.0f} MB, "
      f"broadcast {broadcast_net / 1e6:.1f} MB; tie at a {crossover / 1e6:.1f} MB small table")
assert big == 114_000_000 and small == 2_400_000
assert round(shuffle_net / 1e6) == 111 and round(broadcast_net / 1e6, 1) == 50.4
assert round(crossover / 1e6, 1) == 5.4 and broadcast_net < shuffle_net / 2
assert {len(p) for p in reads} == {n_rows // n_reads}         # broadcast tasks keep the read sizes: skew never forms
```

| Strategy | What crosses the network | Bytes here | Largest task |
|---|---|---|---|
| Shuffle join | Both tables, about (E−1)/E of each | 111 MB | The hot user's partition, 176,441 events |
| Broadcast join | The small table, once to the driver and once per executor | 50.4 MB | A read partition, 125,000 events |

Both strategies return the same 2,000,000 rows, but the broadcast moves less than half the bytes and its tasks keep the even sizes of the read. The saving holds only while the small side is small: with 20 executors and this 114 MB events table, broadcasting a table above 5.4 MB moves more bytes than shuffling, and every executor must also hold the whole copy in memory. That is the reason for a threshold, not a rule that broadcast always wins; the bytes model here ignores the shuffle's disk writes and sorting, which favour broadcasting further.

```plaintext
joined = events.join(F.broadcast(users), on="user_id", how="left")    # users is copied; events never move
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 50 * 1024 * 1024)   # default 10 MB; -1 turns it off
```

## collect() and toPandas(): everything on one machine
`collect()` and `toPandas()` are actions that bring every row to the driver, one process with 1 GiB of memory by default (`spark.driver.memory`). Spark also aborts an action whose serialized results pass `spark.driver.maxResultSize`, 1 GiB by default. Below, the size of the events table as pandas columns, and as the Python tuples `collect()` hands back, one object per row and per value.

```python
col_per_row = nbytes(ev) / n_rows
py_rows = list(zip(*(ev[c].head(10_000).tolist() for c in ev.columns)))   # like collect(): a tuple per row
objects = {id(o): sys.getsizeof(o) for row in py_rows for o in (row, *row)}  # each object counted once
py_per_row = sum(objects.values()) / len(py_rows)
print(f"{col_per_row:.0f} bytes per row in columns, {py_per_row:.0f} as Python objects")
assert col_per_row == 57 and round(py_per_row) == 408          # 7.2 times as much
```

At 57 bytes a row, a billion-row table needs 57 GB as pandas columns, and about 408 bytes a row, 7.2 times as much, as the Python objects `collect()` returns: far past a driver sized for orchestration. The data should stay distributed until it is small:

```plaintext
features.write.mode("overwrite").parquet("data/user_features/")       # the result stays on the cluster's storage
top = features.orderBy(F.desc("clicks")).limit(10_000).toPandas()      # bounded before it reaches the driver
peek = features.sample(fraction=0.001, seed=0).toPandas()             # a sample for plots and sanity checks
```

## Pitfalls
- **Calling an action inside a loop.** Each `count()` or `toPandas()` can rerun the whole plan from the files; `cache()` a DataFrame that several actions reuse, and `unpersist()` it after.
- **Reading `explain()` too late.** Count the `Exchange` nodes before running a job on the full table, not after it has run for an hour.
- **Python UDFs for what a built-in does.** A row-at-a-time Python UDF calls Python once per row and ships the data between the JVM and Python workers; `pyspark.sql.functions` usually has the operation.
- **Too many tiny output files.** 200 shuffle partitions written by a daily job can make 200 files a day; `coalesce` before writing.
- **Trusting the mean task time.** The stage waits for the maximum; look at the slowest tasks first.

The numbers come from a simulation, not a cluster. Task time is taken as proportional to input rows, cores are interchangeable, and network bytes are counted from pandas column sizes, while Spark serializes and compresses its shuffle files, spills to disk when memory runs short, and uses Murmur3 rather than the hash here. The Zipf exponent of 1 is an assumption; a real event log's hottest key may hold far less or far more than 8% of the rows. The simulation shows which way each effect points and roughly how large it is, not what a given cluster will measure.

What stays open is measuring these on real jobs: the task-size summary in the Spark UI says whether a slow stage is skewed, and the shuffle read and write bytes say whether a broadcast or a pre-aggregation would pay. How far AQE's automatic skew handling closes the gap to hand salting on a given workload is a question this post cannot answer without a cluster.
