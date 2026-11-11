# Docker for ML: containers, layer caching and reproducible builds

Add the same 100,000 float32 values in four ways and you get four different sums, so a model's numbers can shift with the libraries installed around it. I show that, then simulate Docker's build cache, where the order of a Dockerfile decides whether a one-line code edit rebuilds in under two seconds or pays the full dependency install again. A container is an ordinary Linux process that has been given its own view of the system: its own root filesystem, its own process list, its own network interfaces. Nothing is booted and nothing is emulated. The host kernel runs the process directly, and the only difference from any other process is what that process is allowed to see, starting with which directory it believes is `/`.

Several practical facts follow. Starting a container costs little more than starting the process inside it. An image is mostly just the files that process needs, so it is small next to a disk image. And a Linux container needs a Linux kernel, which is why Docker on a Mac or on Windows quietly runs a small Linux virtual machine underneath.

This post covers what that means for shipping models: why ML code is unusually sensitive to its environment, how to write a Dockerfile whose build cache works for you, why a cached build is not a reproducible one, and what a lockfile fixes.

## What a container is, and is not
The isolation comes from two Linux kernel features. **Namespaces** give a process its own view of a resource (mount points, process IDs, network, hostname, users), and **cgroups** cap what it may consume (CPU, memory, I/O). An **image** is a frozen, layered filesystem plus a little metadata such as the default command; a **container** is one running instance of an image, with a thin writable layer on top that disappears when the container is removed.

| | Container | Virtual machine |
|---|---|---|
| Kernel | shares the host's | runs its own |
| Start-up | about the cost of starting the process | seconds to boot an OS |
| Typical size | tens to hundreds of MB | several GB |
| Isolation | process level: namespaces and cgroups | hardware level: a hypervisor |
| Runs a different OS kernel | no | yes |

The trade is weaker isolation for a much lower cost. For serving a model or running a training job on infrastructure you control, process-level isolation is usually enough. For running untrusted code from strangers it is not, and that is where VMs or sandboxed runtimes earn their overhead.

## Why ML cares more than most code
A model's numerical behaviour depends on things that never appear in its source: the Python version, the NumPy and PyTorch versions, the BLAS library NumPy links against, the CUDA and cuDNN versions, sometimes the CPU instruction set. Change any of them and the same code can produce different numbers.

The root cause is that floating-point addition is not associative, so any library change that alters the order of a reduction changes the result. Here are four ways to add the same 100,000 float32 values, the precision most models run in:

```python
import math
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(0, 1, 100_000).astype(np.float32)

results = {
    "pairwise (numpy)": float(x.sum()),
    "left to right": float(np.cumsum(x)[-1]),
    "right to left": float(np.cumsum(x[::-1])[-1]),
    "exact, rounded": math.fsum(x.astype(np.float64).tolist()),
}
for name, value in results.items():
    print(f"{name:>16}: {value:.6f}")

exact = results["exact, rounded"]
errors = {name: abs(v - exact) for name, v in results.items()}
ulp = float(np.spacing(np.float32(abs(exact))))         # gap between neighbouring float32 values near the sum
assert len(set(results.values())) == 4                   # same numbers, four different answers
assert all(e < 1e-2 for e in errors.values())            # all close to the truth...
assert max(results.values()) - min(results.values()) > 100 * ulp   # ...but far more than one rounding step apart
assert errors["pairwise (numpy)"] < min(errors["left to right"], errors["right to left"])
```

All four land within about 0.002 of the true sum of roughly −90.83, yet they disagree in the fifth significant digit, by hundreds of float32 rounding steps. Pairwise summation, which NumPy uses for `sum`, lands closest, and a library that switches reduction strategy changes the last digits of every result. In a model those bits move a borderline score across a threshold, break a tie in an argmax, or compound over thousands of training steps into a visibly different checkpoint. "Works on my machine" is the symptom of shipping the code without the environment it was tested in. A container ships both. It does not make GPU kernels deterministic, but it removes a large and avoidable source of drift: different library versions on different machines.

## A Dockerfile for a model service
Here is a Dockerfile for a small scoring API in a Python project managed by uv, following the pattern in uv's Docker guide:

```plaintext
# syntax=docker/dockerfile:1
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# 1. dependencies only: they change rarely and are slow to install
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# 2. the project itself: it changes constantly and installs in a second
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "scorer.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

The structure is three tiers: a slim Debian-based Python image, the third-party packages, and finally the project's own code. Two details are easy to get wrong. The first `uv sync` uses `--no-install-project`, because the project's own source is not in the image yet; without that flag, uv tries to build the package from a `src/` directory that does not exist (and if `pyproject.toml` names a `readme`, that file has to be copied in before the second sync as well). And the server binds to `0.0.0.0`, for reasons covered below.

## Layer caching: the one Dockerfile skill
Docker builds an image one instruction at a time and caches the result of each step. On the next build, it reuses a step when the step is unchanged and every step before it was reused too. The first miss forces every later step to rerun. What "unchanged" means depends on the instruction:

- `COPY` and `ADD` compare a checksum of the files being copied. Contents count; modification times do not, so touching a file without changing it keeps the cache.
- `RUN` compares only the **command string**. Docker does not look at what the command would download.
- The others, such as `WORKDIR` and `ENV`, compare the instruction itself.

That makes ordering everything: put **cheap and stable steps before expensive and volatile ones**. The simulation below models the cache as a chain of hashes, each step's key built from its parent's key, the instruction and, for `COPY`, the contents of the files it copies. It compares the Dockerfile above with the common shortcut of `COPY . .` followed by one install.

```python
import fnmatch
import hashlib


def digest(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode() + b"\0")
    return h.hexdigest()[:16]


def build(dockerfile, files, cache, ignore=()):
    """Return (seconds spent, hit/MISS per step). cache is a set of step keys, shared across builds."""
    context = {p: c for p, c in files.items() if not any(fnmatch.fnmatch(p, pat) for pat in ignore)}
    key, seconds, log = "scratch", 0.0, []
    for instr, cost in dockerfile:
        op, *args = instr.split()
        if op == "COPY":
            sources = args[:-1]
            copied = sorted(p for p in context if any(s == "." or p == s or p.startswith(s) for s in sources))
            key = digest(key, instr, *(f"{p}={context[p]}" for p in copied))
        else:
            key = digest(key, instr)
        if key in cache:
            log.append("hit")
        else:
            cache.add(key)
            seconds += cost
            log.append("MISS")
    return seconds, log


ORDERED = [("FROM python:3.11-slim", 8), ("WORKDIR /app", 0),
           ("COPY pyproject.toml uv.lock ./", 0.1), ("RUN uv sync --locked --no-install-project", 90),
           ("COPY src/ ./src/", 0.1), ("RUN uv sync --locked", 1.5)]
SHORTCUT = [("FROM python:3.11-slim", 8), ("WORKDIR /app", 0),
            ("COPY . .", 0.5), ("RUN uv sync --locked", 91)]

files = {"pyproject.toml": "numpy>=1.24", "uv.lock": "numpy==1.26.4", "README.md": "v1",
         "src/scorer/api.py": "v1", "src/scorer/score.py": "v1", ".venv/lib/site.py": "local-env-1"}

timings = {}
for name, dockerfile in [("ordered", ORDERED), ("shortcut", SHORTCUT)]:
    cache, f = set(), dict(files)
    cold, _ = build(dockerfile, f, cache)
    f["src/scorer/score.py"] = "v2"                      # edit one line of model code
    code_edit, log = build(dockerfile, f, cache)
    f["uv.lock"] = "numpy==2.1.0"                        # bump a dependency
    dep_bump, _ = build(dockerfile, f, cache)
    timings[name] = (cold, code_edit, dep_bump)
    if name == "ordered":
        assert log == ["hit"] * 4 + ["MISS"] * 2             # the expensive dependency step stays cached
    print(f"{name:>8}: cold {cold:5.1f}s, code edit {code_edit:5.1f}s {log}, dependency bump {dep_bump:5.1f}s")

assert timings["ordered"][1] < 2 and timings["shortcut"][1] > 90    # code edit: seconds vs a full reinstall
assert timings["ordered"][2] > 90 and timings["shortcut"][2] > 90    # a real dependency change costs both
```

With the ordered file, editing `score.py` first misses at `COPY src/`, so only that copy and the quick project-only sync after it rerun, and the rebuild takes under two seconds. With the shortcut, the same edit changes the checksum of `COPY . .`, which invalidates the install that sits on top of it, so a one-line fix pays for the full dependency install again. Changing the lockfile costs a full install either way, which is correct: that is when the install really has to run.

The shortcut has a second problem. Its `COPY . .` also hashes whatever junk is in the directory, such as a local virtual environment. Recreating `.venv` on the laptop then busts the cache without a single code change. A `.dockerignore` file removes those paths from the build context altogether:

```python
cache = set()
f = dict(files)
build(SHORTCUT, f, cache)
f[".venv/lib/site.py"] = "local-env-2"                   # reinstall the laptop's virtualenv, touch no code
assert build(SHORTCUT, f, cache)[0] > 90                 # cache busted by files the image should never contain

cache, f = set(), dict(files)
build(SHORTCUT, f, cache, ignore=[".venv/*"])
f[".venv/lib/site.py"] = "local-env-2"
assert build(SHORTCUT, f, cache, ignore=[".venv/*"])[0] == 0
```

Even with careful `COPY` lines, keep a `.dockerignore` listing `.venv/`, `.git/`, data directories, checkpoints and outputs. The build context has to be scanned and shipped to the builder (all of it with the classic builder; with BuildKit, whatever the `COPY` steps ask for), so a few gigabytes of checkpoints can make a build slow before a single step runs, especially against a remote builder. A copied `.venv` built on the host can also clash with the environment the image installs.

## A cached build is not a reproducible build
Because a `RUN` step is keyed on its command text, `RUN pip install "numpy>=1.24"` stays cached on your machine until the cache is pruned or an earlier step changes, while a CI runner with an empty cache runs the same line months later and gets whatever is newest that day. Same Dockerfile, different images. The fix is a **lockfile**: resolve versions once, commit the exact pins, and make every install read the pins rather than re-resolve. The toy package index below uses release dates close to the real ones.

```python
INDEX = {"numpy": [("1.26.4", "2024-02-05"), ("2.0.0", "2024-06-16"), ("2.1.0", "2024-08-18")],
         "scipy": [("1.12.0", "2024-01-20"), ("1.14.1", "2024-08-22")]}


def vtuple(v):
    return tuple(int(part) for part in v.split("."))


def resolve(requirements, today):
    """Newest release on `today` satisfying each 'name>=version' requirement."""
    pins = {}
    for req in requirements:
        name, low = req.split(">=")
        pins[name] = max((v for v, released in INDEX[name] if released <= today and vtuple(v) >= vtuple(low)),
                         key=vtuple)
    return pins


def lock(requirements, today):
    return {"manifest": digest(*sorted(requirements)), "pins": resolve(requirements, today)}


def sync(requirements, lockfile, mode):
    """'locked' refuses a stale lockfile; 'frozen' installs the pins without checking."""
    if mode == "locked" and lockfile["manifest"] != digest(*sorted(requirements)):
        raise RuntimeError("uv.lock needs to be updated")
    return dict(lockfile["pins"])


reqs = ["numpy>=1.24"]
laptop, ci = resolve(reqs, "2024-03-01"), resolve(reqs, "2024-09-30")
assert laptop != ci                                      # unpinned: the build date picks the version

lk = lock(reqs, "2024-03-01")
assert sync(reqs, lk, "locked") == sync(reqs, lk, "locked") == {"numpy": "1.26.4"}

reqs2 = reqs + ["scipy>=1.11"]                           # add a dependency, forget to re-lock
assert "scipy" not in sync(reqs2, lk, "frozen")          # frozen: builds, silently missing scipy
try:
    sync(reqs2, lk, "locked")
    raise AssertionError("a stale lockfile should fail the build")
except RuntimeError:
    pass
```

The last two asserts are the difference between uv's two flags. `--frozen` installs exactly what the lockfile says without checking whether the lockfile still matches `pyproject.toml`, so a forgotten re-lock produces an image without the new dependency. `--locked` checks first and fails the build if the lockfile is out of date. In a Dockerfile you want the build to fail, so use `--locked`. The same reasoning applies to the base image: a tag such as `python:3.11-slim` moves as it gets patched, and pinning it by digest (`python:3.11-slim@sha256:...`) is what fixes it to one exact image.

## Running it
```plaintext
docker build -t scorer:0.3.1 .
docker run --rm -p 8000:8000 scorer:0.3.1
docker run --rm --gpus all -v "$PWD/models:/models:ro" scorer:0.3.1
docker run --rm scorer:0.3.1 python -c "import numpy; print(numpy.__version__)"
```

`--rm` removes the container when it exits; the image stays. `-p 8000:8000` forwards a host port to the container's port. The last line replaces the default command, which is a quick way to check what is actually inside an image.

Model weights can be baked into the image or mounted at run time with `-v`. Baking makes the image a single self-contained artifact that can be rolled back as one unit, at the cost of multi-gigabyte images and slow pulls. Mounting or downloading at start-up keeps images small but makes the weights a second thing to version. Small models usually go in the image; large ones usually do not.

For GPUs the split is precise: the host provides the NVIDIA driver, and the image provides the CUDA libraries (PyTorch's wheels bundle their own). The host needs the NVIDIA Container Toolkit for `--gpus` to work, and its driver must be new enough for the image's CUDA version.

## Pitfalls
- **Binding to `127.0.0.1`.** Inside a container, localhost is the container's own loopback interface, so a server listening there is unreachable through `-p`. Bind to `0.0.0.0`. This is the most common reason a container that built fine cannot be reached.
- **Full base images.** `python:3.11` is roughly a gigabyte; `python:3.11-slim` is roughly 150 MB. Pull time is real on a cold deploy or an autoscaling event.
- **Slim images lack compilers.** A package with no wheel for your platform fails to build. Use a multi-stage build: compile in a full image, copy only the virtual environment into the slim one. Alpine is a poor default for ML, since many scientific wheels target glibc and Alpine uses musl.
- **Architecture mismatch.** An image built on an Apple-silicon laptop is `arm64`; most cloud machines are `amd64`. Build with `--platform linux/amd64` or in CI.
- **Unpinned installs and moving tags.** Covered above: cached is not the same as reproducible.
- **Running as root.** Add a non-root user with `USER` so a compromised server process does not have root inside the container.
- **Secrets in layers.** A file copied in and deleted in a later step still exists in the earlier layer. Pass secrets with BuildKit's `--mount=type=secret` or at run time, never with `COPY`.

## When to reach for it
Containerize anything that runs somewhere other than the laptop it was written on: a model server, a batch scoring job, a training job on a cluster, a reproducible evaluation. For exploratory notebooks a lockfile alone is often enough. The deployment questions around the container (rollout, autoscaling, monitoring) belong to the wider [ML infrastructure](../ml-infrastructure/) and [system design](../ml-system-design-framework/) posts.

The simulations model only the cache key and a toy package index, and the build times are costs I assign to each step, so they show which steps rerun, not how long a real build takes. One source of drift stays open: an image fixes the library versions it was built with but does not make GPU kernels deterministic, and none of the code here runs on a GPU.

The layer caching rules follow Docker's documentation on the build cache, and the uv pattern (`--no-install-project`, cache mounts, `--locked`) follows the Docker integration guide in uv's documentation.
