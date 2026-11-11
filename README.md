# r03922123.github.io

Personal site: a 3D common room as the home page, technical posts on machine learning, a LeetCode patterns series
and an About page. Static HTML served by GitHub Pages from `main`; a Python build turns Markdown sources into the
pages, and the generated HTML is committed.

- **Where things live and how a change goes live:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **Rules and gates:** [`CONSTRAINTS.md`](CONSTRAINTS.md). **Decisions:** [`docs/adr/`](docs/adr/).
- **What each part should do:** the specs, indexed by [`CAPABILITY-MAP.md`](CAPABILITY-MAP.md).

## Checks

```
sh scripts/gates.sh              # every gate, as CI runs it; about 4 minutes
python3 -m http.server 8000      # preview at http://localhost:8000
```

`scripts/check.py` scans every served file for private terms listed in `scripts/denylist.local.txt`. That file is
gitignored on purpose, since everything in this repo is public, and the check refuses to run without it. To recreate
it on a new machine, write one term per line (lines starting with `#` are comments): personal contact details, and
anything else that must never appear on the site.
