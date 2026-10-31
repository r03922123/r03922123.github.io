# r03922123.github.io

Personal site: an interactive desk-at-night home page, long-form technical posts, a LeetCode patterns
series and an About page. Plain static HTML served by GitHub Pages from `main`, with no build step.

## Layout

```
index.html              → home: the desk scene (and its lo-fi text view)
posts/index.html        → posts index; posts/<slug>/ holds one long-form post each
leetcode/index.html     → LeetCode series page; leetcode/<slug>/ holds one pattern post each (generated)
about.html, 404.html    → About, not-found page
style.css, post.js      → the one stylesheet; KaTeX + highlight.js setup for post pages
tools/leetcode/         → Markdown sources and the renderer for the LeetCode series
scripts/check.py        → site checks (privacy, links, page head, pinned CDNs, listings)
SPEC*.md, CAPABILITY-MAP.md, tasks/ → specs and plans
```

## Adding a post

- **Long-form, in this repo:** create `posts/<slug>/index.html` (copy an existing post for the head, header
  and footer), then add an entry to `posts/index.html`. The check fails until it is listed.
- **Long-form, in its own repo:** the project repo deploys itself at `/<repo>/`; add an entry to
  `posts/index.html` linking there. Never create a top-level folder with the same name as a project repo:
  it would shadow that project's site.
- **LeetCode pattern:** write `tools/leetcode/src/<slug>.md` (front matter: title, summary, group, order,
  tags), then run `python3 -I tools/leetcode/build.py .` from the repo root. The renderer runs every Python
  block in the post and stops if an assert fails; it rewrites the series page too.

## Checks

```
python3 scripts/check.py                       # exits 0 when clean
(cd scripts && python3 -m unittest test_check) # tests for the checks and the theme's contrast
python3 -m http.server 8000                    # preview at http://localhost:8000
```

`check.py` also scans every served file for private terms listed in `scripts/denylist.local.txt`. That file
is gitignored on purpose, since everything in this repo is public; it refuses to run without it. To recreate
it on a new machine, write one term per line (lines starting with `#` are comments): personal contact
details, and anything else that must never appear on the site.
