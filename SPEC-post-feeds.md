# Spec: post-feeds

Module of `CAPABILITY-MAP.md` (version 4). Depends on `site-baseline`. Status: **approved** 2026-11-10.

## Objective

Readers, search engines and language models can find every post without rendering a page. Nothing a visitor sees
changes: this module adds machine-readable files and tags in the page head.

Users: feed readers, crawlers, LLM tools, and people who want a post as Markdown.

| Requirement | Done means |
|---|---|
| `rss.xml` | 92 items (90 posts, 2 MusicGen), newest first, valid RSS 2.0 |
| `sitemap.xml` | 96 absolute URLs: home, About, 2 lists, 90 posts, 2 MusicGen. No 404 page |
| `robots.txt` | Allows everything and names the sitemap |
| `llms.txt` | The site's name, a one-line summary, and every post linked by section |
| Markdown beside each post | `posts/<slug>/index.md` and `leetcode/<slug>/index.md` for the 86 posts with a source, with `# <title>` in place of the front matter |
| Page head | Canonical link on 94 pages; feed link; on posts, JSON-LD, and a Markdown alternate on the 86 |
| Same output every run | Two rebuilds in a row change nothing |

## What the code and the standards force

| Finding | Consequence |
|---|---|
| RSS defines item `author` as an email address and says to omit it for a single-author weblog | No `author`, so the owner's email stays out of the feed |
| Google ignores `priority` and `changefreq`, and uses `lastmod` only if it is consistently accurate | The sitemap has `loc` only. There is no accurate per-post modified date |
| The rebuild gate fails on anything that changes between runs | No "now" timestamp anywhere: no `lastBuildDate`, and ties sort by listing order |
| llms.txt's convention for a URL with no filename is `index.md` beside the page | That path, with `rel="alternate" type="text/markdown"` |
| Posts are dated by constants: topic posts November 1, LeetCode October 31; long-form and MusicGen dates are in their listings. First-commit dates match | Use those dates. Real per-post dates are out of scope |
| Article structured data has no required property; `dateModified` is only recommended | `BlogPosting` with `headline`, `datePublished` at +08:00, `description`, and `author` as a `Person` with name and About URL. No `dateModified` or `image` |
| `text_files` in `check.py` scans every served text file | The denylist already covers the new files. No change needed |
| `.nojekyll` also serves the specs and `tools/` | The sitemap lists real pages only, so they are not advertised |
| The templates' `assets` parameter is already inserted into the head | Add tags through it; the `HEAD` strings and their `assert` stay as they are |
| `posts/<slug>/` links in sources are relative, and `index.md` sits at the same depth as `index.html` | Links in the Markdown resolve as they do on the page |

## Tech Stack

Stdlib Python. `tools/feeds/build.py` loads the post lists from `tools/posts/build.py` and `tools/leetcode/build.py` with
`importlib`, each separately, because the loader at `tools/posts/build.py:14-17` points `lc.HERE` at the posts folder. It
writes every feed file. It runs after both builds.

## Commands

```
All gates:   sh scripts/gates.sh
Feeds:       python3 -I tools/feeds/build.py .
Check tests: (cd scripts && python3 -m unittest test_check)
Live check:  curl -sI https://r03922123.github.io/rss.xml
```

## Project Structure

```
tools/feeds/build.py         → rss.xml, sitemap.xml, robots.txt, llms.txt, each index.md
tools/leetcode/build.py      → head_tags(); passed through assets on post and list pages
tools/posts/build.py         → the same
posts/<slug>/index.html      → the 4 long-form pages, by hand
index.html, about.html       → head tags only
scripts/check.py             → feed rules, active once tools/feeds/build.py exists
scripts/test_check.py        → one failing case per rule
scripts/gates.sh             → runs the feeds build; rebuild gate covers the new paths
```

Two PRs, each opening with `Takeover phase: build (owner-approved exception; repo is in understand)`: the feed files and
their checks, then the head tags across pages.

## Code Style

Match the build scripts: stdlib, escape everything that goes into XML, comments that give the reason.

```python
def rss_date(text):
    """'November 1, 2026' to RFC 822, at midnight Taipei time, so the same input always gives the same bytes."""
    return datetime.strptime(text, "%B %d, %Y").strftime("%a, %d %b %Y 00:00:00 +0800")
```

## Testing Strategy

- Test first, one failing case each in `test_check.py`: a listed post missing from the sitemap; malformed XML; an item
  with no `pubDate`; an email in the feed; a canonical link that does not match the page's URL; JSON-LD that does not
  parse; a post source with no `index.md`; a `.md` that keeps its front matter; a page with no canonical link; a page with no
  feed link; a post with no JSON-LD; a source-backed post with no Markdown alternate.
- The check requires the outputs whenever `tools/feeds/build.py` exists, so deleting a feed fails rather than passing.
- Stability: run the build twice and show `git status` is clean after the second.
- After merge, on the live site: each file returns 200 with a sensible content type, and the W3C Feed Validator and
  Google's Rich Results Test pass. Results go in the learning log.

## Boundaries

- Always: use absolute `https://r03922123.github.io/` URLs in feeds and canonical links.
- Ask first: a new crawler rule in `robots.txt`; full post text in the feed; a new package.
- Never: put an email in a feed or JSON-LD. Invent dates. Change what a visitor sees, or anything in the MusicGen repo.

Out of scope: real per-post dates, Open Graph images, search, and security headers (Pages cannot set them).

## Success Criteria

- [ ] `site-baseline` is merged, and `sh scripts/gates.sh` exits 0 on the branch tip
- [ ] Each new check failed in its test before the rule existed, and passes now
- [ ] `rss.xml` has 92 items and `sitemap.xml` 96 URLs, both parsed by `xml.etree`
- [ ] 86 `index.md` files exist, none with front matter
- [ ] 94 pages have one canonical link and a feed link; 90 posts have JSON-LD; the 86 have a Markdown alternate
- [ ] Two consecutive rebuilds leave `git status` clean

Post-merge follow-up, not a merge gate: the live checks above.

## Open Questions

None.
