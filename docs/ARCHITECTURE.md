# Architecture

A map of the site, top down: what is served, what writes it, what checks it, and how a change reaches the live
site. Written 2026-11-11 from the code. When the code and this file disagree, the code is right; fix this file.

Specs say why each piece looks the way it does (`CAPABILITY-MAP.md` indexes them); `CONSTRAINTS.md` holds the rules;
`docs/adr/` holds the decisions. This file only says where things are.

## The short version

- **Static files, served as they are.** GitHub Pages serves `main` with `.nojekyll`, so every file in the repo is a
  public URL and nothing runs on a server. No framework, no bundler, no `package.json`.
- **Markdown in, committed HTML out.** Four Python scripts in `tools/` turn Markdown sources into pages, feeds and a
  search index. Their output is committed, and a gate fails if it differs from a fresh build.
- **One gate script.** `scripts/gates.sh` runs every check, the same way on a laptop and in CI. `main` is protected:
  a PR needs a green `gates` check before it can merge, and merging is deploying.

```
tools/posts/src/*.md ──┐                                   ┌─> posts/<slug>/index.html, posts/index.html
tools/leetcode/src/*.md┴─> tools/{posts,leetcode}/build.py ┴─> leetcode/<slug>/index.html, leetcode/index.html
                                   │ (both lists)
                                   ├─> tools/feeds/build.py  ─> rss.xml, sitemap.xml, robots.txt, llms.txt,
                                   │                            <section>/<slug>/index.md
                                   └─> tools/search/build.py ─> search-index.json
hand-written: index.html + room.js (home), about.html, 404.html, posts/<long-form slug>/index.html, style.css

branch ─> sh scripts/gates.sh ─> push ─> PR ─> CI runs gates.sh ─> rebase merge to main ─> Pages serves main
```

## Pages

| Page | File | Written by | Notes |
|---|---|---|---|
| Home, "the common room" | `index.html`, `room.js`, `vendor/three/` | hand | A low-poly 3D room drawn with three.js (`SPEC-room.md`). Its objects are real `<a>` links laid over the canvas, so navigation never depends on the picture. A lo-fi text view (`SPEC-lofi.md`) shows for reduced motion, without WebGL, or by the toggle; the choice is kept in `localStorage`. three.js loads only in the room view. |
| Posts list | `posts/index.html` | `tools/posts/build.py` | Topic cards, then one `section.group` per topic (`SPEC-posts-index.md`). The URL `#fragment` is the filter, in CSS only. Long-form posts first, then the topic groups, then MusicGen. |
| LeetCode list | `leetcode/index.html` | `tools/leetcode/build.py` | Same card-and-group layout. |
| Topic posts (55) | `posts/<slug>/index.html` | `tools/posts/build.py` from `tools/posts/src/<slug>.md` | Reading shell (`SPEC-reading-shell.md`): title block, side column (details and contents), one text column. |
| LeetCode posts (31) | `leetcode/<slug>/index.html` | `tools/leetcode/build.py` from `tools/leetcode/src/<slug>.md` | Same shell. |
| Long-form posts (4) | `posts/<slug>/index.html` | hand | Listed by `LONGFORM` in `tools/posts/build.py`. No Markdown source, so no `index.md` copy. |
| Unlisted pages (6) | one folder, not named here | `tools/book/build.py` | An unlisted page reached from an object in the room. `noindex`, and in no feed, index or list; `check.py` fails if any page but the room links it. |
| MusicGen posts (2) | another repo, served at `/musicgen-prune-inference/` | that repo | Listed by `MUSICGEN` in `tools/posts/build.py`; `EXTERNAL` in `scripts/check.py` lets links to it pass. |
| About | `about.html` | hand | Each role is a `section` with an id; the room's three portraits link to those ids. |
| Not found | `404.html` | hand | Pages serves it for any missing path. |
| Feeds | `rss.xml`, `sitemap.xml`, `robots.txt`, `llms.txt`, `<section>/<slug>/index.md` | `tools/feeds/build.py` | `SPEC-post-feeds.md`. Deterministic: no clock is read, so a rebuild writes the same bytes. |
| Search | `search.mjs`, `search-core.mjs`, `search-index.json` | hand; the index by `tools/search/build.py` | `SPEC-site-search.md`. On the two list pages only. `search.mjs` inserts the box (none without JavaScript), fetches the index on first use, and ranks with `search-core.mjs`, which has no DOM so `node --test` can check it. Titles, summaries, tags and headings only, not full text. |

Shared by every page:

- `style.css`, the one stylesheet. Reading-page tokens sit on `:root`; the room's own palette and fonts sit under
  `body.home` (`SPEC-blog-theme.md`). Its header comment lists its sections in order.
- `fonts/`: self-hosted OFL fonts. Source Serif 4 and Inter for reading pages (both preloaded on every page but the
  home), Cinzel for the room.
- `post.js`: on post pages only, renders `\( \)` and `\[ \]` maths with KaTeX and highlights code with highlight.js.
  Both come from CDNs at pinned versions; `check.py` fails an unpinned one.

## The builds (`tools/`)

Run from the repo root. The feeds and search builds read both post lists, so they run last. The posts build counts
the `leetcode/*/` folders for its "has N short posts" line, so after adding a LeetCode post, run the LeetCode build
first. `gates.sh` runs posts first, which works because `leetcode/` is committed.

```
python3 -I tools/leetcode/build.py .  [slug ...]   # no slugs = every source
python3 -I tools/posts/build.py .     [slug ...]
python3 -I tools/book/build.py .                     # the unlisted pages; no other build reads its sources
python3 -I tools/feeds/build.py .
python3 -I tools/search/build.py .
```

- **`tools/leetcode/build.py` is the engine.** It holds the page template (`HEAD`, `FOOT`, `article`), the small
  Markdown renderer (`render`, `inline`), the head tags (canonical link, feed link, JSON-LD), the list layout
  (`topic_list`, `row`), and `run_code`. The renderer handles only what the sources use: `##`/`###` headings,
  paragraphs, `-` and `1.` lists, fenced code, tables, `**bold**`, `[text](url)`, `` `code` ``, and `\( \)`/`\[ \]` maths.
- **`run_code` runs every post's Python.** All `python` blocks of one post run as one script, with each numeric
  library held to one thread and a 120 s timeout. An error or a failing assert stops the build, so every asserted
  claim is re-checked on each build. A printed number with no assert is not checked.
- **`tools/posts/build.py` reuses that engine.** It loads its own copy of the LeetCode build and points it at the posts
  sources, and adds `GROUPS`, `LONGFORM` and `MUSICGEN`. `tools/book/build.py` does the same for the unlisted
  pages, without the head tags.
- **`tools/feeds/build.py` and `tools/search/build.py`** read the builds' lists: the feeds build loads each build on
  its own, the search build loads the posts build and reaches the LeetCode one through it. A new post appears in the
  feeds and the search index with no change to either.
- A source file is front matter (`title`, `summary`, `group`, `tags`, optional `order`) between `---` lines, then the
  body. A `group` that is not in its build's `GROUPS` still gets a page but no list row, so `check.py` reports the post
  as not listed; `order` sorts within the group.

`tools/dev/` holds helpers that the site does not use: `same_articles.py` (compares article text with a git ref, run
after any layout change), `shoot.sh` and `shoot3d.sh` (headless Chrome screenshots), `frame.html` (a phone frame for
them), `hooks/git-guard.sh` (a Claude Code Stop hook, wired in `.claude/settings.json`, that warns about uncommitted
work) and `workflows/` (the agent workflows that wrote the room and the topic posts).

## The gates (`scripts/`, CI)

`sh scripts/gates.sh` is the only place the gates are spelled out (`CONSTRAINTS.md` names them). In order:

| Step | What it runs | Fails when |
|---|---|---|
| privacy | `scripts/check.py --denylist-only` | a private term from `scripts/denylist.local.txt` is in any served file's text or name |
| site | `scripts/check.py` | a broken link or `#fragment`, an incomplete page head, an unpinned CDN, a post missing from its list, a card count that differs from its rows, a feed or search entry missing, `index.html` plus `style.css` over 150 KB (`room.js` and three.js are not counted), a hand-written count out of date |
| check tests | `python3 -m unittest test_check` | the checker regressed, or the theme's tokens lost contrast or the room's pinned colours changed |
| search tests | `node --test scripts/test_search.mjs` | ranking cases or golden queries fail |
| post wording | `grep` over `tools/posts/src` | vault or Obsidian wording |
| post, leetcode, feeds, search builds | the four builds above | a post's code fails |
| rebuild clean | `git status --short` | the builds changed a committed file: generated HTML was edited by hand or not rebuilt |

Privacy runs first and stops the run, so no later step can print a private term. `check.py` scrubs every line it
prints; commands outside it do not, which is why the wording step prints only names and counts.

The denylist never enters git (`.gitignore`). CI gets it from the `SITE_DENYLIST` secret, base64-encoded, written
to disk in one step. An empty list fails closed. `check.py` refuses `--show-terms` when `CI` is set.

`.github/workflows/gates.yml` runs `gates.sh` on every PR to `main` and on every push to `main`, on Python 3.12 and
Node 26. Actions are pinned to commit SHAs, and the post code's packages install from `requirements-ci.txt`, locked
by hash and wheels only, because post code runs while the denylist is on disk. CI checks; it does not deploy.

## How a change reaches the live site

1. Branch from `origin/main` and unset the upstream, so a bare `git push` cannot target `main`.
2. Edit a source, then run its build (`build.py . <slug>` for one post), then the feeds and search builds. Commit the
   source and the generated files together, with the noreply identity (see `CLAUDE.md`).
3. After a layout change, run `python3 tools/dev/same_articles.py origin/main` to show no article text moved.
4. Run `sh scripts/gates.sh` (about 4 minutes). A pushed branch is public before CI runs, so this is the first check.
5. Push and open a PR. CI runs the same script with the secret denylist; `main`'s protection requires it green.
6. Merge with rebase, not squash. GitHub Pages then publishes `main` as it is.

Common changes:

- **New topic or LeetCode post:** add `tools/<section>/src/<slug>.md`, then run all four builds. The list, feeds and
  search index follow. A new LeetCode post also changes the hand-written counts on the home page; the `site` check
  names each one that is out of date.
- **New long-form post:** write `posts/<slug>/index.html` in the reading shell (copy one), add it to the top of
  `LONGFORM`, run the posts, feeds and search builds (the search build reads the page's headings), and make the home's
  `.lofi-latest` match the first three posts of the posts list (`check.py` checks).
- **Another project repo's posts:** a code change, since `MUSICGEN` serves only the MusicGen repo. Its name goes in
  `EXTERNAL` in `scripts/check.py`, and its posts need a section of their own in `tools/posts/build.py`, a heading in
  `tools/feeds/build.py` and a label in `tools/search/build.py`, as MusicGen has, and its URLs a tuple like
  `MUSICGEN` in `scripts/check.py`, which is how the feed, sitemap and search checks see them. Never create a top-level
  folder with a project repo's name: it would hide that project's site.

## Follow-ups

Known gaps, with why each is not done yet.

- **No security headers.** Pages cannot set response headers (ADR 0001). Accepted for a static site with no
  sign-in and no forms.
- **Unlisted is not private.** The unlisted pages rely on `noindex`, which only well-behaved crawlers honour, and
  their link and sources are public in this repo. Accepted: they hold nothing that may not be public.
- **Performance is measured once, not checked.** First paint, draw calls and frame rate are in `CONSTRAINTS.md` under
  "Measured, not yet enforced". A Lighthouse step in CI would need a browser in the job.
