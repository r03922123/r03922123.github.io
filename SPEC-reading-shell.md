# Spec: reading-shell

Module of `CAPABILITY-MAP.md` (version 4). Depends on `blog-theme`. Status: **approved** 2026-11-10.

## Objective

Every article page has the layout of a research publication: a centred title, one 720 px text column, and a side
column for the post's details and contents. Code, the heart of these posts, gets the extra width it needs.

Users: readers of 15-minute technical posts, on phones and on wide screens. The common room is unchanged.

| Requirement | Done means |
|---|---|
| Shared top bar | Reading pages have a sticky bar: links at left, name centred, GitHub at right. Home's header looks as it does today |
| Title block | Topic link and `h1`, centred above the text column. Long-form posts have no topic group, so theirs reads "Long-form" and links to `/posts/` |
| One text column | Body text is at most 720 px wide at every screen size |
| Side column | From 1200 px: published, reading time, topic, tags and contents, sticky at the right. Below: details under the title, contents as today's card |
| Code and wide tables get room | From 1200 px, `pre` and tables of 5 or more columns hang left of the text column, up to 1040 px wide. Maths stays in the text column and scrolls |
| Links land correctly | A contents link or a room door puts its heading below the sticky bar |
| Doors cannot break silently | `check.py` fails on a local `#fragment` with no matching id |

## What the code forces

| Finding | Consequence |
|---|---|
| 530 of 585 code blocks have a line over 79 characters, about what 720 px holds. Median longest line 101, 90th percentile 115 | Without extra width, 9 blocks in 10 scroll sideways. 1040 px holds about 116 characters |
| Right of the text is taken by the side column, and left is empty | `pre` extends left only, so it never runs under the side column |
| 14 of 106 tables have 5 or more columns | `table:has(th:nth-child(5))` makes those wide with CSS alone. Without `:has`, they scroll as they do today |
| Two templates produce 88 pages (`tools/leetcode/build.py`, `tools/posts/build.py`); the 4 long-form pages, About and 404 are hand-written | Change two templates, regenerate, and edit the long-form pages by hand in the same PR, or they would meet the new grid with the old markup |
| `.wrap` caps every page at 760 px, and home uses it 4 times | Scope the wider layout to article pages, `body:has(.post-body)`. Home, the two list pages, About and 404 keep 760 px |
| `style.css:60` sets `scroll-margin-top: 24px` | A sticky bar would cover the target heading. Raise it to the bar's height plus a gap |
| `link_ok` in `scripts/check.py:78` drops the fragment. Home links to `about.html#kkcompany`, `#tiktok` and `#htc` | Add the fragment check. Today 708 local fragment links resolve and 0 are broken, so it starts green |
| Long-form pages place the contents after the intro paragraph | Move it into the side column; the intro stays first in the text |
| `post-voice` edits the same 4 long-form files | Whichever merges second rebases. The shell changes only the header and contents, not paragraphs |

## Tech Stack

`style.css` with CSS grid, the two Python templates, and no new JavaScript or package. The fonts from `blog-theme` get
preload tags here, for Source Serif 4 upright and Inter only, since italic is used on few pages. They go through the templates' `assets`
parameter, and by hand on the 6 hand-written pages. The bar's elements are reordered to match the screen (links, name, GitHub) in the
template and on those 6 pages. Keep the two link strings that `tools/posts/build.py:22-24` replaces, since it asserts on them.

## Layout

```
1200 px and wider                                      Under 1200 px
                 [ links    name    GitHub ]           [ links  name  GitHub ], wrapping on phones
                      Topic · Title                    Topic · Title
 [ pre, up to 1040 px     ][ text 720 ][ side 240 ]    details row
                                                       contents card
                                                       text, full width up to 720 px
```

Below 700 px the bar is not sticky, because on a phone it would take space from the text. Not taken from the
reference: dropdown pill menus, hero image, fade-in on scroll, related articles, the dark end band.

## Commands

```
All gates:   sh scripts/gates.sh
Check tests: (cd scripts && python3 -m unittest test_check)
Rebuild:     python3 -I tools/posts/build.py . && python3 -I tools/leetcode/build.py .
Preview:     python3 -m http.server 8000
```

## Project Structure

```
style.css                    → bar, grid, title block, side column, wide code, scroll margin
tools/leetcode/build.py      → HEAD and post template; regenerates 31 posts
tools/posts/build.py         → post template; regenerates 55 posts
posts/<slug>/index.html      → the 4 long-form pages, by hand
about.html, 404.html         → plain headings in place of the frames; ids unchanged
scripts/check.py             → fragment check
scripts/test_check.py        → its failing cases first
tools/dev/same_articles.py   → drops header, aside, nav and .crumb before comparing
```

Two PRs, each opening with `Takeover phase: build (owner-approved exception; repo is in understand)`:
the fragment check alone, then the shell. The 300-line limit counts sources; generated pages are mirrors.

## Code Style

Semantic markup with no layout classes in the content, generated the same way for every post.

```html
<article class="post-body">
  <header class="post-head"><p class="crumb">…</p><h1>…</h1></header>
  <aside class="post-side">
    <dl class="details"><dt>Published</dt><dd>November 1, 2026</dd><dt>Topic</dt><dd>Probability</dd></dl>
    <nav class="toc" aria-label="Contents">…</nav>
  </aside>
  <div class="prose">…</div>
</article>
```

## Testing Strategy

- Test first: `test_check.py` cases link to `page.html#missing`, `/dir/#missing` and a same-page `#missing`, and each
  must fail before the check exists. Same-page links need their own case: `link_ok` returns early on them, and they are
  703 of the 708.
- Browser check on the local preview, on one topic post, one LeetCode post, one long-form post (`posts/position-bias/`, which has a wide table and a long code line), one list page, About and
  404, at
  375, 768, 1200, 1440 and 1920 px, in light and dark:
  - The page never scrolls sideways: `scrollWidth <= innerWidth`.
  - On posts, the text column is at most 720 px at every width, and exactly 720 px from 1200 px up.
  - At 1440 px, an overflowing `pre` and a table of 5 or more columns are wider than 720 px, end at the text column's
    right edge, stay clear of the side column, and start at least 16 px from the page edge.
  - From 1200 px, `.post-side` is visible and stays in view while scrolling. Below 1200 px, the details row and
    contents card show instead.
  - The list page stays at 760 px.
  - Clicking a contents link, and opening `about.html#tiktok`, leaves the heading below the bar.
- Home: computed styles and lo-fi screenshots match `main`, as in `blog-theme`.
- Screenshots of each page at 375 and 1440 px in the PR.

## Boundaries

- Always: keep heading ids and About's role ids, since links and the room's doors depend on them. Keep the same reading
  order in the markup as on screen, for keyboard and screen-reader users.
- Ask first: new JavaScript, such as highlighting the current section; related articles or any other new content.
- Never: change `.rm-*` rules or home's markup. Change post text, code or tables. Copy the reference's logo or menus.

Out of scope: list pages (`posts-index`), real per-post dates, search.

## Success Criteria

- [ ] `blog-theme` is merged, and `sh scripts/gates.sh` exits 0 on the branch tip
- [ ] The fragment test failed before the check existed and passes after it
- [ ] Every browser check above passes at all five widths, with output in the PR
- [ ] Home: computed styles and lo-fi screenshots match `main`
- [ ] The rebuild leaves no uncommitted generated page
- [ ] `python3 tools/dev/same_articles.py main` reports no change. It needs the edit above first, because today it
  compares the whole `<article>` and the new side column would flag every post

## Open Questions

None.
