# Spec: posts-index

Module of `CAPABILITY-MAP.md` (version 4). Depends on `blog-theme`. Status: **draft**, 2026-11-10.

## Objective

The Posts and LeetCode pages let a reader see what is here and narrow it to one topic, the way a research publication
list does. The lists stay plain text: the site has no images, and these posts sell on their titles and summaries.

Users: a visitor deciding what to read, and anyone sharing a link to one topic.

| Requirement | Done means |
|---|---|
| Topic cards | One card per section, with name and post count: 9 on Posts, 6 on LeetCode |
| Article count | The page states its total: 61 on Posts, 31 on LeetCode |
| Filter | Choosing a card shows only that topic; "All topics" shows everything |
| Shareable and reversible | `/posts/#probability` opens filtered; the browser's back button undoes a filter |
| Rows | Every post is a row: title and summary, plus date and length where the build has them |
| Nothing is lost | Without `:has` support, every section stays visible and cards jump to their section |

## What the code forces

| Finding | Consequence |
|---|---|
| `check.py` compares `.lofi-latest` with the first three `/posts/<slug>/` links on the index | Long-form rows stay first. Cards link to `#fragments`, so they do not count as post links |
| `check.py` requires a `#musicgen` section that links both MusicGen posts | It stays, as a filterable section |
| The 4 long-form posts sit outside any section, as cards | Wrap them in `<section class="group" id="long-form">` so they can be filtered |
| LeetCode sections are a bare `<h2 id>` followed by a list | Change to the same `<section class="group">`. Heading ids do not change |
| `class="post"` is used only on `posts/index.html` | Its CSS is removed with the cards |
| Nothing links into a list section today | Making `#musicgen` show one section breaks no existing link |
| Summaries already exist for all topic and long-form posts | No new copy is needed; cards have no descriptions, unlike the reference's |
| 55 posts share one hard-coded date | No "Most recent" sort; sections keep `order` |

## Tech Stack

The two Python templates and `style.css`. No JavaScript: the URL fragment is the filter state, which gives shareable
links and a working back button for free.

```css
/* A section is the URL's target, so hide its siblings. "All topics" links to #topics, which is not a section. */
.groups:has(.group:target) .group:not(:target) { display: none; }
```

Each page also gets one generated rule that marks the chosen card, because its selectors come from the section list.

## Commands

```
All gates:   sh scripts/gates.sh
Check tests: (cd scripts && python3 -m unittest test_check)
Rebuild:     python3 -I tools/posts/build.py . && python3 -I tools/leetcode/build.py .
Preview:     python3 -m http.server 8000
```

## Project Structure

```
tools/posts/build.py     → cards, count, long-form and MusicGen as rows; regenerates posts/index.html
tools/leetcode/build.py  → cards, count, sections; regenerates leetcode/index.html
style.css                → topic cards, rows, filter rule; removes .post
scripts/check.py         → every section has a card, and each card's count equals its rows
scripts/test_check.py    → its failing cases first
```

One PR, opening with `Takeover phase: build (owner-approved exception; repo is in understand)`.

## Code Style

Generated markup, the same on both pages.

```html
<nav class="topics" id="topics" aria-label="Topics">
  <a href="#probability"><b>Probability</b><span>9 posts</span></a>
</nav>
<p class="count">61 articles</p>
<div class="groups">
  <section class="group" id="probability">
    <h2>Probability</h2><a class="all" href="#topics">All topics</a>
    <ul class="series"><li><a href="…">Title</a><span>Summary</span></li></ul>
  </section>
</div>
```

## Testing Strategy

- Test first, in `test_check.py`: a section with no card fails, and a card saying 3 posts over 2 rows fails.
- Browser check on the local preview, on both pages, at 375, 768 and 1440 px, in light and dark:
  - Each card leaves only its section visible, with the right number of rows, and marks itself as chosen.
  - "All topics" and the back button restore every section.
  - Opening `/posts/#probability` directly starts filtered; the page never scrolls sideways.
  - Cards work with the keyboard and show a focus ring.
- Screenshots of the full list and one filtered view in the PR.

## Boundaries

- Always: keep long-form rows first, and keep every section id.
- Ask first: JavaScript on these pages; a new section or a changed order.
- Never: add a gallery, carousel or cover image (the map rules them out for now), a date sort on one shared date, or
  post text changes.

Out of scope: the search box (`site-search`), real per-post dates, the top bar and page width (`reading-shell`).

## Success Criteria

- [ ] `blog-theme` is merged, and `sh scripts/gates.sh` exits 0 on the branch tip
- [ ] Both new `check.py` cases failed before the rule existed and pass after it
- [ ] Every browser check above passes at all three widths, with output in the PR
- [ ] `check.py` still confirms every post is listed, `#musicgen` is intact and `.lofi-latest` matches
- [ ] The rebuild leaves no uncommitted generated page

## Open Questions

None.
