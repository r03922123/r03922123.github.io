# Spec: site-search

Module of `CAPABILITY-MAP.md` (version 4). Depends on `posts-index`. Status: **approved** 2026-11-10.

## Objective

A reader who knows the term they want, such as "CUPED" or "knapsack", reaches the right post in one step. Search runs
in the browser from an index built with the site, so there is no server and no query leaves the page.

Users: readers on the Posts and LeetCode pages, across all 92 entries.

| Requirement | Done means |
|---|---|
| Search box | On both list pages; it finds posts from either section |
| Results | Ranked rows with title, summary and section, at most 20, with a spoken result count |
| Finds the right post | The golden queries below return the expected post in the top 3 |
| Stays light | The index is under 100 KB uncompressed, and is fetched only when the box is first used |
| Shareable | `?q=cuped` opens with results; typing does not flood the back button |
| Nothing depends on it | Without JavaScript, or if the index fails to load, the lists work as before and no dead box remains |

## What the prototype and code force

Prototype on the real posts, 2026-11-10, over 90 entries (the 2 MusicGen entries add little):

| Finding | Consequence |
|---|---|
| Title, summary, tags, section and headings: 62.9 KB, 21.0 KB gzipped. With full text: 1514 KB, 522 KB gzipped | Index the first set. Full text costs 25 times as much |
| Exact and prefix matching, every query word required, weights title 5, tags 3, headings 2, summary 1: 13 of 13 expected posts in the top 3 | No search library or new package. The 13 queries were chosen by the author of the design, so they guard against regressions and do not prove quality |
| Titles and summaries hold code spans and `\( \)` maths | Strip that markup when building the index, so results show plain text |
| `posts-index` makes the URL fragment the topic filter | Search uses `?q=`. While a query is present, results replace the sections and the filter is ignored |
| `posts-index` has no JavaScript, and lists it under "ask first" | Approving this spec is that approval, for these two pages only |
| The repo has no JS tests; Node's built-in runner needs no package | `node --test`. CI adds `actions/setup-node`, pinned by SHA, with Node 26, so its version cannot drift |
| The page's title and summary text comes from files in this repo, but a query comes from the URL | Render with `textContent` only, so a crafted `?q=` cannot inject markup |
| `check.py` scans every served text file for private terms | The index is covered already |

## Tech Stack

Vanilla ES modules with no dependency, loaded only on the two list pages through the templates' `assets` parameter.
`tools/search/build.py`, stdlib Python, writes the index after the post builds, with no timestamps.

## Commands

```
All gates:    sh scripts/gates.sh
Index:        python3 -I tools/search/build.py .
Search tests: node --test scripts/test_search.mjs
Check tests:  (cd scripts && python3 -m unittest test_check)
Preview:      python3 -m http.server 8000
```

## Project Structure

```
search-core.mjs              → rank(entries, query): pure, so Node can test it
search.mjs                   → box, lazy fetch, results, ?q= handling
search-index.json            → generated; committed like the other generated files
tools/search/build.py        → builds the index
tools/posts/build.py, tools/leetcode/build.py → add the module to the list pages
style.css                    → box and result rows, using existing tokens
scripts/test_search.mjs      → ranking cases and golden queries
scripts/check.py             → every listed post is in the index
scripts/gates.sh, .github/workflows/gates.yml → run the search tests and the index build
```

One PR, opening with `Takeover phase: build (owner-approved exception; repo is in understand)`.

## Code Style

Pure ranking, apart from the DOM, with reasons in comments.

```js
// Every query word must match, so adding a word narrows the results.
export function rank(entries, query) {
  const words = tokens(query);
  if (!words.length) return [];
  return entries.map((e) => [score(e, words), e]).filter(([s]) => s > 0)
    .sort((a, b) => b[0] - a[0] || a[1].u.localeCompare(b[1].u)).slice(0, 20).map(([, e]) => e);
}
```

## Testing Strategy

- Test first, in `test_search.mjs`: an exact match outranks a prefix match; a title outranks a summary; all words are
  required; matching ignores case; an empty or punctuation-only query returns nothing; ties sort the same way every run.
- Golden queries on the real index, each with the post that must appear in the top 3:

| Query | Post | Query | Post |
|---|---|---|---|
| `dropout` | `dropout` | `two tower` | `two-tower-retrieval` |
| `cuped` | `ab-testing` | `vector database` | `vector-databases` |
| `position bias` | `position-bias` | `bayes` | `conditional-probability-bayes` |
| `knapsack` | `knapsack` | `poisson` | `poisson-distribution` |
| `union find` | `union-find` | `flashattention` | `flashattention` |
| `peeking` | `ab-testing` | `sql window` | `sql-cheatsheet` |
| `cold start` | `cold-start` | | |

- Test first, in `test_check.py`: a listed post missing from the index fails.
- Browser check on the local preview, on both pages, at 375 and 1440 px, in light and dark:
  - The index is requested only after the box gets focus or the URL has `?q=`.
  - `?q=<img src=x onerror=alert(1)>` shows as text and runs nothing.
  - Clearing the box restores the sections, and any topic filter in the URL.
  - With the index URL blocked, a message appears and the lists still work; with JavaScript off, no box is shown.
  - A screen reader's live region announces the result count.

## Boundaries

- Always: render results with `textContent`. Keep every query in the browser.
- Ask first: a search library; full-text indexing; search on pages other than the two lists.
- Never: add analytics or send a query anywhere. Use `innerHTML` with index or query text.

Out of scope: typo tolerance, stemming, highlighting matches, and Chinese-language search (the posts are in English).

## Success Criteria

- [ ] `posts-index` is merged, and `sh scripts/gates.sh` exits 0 on the branch tip, locally and in CI
- [ ] Each ranking test and the check rule failed before the code existed, and pass now
- [ ] All 13 golden queries pass on the generated index
- [ ] `search-index.json` is under 100 KB, and two rebuilds leave `git status` clean
- [ ] Every browser check above passes, with output in the PR
- [ ] The home page's network log shows no search file

## Open Questions

None.
