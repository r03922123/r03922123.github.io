# Spec: resume-book

Module of `CAPABILITY-MAP.md`. Depends on `room`, `reading-shell`. Status: **approved** 2026-11-11 (owner, in chat;
recorded in `docs/adr/0003-unlisted-resume-book.md`).

## Objective
An old, worn purple book lies on the mantel of the common room. Clicking it, or pressing Enter on it, opens
`/book/`: a short shelf of résumé-preparation posts, rewritten from the owner's private notes. The pages are public
but unlisted: nothing else on the site links them, and they ask search engines not to index them.

Done means:
- The book is a door like the others: an `<a>` in `.rm-doors` over the object, focusable, named "Old book".
  It is not in the text door list, the lo-fi list or the hint line, so it stays discreet.
- `/book/` lists every book page; each page renders in the reading shell and carries `noindex`.
- No book page appears in `sitemap.xml`, `rss.xml`, `llms.txt`, `search-index.json`, the Posts or LeetCode lists,
  or About; the home links it once, from the door. `check.py` fails if any of this breaks.
- Every page is rewritten, never pasted, and carries no personal identifier (see Boundaries).

## Tech Stack
Same as `posts-vault`: Markdown sources in `tools/book/src/`, rendered by `tools/book/build.py` with the LeetCode
renderer (`tools/leetcode/build.py`), Python blocks run with asserts. The book is a few boxes in `room.js`.

## Commands
```
Build:  python3 -I tools/book/build.py .
Check:  python3 scripts/check.py && (cd scripts && python3 -m unittest test_check)
```

## Project Structure
```
tools/book/src/<slug>.md  → one source per page (front matter: title, summary, group, order, tags)
tools/book/build.py       → book/<slug>/index.html and book/index.html, each with <meta name="robots" content="noindex">
room.js, index.html       → the book on the mantel and its door
```
The feed and search builds read only `tools/posts/src` and `tools/leetcode/src`, so the book stays out of them
without any change there; `check.py` checks that it does.

## Testing Strategy
- `test_check.py`: a book page without `noindex`, a book URL in a feed or the search index, a list page or About
  linking the book, and a book page missing from `/book/` each give a finding.
- Gate `book wording`: no mention of the notes' source (`obsidian`, `vault`) in `tools/book/src`.
- Browser: the door is reachable by Tab and opens `/book/`; on the live site after deploy.

## Boundaries
- Always: rewrite in new words; drop names of people, recruiter and HR contacts, companies tied to job
  applications, compensation, contact details, application dates and statuses, health and family matters.
  Former employers already named on About may be named. A number appears only as the notes validate it, with
  how it was obtained.
- Ask first: adding notes from other private folders.
- Never: paste note text; link the book from any list, feed or index.

## Success Criteria
- [ ] The book opens `/book/` by mouse, touch and keyboard; the room's camera and other doors are unchanged
- [ ] Every `/book/` page has `noindex` and is absent from the sitemap, feeds, `llms.txt` and the search index
- [ ] `sh scripts/gates.sh` passes; the PR lists the notes used and skipped
