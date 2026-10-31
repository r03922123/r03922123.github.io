# Spec: reskin

Module of `CAPABILITY-MAP.md`. Depends on `theme`. Status: **approved** 2026-10-31.

## Objective
Every reading page wears the theme and leads back to the desk, while staying a plain, fast, shareable
page. Because the home becomes the scene, the post list moves to a new **posts index**.

| Page | Change |
|---|---|
| `posts/index.html` (new) | Long-form posts, newest first, then a **MusicGen** section (`#musicgen`) with the two external posts. Target of the CRT and the radio |
| `posts/<slug>/` (4) | Themed header/footer; title block with a mono date/tag line; TOC as a side rail on wide screens |
| `leetcode/` + 31 posts | Same, through the template in `tools/leetcode/build.py`, then a full rebuild |
| `about.html` | Each role gets an id (`kkcompany`, `tiktok`, `htc`) and a tape-label heading matching the scene |
| `404.html` | "Lost in the city" with a small skyline and a link back to the desk |

## Tech Stack
Existing static HTML and `style.css`; the LeetCode pages are regenerated with the stdlib renderer.

## Commands
```
Rebuild LeetCode: python3 -I tools/leetcode/build.py .
Preview:          python3 -m http.server 8000
Check:            python3 scripts/check.py && (cd scripts && python3 -m unittest test_check)
```

## Project Structure
```
posts/index.html            → new posts index
posts/<slug>/index.html     → 4 long-form posts (header/footer/title block edited in place)
tools/leetcode/build.py     → HEAD/FOOT templates updated; 31 pages regenerated
about.html, 404.html        → edited in place
scripts/check.py            → posts/ entries must be listed on posts/index.html (was: index.html)
```

## Code Style
Shared header on every page, `Desk` linking home:
```html
<header class="site">
  <a class="home" href="/">r03922123</a>
  <nav><a href="/">Desk</a><a href="/posts/">Posts</a><a href="/leetcode/">LeetCode</a><a href="/about.html">About</a><a href="https://github.com/r03922123">GitHub</a></nav>
</header>
```
The site header gets a class (`header.site`), so the post title block (`header.post-head`) can no longer
inherit its layout by accident.

## Testing Strategy
- Test first: `scripts/test_check.py` gains a case that a `posts/<slug>/` missing from `posts/index.html`
  fails; then `check.py` moves the rule.
- `tools/leetcode/build.py` rebuild: all 31 posts' code still passes.
- Browser: every page type at 375px and 1440px, light and dark; KaTeX and code blocks render; the TOC rail
  does not overlap text; no horizontal scroll.
- Content unchanged: a script compares each page's article text before and after the reskin (ignoring
  header/footer) and must report no differences.

## Boundaries
- Always: content stays byte-for-byte the same inside each article; only chrome changes.
- Ask first: changing a post's URL.
- Never: restyle the external MusicGen posts (they are deployed by their own repo).

## Success Criteria
- [ ] Posts index lists 4 long-form posts and the 2 MusicGen posts, and the check enforces it
- [ ] Every reading page uses the shared header and the theme; article text unchanged
- [ ] 31 LeetCode pages regenerated with passing code
- [ ] About roles have ids for the tapes; 404 links back to the desk
