# Spec: posts-vault

Module of `CAPABILITY-MAP.md`. Status: **approved** 2026-11-01 (scope and quality chosen by the owner).

## Objective
Every technical note becomes one public post: 12 machine-learning fundamentals, 9 probability, 9 statistics and
experimentation, 10 LLM and generative-AI, 6 recommender-system, 5 ML-systems and 4 coding-and-data posts (55).
Notes already published as long-form posts (position bias, text-to-music, the LoRA tutorial, evaluating ML
systems) and the folders' index pages are not repeated.

Each post is **rewritten** for a public reader, never pasted: no interview framing, no hiring-target companies,
nothing personal. Notes built on outside material (a blog post, a course, a problem site) are rewritten in original
words with credit. Every executable code block runs with asserts; a second agent reviews every post adversarially.

## Tech Stack
Markdown sources in `tools/posts/src/`, rendered by `tools/posts/build.py`, which reuses the LeetCode renderer's
Markdown engine (`tools/leetcode/build.py`). Python standard library plus numpy, pandas, scipy, scikit-learn,
torch and sqlite3 for the executed examples. Output is committed HTML; the site keeps no build step.

## Commands
```
Build one post:  python3 -I tools/posts/build.py . <slug>
Build all:       python3 -I tools/posts/build.py .
Check:           python3 scripts/check.py && (cd scripts && python3 -m unittest test_check)
```

## Project Structure
```
tools/posts/src/<slug>.md   → one source per post (front matter: title, summary, group, order, tags)
tools/posts/build.py        → renders posts/<slug>/index.html and regenerates posts/index.html
posts/index.html            → long-form posts first, then the seven topic groups, then MusicGen (#musicgen)
```

## Code Style
Same source conventions as the LeetCode series: `##`/`###` headings, `\( \)` and `\[ \]` math, executed
` ```python ` blocks with asserts, ` ```plaintext ` for anything not executed, sibling links as `../<slug>/`.

## Testing Strategy
- The renderer runs every Python block and stops on the first failing assert.
- `scripts/check.py`: privacy denylist, links, listings (every `posts/<slug>/` on the posts index), counts.
- A reviewer agent per post re-checks claims, citations, code and privacy, and fixes what it finds.
- Spot review of rendered posts in the browser before commit.

## Boundaries
- Always: rewrite; credit outside sources; run every code block; keep the long-form posts first on the index.
- Ask first: publishing anything from the private folders.
- Never: paste vault text; publish interview coaching, résumé details beyond the About page, or personal notes.
  Exception (2026-11-11, ADR 0003): résumé preparation may be published under `/book/`, unlisted, as
  `SPEC-resume-book.md` sets out. The rule holds for `/posts/`.

## Success Criteria
- [ ] 55 posts render, each with passing code (or no executable code where none fits)
- [ ] Every post passed its review; open issues listed in the commit message
- [ ] Posts index lists all of them by topic; `check.py` passes
