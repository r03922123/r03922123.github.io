# Implementation Plan: Personal site — educational writing + work experience

Spec: [`SPEC.md`](../SPEC.md) (approved 2026-10-28). Task checklist: [`todo.md`](todo.md).

## Overview

Add an About page and four in-repo posts to the existing static hub, guarded by a local check
script. The privacy gate comes first, so nothing written later can leak. The first post sets
the post template (TOC, math, code, references, citation); the other three reuse it.

## Dependency graph

```
scripts/check.py + denylist.local.txt   (gate for everything below)
        │
        ├── nav + about.html             (independent of posts)
        │
        └── post template (built inside post 1: position bias)
                │
                ├── post 2: text-to-music 2026   (riskiest content → early)
                ├── post 3: LoRA from scratch
                └── post 4: evaluating ML systems
                        │
                        └── README + final full-site verification
```

## Architecture Decisions

- **Denylist is local and gitignored.** The repo is public, so a committed list of private
  terms would publish them. `check.py` fails loudly if the local file is missing.
- **Post template is extracted from a real post, not designed up front.** Post 1 adds the post CSS
  and HTML skeleton; posts 2–4 copy the skeleton. With no build step, copying a ~20-line
  head/header block beats adding a templating tool.
- **KaTeX + highlight.js load per page, pinned.** Only post pages include them; the home and
  About pages stay free of third-party scripts. KaTeX auto-render uses `\( \)` / `\[ \]` delimiters
  (not `$`), so dollar signs in prose never get mangled.
- **Content is rewritten from the vault, never pasted.** Each content task starts by reading the
  source note and ends with a check run plus a manual claim review.
- **Riskiest content goes early.** Text-to-music (translation + 2026 models + paper fact-marking)
  is post 2, right after the template exists.

## Task List

### Phase 1: Foundation
- [x] Task 1: Privacy and integrity check script
- [x] Task 2: Nav and About page

### Checkpoint A — after Tasks 1–2 (review with you)
- [ ] `check.py` exits 0; About page reviewed against résumé + `Never claim:` lines

### Phase 2: Posts
- [x] Task 3: Post template + post 1 (position bias)
- [x] Task 4: Post 2 (text-to-music 2026)

### Checkpoint B — after Tasks 3–4 (review with you)
- [ ] Two posts render math, code, TOC at 375px, light + dark

- [x] Task 5: Post 3 (LoRA from scratch)
- [x] Task 6: Post 4 (evaluating ML systems)

### Phase 2b: LeetCode series (added 2026-10-31)
- [x] Task 8: Series page, check-script support, first group (arrays and pointers)
- [x] Task 9: Search, sorting and selection
- [x] Task 10: Stacks, queues, heaps and hashing
- [ ] Task 11: Trees, tries and range queries
- [ ] Task 12: Graphs
- [ ] Task 13: Dynamic programming, bits and game theory

### Phase 3: Finish
- [ ] Task 7: README and full-site verification

### Checkpoint C — complete
- [ ] All spec success criteria checked; ready to push (push needs your OK)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Private vault content leaks into a page | High | Check script gate on every commit; manual review at Checkpoint A |
| About page overclaims vs. a `Never claim:` line | High | Claim-by-claim table in Task 2, reviewed by you |
| Text-to-music facts about 2026 models are wrong after translation | Med | Keep the note's `[paper]`/`[illustrative]` marks; cite each paper; flag anything unverifiable |
| LoRA tutorial code no longer runs once rewritten | Med | Extract code to scratchpad and run it (CPU, tiny model) before publishing |
| CDN script breaks or changes | Low | Exact pinned versions; pages stay readable without JS (math shows as raw TeX) |
| `posts/` collides with a future repo named `posts` | Low | README rule already forbids it; noted again in README |

## Open Questions

None blocking. Decisions taken at spec approval are recorded in `SPEC.md`.
