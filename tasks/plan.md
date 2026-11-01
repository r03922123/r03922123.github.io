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
- [x] Task 11: Trees, tries and range queries
- [x] Task 12: Graphs
- [x] Task 13: Dynamic programming, bits and game theory

### Phase R: Interactive redesign (added 2026-10-31; see CAPABILITY-MAP.md)
- [x] Task R1 `theme`: tokens, header/footer, AA contrast test
- [x] Task R2 `reskin`: posts index, nav, posts/LeetCode/About/404 restyled, article text unchanged
- [x] Task R3 `scene`: design panel (3 drafts, 3 judges, synthesis), then integrate into index.html
- [x] Task R4 `lofi`: toggle, pre-paint mode script, plain home
- [x] Task R5 review: multi-lens review workflow over the redesign, verified findings fixed

### Checkpoint R
- [ ] All module success criteria checked; you review the live preview before any push

### Phase 3: Finish
- [x] Task 7: README and full-site verification

### Checkpoint C — complete
- [ ] All spec success criteria checked; ready to push (push needs your OK)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Private vault content leaks into a page | High | Check script gates every merge and push (not local commits, see CONSTRAINTS.md); manual review at Checkpoint A |
| About page overclaims vs. a `Never claim:` line | High | Claim-by-claim table in Task 2, reviewed by you |
| Text-to-music facts about 2026 models are wrong after translation | Med | Keep the note's `[paper]`/`[illustrative]` marks; cite each paper; flag anything unverifiable |
| LoRA tutorial code no longer runs once rewritten | Med | Extract code to scratchpad and run it (CPU, tiny model) before publishing |
| CDN script breaks or changes | Low | Exact pinned versions; pages stay readable without JS (math shows as raw TeX) |
| `posts/` collides with a future repo named `posts` | Low | README rule already forbids it; noted again in README |

## Open Questions

None blocking. Decisions taken at spec approval are recorded in `SPEC.md`.

## Phase V2: version 2 (wizarding-game common room) — HANDOFF, 2026-11-01

Live site: https://r03922123.github.io/ still serves **version 1** (Tokyo desk, pushed fcf76c0). Nothing of
version 2 is pushed. Specs: `CAPABILITY-MAP.md` v2, `SPEC-posts-vault.md`, `SPEC-room.md`, `SPEC-theme.md` v2,
`SPEC-lofi.md` / `SPEC-reskin.md` amendments — all approved by the owner.

- [x] Specs v2, posts renderer (`tools/posts/build.py`), vendored three.js r169 + Cinzel (80b502a)
- [x] Theme v2 + reskin v2: parchment/candlelit, Cinzel, "Common room" nav, gilt frames, castle 404 (dafb657)
- [x] **posts-vault**: all 55 written and reviewed in 3 batches (2026-11-01) and committed on branch `v2` (0692cf9).
      MusicGen was removed from the home page (the owner studied it and is not its author); the gramophone
      stays as decoration (ce0eb9e, 74ba115).
- [x] **room**: `room.js` + room markup in `index.html` + `rm-*` section of `style.css`, committed on `v2` (1f87a26).
      Checked in real Chrome: hover/Tab labels, focus no longer scrolls the stage (`overflow: clip`), no console
      errors, phone framing. Code review (J2) returned 11 findings, to be triaged in J5.
- [x] **lofi** amendment: implemented in `room.js` (no WebGL → lo-fi + toggle hidden; lo-fi never fetches three.js,
      verified via resource timing).

### Ready jobs (queued 2026-11-01): run in order, without asking

All v2 work is on branch `v2`, nothing pushed or merged. `main` is unchanged at 851b0b7.

The owner's rule (memory `decide-next-step-by-role`): pick the best-practice step, give it to its role, and state a
one-line reason grounded in real industry practice plus this project's evidence.

- [x] **J1 Rebuild + checks** (done: fixed a denylisted company name, a realistic phone number, and an interview-URL credit): role: main session.
      `python3 -I tools/posts/build.py . && python3 -I tools/leetcode/build.py .`, then `python3 scripts/check.py`,
      the unit tests, and a grep of `tools/posts/src` for interview|obsidian|vault|musicgen.
      Why: every change is built and tested before review (Google's presubmit on TAP); here, batch 1-2 posts dropped
      links to posts that didn't exist yet.
- [x] **J2 Room code review** (done: 11 findings, 1 high): role: `agent-skills:code-reviewer`, on `room.js`, the room markup in `index.html`
      and the `rm-*` CSS, against `SPEC-room.md` and `SPEC-lofi.md`.
      Why: no change lands without a reviewer (Google eng-practices, "The Standard of Code Review"); here, `room.js`
      has had no review.
- [x] **J3 Privacy audit, all 55 posts** (done: 'publishable after the listed fixes'; 5 findings): role: `agent-skills:security-auditor`, on `tools/posts/src/*.md` plus the
      rendered `posts/`. Look for interview framing, hiring-target companies, personal details, claims of authorship
      over others' work (MusicGen and the like), and copied third-party text.
      Why: identify and control personal data before release (NIST Privacy Framework); here, each per-post
      reviewer only saw its own post.
- [x] **J4 Home performance audit** (done: 9 findings, 1 high): role: `agent-skills:web-performance-auditor`, on `index.html`, `room.js` and
      `style.css`: first paint before three.js, the 30 fps cap, pausing when hidden or off-screen, and phone cost.
      Why: set performance budgets and measure against them (web.dev, Core Web Vitals); here, `SPEC-room.md`'s
      30 fps cap has never been measured.
- [x] **J5 Fix findings** (done on `v2`: 21 fixed, 2 rejected, 1 partly fixed with the rest open, and 1 regression found and fixed while verifying): role: main session. Apply confirmed findings from J2-J4, then rerun J1.
- [ ] **J6 Owner preview, then stop**: role: owner. Serve locally, give the owner a short summary, and push only
      with the owner's explicit OK.
      Why: a human approves before a release reaches users (Google SRE book, "Release Engineering"); here, the
      handoff rules say "push only with the owner's explicit OK".
J2, J3 and J4 are independent: launch them together, in parallel.

Dev tools: `tools/dev/shoot.sh` (2D screenshots, light/dark, phone frame), `tools/dev/shoot3d.sh` (WebGL),
`tools/dev/same_articles.py <git-ref>` (article text unchanged check). Scratch output goes to `.cache/` (gitignored).
