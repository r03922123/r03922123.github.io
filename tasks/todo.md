# Tasks — personal site

Spec: `SPEC.md` · Plan: `tasks/plan.md`. Every task ends with `python3 scripts/check.py` exiting 0.

## Task 1: Privacy and integrity check script

**Description:** Add `scripts/check.py` (stdlib only) implementing the four checks in SPEC →
Testing Strategy, plus a gitignored `scripts/denylist.local.txt` built from the vault's private
terms (phone number, target companies, private job-search notes, employer-internal identifiers).

**Acceptance criteria:**
- [ ] Exits 0 on the current site; exits non-zero naming file:line for each finding
- [ ] Fails with a clear message if `denylist.local.txt` is missing
- [ ] `git check-ignore scripts/denylist.local.txt` confirms it is ignored

**Verification:**
- [ ] `python3 scripts/check.py` → 0 on the current site
- [ ] Planted faults in a scratch copy (a `[[link]]`, the phone number, a broken href, an unpinned
      CDN URL, an unlisted `posts/x/`) each make it fail
- [ ] `git status` shows the denylist untracked and ignored

**Dependencies:** None
**Files:** `scripts/check.py`, `.gitignore`, `scripts/denylist.local.txt` (local only)
**Scope:** S

## Task 2: Nav and About page

**Description:** Add `about.html` (bio, experience, education, competitions, contact) and change
the nav on `index.html` and `404.html` to Posts · About · GitHub. Facts come only from
`resume-mle.tex` (APTOS from `resume.tex`); bullets are rewritten for a web reader, without
résumé compression.

**Acceptance criteria:**
- [ ] Three roles with titles, dates, places; NTU M.S. and NTPU B.S.; RSNA top 3% and APTOS top 4%
- [ ] Contact: email, LinkedIn, GitHub; no phone, no timeline/family narrative, no positioning
- [ ] A claim table (in the PR description, not on the site) maps each sentence to its résumé
      line and shows that no `Never claim:` line is contradicted

**Verification:**
- [ ] `python3 scripts/check.py` → 0
- [ ] Browser: 375px wide with no horizontal scroll; light and dark
- [ ] You review the claim table at Checkpoint A

**Dependencies:** Task 1
**Files:** `about.html`, `index.html`, `404.html`, `style.css`
**Scope:** M

## Checkpoint A
- [ ] Check passes · About page approved by you · commit

## Task 3: Post template + post 1 — position bias

**Description:** Write `posts/position-bias/index.html` from `03-ml-depth/recsys/Positional Bias.md`,
and in doing so set up the post layout: title block, TOC, KaTeX auto-render, highlight.js, numbered
references to primary papers, a "Cite as" BibTeX block, and post typography in `style.css`. Add the
entry to `index.html`, with tags.

**Acceptance criteria:**
- [ ] No target-company mapping or interview framing; every method cites its paper
- [ ] Math renders via KaTeX; code is highlighted; TOC anchors work
- [ ] Index entry shows date, reading time (words/230), tags

**Verification:**
- [ ] `python3 scripts/check.py` → 0
- [ ] Browser: math, code, TOC; 375px; light and dark; readable with JS disabled

**Dependencies:** Task 1
**Files:** `posts/position-bias/index.html`, `style.css`, `index.html`
**Scope:** M

## Task 4: Post 2 — text-to-music in 2026

**Description:** Translate and rewrite `03-ml-depth/text-to-music-2026-model-reward-dpo.md`
into English at `posts/text-to-music-2026/`, keeping the tensor-shape annotations and the
`[paper]` / `[illustrative]` fact-marking. Link to the two existing MusicGen posts where they overlap.

**Acceptance criteria:**
- [ ] Every `[paper]` fact has a numbered reference; `[illustrative]` values stay marked as such
- [ ] Sections: building blocks → generators → reward models → DPO, as in the source
- [ ] Index entry added

**Verification:**
- [ ] `python3 scripts/check.py` → 0
- [ ] Each cited paper's title and venue checked against its arXiv/proceedings page
- [ ] Browser: 375px; light and dark

**Dependencies:** Task 3
**Files:** `posts/text-to-music-2026/index.html`, `index.html`
**Scope:** S

## Checkpoint B
- [ ] Check passes · two posts reviewed by you · commit

## Task 5: Post 3 — LoRA from scratch

**Description:** Rewrite `03-ml-depth/llm-genai/PEFT LoRA Fine-tuning Tutorial.md` at
`posts/lora-from-scratch/`, merging in the equation section of `LoRA.md`. No wikilinks.

**Acceptance criteria:**
- [ ] Code blocks, run in order, reproduce the post's assert results
- [ ] References the LoRA paper (Hu et al., 2021); index entry added

**Verification:**
- [ ] Code extracted to the scratchpad and run on CPU: all asserts pass
- [ ] `python3 scripts/check.py` → 0 · Browser: 375px; light and dark

**Dependencies:** Task 3
**Files:** `posts/lora-from-scratch/index.html`, `index.html`
**Scope:** S

## Task 6: Post 4 — evaluating ML systems

**Description:** Rewrite `03-ml-depth/system-design/ML System Design - Evaluation.md` as a
reference article at `posts/evaluating-ml-systems/`: metrics and methods for classification,
ranking, search and generative models. Drop the interview playbook and company mapping.

**Acceptance criteria:**
- [ ] No interview or company framing; each metric defined with a formula
- [ ] References for non-standard metrics (e.g. NDCG, FAD, CLAP score); index entry added

**Verification:**
- [ ] `python3 scripts/check.py` → 0 · Browser: 375px; light and dark

**Dependencies:** Task 3
**Files:** `posts/evaluating-ml-systems/index.html`, `index.html`
**Scope:** S

## Task 7: README and full-site verification

**Description:** Update `README.md` to cover both kinds of post (in-repo under `posts/`, and
external project repos), the check command, and the local denylist. Then run the full spec
success-criteria list.

**Acceptance criteria:**
- [ ] Every item under SPEC → Success Criteria is checked, with evidence
- [ ] README explains how to add a post of each kind and how to recreate the denylist

**Verification:**
- [ ] `python3 scripts/check.py` → 0
- [ ] Every page clicked through in the browser, light and dark, at 375px and desktop widths

**Dependencies:** Tasks 2–6
**Files:** `README.md`
**Scope:** XS

## Checkpoint C
- [ ] Ready to push; pushing to `main` needs your explicit OK

## Phase 2b — LeetCode series (added 2026-10-31)

Every note in `01-coding/patterns/` becomes one post at `leetcode/<slug>/`. Shared rules for
Tasks 8–13: Python code is extracted and run (asserts on the worked examples) before publishing;
no interview framing; outside-sourced material rewritten with original code and cited; each post
links its LeetCode problems; each is listed on `leetcode/index.html`.

## Task 8: Series page, check support, arrays and pointers
**Files:** `leetcode/index.html`, `scripts/check.py`, `scripts/test_check.py`, `index.html`, posts:
Two Pointer, Sliding Window, Prefix State Map, Sweep Line, Misc (Boyer–Moore, sieve)
**Acceptance:** check fails when a `leetcode/` post is missing from the series page (test first);
home has one series card; 5 posts. **Scope:** M

## Task 9: Search, sorting and selection
Binary Search, Sorting, Quick Select, Bucket Sort, Reverse Pair, Rolling Hash, String Matching. **Scope:** M

## Task 10: Stacks, queues, heaps and hashing
Monotonic Stack, Monotonic Queue, Heap, Design (hash set/map). **Scope:** S

## Task 11: Trees, tries and range queries
Tree, Trie, Segment Tree, Union Find. **Scope:** M

## Task 12: Graphs
BFS, DFS, Dijkstra, Shortest Distance, Topological Order, Graph. **Scope:** M

## Task 13: Dynamic programming, bits and game theory
Dynamic Programming, Knapsack, Bitmasking DP, Bit Manipulation, Game Theory. **Scope:** M
