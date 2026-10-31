# Spec: Personal site — educational writing + work experience

Status: **approved** · 2026-10-28

## Objective

Turn the `r03922123.github.io` hub into a personal site in the style of Lil'Log / Distill:
long-form, educational technical writing is the front page, and a single About page tells who
wrote it and what they have built. No job-market "positioning" or lane statement.

**Readers:** ML engineers and researchers who arrive at a post, and hiring managers or
recruiters who then click "About" to see the person behind it.

**Content sources:** `~/Documents/Obsidian Vault` (read-only), specifically:

| Vault source | Becomes |
|---|---|
| `resume/resume-mle.tex` (newest, 2026-10-12) | About page: bio, experience, education, competitions |
| `04-bullets/*.md` — only their `Never claim:` lines | Hard limits on what the About page may say |
| 4 curated study notes (below) | 4 new posts, **rewritten** for a public reader |

**Curated notes → posts** (chosen for originality and fit with the existing MusicGen posts):

| # | Vault note | Post (working title) | Work needed |
|---|---|---|---|
| 1 | `03-ml-depth/text-to-music-2026-model-reward-dpo.md` | Text-to-music in 2026: models, reward models, DPO | Translate from Chinese; keep its `[paper]` / `[illustrative]` fact-marking |
| 2 | `03-ml-depth/llm-genai/PEFT LoRA Fine-tuning Tutorial.md` | LoRA from scratch, proven with asserts | Strip wikilinks; code blocks stay runnable |
| 3 | `03-ml-depth/recsys/Positional Bias.md` | Position bias: why click logs lie and how to correct them | Remove target-company mapping callouts |
| 4 | `03-ml-depth/system-design/ML System Design - Evaluation.md` | Evaluating ML systems: classification, ranking, search, generative | Remove interview framing; reframe as a reference article |

Excluded on purpose: the softmax note (adapted from Jay Mody's blog, not original),
`Stanford CME295` (course notes), interview playbooks, the non-LeetCode coding notes (`Coding Flow`,
`ML Coding`, SQL and pandas cheatsheets), `00-*`, `05-*`, `06-*`.

**LeetCode series (added 2026-10-31 at the user's request).** Every note in `01-coding/patterns/`
(31, excluding the `dsa.md` index) becomes its own post under `leetcode/<slug>/`, listed on a
series page `leetcode/index.html` grouped by technique; the home page links the series with one card.
Each post: the idea in prose, a reusable Python template, worked problems with complexity, and
links to the LeetCode problems. Thin notes are expanded; notes built on outside material (StackOverflow,
MIT 6.006, EPI, cp-algorithms) are rewritten with original code and cite their source. All code is
run before publishing.

### User stories / acceptance criteria

1. **Home is the post list.** As a reader, I land on a list of all six posts (2 existing,
   4 new), newest first, each showing title, a 1–2 sentence summary, date, reading time, and tags.
2. **Posts read like Lil'Log.** Each new post has a title block (date, reading time, tags),
   a table of contents, rendered math, syntax-highlighted code, numbered references, and a
   "Cite as" block with a BibTeX entry.
3. **About page.** As a hiring manager, I can read a short bio, three roles (KKCompany,
   TikTok, HTC) with dates and outcome-first bullets, education (NTU M.S., NTPU B.S.), Kaggle
   results (RSNA top 3%, APTOS top 4%), and contact links: email, LinkedIn, GitHub.
4. **Nothing private leaks.** No phone number, no career-gap or family/health narrative, no
   target companies, no employer-internal identifiers, no vault wikilinks or callout syntax.
5. **Every claim is defensible.** Numbers come only from `resume-mle.tex`; nothing contradicts a
   `Never claim:` line in the vault's project notes.

## Tech Stack

- Plain static HTML + the one shared `style.css`. No framework, no build step — unchanged.
- **New dependencies (pinned, CDN, loaded only on post pages that need them):**
  - KaTeX `0.16.11` (cdn.jsdelivr.net) for math
  - highlight.js `11.10.0` (cdnjs) for code
- GitHub Pages from `main`, unchanged.
- Python 3 stdlib for the check script (no pip installs).

## Commands

```
Preview:  python3 -m http.server 8000        # then open http://localhost:8000
Check:    python3 scripts/check.py            # exits 0 when clean, non-zero with findings
Deploy:   git push origin main                # GitHub Pages serves main
```

## Project Structure

```
index.html              → Post list (home). Existing + new post entries.
about.html              → Bio, experience, education, competitions, contact
posts/<slug>/index.html → One folder per in-repo post (the 4 new ones)
leetcode/index.html     → LeetCode series page, posts grouped by technique
leetcode/<slug>/index.html → One folder per LeetCode pattern post (31)
style.css               → Shared styles: hub, post typography, TOC, about page
404.html                → Unchanged except nav
scripts/check.py        → Privacy + link + structure checks
SPEC.md                 → This file
README.md               → Updated: how to add an in-repo post vs. an external one
```

Existing external posts stay in their own repos (`/musicgen-prune-inference/…`). `posts/` must
never share a name with a project repo (README rule).

## Code Style

Match the existing hub: hand-written semantic HTML, no classes where an element suffices,
CSS custom properties on `:root` with a dark-mode override, one comment per file explaining intent.

```html
<article class="post-body">
<header class="post-head">
  <h1>Position bias: why click logs lie and how to correct them</h1>
  <div class="meta">October 28, 2026 · 12 min read · ranking · evaluation</div>
</header>
<nav class="toc"><h2>Contents</h2><ol><li><a href="#ips">Inverse propensity weighting</a></li></ol></nav>
<h2 id="ips">Inverse propensity weighting</h2>
<p>Weight each click by \(1/p(\text{examined} \mid \text{position})\) …</p>
<ol class="refs"><li id="ref-1">Joachims et al. (2017). <i>Unbiased Learning-to-Rank with Biased Feedback.</i> WSDM.</li></ol>
</article>
```

Conventions: kebab-case slugs; dates as "Month D, YYYY"; reading time = words / 230, rounded;
prose in English, first person sparingly; references cite the original papers, not blog posts.

## Testing Strategy

`scripts/check.py` runs over every `*.html` in the repo and fails on:

1. **Privacy denylist** — generic patterns in the script (`\[\[` wikilinks, `> [!` callouts,
   phone-number shapes, `s3://`, `.azure.`, `confluence`), plus specific terms (the phone number,
   target-company names, private job-search notes) in `scripts/denylist.local.txt`, which is
   **gitignored** — the repo is public, so a committed denylist would publish what it guards.
   The script fails if that file is missing.
2. **Broken internal links** — every relative `href`/`src` resolves to a file in the repo.
3. **Structure** — every page has `<title>`, `meta description`, `lang`, viewport; every post
   in `posts/` is listed in `index.html`, and every post in `leetcode/` in `leetcode/index.html`.
4. **Pinned CDN** — every external script URL contains an exact version.

Manual checks per page (recorded in the PR): renders at 375px wide with no horizontal scroll,
light and dark mode, math and code render, and a side-by-side read of the About page against
`resume-mle.tex` and each `Never claim:` line.

## Boundaries

- **Always:** take facts and numbers only from `resume-mle.tex`; rewrite content, never paste
  vault notes; run `scripts/check.py` before each commit; keep a post's references to primary sources.
- **Ask first:** adding any dependency beyond KaTeX/highlight.js; publishing any note not in the
  table above; any wording about a role that goes beyond the résumé bullet; pushing to `main`.
- **Never:** write to the Obsidian vault; push the vault or copy its git history; publish the phone
  number, the timeline/family narrative, mental-health content, target-company lists, or interview
  coaching; invent a metric or resolve a "not measured" item into a number.

## Success Criteria

- [ ] 31 LeetCode posts under `leetcode/<slug>/`, each listed on `leetcode/index.html`, code run

- [ ] `index.html` lists 6 posts with date, reading time, tags; nav = Posts · About · GitHub
- [ ] 4 posts under `posts/<slug>/`, each with TOC, KaTeX math, highlighted code, references, BibTeX
- [ ] `about.html` matches `resume-mle.tex` facts; zero conflicts with any `Never claim:` line
- [ ] `python3 scripts/check.py` exits 0
- [ ] Every page verified at 375px, light and dark
- [ ] README describes both kinds of post

## Decisions (defaults taken at spec approval, 2026-10-28)

1. **Author name:** "Jie-Hong Lin" on About and in BibTeX; the header keeps the handle `r03922123`.
2. **Post dates:** the date each post ships.
3. **Post 1 language:** English only.
4. **Résumé PDF:** not on the site.
