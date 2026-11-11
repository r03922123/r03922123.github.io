# Capability Map: interactive redesign

Version 4, 2026-11-10. Content is specified in `SPEC.md`; this map covers the redesign and the vault posts.
The map, not file names, is the index of what exists.

Status, 2026-11-11: every module of versions 3 and 4 is built and merged. `docs/ARCHITECTURE.md` maps the site as
it is now; this map keeps what each module was for.

## Version 4: reading pages after an Anthropic research article (approved 2026-11-10)

Changes two things in version 3. The visual reference for `blog-theme`, `reading-shell` and `posts-index` moves from
claude.dev to `anthropic.com/research/components-of-agent-memory` (article page) and `anthropic.com/research` (lists).
A new module, `post-voice`, takes the writing craft of Anthropic's research and engineering posts. Existing module ids
and the common room are unchanged.

| New module id | Responsibility | Depends on | Spec |
|---|---|---|---|
| `post-voice` | Reframe the 59 topic and long-form posts: opening, main claim, limits, captions, ending. Code, numbers and asserts stay as they are | `site-baseline` | `SPEC-post-voice.md` |

Build order: `site-baseline` → `post-feeds`, `blog-theme`, `post-voice` → `reading-shell`, `posts-index` → `site-search`.

| Module id | Taken from the reference | Adapted for this site |
|---|---|---|
| `blog-theme` | Serif body, sans headings, warm off-white page, near-black text, links that are underlined and keep the text colour | Open-licensed self-hosted fonts and our own palette; dark mode, which the reference page does not show |
| `reading-shell` | Centred title; a side column for published date, topic and tags; one 720 px text column; figures wider than the text | No hero image; the contents rail joins the side column; code and tables of 5 or more columns, which the reference article has none of, use the wider column; maths stays in the text column |
| `posts-index` | Topic cards with post counts, an article count, a filter bar, rows of topic and title and summary | Text rows only, no gallery or carousel; ordered by `group` and `order`, since 55 posts share one date |

Measured on the reference at a 1920 px viewport: title 36 px bold sans, body 17 px on a 26 px line in a 720 px column,
figures 1158 px wide, hero 865 px wide.

Writing sample, 2026-11-10: 9 posts (the article above, 4 from the Interpretability team, 4 from Engineering) against
this site's 55 topic posts, with one script for both. Nine posts is a small sample, so these are leads, not laws.

| Per 1000 words unless noted | Reference (9) | Topic posts (55) | Reading |
|---|---|---|---|
| Mean sentence length, words | 19.6 | 19.6 | Same; leave alone |
| Words per paragraph | 59.9 | 55.2 | Close; leave alone |
| First person (we/our/I) | 24.6 | 0.6 | Largest gap: the reference says who did what |
| Headings | 4.6 | 6.7 | Reference sections run longer |
| Code blocks | 0.2 | 5.7 | Keep ours: code is what these posts are for |
| Figures | 2.6 | 0 | Tables and asserts play this role here |

| Move | Reference | Topic posts today | `post-voice` |
|---|---|---|---|
| Opening | A concrete situation, stake or question (7 of 8 read closely) | 27 of 55 open by defining the term | Open on the problem; define the term next |
| Limits | Stated in all 4 Engineering posts and in the Interpretability result posts | About 14 of 55 near the end (rough keyword count) | Every post says what it does not show |
| Ending | Open questions, implications or a takeaway | 26 of 55 end on Pitfalls | Keep Pitfalls; close with what remains open |
| Terms defined at first use | All 8 | Already done | Keep |

Not taken: the team "we" (one author here, so "I" for what was run), hiring calls, a longer word count, em dashes
(the posts' prose uses none today), and any sentence from the reference.

Assumptions approved with this version:
- `post-voice` is a framing pass, not a full rewrite: the code blocks of each post stay byte-identical, checked by hash.
- The 31 LeetCode notes are out of scope. At a median of about 305 words of paragraph text they are a different form, and the sample has no match.
- Pilot of 3 posts first, then batches by group, because 59 posts cannot fit the 300-line limit per change.
- No cover images: the repo has none. Covers would be a separate module later.
- The version 3 rule still holds: structure only, no Anthropic fonts, logo, footer content or identical palette.

## Version 3: reading pages after claude.dev (approved 2026-11-10)

Everything except the common room (all of `index.html`, lo-fi included) takes its structure from claude.dev.
Module specs are written one at a time, in build order.

| Module id | Responsibility | Depends on | Spec |
|---|---|---|---|
| `site-baseline` | Project CLAUDE.md, learning log, ADRs, CI that runs the CONSTRAINTS.md gates, protected `main` | — | `SPEC-site-baseline.md` |
| `post-feeds` | Generated at build time: `rss.xml`, `sitemap.xml`, `robots.txt`, `llms.txt`, a `.md` beside each post that has a Markdown source (86 of 90), canonical links, JSON-LD | `site-baseline` | `SPEC-post-feeds.md` |
| `blog-theme` | New tokens and type for reading pages; the parchment tokens move under `body.home` so the common room is unchanged | `site-baseline` | `SPEC-blog-theme.md` |
| `reading-shell` | Shared nav, title block, contents rail, tables and code, across posts, About and 404 | `blog-theme` | `SPEC-reading-shell.md` |
| `posts-index` | Posts and LeetCode lists as rows, with category filters | `blog-theme` | `SPEC-posts-index.md` |
| `site-search` | Index generated at build time, searched in the browser | `posts-index` | `SPEC-site-search.md` |

Build order: `site-baseline` → `post-feeds`, `blog-theme` → `reading-shell`, `posts-index` → `site-search`.

Decisions recorded 2026-11-10:
- Hosting stays on GitHub Pages with the Python build. Only claude.dev's public outputs are adopted, not its framework.
  Pages cannot set custom response headers, so its security headers are out of reach here.
- Baseline first; the understand and tidy phases do not block this redesign (owner-approved exception).
- Structure only. No Anthropic fonts, mascot, footer content or identical palette: this is a personal site and must
  not read as an Anthropic property.
- Not taken: Mods, Docs and Terminal pages, the "Try Claude Code" button, interactive figures.
- Supersedes for reading pages: `reskin`, and the version 2 decision "parchment theme everywhere". `theme` stays
  for the common room only. The external MusicGen posts stay untouched.

Notes for later module specs:
- `blog-theme`: `scripts/test_check.py` reads the theme tokens from `:root`, so moving them under `body.home` changes
  a gate test. Say so in that spec, because `CONSTRAINTS.md` forbids quietly weakening a gate.
- `reading-shell`: `scripts/check.py` strips fragments before checking links, so a rewrite of About could break the
  common room's portrait doors (`about.html#kkcompany` and its siblings) with no failing check. That spec needs its
  own check that those ids still exist.

## Version 2 modules

| Module id | Responsibility | Depends on | Spec |
|---|---|---|---|
| `posts-vault` | 55 topic posts, one per technical or coding note in the vault, plus the generated posts index | — | `SPEC-posts-vault.md` |
| `theme` | Parchment (light) and candlelit (dark) tokens, serif display headings, game-menu chrome | — | `SPEC-theme.md` |
| `room` | Home page: a real-time low-poly 3D castle common room in a 2000s console-game style; objects are the navigation | `theme` | `SPEC-room.md` |
| `lofi` | Plain text home and a remembered toggle; automatic for reduced motion and when WebGL is unavailable | `room` | `SPEC-lofi.md` |
| ~~`reskin`~~ | Superseded by `blog-theme` and `reading-shell` in version 3: every reading page in the parchment theme | `theme` | `SPEC-reskin.md` |
| ~~`scene`~~ | Superseded by `room`: the Tokyo-night SVG desk (version 1). Kept in git history | — | `SPEC-scene.md` |

Build order: `posts-vault` (content, in parallel) and `theme` → `reskin`, `room` → `lofi`.

Decisions recorded 2026-11-01:
- Posts: every note in 02-ml-breadth and 03-ml-depth, plus the 01-coding notes outside the LeetCode patterns, each
  rewritten for the public and verified (code run, claims reviewed). Private folders are never published: résumé,
  behavioral/HR, companies, plan/state, project-bullet coaching.
- Direction: a 2000s console-game look (low-poly, low resolution, chunky serif menus), rendered live in 3D.
  The room is an original castle common room. No Harry Potter names, crests, characters or recreations of game
  screens; no copying of felixrieseberg.com's room, code or assets.
- Objects: bookshelf → posts; open spellbook on a lectern → LeetCode; three portraits above the mantel (KK, TT, HTC)
  → the roles on About; gramophone → the MusicGen posts.
- Reading pages: parchment theme everywhere, still ordinary pages, each with a way back to the room.
- Extras: lo-fi plain mode only. No sound.
- Version 1 (Tokyo desk, Tokyo-night theme) shipped publicly on 2026-11-01 and stays live until version 2 is approved.
