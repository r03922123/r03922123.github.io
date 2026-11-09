# Capability Map: interactive redesign

Version 3, 2026-11-10. Content is specified in `SPEC.md`; this map covers the redesign and the vault posts.
The map, not file names, is the index of what exists.

## Version 3: reading pages after claude.dev (approved 2026-11-10)

Everything except the common room (all of `index.html`, lo-fi included) takes its structure from claude.dev.
Module specs are written one at a time, in build order.

| Module id | Responsibility | Depends on | Spec |
|---|---|---|---|
| `site-baseline` | Project CLAUDE.md, learning log, ADRs, CI that runs the CONSTRAINTS.md gates, protected `main` | — | `SPEC-site-baseline.md` |
| `post-feeds` | Generated at build time: `rss.xml`, `sitemap.xml`, `robots.txt`, `llms.txt`, a `.md` beside each post that has a Markdown source (86 of 90), canonical links, JSON-LD | `site-baseline` | not yet written |
| `blog-theme` | New tokens and type for reading pages; the parchment tokens move under `body.home` so the common room is unchanged | `site-baseline` | not yet written |
| `reading-shell` | Shared nav, title block, contents rail, tables and code, across posts, About and 404 | `blog-theme` | not yet written |
| `posts-index` | Posts and LeetCode lists as rows, with category filters | `blog-theme` | not yet written |
| `site-search` | Index generated at build time, searched in the browser | `posts-index` | not yet written |

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
