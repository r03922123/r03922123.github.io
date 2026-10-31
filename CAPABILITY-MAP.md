# Capability Map: interactive redesign

Version 2, 2026-11-01. Content is specified in `SPEC.md`; this map covers the redesign and the vault posts.
The map, not file names, is the index of what exists.

| Module id | Responsibility | Depends on | Spec |
|---|---|---|---|
| `posts-vault` | 55 topic posts, one per technical or coding note in the vault, plus the generated posts index | — | `SPEC-posts-vault.md` |
| `theme` | Parchment (light) and candlelit (dark) tokens, serif display headings, game-menu chrome | — | `SPEC-theme.md` |
| `room` | Home page: a real-time low-poly 3D castle common room in a 2000s console-game style; objects are the navigation | `theme` | `SPEC-room.md` |
| `lofi` | Plain text home and a remembered toggle; automatic for reduced motion and when WebGL is unavailable | `room` | `SPEC-lofi.md` |
| `reskin` | Every reading page (long-form, topic and LeetCode posts, About, 404) in the parchment theme | `theme` | `SPEC-reskin.md` |
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
