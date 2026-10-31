# Capability Map: interactive redesign

Approved 2026-10-31. Content is specified in `SPEC.md`; this map covers the redesign only.
The map, not file names, is the index of what exists.

| Module id | Responsibility | Depends on | Spec |
|---|---|---|---|
| `theme` | Night-palette design tokens, type scale, shared header/footer, light/dark for reading pages | — | `SPEC-theme.md` |
| `scene` | Home page: an illustrated SVG desk in front of a window onto Tokyo at night; desk objects are the navigation | `theme` | `SPEC-scene.md` |
| `lofi` | Plain text home and a remembered toggle; automatic for reduced-motion visitors | `scene` | `SPEC-lofi.md` |
| `reskin` | Every reading page (long-form posts, LeetCode series, About, 404) restyled to the theme, plus a posts index | `theme` | `SPEC-reskin.md` |

Build order: `theme` → `reskin`, `scene` (independent of each other) → `lofi`.

Decisions recorded at approval:
- Direction: illustrated 2D scene (SVG + CSS), not 3D and not a desktop OS. Inspired by the idea behind
  felixrieseberg.com (objects as navigation, a lo-fi escape hatch); the art and layout are original.
- Motif: Tokyo night view through the window.
- Objects: CRT monitor → long-form posts; notebook → LeetCode series; three labeled tapes → About and
  experience; radio → the MusicGen posts.
- Reading: posts stay ordinary pages, themed, each with a way back to the desk.
- Extras: lo-fi plain mode only. No sound, no keyboard shortcuts, no easter eggs.
- The LeetCode renderer and sources live in `tools/leetcode/`; the site itself has no build step.
