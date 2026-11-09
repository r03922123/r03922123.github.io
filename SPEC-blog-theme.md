# Spec: blog-theme

Module of `CAPABILITY-MAP.md` (version 4). Depends on `site-baseline`. Status: **approved** 2026-11-10.

## Objective

Reading pages get a calm publication look: serif body, sans headings, warm off-white paper. The common room looks
exactly as it does today. This module changes colours and fonts only; layout belongs to `reading-shell`.

Users: readers of the 90 reading pages, and visitors to the room, who should notice nothing.

| Requirement | Done means |
|---|---|
| Reading pages use the new tokens | Posts, LeetCode, About, 404 and both indexes render with the tokens below, in light and dark |
| The room is unchanged | `body.home` resolves to today's parchment and candlelit values; the home page requests no new font |
| The gate gets stronger, not weaker | Contrast is tested on all four palettes; the room's values are pinned by a test |
| It reads as a personal site | No Anthropic font, logo or identical colour |

## What the code forces

| Finding | Consequence |
|---|---|
| `scripts/test_check.py:250-252` reads only the two `:root` blocks | After the move, the room's colours would leave the contrast test unnoticed. Extend `variants()` to four palettes |
| `scripts/test_check.py:265` scans for colours after the second `:root` block | The `body.home` blocks would fail it. Start the scan after the last token block, and keep every other line covered |
| No JS reads a colour token (`room.js`, `post.js`, `index.html`) | Moving tokens under `body.home` cannot break the room's code |
| `index.html:29` is the only page with `class="home"`, and it also holds lo-fi mode | One selector covers the room and lo-fi; no HTML changes |
| Most rules use `var(--font-display)` and `var(--font-body)` | Swapping those two tokens restyles reading pages without touching their rules |
| Browsers download a `@font-face` file only when text uses it | The home page pays nothing for the new fonts; italic is fetched only by pages with `<em>` or `<i>`: 16 `<em>` in 3 files, and `<i>` on all 4 long-form pages |
| Five characters fall outside the Latin subset: `→ λ ≤ α ≈`, on at most 3 pages each | They use the next font in the stack. Accepted; a wider subset would cost every page |

## Tech Stack

The single `style.css`, stdlib `unittest`, and three self-hosted variable `woff2` files, Latin subset, SIL OFL 1.1:

| Role | Font | Source (npm, pinned) | Size |
|---|---|---|---|
| Body | Source Serif 4, upright | `@fontsource-variable/source-serif-4@5.3.1`, `latin-wght-normal` | 50.9 KB |
| Body emphasis | Source Serif 4, italic | same, `latin-wght-italic` | 51.6 KB |
| Headings, labels | Inter | `@fontsource-variable/inter@5.3.0`, `latin-wght-normal` | 48.3 KB |

The `wght` file is used instead of `opsz` (122 KB) because one axis is enough at a 17 px body size.

## Tokens

The names stay, so existing rules keep working. All ratios were computed with the formula in `test_check.py`.

| Token | Light | Dark | Lowest contrast (bg, card) |
|---|---|---|---|
| `--bg` | `#f8f5ec` | `#191815` | |
| `--card` | `#efebdf` | `#252320` | |
| `--fg`, `--link` | `#1d1b16` | `#ede9df` | 14.4 light, 12.9 dark |
| `--muted` | `#5e594e` | `#aaa597` | 5.8 light, 6.4 dark |
| `--accent`, `--signal`, `--gold` | `#1b6554` | `#7ccdb6` | 5.8 light, 8.4 dark |
| `--line` | `#e3dfd2` | `#35332e` | decorative |
| `--code-builtin` | `#8a4b08` | `#ffa657` | 5.7 light, 8.1 dark |
| `--code-keyword` | `#a3172f` | `#ff7b72` | 6.5 light, 6.2 dark |

Green is the accent on purpose: it is far from the reference's clay orange and from the room's burgundy and gold.

Type, measured from the reference and adapted: body 17 px on a 26 px line; `h1` 36 px bold sans on a 41 px line, down
to 28 px on phones; `h2` 22 px; `h3` 18 px. Headings lose the Cinzel letter spacing.

## Commands

```
All gates:   sh scripts/gates.sh
Check tests: (cd scripts && python3 -m unittest test_check)
Preview:     python3 -m http.server 8000
```

## Project Structure

```
style.css                  → reading tokens at :root; today's values move to body.home; @font-face; type rules
fonts/source-serif-4/      → upright and italic woff2, OFL.txt
fonts/inter/               → woff2, OFL.txt
scripts/test_check.py      → four palettes, pinned room values, scan after the last token block
```

The PR opens with `Takeover phase: build (owner-approved exception; repo is in understand)`.

## Code Style

Role-named tokens, swapped per surface, with comments that give the reason.

```css
:root { --bg: #f8f5ec; --font-body: "Source Serif 4", Georgia, serif; }
/* The room keeps its own palette: its art and lo-fi page were drawn for parchment. */
body.home { --bg: #f3e9d2; --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
```

## Testing Strategy

- Test first, in `test_check.py`: the room's 22 colour values equal today's literal values. It fails before the move
  because `body.home` does not exist yet.
- Contrast: the existing AA test runs over four palettes, so no surface leaves it.
- Browser check on the local preview, as evidence in the PR, in light and dark:
  - Home: computed colours and fonts on `body` match `main`; the network log shows no Source Serif or Inter request.
  - A post with `<em>`: `document.fonts.check` is true for all three faces.
  - Screenshots of the lo-fi home (before and after) and of a post, About and 404.

## Boundaries

- Always: keep every colour in a token block outside the scene art. Ship each `OFL.txt` beside its font.
- Ask first: another font or weight file; any change to `.rm-*` rules; a palette change after approval.
- Never: use Anthropic's fonts, logo or exact colours. Lower a contrast threshold. Change post text or generated HTML.

Out of scope: column width, title block, side column and contents rail (`reading-shell`); list pages (`posts-index`);
font preload tags, which would touch every generated page (`reading-shell`).

Between this PR and `reading-shell`, the live site shows new colours and fonts on the old layout. That is acceptable
because every page stays readable; the PR's screenshots are where to object.

## Success Criteria

- [ ] `site-baseline` is merged, and `sh scripts/gates.sh` exits 0 on the branch tip
- [ ] The pinned-room test failed before the move and passes after it
- [ ] The contrast test covers four palettes, each text token at 4.5:1 or better on both surfaces
- [ ] The colour scan still fails when a stray hex is added after the token blocks (shown once, then reverted)
- [ ] Home: computed `body` styles match `main`, no new font requested, and lo-fi screenshots match
- [ ] A post renders all three faces in light and dark
- [ ] `git diff --name-only main` lists no generated page, and every `.rm-` rule is unchanged: the result of
  `re.findall(r"[^{}]*\.rm-[^{}]*\{[^{}]*\}", css)` is equal on `git show main:style.css` and on the working tree. `@media`
  and `@keyframes` wrappers around those rules are outside that comparison, so read them in the diff

## Open Questions

None. The colours can be adjusted from the PR's screenshots before merge.
