# Spec: theme

Module of `CAPABILITY-MAP.md`. Version 2 (parchment), 2026-11-01. Status: **approved** 2026-11-01.
Version 1 (Tokyo-night paper/night tokens) shipped on 2026-11-01 and is replaced by this.

## Objective
One look for the whole site, taken from the common room: **parchment** pages (aged paper, iron-gall ink, a burgundy
and gold accent) in light mode and **candlelit** pages (warm dark wood, cream text, gold accents) in dark mode,
chosen by the OS setting. Headings use a carved-serif display face; body text stays a highly readable font, because
posts run 15 minutes. Small labels (dates, tags, menu items) look like a 2000s console-game menu: small caps,
generous letter spacing.

## Tech Stack
CSS custom properties in the single `style.css`. Display font **Cinzel** (SIL Open Font Licence), **self-hosted**
as one variable `woff2` in `fonts/` with its licence, so no font request leaves the site; system serif fallback
(`"Iowan Old Style", "Palatino Linotype", Georgia, serif`). Body: the system sans stack as today.

## Commands
```
Check: python3 scripts/check.py && (cd scripts && python3 -m unittest test_check)
```

## Project Structure
```
style.css          → tokens (parchment, candlelit), base, header/footer, lists, reading pages, About, LeetCode, room, lo-fi
fonts/cinzel/      → Cinzel variable woff2 + OFL.txt
```

## Code Style
Role-named tokens, swapped per variant, as in version 1:
```css
:root {
  --bg: #f3e9d2; --fg: #2b2118; --muted: #6b5a45; --line: #dccfb0; --card: #f8f1df;
  --accent: #8a2a1a; --link: #7a2414; --signal: #8a6d1f; --gold: #b8860b;
  --font-display: "Cinzel", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #1a1410; --fg: #efe3c8; --muted: #b9a88a; --line: #3a2e24; --card: #231b15;
          --accent: #e0a85a; --link: #e8bf7a; --signal: #f0c869; --gold: #d9a441; }
}
```
- Headings (`h1`, `h2`) and the wordmark in `--font-display`; menus and meta lines in small caps.
- Header: wordmark, then nav **Common room · Posts · LeetCode · About · GitHub**; the lo-fi toggle on the home.
- Footer: one line with a small candle glyph replacing the skyline.

## Testing Strategy
- The AA contrast test covers every text token on every surface in both variants (exact values above may be tuned
  to pass; the test decides).
- No literal colours outside the token blocks and the room section (existing test).
- Browser: each page type at 375 and 1440 px, both variants; KaTeX and code readable on parchment.

## Boundaries
- Always: colours through tokens; underlined prose links; focus visible on every control.
- Ask first: any further font or external stylesheet.
- Never: display font for body text; text below AA contrast.

## Success Criteria
- [ ] Every page reads from the parchment/candlelit tokens; AA test passes for both
- [ ] Cinzel served from the site; no external font request
- [ ] Header and footer consistent on every page at 375 and 1440 px
