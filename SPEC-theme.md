# Spec: theme

Module of `CAPABILITY-MAP.md`. Status: **approved** 2026-10-31.

## Objective
One visual language for the whole site, taken from the home scene: a Tokyo night of deep indigo,
sodium-lamp amber and the red-white of Tokyo Tower, with neon cyan and pink as small accents. Reading
pages must stay comfortable for 15-minute posts, so the theme has a **paper** (light) and a **night**
(dark) variant, chosen by the visitor's OS setting as today. The scene itself is always night.

## Tech Stack
Plain CSS custom properties in the existing single `style.css`. System font stacks only, no web fonts,
so pages make no extra requests for type.

## Commands
```
Preview: python3 -m http.server 8000
Check:   python3 scripts/check.py && (cd scripts && python3 -m unittest test_check)
```

## Project Structure
```
style.css   → sections in order: tokens, base, header/footer, reading pages, LeetCode, scene, lo-fi
```

## Code Style
Tokens are named by role, never by colour, so the light and dark variants swap values only:
```css
:root {
  --bg: #f7f5f0; --fg: #1f2130; --muted: #5d6070; --line: #e2ded4; --card: #fffdf8;
  --accent: #c2410c;          /* sodium amber, darkened for contrast on paper */
  --link: #1d4ed8; --signal: #0e7490; --tower: #dc2626;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1020; --fg: #e8e6f0; --muted: #9ba0b8; --line: #262b45; --card: #141933;
          --accent: #f5b14c; --link: #8ab4ff; --signal: #5eead4; --tower: #f87171; }
}
```
- Headings use the body stack at heavier weights; small labels (dates, tags, the wordmark) use `--font-mono`
  in small caps-like uppercase with letter spacing, echoing signage.
- Header: wordmark `r03922123` on the left; nav **Desk · Posts · LeetCode · About · GitHub** on the right;
  wraps cleanly at 375px.
- Footer: one line, plus a small skyline glyph (inline SVG, a few hundred bytes) as a signature.

## Testing Strategy
- `scripts/check.py` keeps passing (head fields, links, privacy, pinned CDNs).
- Contrast: every text/background token pair meets WCAG AA (4.5:1 body, 3:1 large text), computed by a
  small test added to `scripts/test_check.py` that parses the tokens from `style.css`.
- Browser: home, a long-form post, a LeetCode post and About, at 375px and 1440px, in light and dark.

## Boundaries
- Always: change colours through tokens; keep KaTeX and code highlighting readable in both variants.
- Ask first: adding any web font or external stylesheet.
- Never: colour-only meaning (links stay underlined on hover/focus), body text below AA contrast.

## Success Criteria
- [ ] All pages read from the same tokens; no hard-coded colours outside the token block and the scene art
- [ ] AA contrast test passes for both variants
- [ ] Header nav fits at 375px without horizontal scroll on every page
