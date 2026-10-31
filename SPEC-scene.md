# Spec: scene

Module of `CAPABILITY-MAP.md`. Depends on `theme`. Status: **approved** 2026-10-31.

## Objective
The home page is a hand-drawn SVG scene: a desk at night in front of a large window, with Tokyo spread out
behind it. Visitors explore by pointing at things. Four desk objects are doors:

| Object | Opens | Hover/focus behaviour |
|---|---|---|
| CRT monitor | `posts/` (long-form posts index) | screen brightens, scanline flicker, shows "POSTS" |
| Notebook | `leetcode/` | cover lifts slightly, shows "LEETCODE · 31" |
| Three tapes labeled KK, TT, HTC | `about.html#kkcompany`, `#tiktok`, `#htc` | the hovered tape slides out of the stack |
| Radio | `posts/#musicgen` (the two MusicGen posts) | tuning needle sweeps, small signal bars pulse |

Ambient life, all CSS: lit windows in distant buildings that switch on and off at random-looking offsets,
Tokyo Tower's beacon blinking, a train crossing the middle distance every ~20 s, steam from a mug, a slow
glow cycle on the desk lamp. The window's sky tint follows the visitor's local hour (dusk, night, late
night) via one class set by a few lines of JS; without JS it is night.

A one-line greeting sits on the scene ("Jie-Hong Lin — machine learning engineer. Pick something on the
desk.") and a text list of the same four doors sits below it, so the scene never hides navigation.

## Tech Stack
Inline SVG inside `index.html` (no image requests), CSS animations in `style.css`, under 1 KB of inline JS
for the sky tint. No libraries.

## Commands
```
Preview: python3 -m http.server 8000      # then open http://localhost:8000
Check:   python3 scripts/check.py && (cd scripts && python3 -m unittest test_check)
```

## Project Structure
```
index.html            → greeting, inline SVG scene, text door list, footer
style.css (scene)     → scene layout, object hover/focus states, ambient keyframes
posts/index.html      → created by `reskin`; the CRT and radio point at it
```

## Code Style
Each door is a real link inside the SVG, focusable and labeled, with the label shown on hover and focus:
```html
<a href="leetcode/" class="door" aria-label="LeetCode patterns: 31 posts">
  <g class="notebook"> <!-- paths --> </g>
  <text class="door-label" x="980" y="610">LEETCODE · 31</text>
</a>
```
- `viewBox="0 0 1600 900"`; groups named by object (`.sky`, `.city`, `.tower`, `.train`, `.desk`, `.crt`...).
- Ambient animations use `transform` and `opacity` only, so they stay off the main thread's layout work.
- Every animation is wrapped in `@media (prefers-reduced-motion: no-preference)`.

## Testing Strategy
- `scripts/check.py`: links resolve (doors included); new test that every `<a>` inside the scene has an
  `aria-label`; page weight of `index.html` + `style.css` under 150 KB uncompressed.
- Browser at 375px, 768px and 1440px: no horizontal scroll; all four doors visible and clickable; Tab
  reaches each door in a sensible order with a visible focus ring; labels appear on focus; no console errors.
- Performance: the home makes no third-party requests; first render does not wait on JS.
- Art review: a screenshot at each width, checked against the objective's object list.

## Boundaries
- Always: original artwork drawn as SVG paths here; doors are plain links that work without JS.
- Ask first: raster images, any library, any sound.
- Never: recreate felixrieseberg.com's room, objects or layout; autoplay anything; trap keyboard focus.

## Success Criteria
- [ ] Four working doors with hover and focus states; text door list below the scene
- [ ] Ambient animation present, and fully off under reduced motion
- [ ] Works and stays navigable at 375px; no horizontal scroll at any tested width
- [ ] Home weight under 150 KB; no third-party requests; no console errors
