# Spec: lofi

Module of `CAPABILITY-MAP.md`. Depends on `scene`. Status: **approved** 2026-10-31.

## Objective
An escape hatch from the scene: a plain, fast, text-only home with the same content. A **lo-fi** toggle
in the corner switches between the scene and the plain home; the choice is remembered on that device.
Visitors whose OS asks for reduced motion start in lo-fi, and can still switch to the (motionless) scene.

The plain home shows: the greeting, the four sections as a list (Posts, LeetCode, About, MusicGen), and
the latest three posts.

## Tech Stack
A `data-mode="scene|lofi"` attribute on `<html>`, set by an inline script in `<head>` before first paint
(no flash), and CSS that shows one view or the other. Storage through `localStorage`, every access
wrapped in try/catch so a private window or blocked storage falls back to the default.

## Commands
```
Preview: python3 -m http.server 8000
Check:   python3 scripts/check.py && (cd scripts && python3 -m unittest test_check)
```

## Project Structure
```
index.html        → <head> mode script; the toggle button; the lo-fi view next to the scene
style.css (lofi)  → [data-mode="lofi"] rules
```

## Code Style
```html
<script>
  // Pick the home view before first paint: stored choice, else lo-fi for reduced-motion visitors.
  (function () {
    var mode = null;
    try { mode = localStorage.getItem("home-mode"); } catch (e) {}
    if (mode !== "scene" && mode !== "lofi")
      mode = matchMedia("(prefers-reduced-motion: reduce)").matches ? "lofi" : "scene";
    document.documentElement.dataset.mode = mode;
  })();
</script>
```
The toggle is a `<button aria-pressed>` labeled "lo-fi", hidden when JS is off (both views are then
available: the scene renders, and the text door list beneath it covers the lo-fi need).

## Testing Strategy
- Browser: toggling switches views and updates `aria-pressed`; reload keeps the choice; clearing storage
  returns to the default; with `localStorage` throwing (simulated by overriding it), the page still renders.
- Reduced motion: with the media query forced to match (by overriding `matchMedia` before the script runs),
  the default is lo-fi.
- No flash: the mode attribute is present before the body renders (script is inline in `<head>`).

## Boundaries
- Always: both views reachable by keyboard; the lo-fi view needs no JS to read.
- Never: store anything but the one mode string; use cookies.

## Success Criteria
- [ ] Toggle works, is remembered, and survives blocked storage
- [ ] Reduced-motion visitors start in lo-fi
- [ ] No flash of the wrong view on load
