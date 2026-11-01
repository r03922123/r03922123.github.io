# Spec: room

Module of `CAPABILITY-MAP.md`. Depends on `theme`. Supersedes `scene`. Status: **approved** 2026-11-01.

## Objective
The home page is a small castle common room at night, drawn live in low-poly 3D with the look of a 2000s
console game: a low internal resolution scaled up with hard pixels, slightly wobbling vertices, flat-shaded stone,
wood and fabric, warm firelight against cool moonlight. The camera breathes slowly; nothing needs to be learned.

| Object | Opens | On hover / focus |
|---|---|---|
| Bookshelf | `posts/` | books glow, label "POSTS" (the bookshelf) |
| Open spellbook on a lectern | `leetcode/` | a page turns, label "LEETCODE" (the spellbook · 31 pattern notes) |
| Three portraits above the mantel: KK, TT, HTC | `about.html#kkcompany`, `#tiktok`, `#htc` | that frame lights up, label "ABOUT · KKCOMPANY" etc. |

Ambience (no links): a gramophone on a side table, a crackling fireplace (flickering light and a few rising embers), candles, two armchairs and a rug,
a tall window with moon and stars, a tapestry. Portraits show emblems (a note, a play triangle, goggles), not faces.
A game-style title card ("Jie-Hong Lin — machine learning engineer. Choose something in the room.") sits over the
top of the view and never blocks it. A plain text list of the same doors sits below.

Everything is original: no Harry Potter names, crests or characters, no recreation of a specific game's room, and
nothing taken from felixrieseberg.com.

## Tech Stack
- three.js r169, **vendored** at `vendor/three/0.169.0/three.module.min.js` with its MIT licence (no third-party
  request on the home page), loaded by `room.js` as an ES module.
- The room is built in code from boxes, cylinders and a few extruded shapes; textures are drawn at load time on small
  canvases (stone, planks, cloth) with nearest-neighbour filtering. No model or image files.
- Era look: render the canvas itself at about 270 px tall and scale it up by a whole number with `image-rendering: pixelated`;
  a vertex-snapping tweak on the materials for the slight PS1-style wobble; fog for depth.

## Commands
```
Preview: python3 -m http.server 8001
Check:   python3 scripts/check.py && (cd scripts && python3 -m unittest test_check)
```

## Project Structure
```
index.html                         → title card, <canvas>, door overlay links, text door list, lo-fi view
room.js                            → scene construction, render loop, picking, door overlay positioning
vendor/three/0.169.0/              → three.module.min.js + LICENSE
style.css (room)                   → canvas, overlay, title card
```

## Code Style
Doors are real links even though the objects are pixels: an overlay of `<a>` elements is positioned over each object
by projecting its centre every frame; they are focusable, labeled, and show the same label as hover.
```html
<nav class="rm-doors" aria-label="Things in the room">
  <a class="rm-door" href="posts/" data-object="bookshelf">Posts</a>
  ...
</nav>
```
`room.js` is plain ES-module JavaScript, no build step; every class name for the room is prefixed `rm-`.

## Testing Strategy
- Browser at 375, 768 and 1440 px: the five door links (bookshelf, spellbook, three portraits) are visible, hover and Tab show their labels, Enter opens the
  right page, no horizontal scroll, no console errors.
- Performance: render loop capped near 30 fps (the era's frame rate), paused when the tab is hidden or the canvas is
  off-screen; first paint of the page (title card, text list) does not wait for three.js.
- Reduced motion: a single still frame, no camera drift, no flicker. No WebGL: lo-fi view automatically.
- `check.py`: links, page weight of `index.html` + `style.css` under 150 KB (three.js is counted separately and
  lazy-loaded), named overlay links.

## Boundaries
- Always: the overlay links and the text list make every door reachable without the canvas.
- Ask first: sound, external models or textures, any other library.
- Never: Harry Potter or other franchise marks; copying another site's art or code; autoplay of anything.

## Success Criteria
- [ ] Room renders with the era look on desktop and phone; five door links work by mouse, touch and keyboard
- [ ] 30 fps cap, pause when hidden, still frame under reduced motion, lo-fi when WebGL is missing
- [ ] No third-party requests; home HTML + CSS under 150 KB
