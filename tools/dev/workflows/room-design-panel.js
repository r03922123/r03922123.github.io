export const meta = {
  name: 'room-design-panel',
  description: 'Two designers build candidate low-poly 3D common rooms in three.js, two judges score them, a builder refines the winner and a critic hardens it',
  phases: [
    { title: 'Draft', detail: 'two independent rooms from different art directions, self-checked with WebGL screenshots' },
    { title: 'Judge', detail: 'art and charm; usability, accessibility, spec and performance' },
    { title: 'Refine', detail: 'winner plus grafts, then an adversarial critic' },
  ],
}

const S = '/Users/linjiehong/Desktop/projects/r03922123.github.io/.cache'
const SITE = '/Users/linjiehong/Desktop/projects/r03922123.github.io'
const BRIEF = `
You are building the interactive home page of a personal site (an ML engineer's portfolio and educational blog), as a
standalone draft. Read the approved spec first: ${SITE}/SPEC-room.md, and for context ${SITE}/SPEC-theme.md (parchment /
candlelit palette, Cinzel display font) and ${SITE}/CAPABILITY-MAP.md.

THE ROOM: a small castle common room at night, rendered LIVE in low-poly 3D with three.js, with the look and feel of a
2000s console game set in a wizarding school (think PS2 era): low internal resolution scaled up with hard pixels
(image-rendering: pixelated), a subtle PS1-style vertex wobble (snap clip-space positions in the vertex shader via
material.onBeforeCompile), flat/Lambert shading, warm flickering firelight against cool moonlight, fog for depth,
small procedural canvas textures (stone blocks, wood planks, cloth weave, book spines) with NearestFilter. Built only
from three.js primitives and shapes in code: NO model files, NO image files, NO external requests.
Game-like UI: a title card at the top over the view ("Jie-Hong Lin — machine learning engineer. Choose something in
the room.") styled as a parchment panel with a gold border in the Cinzel font; hover/focus labels styled the same way.
Four objects are the navigation ("doors"):
- Bookshelf -> href "posts/"  label "POSTS"  (books glow)
- Open spellbook on a lectern -> href "leetcode/"  label "LEETCODE · 31"  (pages flutter)
- Three portraits above the mantel, emblems not faces (KK: a music note; TT: a play triangle; HTC: VR goggles)
  -> hrefs "about.html#kkcompany", "about.html#tiktok", "about.html#htc"; labels "ABOUT · KKCOMPANY", "ABOUT · TIKTOK", "ABOUT · HTC"
- Gramophone -> href "posts/#musicgen"  label "MUSICGEN"  (horn glows)
Ambience: a crackling fireplace (flickering light, a few rising embers), candles, two armchairs and a rug, a tall arched
window with moon and stars, an original tapestry/banner pattern. Slow camera breathing / gentle sway.
STRICTLY ORIGINAL: no Harry Potter names, house crests, characters, lion/serpent/badger/eagle house emblems, or
recreations of any specific game's room; nothing copied from felixrieseberg.com.

ACCESSIBILITY AND STRUCTURE (required):
- Doors must be real links: an overlay <nav class="rm-doors" aria-label="Things in the room"> of <a class="rm-door"> elements,
  positioned over each object by projecting the object's centre to screen every frame; focusable in this order: bookshelf,
  spellbook, KK, TT, HTC, gramophone; :focus-visible and :hover on a link show the same label/highlight as hovering the
  object in 3D (raycast picking on pointermove; click on an object navigates). Overlay links are invisible boxes by default
  (the 3D object is the visual), with a clear visible focus ring.
- Below the view: a plain text list of the same doors (Posts, LeetCode, About, MusicGen) with one-line descriptions.
- Reduced motion (prefers-reduced-motion: reduce): render one still frame, no sway, no flicker, no embers.
- No WebGL: show the text list only (the room area shows a short parchment note), no errors.
- Performance: cap the render loop near 30 fps, pause when document.hidden or the canvas is off-screen (IntersectionObserver),
  handle resize, keep geometry modest (it must run on a phone).
- Phone (375 px wide): portrait framing (adjust camera/FOV) so all four doors stay visible and tappable (>= 44 px targets);
  no horizontal scroll.
- Every class name prefixed "rm-" (it will be merged into a site stylesheet that uses .post, .intro, .label, .toc, .series,
  .role, .wrap). Plain ES-module JavaScript, no build step, no other libraries.

FILES: a draft folder ${S}/room/<id>/ containing index.html and room.js. Import three with
  import * as THREE from "../shared/three.module.min.js";
and use the font at ../shared/cinzel-latin.woff2 via @font-face. A static server already serves ${S}/room at http://localhost:8767/
(so your draft is at http://localhost:8767/<id>/). Do not touch anything in ${SITE}.

SCREENSHOTS (WebGL works headless via SwiftShader, but the machine is heavily loaded, so each shot can take up to ~75 s):
  /Users/linjiehong/Desktop/projects/r03922123.github.io/tools/dev/shoot3d.sh <name> desktop http://localhost:8767/<id>/
  /Users/linjiehong/Desktop/projects/r03922123.github.io/tools/dev/shoot3d.sh <name>-phone phone http://localhost:8767/<id>/
It prints the PNG path (or NO SCREENSHOT; then retry once). Read the PNG and look at it. Do this loop at least three times.
For hover/focus states you may add a temporary URL hash like #focus=bookshelf that your code reads to force a highlight,
but remove it from the final code or keep it harmless.`

const ANGLES = [
  { id: 'fireside', angle: 'FIRESIDE: a cozy, warm, intimate room seen from a seated eye level facing a big stone fireplace; portraits above the mantel, bookshelf on the left wall, lectern with spellbook front-right, gramophone on a side table, armchairs angled toward the fire, a tall arched window beside the fireplace with moonlight spilling in.' },
  { id: 'tower', angle: 'TOWER: a round tower common room with high arched windows full of stars, a curving bookshelf wall, candles in iron sconces, the fireplace off to one side with portraits above it, a lectern in the centre on a round rug, the gramophone by the window; more vertical drama and a slow orbiting sway of the camera.' },
]
const DRAFT = { type: 'object', properties: { id: { type: 'string' }, url: { type: 'string' }, bytes: { type: 'number' },
  screenshots: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } }, required: ['id', 'url', 'bytes', 'screenshots', 'notes'] }
const JUDGE = { type: 'object', properties: { lens: { type: 'string' },
  scores: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, score: { type: 'number' }, strengths: { type: 'array', items: { type: 'string' } }, defects: { type: 'array', items: { type: 'string' } } }, required: ['id', 'score', 'strengths', 'defects'] } },
  winner: { type: 'string' }, graft: { type: 'array', items: { type: 'string' } } }, required: ['lens', 'scores', 'winner', 'graft'] }
const FINAL = { type: 'object', properties: { folder: { type: 'string' }, screenshots: { type: 'array', items: { type: 'string' } },
  changes: { type: 'array', items: { type: 'string' } }, open_issues: { type: 'array', items: { type: 'string' } } }, required: ['folder', 'screenshots', 'changes', 'open_issues'] }

phase('Draft')
const drafts = (await parallel(ANGLES.map(a => () => agent(`${BRIEF}\nYour art direction: ${a.angle}\nYour draft folder id: "${a.id}". Screenshot names start with "${a.id}". Return the structured summary.`,
  { label: `draft:${a.id}`, phase: 'Draft', schema: DRAFT })))).filter(Boolean)
log(`${drafts.length} drafts done`)

phase('Judge')
const LENSES = [
  'ART AND CHARM: does it feel like a cozy, magical 2000s console-game common room a visitor wants to explore? Composition, lighting (fire vs moon), the era look (pixelation, wobble, textures), silhouettes, how the four doors read as objects, animation feel, originality (and NO franchise marks).',
  'USABILITY, ACCESSIBILITY, SPEC AND PERFORMANCE: overlay links (order, labels, focus ring, alignment over objects at 375/768/1440 px), raycast hover and click correctness, text door list, reduced motion still frame, no-WebGL fallback, 30 fps cap and pause when hidden/off-screen, phone framing and tap targets, no horizontal scroll, rm- prefixes, no external requests, code clarity and how cleanly it merges into the site.',
]
const judgements = (await parallel(LENSES.map((lens, i) => () => agent(`${BRIEF}\nYou are judge ${i + 1} of 2. Lens: ${lens}\nDrafts (read every file; take your own screenshots, names starting "rjudge${i + 1}-"; you may also probe with headless Chrome over CDP using your own --user-data-dir under ${S}):\n${drafts.map(d => `- ${d.id}: ${d.url} (designer notes: ${d.notes})`).join('\n')}\nScore each 1-10 under your lens, list concrete strengths and defects, name a winner, and list specific ideas from the other draft worth grafting.`,
  { label: `judge:${i ? 'spec' : 'art'}`, phase: 'Judge', schema: JUDGE })))).filter(Boolean)
const totals = {}
for (const j of judgements) for (const s of j.scores) totals[s.id] = (totals[s.id] || 0) + s.score
const winner = Object.entries(totals).sort((a, b) => b[1] - a[1])[0][0]
log(`scores ${JSON.stringify(totals)}; winner ${winner}`)

phase('Refine')
const final = await agent(`${BRIEF}\nTwo drafts were judged. Totals: ${JSON.stringify(totals)}. Winner: ${winner}.\nDrafts: ${drafts.map(d => d.id + ' at ' + S + '/room/' + d.id + '/').join('; ')}\nJudgements (verbatim): ${JSON.stringify(judgements)}\nBuild the final room in ${S}/room/final/ (index.html + room.js): start from the winner, fix every defect the judges listed for it, and graft the listed ideas from the other draft where they improve it. Screenshot loop (names "rfinal") until desktop and phone both look excellent. Return what changed and anything open.`,
  { label: 'refine', phase: 'Refine', schema: FINAL })
const critic = await agent(`${BRIEF}\nAdversarial final check of ${S}/room/final/ against SPEC-room.md and every requirement in this brief. Assume something is wrong. Verify by reading the code AND by screenshots (names "rcritic") and, where useful, by driving headless Chrome over CDP (real Tab presses, pointer moves, emulated reduced motion, a run with WebGL disabled via --disable-webgl, document.hidden pause). Fix every real issue in the final folder and re-screenshot. The refiner reported: ${JSON.stringify(final)}`,
  { label: 'critic', phase: 'Refine', schema: FINAL })
return { totals, winner, final, critic }