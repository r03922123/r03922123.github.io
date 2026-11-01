// room.js: the home page's common room (SPEC-room). A castle common room at night, built in code from low-poly
// shapes and tiny canvas textures, drawn at a low resolution and scaled up with hard pixels, snapped vertices and
// fog: the look of a 2000s console game. The doors are the real <a> elements in .rm-doors; this file draws the room,
// moves each link over its object, and lights the object up while its link is hovered or focused.
// three.js is imported only when the room is shown (never in lo-fi mode) and only when the browser has WebGL.
const THREE_URL = "/vendor/three/0.169.0/three.module.min.js";
const root = document.documentElement;
const stage = document.getElementById("rm-stage");
let room = null, starting = false;

const GL_OPTIONS = { antialias: false, alpha: false, stencil: false, powerPreference: "low-power" };
function hasWebGL() {                        // three.js r169 needs WebGL2; release the probe's context at once
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    return !!gl;
  } catch (e) { return false; }
}
function setMode(mode) {                     // the attribute and the toggle's pressed state, kept in step
  root.setAttribute("data-mode", mode);
  document.querySelector(".mode-toggle")?.setAttribute("aria-pressed", String(mode === "lofi"));
}
function giveUp() {                          // no WebGL: the plain home, and no toggle (SPEC-lofi amendment)
  root.classList.add("no-webgl");
  try { localStorage.setItem("home-mode", "lofi"); } catch (e) {}   // so the next visit does not flash the loader
  setMode("lofi");
}
async function sync() {
  if (!stage) return;
  if (root.classList.contains("no-webgl")) { if (root.getAttribute("data-mode") !== "lofi") setMode("lofi"); return; }
  if (root.getAttribute("data-mode") !== "scene") { if (room) room.pause(); return; }
  if (room) { room.resume(); return; }
  if (starting) return;
  starting = true;
  let gl = null;                             // made on the room's own canvas and handed to three.js: one context
  try { gl = stage.querySelector(".rm-canvas").getContext("webgl2", GL_OPTIONS); } catch (e) {}
  if (!gl) { giveUp(); return; }
  let THREE;
  try { THREE = await import(THREE_URL); }
  catch (e) { starting = false; setMode("lofi"); return; }       // a network failure, not a missing GPU: the toggle retries
  try { room = build(THREE, gl); }
  catch (e) { gl.getExtension("WEBGL_lose_context")?.loseContext(); giveUp(); return; }
  if (root.getAttribute("data-mode") !== "scene") room.pause();   // switched to lo-fi while three.js was loading
}
new MutationObserver(sync).observe(root, { attributes: true, attributeFilter: ["data-mode"] });
// Creating a WebGL context can take seconds while a cold GPU wakes up, so let the title card and door list paint first.
requestAnimationFrame(() => setTimeout(sync));
// A visitor who starts in lo-fi never reaches the check above, so probe once when the page is idle: without WebGL
// the toggle would only lead to a room that cannot render.
if (stage && root.getAttribute("data-mode") === "lofi")
  (window.requestIdleCallback || setTimeout)(() => { if (!room && !starting && !hasWebGL()) giveUp(); });

function rng(seed) {                         // mulberry32: the same room on every visit
  return function () {
    seed |= 0; seed = seed + 0x6d2b79f5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function build(THREE, context) {
  const rand = rng(20011116);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  const canvas = stage.querySelector(".rm-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, context, ...GL_OPTIONS });
  renderer.setPixelRatio(1);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07060b);
  scene.fog = new THREE.Fog(0x0b0812, 10, 22);
  const camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 40);

  // ---- The era look: every vertex snaps to the low-resolution pixel grid, so edges shimmer as the camera drifts.
  const snap = { value: new THREE.Vector2(240, 135) };
  function ps1(m) {
    m.onBeforeCompile = (s) => {
      s.uniforms.uSnap = snap;
      s.vertexShader = "uniform vec2 uSnap;\n" + s.vertexShader.replace("#include <project_vertex>",
        "#include <project_vertex>\n\tgl_Position.xy = floor(gl_Position.xy / gl_Position.w * uSnap + .5) / uSnap * gl_Position.w;");
    };
    return m;
  }
  const lam = (o) => ps1(new THREE.MeshLambertMaterial({ flatShading: true, ...o }));
  const flat = (o) => ps1(new THREE.MeshBasicMaterial(o));
  const glow = (o) => new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, fog: false,
                                                    blending: THREE.AdditiveBlending, ...o });

  // ---- Textures: drawn once on tiny canvases, sampled with nearest-neighbour filtering and no mipmaps.
  function tex(w, h, draw, rx = 1, ry = 1) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    draw(c.getContext("2d"), w, h);
    const t = new THREE.CanvasTexture(c);
    t.magFilter = t.minFilter = THREE.NearestFilter;
    t.generateMipmaps = false;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(rx, ry);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }
  const px = (g, x, y, w, h, c) => { g.fillStyle = c; g.fillRect(x, y, w, h); };
  const hsl = (h, s, l) => `hsl(${h},${s}%,${l}%)`;
  function speckle(g, w, h, n, a) {
    for (let i = 0; i < n; i++)
      px(g, rand() * w | 0, rand() * h | 0, 1, 1, rand() < .55 ? `rgba(0,0,0,${a})` : `rgba(255,236,210,${a * .6})`);
  }
  function wrapRect(g, x, y, w, h, c, W) { px(g, x, y, w, h, c); if (x + w > W) px(g, x - W, y, w, h, c); if (x < 0) px(g, x + W, y, w, h, c); }
  const stone = (hue, sat, base) => (g, w, h) => {
    px(g, 0, 0, w, h, hsl(hue, sat, base - 18));
    for (let y = 0; y < h; y += 8) {
      let x = rand() * 16 | 0;
      const end = x + w;
      while (x < end) {
        const bw = 9 + (rand() * 10 | 0), l = base + rand() * 12 - 6, bx = x % w;
        wrapRect(g, bx + 1, y + 1, bw - 1, 7, hsl(hue, sat, l), w);
        wrapRect(g, bx + 1, y + 1, bw - 1, 1, hsl(hue, sat, l + 8), w);
        wrapRect(g, bx + 1, y + 7, bw - 1, 1, hsl(hue, sat, l - 9), w);
        x += bw;
      }
    }
    speckle(g, w, h, 520, .2);
  };
  const planks = (hue, sat, base, n = 4) => (g, w, h) => {
    const pw = w / n;
    for (let i = 0; i < n; i++) {
      const l = base + rand() * 8 - 4, x0 = i * pw;
      px(g, x0, 0, pw, h, hsl(hue, sat, l));
      for (let k = 0; k < 6; k++) px(g, x0 + 1 + (rand() * (pw - 2) | 0), 0, 1, h, hsl(hue, sat, l - 4 - rand() * 5));
      px(g, x0, 0, 1, h, hsl(hue, sat, l - 15));
      px(g, x0, rand() * h | 0, pw, 1, hsl(hue, sat, l - 15));
      px(g, x0 + 3 + (rand() * (pw - 6) | 0), rand() * h | 0, 2, 2, hsl(hue, sat, l - 12));
    }
    speckle(g, w, h, 160, .12);
  };
  const wallT = tex(64, 64, stone(28, 9, 36), 3.7, 2.2);
  const sideT = tex(64, 64, stone(28, 9, 36), 4.5, 2.2);
  const breastT = tex(64, 64, stone(30, 12, 44), 1.2, 2.2);
  const hearthT = tex(64, 64, stone(30, 10, 48), .8, .5);
  const floorT = tex(64, 64, planks(26, 34, 26), 3.7, 4.5);
  const ceilT = tex(64, 64, planks(24, 26, 13), 3.7, 4.5);
  const woodT = tex(32, 32, planks(20, 32, 22, 2));
  const lightWoodT = tex(32, 32, planks(18, 45, 32, 2));
  const fabricT = tex(16, 16, (g, w, h) => {
    px(g, 0, 0, w, h, "#6a1a20");
    for (let y = 0; y < h; y += 2) px(g, 0, y, w, 1, "#581418");
    for (let x = 0; x < w; x += 4) px(g, x, 0, 1, h, "#74222a");
    speckle(g, w, h, 50, .2);
  }, 2, 2);
  const rugT = tex(48, 32, (g, w, h) => {
    px(g, 0, 0, w, h, "#4a1012"); px(g, 2, 2, w - 4, h - 4, "#c08d34"); px(g, 3, 3, w - 6, h - 6, "#2c0a0d");
    px(g, 5, 5, w - 10, h - 10, "#721a1a");
    for (let x = 6; x < w - 6; x += 3) { px(g, x, 4, 1, 1, "#d9a441"); px(g, x, h - 5, 1, 1, "#d9a441"); }
    for (let y = 6; y < h - 6; y += 3) { px(g, 4, y, 1, 1, "#d9a441"); px(g, w - 5, y, 1, 1, "#d9a441"); }
    const cx = w / 2, cy = h / 2;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const d = Math.abs(x + .5 - cx) / 1.5 + Math.abs(y + .5 - cy);
      if (d < 9.5 && d > 8.2) px(g, x, y, 1, 1, "#d9a441");
      else if (d < 6) px(g, x, y, 1, 1, d < 2.2 ? "#e2b04f" : d < 3.4 ? "#721a1a" : "#1f2c4a");
    }
    for (const [x, y] of [[10, 8], [w - 11, 8], [10, h - 9], [w - 11, h - 9]]) px(g, x - 1, y - 1, 3, 3, "#1f2c4a");
    speckle(g, w, h, 140, .16);
  });
  const pageT = tex(24, 32, (g, w, h) => {
    px(g, 0, 0, w, h, "#e9dab0");
    px(g, 3, 3, 5, 5, "#8a2a1a");
    for (let y = 4; y < h - 3; y += 3)
      for (let x = y < 9 ? 10 : 3; x < w - 3; x++) if (rand() < .78) px(g, x, y, 1, 1, rand() < .3 ? "#8c7a5c" : "#5b4a36");
    px(g, w - 1, 0, 1, h, "#c9b68a");
    speckle(g, w, h, 30, .08);
  });
  const winT = tex(32, 64, (g, w, h) => {
    const sky = g.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, "#0a1230"); sky.addColorStop(.7, "#22336a"); sky.addColorStop(1, "#3c4a7c");
    g.fillStyle = sky; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 40; i++) px(g, rand() * w | 0, rand() * h * .7 | 0, 1, 1, rand() < .3 ? "#ffffff" : "#9fb0e8");
    g.fillStyle = "#f4ecd0"; g.beginPath(); g.arc(21.5, 15.5, 5, 0, 7); g.fill();
    px(g, 19, 14, 2, 2, "#d8cfae"); px(g, 23, 17, 1, 1, "#d8cfae");
    g.fillStyle = "#0b1024"; g.beginPath(); g.moveTo(0, h);
    for (let x = 0; x <= w; x += 4) g.lineTo(x, h - 9 - Math.abs(Math.sin(x * .21)) * 9 - (x > 18 && x < 26 ? 8 : 0));
    g.lineTo(w, h); g.fill();
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
      if ((x + y) % 8 === 0 || (x - y + 80) % 8 === 0) px(g, x, y, 1, 1, "rgba(14,10,12,.55)");
    px(g, w / 2 - 1, 0, 2, h, "#2c2522"); px(g, 0, h * .42 | 0, w, 2, "#2c2522");
    g.globalCompositeOperation = "destination-in";
    g.beginPath(); g.moveTo(0, h); g.lineTo(0, w / 2); g.arc(w / 2, w / 2, w / 2, Math.PI, 0); g.lineTo(w, h); g.fill();
  });
  const archT = tex(32, 64, (g, w, h) => {
    stone(30, 12, 46)(g, w, h);
    g.globalCompositeOperation = "destination-in";
    g.beginPath(); g.moveTo(0, h); g.lineTo(0, w / 2); g.arc(w / 2, w / 2, w / 2, Math.PI, 0); g.lineTo(w, h); g.fill();
  });
  const latticeT = tex(32, 32, (g, w, h) => {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
      if (!((x + y) % 7 === 0 || (x - y + 70) % 7 === 0 || x === 15 || x === 16)) px(g, x, y, 1, 1, "#7f95d8");
  });
  const shaftT = tex(4, 32, (g, w, h) => {
    const s = g.createLinearGradient(0, 0, 0, h);
    s.addColorStop(0, "#8ea4e8"); s.addColorStop(1, "#000000");
    g.fillStyle = s; g.fillRect(0, 0, w, h);
  });
  const bannerT = tex(24, 48, (g, w, h) => {
    px(g, 0, 0, w, h, "#1b2a58"); px(g, 1, 1, w - 2, h - 2, "#c39236"); px(g, 2, 2, w - 4, h - 4, "#1b2a58");
    for (let y = 4; y < h - 8; y += 4) { px(g, 3, y, 1, 1, "#c39236"); px(g, w - 4, y, 1, 1, "#c39236"); }
    g.fillStyle = "#d9a441"; g.beginPath(); g.arc(12, 15, 6, 0, 7); g.fill();       // a crescent moon
    g.fillStyle = "#1b2a58"; g.beginPath(); g.arc(14.5, 13.5, 5, 0, 7); g.fill();
    for (const [x, y] of [[7, 25], [12, 29], [17, 25]]) {                            // and three stars
      px(g, x, y - 2, 1, 5, "#e6c36a"); px(g, x - 2, y, 5, 1, "#e6c36a"); px(g, x, y, 1, 1, "#fff3c4");
    }
    for (let y = 33; y < 37; y++) px(g, 5, y, w - 10, 1, y % 2 ? "#c39236" : "#7a1c1a");
    g.globalCompositeOperation = "destination-out";                                // swallowtail hem
    g.beginPath(); g.moveTo(4, h); g.lineTo(w / 2, h - 7); g.lineTo(w - 4, h); g.fill();
    speckle(g, w, h, 40, .12);
  });
  function painting(bg, draw) {
    return tex(24, 30, (g, w, h) => {
      px(g, 0, 0, w, h, bg[0]); px(g, 2, 2, w - 4, h - 4, bg[1]); px(g, 5, 5, w - 10, h - 10, bg[2]);
      g.fillStyle = "#e9d49a"; g.strokeStyle = "#e9d49a"; draw(g);
      speckle(g, w, h, 70, .14);
    });
  }
  const kkT = painting(["#1d2a20", "#25382a", "#2c4433"], (g) => {          // KKCompany: a note
    px(g, 13, 7, 2, 13, "#e9d49a"); px(g, 15, 7, 3, 2, "#e9d49a"); px(g, 17, 9, 2, 3, "#e9d49a");
    g.beginPath(); g.ellipse(11.5, 20.5, 3.2, 2.3, -.4, 0, 7); g.fill();
  });
  const ttT = painting(["#2a1424", "#3a1a30", "#48213c"], (g) => {          // TikTok: a play triangle
    g.beginPath(); g.moveTo(9, 8); g.lineTo(18, 15); g.lineTo(9, 22); g.closePath(); g.fill();
    px(g, 7, 24, 11, 1, "#b98d5a");
  });
  const htcT = painting(["#141c30", "#1a2540", "#22304f"], (g) => {         // HTC: goggles
    px(g, 4, 14, 16, 2, "#9a7a4a");
    g.lineWidth = 2;
    g.beginPath(); g.arc(8.5, 15, 3.2, 0, 7); g.stroke();
    g.beginPath(); g.arc(15.5, 15, 3.2, 0, 7); g.stroke();
    px(g, 7, 14, 3, 2, "#7fb3d8"); px(g, 14, 14, 3, 2, "#7fb3d8");
  });
  const fireTs = [0, 1, 2].map((k) => tex(16, 32, (g, w, h) => {
    const img = g.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = y / (h - 1);
      const half = .5 * Math.pow(v, .75) * (.8 + .2 * Math.sin(y * .9 + k * 2.1)) * (v > .82 ? 1 - (v - .82) * 1.6 : 1);
      const a = 1 - Math.abs((x + .5) / w - .5 + Math.sin(y * .45 + k) * .05) / Math.max(half, .001);
      if (a <= 0) continue;
      const heat = a * v + rand() * .08;
      const c = heat > .55 ? [255, 246, 205] : heat > .32 ? [255, 190, 80] : heat > .14 ? [235, 105, 30] : [150, 36, 14];
      const i = (y * w + x) * 4;
      img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }));
  const haloT = tex(32, 32, (g, w, h) => {
    const r = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    r.addColorStop(0, "#ffffff"); r.addColorStop(1, "#000000");
    g.fillStyle = r; g.fillRect(0, 0, w, h);
  });
  [haloT, shaftT].forEach((t) => { t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; });

  // ---- Shapes.
  function box(w, h, d, mat, x, y, z, parent = scene) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z); parent.add(m); return m;
  }
  function cyl(rt, rb, h, seg, mat, x, y, z, parent = scene) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), mat);
    m.position.set(x, y, z); parent.add(m); return m;
  }
  function plane(w, h, mat, x, y, z, parent = scene) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    m.position.set(x, y, z); parent.add(m); return m;
  }

  const stoneM = lam({ map: wallT }), sideM = lam({ map: sideT }), breastM = lam({ map: breastT });
  const hearthM = lam({ map: hearthT }), woodM = lam({ map: woodT }), lightWoodM = lam({ map: lightWoodT });
  const darkM = lam({ color: 0x1a1210 }), sootM = lam({ color: 0x120a08 });
  const fabricM = lam({ map: fabricT }), brassM = lam({ color: 0xc79a42, side: THREE.DoubleSide });
  const candleM = lam({ color: 0xeee0bf, emissive: 0x3a2a12 });

  // The room: 7.4 m wide, the back wall at z = -4, the camera near z = 5.
  const floor = plane(7.4, 9, lam({ map: floorT }), 0, 0, .5); floor.rotation.x = -Math.PI / 2;
  const ceil = plane(7.4, 9, lam({ map: ceilT }), 0, 4.4, .5); ceil.rotation.x = Math.PI / 2;
  plane(7.4, 4.4, stoneM, 0, 2.2, -4);
  const lw = plane(9, 4.4, sideM, -3.7, 2.2, .5); lw.rotation.y = Math.PI / 2;
  const rw = plane(9, 4.4, sideM, 3.7, 2.2, .5); rw.rotation.y = -Math.PI / 2;
  for (const z of [-3.2, -1.2, .8, 2.8]) box(7.4, .26, .3, darkM, 0, 4.27, z);
  box(7.4, .16, .06, darkM, 0, .08, -3.97);

  // Fireplace, mantel, and the fire.
  box(2.4, 4.4, .35, breastM, 0, 2.2, -3.825);
  box(.36, 1.32, .55, hearthM, -.86, .66, -3.55);
  box(.36, 1.32, .55, hearthM, .86, .66, -3.55);
  box(2.08, .3, .58, hearthM, 0, 1.45, -3.55);
  box(1.36, 1.3, .06, sootM, 0, .65, -3.62);
  box(2.7, .08, .95, hearthM, 0, .04, -3.3);
  box(2.7, .12, .62, woodM, 0, 1.66, -3.45);
  const logs = cyl(.08, .08, .95, 6, lightWoodM, 0, .15, -3.42); logs.rotation.z = Math.PI / 2;
  const log2 = cyl(.07, .07, .8, 6, woodM, .04, .27, -3.47); log2.rotation.set(0, .35, Math.PI / 2);
  const flames = [[-.27, .42, .62, -3.42], [0, .58, .9, -3.37], [.25, .42, .6, -3.4], [.08, .3, .44, -3.33], [-.1, .28, .4, -3.34]]
    .map(([x, w, h, z], i) => {
      const g = new THREE.PlaneGeometry(w, h); g.translate(0, h / 2, 0);
      const m = new THREE.Mesh(g, glow({ map: fireTs[i % 3] }));
      m.position.set(x, .16, z); m.userData = { h, phase: i * 1.7, k: i }; scene.add(m); return m;
    });
  const halo = plane(2.6, 2, glow({ map: haloT, color: 0xff7a2a, opacity: .38 }), 0, .7, -3.2);
  const floorGlow = plane(2.8, 2.2, glow({ map: haloT, color: 0xff7a2a, opacity: .22 }), 0, .02, -2.6);
  floorGlow.rotation.x = -Math.PI / 2;

  // Candles: on the mantel and in two wall sconces.
  const candleFlames = [];
  function candle(x, y, z, h = .22, parent = scene) {
    cyl(.035, .035, h, 6, candleM, x, y + h / 2, z, parent);
    const f = plane(.07, .13, glow({ map: fireTs[candleFlames.length % 3] }), x, y + h + .06, z + .01, parent);
    const hl = plane(.36, .36, glow({ map: haloT, color: 0xffb060, opacity: .3 }), x, y + h + .06, z + .02, parent);
    candleFlames.push({ f, hl, phase: rand() * 6 });
  }
  for (const x of [-1.18, -1.02, 1.02, 1.18]) candle(x, 1.72, -3.4, x * x > 1.2 ? .26 : .18);
  function sconce(x, z, face) {
    const g = new THREE.Group(); g.position.set(x, 2.05, z); g.rotation.y = face; scene.add(g);
    box(.1, .26, .05, brassM, 0, 0, 0, g); box(.06, .04, .22, brassM, 0, -.08, .11, g);
    cyl(.07, .04, .05, 6, brassM, 0, -.04, .22, g);
    candle(0, -.02, .22, .2, g);
  }
  sconce(-3.68, -2.2, Math.PI / 2);
  sconce(3.68, -1.35, -Math.PI / 2);

  // Rug and two armchairs facing the fire.
  const rug = plane(3.3, 2.2, lam({ map: rugT }), 0, .012, -1.45); rug.rotation.x = -Math.PI / 2;
  function chair(x, z, ry) {
    const g = new THREE.Group(); g.position.set(x, 0, z); g.rotation.y = ry; scene.add(g);
    box(.82, .26, .78, woodM, 0, .13, 0, g);
    box(.74, .16, .7, fabricM, 0, .34, .02, g);
    box(.82, .9, .2, fabricM, 0, .78, -.32, g);
    box(.84, .14, .24, fabricM, 0, 1.22, -.33, g);
    box(.15, .38, .78, fabricM, -.4, .48, 0, g);
    box(.15, .38, .78, fabricM, .4, .48, 0, g);
    box(.12, .3, .3, fabricM, -.13, .53, -.2, g).rotation.z = .3;          // a cushion
  }
  chair(-1.4, -2.05, .8);
  chair(1.4, -2.05, -.8);

  // Window with the moon, its light on the floor, and a tapestry.
  const arch = plane(1.5, 3, flat({ map: archT, alphaTest: .5, color: 0x8a7f74 }), 2.55, 2.32, -3.98);
  const win = plane(1.22, 2.6, flat({ map: winT, alphaTest: .5, fog: false }), 2.55, 2.25, -3.96);
  box(1.56, .1, .26, hearthM, 2.55, .9, -3.88);
  const shaftG = new THREE.BufferGeometry();
  shaftG.setAttribute("position", new THREE.Float32BufferAttribute([2, 3.3, -3.9, 3.1, 3.3, -3.9, .95, .03, -1.3, 2.05, .03, -1.3], 3));
  shaftG.setAttribute("uv", new THREE.Float32BufferAttribute([0, 1, 1, 1, 0, 0, 1, 0], 2));
  shaftG.setIndex([0, 2, 1, 1, 2, 3]);
  const shaft = new THREE.Mesh(shaftG, glow({ map: shaftT, opacity: .14, side: THREE.DoubleSide }));
  scene.add(shaft);
  const pool = plane(1.1, 1.9, glow({ map: latticeT, opacity: .22 }), 1.5, .016, -1.5);
  pool.rotation.set(-Math.PI / 2, 0, .38);
  const banner = plane(.9, 1.8, lam({ map: bannerT, alphaTest: .5, side: THREE.DoubleSide }), 3.67, 2.25, -2.7);
  banner.rotation.y = -Math.PI / 2;
  const rod = cyl(.025, .025, 1.1, 6, brassM, 3.66, 3.18, -2.7); rod.rotation.x = Math.PI / 2;

  // ---- The doors' objects.
  const doorObjects = {};

  // Bookshelf (Posts).
  const shelf = new THREE.Group(); shelf.position.set(-2.55, 0, -3.775); scene.add(shelf);
  box(.08, 3.3, .45, woodM, -.76, 1.65, 0, shelf); box(.08, 3.3, .45, woodM, .76, 1.65, 0, shelf);
  box(1.62, .1, .48, woodM, 0, 3.32, 0, shelf); box(1.52, 3.3, .03, darkM, 0, 1.65, -.21, shelf);
  const rows = [.1, .65, 1.2, 1.75, 2.3, 2.85];
  rows.forEach((y) => box(1.52, .05, .43, woodM, 0, y, 0, shelf));
  const leather = [0x5a1a1a, 0x1f3a2a, 0x2a2a58, 0x6b4a1a, 0x3a1a3a, 0x7a5a2a, 0x213a4a, 0x4a2a1a, 0x8a2a1a, 0x2f4a2a];
  const books = [];
  rows.forEach((y0, r) => {
    let x = -.71;
    const space = (rows[r + 1] ?? 3.27) - y0 - .07;
    while (x < .68) {
      const w = .045 + rand() * .05, h = Math.min(space, .3 + rand() * .17);
      if (rand() < .08) { x += .08; continue; }                              // a gap
      const lean = rand() < .07 && x < .55 ? .22 : 0;
      books.push({ x: x + w / 2 + (lean ? .04 : 0), y: y0 + .025 + h / 2, w, h, d: .28 + rand() * .08, lean,
                   c: leather[rand() * leather.length | 0] });
      x += w + .004 + (lean ? .06 : 0);
    }
  });
  const bookMat = lam({ color: 0xffffff });
  const bookMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), bookMat, books.length);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), col = new THREE.Color();
  books.forEach((b, i) => {
    q.setFromEuler(e.set(0, 0, -b.lean));
    m4.compose(new THREE.Vector3(b.x, b.y, .02), q, new THREE.Vector3(b.w, b.h, b.d));
    bookMesh.setMatrixAt(i, m4); bookMesh.setColorAt(i, col.setHex(b.c).offsetHSL(0, 0, rand() * .06 - .03));
  });
  shelf.add(bookMesh);
  const bandMat = lam({ color: 0xc79a42 });
  books.forEach((b) => { if (rand() < .45 && !b.lean) box(b.w + .004, .012, .01, bandMat, b.x, b.y + b.h * .3, .02 + b.d / 2, shelf); });
  doorObjects.bookshelf = { group: shelf, mats: [bookMat, bandMat], hot: 0xffc070, k: .07 };

  // Lectern with an open spellbook (LeetCode).
  const lectern = new THREE.Group(); lectern.position.set(-1.7, 0, .45); lectern.rotation.y = .38; scene.add(lectern);
  cyl(.3, .34, .08, 6, woodM, 0, .04, 0, lectern);
  cyl(.06, .08, 1.02, 6, lightWoodM, 0, .56, 0, lectern);
  const top = new THREE.Group(); top.position.set(0, 1.1, 0); top.rotation.x = .5; lectern.add(top);
  box(.66, .05, .5, lightWoodM, 0, 0, 0, top);
  const coverM = lam({ color: 0x5a1416 }), pagesM = lam({ color: 0xe0cf9e }), pageM = lam({ map: pageT, side: THREE.DoubleSide });
  pageM.emissiveMap = pageT;
  box(.62, .025, .44, coverM, 0, .037, 0, top);
  const lp = box(.29, .035, .4, pagesM, -.148, .066, 0, top); lp.rotation.z = .07;
  const rp = box(.29, .035, .4, pagesM, .148, .066, 0, top); rp.rotation.z = -.07;
  const lpp = plane(.28, .39, pageM, -.148, .086, 0, top); lpp.rotation.set(-Math.PI / 2, 0, .07);
  const rpp = plane(.28, .39, pageM, .148, .086, 0, top); rpp.rotation.set(-Math.PI / 2, 0, -.07);
  box(.025, .2, .006, lam({ color: 0x9a1c1c }), .02, -.06, .24, top);                    // ribbon
  const flip = new THREE.Group(); flip.position.set(0, .09, 0); top.add(flip);
  const leaf = plane(.27, .38, pageM, .135, 0, 0, flip); leaf.rotation.x = -Math.PI / 2;
  doorObjects.spellbook = { group: lectern, focus: top, mats: [pageM, pagesM, coverM], hot: 0xffe2a0 };

  // Three portraits above the mantel (About).
  const portraits = [["kk", -.74, 2.52, kkT], ["tt", 0, 2.66, ttT], ["htc", .74, 2.52, htcT]].map(([key, x, y, t]) => {
    const g = new THREE.Group(); g.position.set(x, y, -3.62); scene.add(g);
    const big = key === "tt" ? 1.12 : 1;
    const frameM = lam({ color: 0xa8792a });
    box(.56 * big, .72 * big, .06, frameM, 0, 0, 0, g);
    box(.48 * big, .64 * big, .07, lam({ color: 0x6e4c18 }), 0, 0, .005, g);
    const artM = lam({ map: t }); artM.emissiveMap = t;
    plane(.44 * big, .58 * big, artM, 0, 0, .045, g);
    doorObjects["portrait-" + key] = { group: g, mats: [frameM, artM], hot: 0xffd28a };
    return g;
  });

  // Gramophone on a side table: decoration only, not a door.
  const table = new THREE.Group(); table.position.set(1.75, 0, .35); scene.add(table);
  cyl(.4, .4, .05, 8, woodM, 0, .78, 0, table);
  cyl(.05, .06, .74, 6, woodM, 0, .4, 0, table);
  cyl(.24, .26, .04, 8, woodM, 0, .02, 0, table);
  const gramo = new THREE.Group(); gramo.position.set(0, .805, 0); gramo.rotation.y = -.35; table.add(gramo);
  box(.44, .17, .42, lightWoodM, 0, .085, 0, gramo);
  box(.46, .02, .44, woodM, 0, .18, 0, gramo);
  const disc = new THREE.Group(); disc.position.set(-.03, .19, .02); gramo.add(disc);
  cyl(.17, .17, .012, 12, lam({ color: 0x121014 }), 0, .006, 0, disc);
  cyl(.05, .05, .014, 8, lam({ color: 0x9a1c1c }), 0, .007, 0, disc);
  box(.015, .015, .08, lam({ color: 0xe6c36a }), .04, .016, 0, disc);                  // the label, so it shows the spin
  const gBrass = lam({ color: 0xc79a42 });
  const arm = box(.018, .02, .2, gBrass, .17, .22, -.03, gramo); arm.rotation.y = .5;
  const crank = cyl(.012, .012, .12, 5, gBrass, .26, .1, 0, gramo); crank.rotation.z = Math.PI / 2;
  const curve = new THREE.CatmullRomCurve3([new THREE.Vector3(.16, .2, -.15), new THREE.Vector3(.16, .38, -.22),
                                           new THREE.Vector3(.06, .55, -.14)]);
  gramo.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 6, .022, 5), gBrass));
  const prof = [];
  for (let i = 0; i <= 8; i++) prof.push(new THREE.Vector2(.022 + .26 * Math.pow(i / 8, 2.6), i / 8 * .5));
  const hornM = lam({ color: 0xc79a42, side: THREE.DoubleSide });
  const horn = new THREE.Mesh(new THREE.LatheGeometry(prof, 10), hornM);
  horn.position.set(.06, .55, -.14);
  horn.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(-.35, .75, .55).normalize());
  gramo.add(horn);

  // ---- Lights: firelight and candles against moonlight. No shadows, like the era.
  scene.add(new THREE.AmbientLight(0x3c3a68, .95));
  const moon = new THREE.DirectionalLight(0x8ea4ff, 1.1); moon.position.set(3.5, 5, -6); scene.add(moon);
  const fireL = new THREE.PointLight(0xff7f30, 16, 13, 1.25); fireL.position.set(0, .8, -2.9); scene.add(fireL);
  const fillL = new THREE.PointLight(0xffb070, 3.2, 7, 1.2); fillL.position.set(0, 2.2, 1.6); scene.add(fillL);
  const sconceL = new THREE.PointLight(0xffb060, 3, 5, 1.2); sconceL.position.set(-3.2, 2.2, -2.2); scene.add(sconceL);
  const spellL = new THREE.PointLight(0xffd28a, 0, 3.2, 1.2); scene.add(spellL);

  // Embers rising from the fire, and sparkles around the chosen object.
  function points(n, size, color) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
    const p = new THREE.Points(g, new THREE.PointsMaterial({ size, sizeAttenuation: false, vertexColors: true,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false, color }));
    p.frustumCulled = false; scene.add(p);
    return { p, pos: g.attributes.position, col: g.attributes.color, life: Array.from({ length: n }, () => rand()) };
  }
  const embers = points(26, 2, 0xffffff), sparks = points(40, 2, 0xffffff);
  let sparkBox = null;
  const emberSeed = Array.from({ length: embers.life.length }, (_, i) => (i * .37) % 1);

  // ---- Doors: each <a> in .rm-doors is moved over the screen box of its object.
  const links = [...stage.querySelectorAll(".rm-door")];
  const doors = links.map((link) => {
    const d = doorObjects[link.dataset.object];
    if (!d) return null;
    d.group.updateWorldMatrix(true, true);
    const b = new THREE.Box3().setFromObject(d.group);
    const corners = [];
    for (let i = 0; i < 8; i++) corners.push(new THREE.Vector3(i & 1 ? b.max.x : b.min.x, i & 2 ? b.max.y : b.min.y, i & 4 ? b.max.z : b.min.z));
    const c = b.getCenter(new THREE.Vector3());
    return { ...d, link, box: b, corners, center: c, glow: 0, hotColor: new THREE.Color(d.hot) };
  }).filter(Boolean);
  const bookDoor = doors.find((d) => d.focus === top);
  let last = 0, running = false, raf = 0, onScreen = true, wanted = true;   // animation state, used from here on

  // The pointer wins while it is over a door; otherwise the keyboard-focused door stays lit with its label.
  let active = null, hovered = null, focused = null;
  function choose() { active = hovered || focused; if (!running) still(); }
  doors.forEach((d) => {
    d.link.addEventListener("pointerenter", (ev) => { if (ev.pointerType !== "touch") { hovered = d; choose(); } });
    d.link.addEventListener("pointerleave", () => { if (hovered === d) { hovered = null; choose(); } });
    d.link.addEventListener("focus", () => { focused = d; choose(); });
    d.link.addEventListener("blur", () => { if (focused === d) { focused = null; choose(); } });
  });

  // ---- Size: render at about 270 px tall and scale up by a whole number, so every pixel is a crisp square.
  const base = new THREE.Vector3(0, 2.05, 6.1), look = new THREE.Vector3(0, 1.55, -4);
  let cssW = 1, cssH = 1, scale = 3;
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    scale = Math.max(2, Math.round(Math.min(h, w * 1.2) / 270));
    const iw = Math.ceil(w / scale), ih = Math.ceil(h / scale);
    renderer.setSize(iw, ih, false);
    cssW = iw * scale; cssH = ih * scale;
    canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
    snap.value.set(iw / 2, ih / 2);
    base.z = iw / ih < 1.1 ? 8 : 6.1;                                      // tall screens: step back so the near objects fit
    scene.fog.near = base.z + 4; scene.fog.far = base.z + 16;
    const aspect = iw / ih, dist = base.z + 4, vHalf = Math.max(2.55, 3.7 / aspect);
    camera.aspect = aspect;
    camera.fov = 2 * Math.atan(vHalf / dist) * 180 / Math.PI;
    camera.updateProjectionMatrix();
    doors.forEach((d) => { d.tipW = 0; d.placed = ""; });
    if (!running) still();
  }

  const aim = { x: 0, y: 0, tx: 0, ty: 0 };
  stage.addEventListener("pointermove", (ev) => {
    if (ev.pointerType === "touch") return;
    const r = stage.getBoundingClientRect();
    aim.tx = (ev.clientX - r.left) / r.width * 2 - 1; aim.ty = (ev.clientY - r.top) / r.height * 2 - 1;
  });
  stage.addEventListener("pointerleave", () => { aim.tx = aim.ty = 0; });
  stage.addEventListener("scroll", () => { stage.scrollTop = stage.scrollLeft = 0; });   // older browsers without overflow: clip

  const v = new THREE.Vector3(), tmp = new THREE.Vector3(), spellOffset = new THREE.Vector3(0, .2, .9);
  function placeDoors() {
    for (const d of doors) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const c of d.corners) {
        v.copy(c).project(camera);
        const sx = (v.x + 1) / 2 * cssW, sy = (1 - v.y) / 2 * cssH;
        x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
      }
      const mw = Math.round(Math.max(44, x1 - x0)), mh = Math.round(Math.max(44, y1 - y0));   // touch targets >= 44 px
      const left = Math.round((x0 + x1 - mw) / 2), top = Math.round((y0 + y1 - mh) / 2);
      const tall = mh > cssH * .45;                                       // tall objects: the label sits inside
      const key = `${left},${top},${mw},${mh}`;
      if (key === d.placed) continue;                                     // the camera drifts slowly: most frames, no change
      d.placed = key;
      d.link.style.transform = `translate(${left}px,${top}px)`;
      d.link.style.width = mw + "px"; d.link.style.height = mh + "px";
      d.link.classList.toggle("rm-mid", tall);
      d.link.classList.toggle("rm-below", !tall && top < cssH * .3);
      d.tipW ||= d.link.querySelector(".rm-tip").offsetWidth;            // measured once, again after a resize or font load
      const half = d.tipW / 2 + 6, centre = left + mw / 2;                 // keep the label inside the stage's clip
      const dx = Math.min(Math.max(centre, half), Math.max(half, cssW - half)) - centre;
      d.link.style.setProperty("--tip-dx", Math.round(dx) + "px");
    }
  }

  // ---- Animation.
  function update(t, dt, frozen) {
    const k = frozen ? 1 : 1 - Math.exp(-dt * 7);
    if (!frozen) {
      aim.x += (aim.tx - aim.x) * (1 - Math.exp(-dt * 2)); aim.y += (aim.ty - aim.y) * (1 - Math.exp(-dt * 2));
      camera.position.set(base.x + Math.sin(t * .11) * .16 + aim.x * .3, base.y + Math.sin(t * .17) * .05 - aim.y * .12, base.z);
      tmp.set(look.x + aim.x * .25, look.y - aim.y * .1, look.z);
    } else {
      camera.position.copy(base); tmp.copy(look);
    }
    camera.lookAt(tmp);

    const flick = frozen ? 1 : .86 + .08 * Math.sin(t * 11.3) + .06 * Math.sin(t * 23.7 + 1) + .04 * rand();
    fireL.intensity = 16 * flick;
    halo.material.opacity = .32 * flick; floorGlow.material.opacity = .2 * flick;
    const frame = frozen ? 0 : Math.floor(t * 9);
    flames.forEach((f) => {
      const u = f.userData;
      f.scale.set(1 + (frozen ? 0 : .06 * Math.sin(t * 7 + u.phase)), 1 + (frozen ? 0 : .14 * Math.sin(t * 9.3 + u.phase) + .06 * Math.sin(t * 17 + u.phase)), 1);
      f.material.map = fireTs[(frame + u.k) % 3];
    });
    candleFlames.forEach((c) => {
      const s = frozen ? 1 : 1 + .12 * Math.sin(t * 13 + c.phase) + .05 * Math.sin(t * 29 + c.phase);
      c.f.scale.set(1, s, 1); c.hl.material.opacity = .26 * s;
    });
    sconceL.intensity = 3 * (frozen ? 1 : .9 + .1 * Math.sin(t * 13 + 2));

    // Embers.
    for (let i = 0; i < embers.life.length; i++) {
      let l = embers.life[i];
      if (!frozen) l += dt * (.35 + (i % 5) * .06);
      if (l >= 1) { l -= 1; emberSeed[i] = rand(); }
      embers.life[i] = l;
      const s = emberSeed[i];
      embers.pos.setXYZ(i, (s - .5) * .8 + Math.sin(l * 6 + i) * .06, .25 + l * 1.15, -3.38 + (i % 3) * .03);
      const a = frozen ? 0 : Math.sin(l * Math.PI) * (.6 + .4 * Math.sin(t * 20 + i));
      embers.col.setXYZ(i, a, a * .5, a * .15);
    }
    embers.pos.needsUpdate = embers.col.needsUpdate = true;

    // Door glow, the spell light, and sparkles around the chosen object.
    for (const d of doors) {
      d.glow += ((d === active ? 1 : 0) - d.glow) * k;
      for (const m of d.mats) m.emissive.copy(d.hotColor).multiplyScalar(d.glow * (d.k ?? .45));
    }
    spellL.intensity = 0;
    if (active) {
      spellL.position.copy(active.center).add(spellOffset);
      spellL.intensity = 2 * active.glow;
    }
    const sb = active ? active.box : null;
    if (sb !== sparkBox) {                   // a new object: every sparkle starts around it, not where the last one was
      sparkBox = sb;
      if (sb) for (let i = 0; i < sparks.life.length; i++) {
        sparks.pos.setXYZ(i, sb.min.x + rand() * (sb.max.x - sb.min.x), sb.min.y + rand() * (sb.max.y - sb.min.y), sb.max.z + .05);
        sparks.life[i] = rand();
      }
    }
    sparks.p.visible = !!sb;                 // nothing chosen: no upload, no draw
    if (sb) for (let i = 0; i < sparks.life.length; i++) {
      let l = sparks.life[i] + (frozen ? 0 : dt * (.5 + (i % 4) * .15));
      if (l >= 1 || !sb) {
        l = l % 1;
        if (sb) sparks.pos.setXYZ(i, sb.min.x + rand() * (sb.max.x - sb.min.x), sb.min.y + rand() * (sb.max.y - sb.min.y), sb.max.z + .05);
      }
      sparks.life[i] = l;
      if (sb && !frozen) sparks.pos.setY(i, sparks.pos.getY(i) + dt * .25);
      const a = sb && !frozen ? Math.sin(l * Math.PI) * (i % 3 ? 1 : .5 + .5 * Math.sin(t * 25 + i)) * (active ? active.glow : 0) : 0;
      sparks.col.setXYZ(i, a, a * .86, a * .55);
    }
    if (sb) sparks.pos.needsUpdate = sparks.col.needsUpdate = true;

    // The record turns; the spellbook's page turns while it is chosen.
    disc.rotation.y -= frozen ? 0 : dt * 1.1;
    if (!frozen && active === bookDoor) flip.rotation.z = Math.PI * Math.min(1, ((t * .9) % 1) * 1.25);
    else flip.rotation.z += (0 - flip.rotation.z) * k;
  }
  function render(t, dt, s) {
    update(t, dt, s);
    renderer.render(scene, camera);
    placeDoors();
    stage.classList.add("rm-ready");
  }
  function still() { render(0, 0, true); }
  function tick(now) {
    raf = requestAnimationFrame(tick);
    if (now - last < 1000 / 30 - 2) return;                                  // the era's 30 frames a second
    const dt = last ? Math.min(.1, (now - last) / 1000) : 1 / 30;
    last = now;
    render(now / 1000, dt, false);
  }
  function loop() {
    const go = wanted && onScreen && !document.hidden && !reduce.matches;
    if (go && !running) { running = true; last = 0; raf = requestAnimationFrame(tick); }
    if (!go && running) { running = false; cancelAnimationFrame(raf); }
    if (!go && wanted) still();
  }
  document.addEventListener("visibilitychange", loop);
  reduce.addEventListener?.("change", loop);
  new IntersectionObserver((es) => { onScreen = es[es.length - 1].isIntersecting; loop(); }).observe(stage);
  new ResizeObserver(resize).observe(stage);
  document.fonts?.ready.then(() => { doors.forEach((d) => { d.tipW = 0; d.placed = ""; }); if (!running) still(); });
  resize();
  loop();
  return { pause() { wanted = false; loop(); }, resume() { wanted = true; resize(); loop(); } };
}
