import { THREE, Stage, Model, loadAsset, evaluate, rng, hash, clamp, lerp, ease, NO_DEFECT, HAS_GL } from './stage.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const smooth = (x, a, b) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const hex = (n) => '#' + n.toString(16).padStart(6, '0');
const S = (n) => `S<sub>${n}</sub>`;
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ================= data of the reported run ================= */
const ROUNDS = [
  { v: 0, gate: 'seed', train: 0.431, tp: 25.0, val: 0.439, vp: 24.2, title: 'Hand-written seed Skill', file: 'SKILL.md',
    diff: ['# Assemble an articulated object', '', '## Input', 'part meshes, reference image, URDF template', '## Output', 'the template with every blank filled', '## Tools', 'the Assembly SDK (place, scale, set_joint, render)'] },
  { v: 1, gate: 'unchanged', train: 0.427, tp: 25.0, title: 'No edit survived the Skill Evolver’s own trials', file: 'SKILL.md',
    diff: ['<cm># three edits drafted, none fixed its failure cluster</cm>', '<cm># on the 3 train objects it was tested on</cm>', '', 'Skill unchanged.'] },
  { v: 2, gate: 'rejected', train: 0.448, tp: 27.5, val: 0.455, vp: 29.2, d: -0.002, z: -0.07, title: 'Cavity-mouth detection for hollow parts', file: 'helpers/cavity.py',
    diff: ['## Hollow parts', '<add>+ Render orthographic depth along each face.</add>', '<add>+ Call cavity.mouth(part) to find the opening,</add>', '<add>+ then point the opening away from the parent.</add>'] },
  { v: 3, gate: 'accepted', train: 0.496, tp: 35.0, val: 0.544, vp: 37.5, d: 0.089, z: 2.59, title: 'Compare the render with the reference image, registered on the root', file: 'helpers/image_compare.py',
    diff: ['## After every placement', '<add>+ Render the assembly from the reference camera.</add>', '<add>+ Register both images on the root part.</add>', '<add>+ image_compare.offsets() gives the projected bounds</add>', '<add>+ and the offset of every link. Correct the largest.</add>'] },
  { v: 4, gate: 'rejected', train: 0.587, tp: 47.5, val: 0.538, vp: 35.8, d: -0.013, z: -0.45, title: 'Pivot axes from paired hardware', file: 'helpers/axis_evidence.py',
    diff: ['## Revolute joints', '<add>+ Pair hinge barrels on parent and child.</add>', '<add>+ axis_evidence.fit() returns the shared axis.</add>', '## Prismatic joints', '<add>+ Take the sign of travel from the measured seats.</add>'] },
  { v: 5, gate: 'rejected', train: 0.549, tp: 37.5, val: 0.550, vp: 35.0, d: 0.001, z: 0.03, title: 'Synchronise joint frames parent-first', file: 'SKILL.md',
    diff: ['## Order of operations', '<del>- Set visual poses, then joint frames.</del>', '<add>+ Set all joint frames parent-first,</add>', '<add>+ then set the visual poses.</add>'] },
  { v: 6, gate: 'accepted', train: 0.694, tp: 50.0, val: 0.601, vp: 38.3, d: 0.061, z: 2.78, title: 'Take the attachment depth from the parent geometry', file: 'SKILL.md',
    diff: ['## Attachment', '<add>+ Constrain depth by the parent cavity or seat</add>', '<add>+ before any screen-space correction.</add>', '## Sliders', '<add>+ Check one-sided endpoints against the engagement</add>', '<add>+ of the guide. A drawer never leaves its rails.</add>'] },
  { v: 7, gate: 'rejected', train: 0.664, tp: 52.5, val: 0.584, vp: 38.3, d: -0.019, z: -0.82, title: 'Fit the scale of pivoted parts from distal landmarks', file: 'SKILL.md',
    diff: ['## Scale of pivoted parts', '<add>+ Anchor the pivot, then fit the scale so that the</add>', '<add>+ far end of the part meets its landmark.</add>'] },
  { v: 8, gate: 'rejected', train: 0.652, tp: 47.5, val: 0.606, vp: 40.8, d: 0.012, z: 0.47, title: 'Correct hinge angles from directed edges', file: 'helpers/projected_direction.py',
    diff: ['## Rest angle of hinged parts', '<add>+ projected_direction.edges(reference) returns</add>', '<add>+ the directed edges of each link.</add>', '<add>+ Rotate about the hinge until they agree.</add>'] },
  { v: 9, gate: 'accepted', train: 0.714, tp: 60.0, val: 0.682, vp: 52.5, d: 0.086, z: 3.0, title: 'Read bearing axes from closed circular rims', file: 'helpers/round_features.py',
    diff: ['## Rotating parts', '<add>+ round_features.rims(part) extracts closed circular</add>', '<add>+ rims. Their normal is the axis, their centre the pivot.</add>', '## Sliders', '<add>+ Constrain the slider transversally first,</add>', '<add>+ then fit its extension.</add>'] },
  { v: 10, gate: 'unchanged', train: 0.716, tp: 57.5, title: 'No edit survived the Skill Evolver’s own trials', file: 'SKILL.md',
    diff: ['<cm># every drafted edit lowered fitness on its trial objects</cm>', '', 'Skill unchanged.'] },
  { v: 11, gate: 'rejected', train: 0.698, tp: 55.0, val: 0.670, vp: 51.7, d: -0.016, z: -0.77, title: 'Check the sign of joint motion numerically', file: 'SKILL.md',
    diff: ['## Sign of motion', '<add>+ Sweep the joint and measure the overlap with</add>', '<add>+ the parent. Flip the axis if it grows.</add>', '<add>+ Run a half-turn test on surface relief.</add>'] },
];
const CHAMP_VAL = { 0: 0.439, 3: 0.544, 6: 0.601, 9: 0.682 };
const championAt = (r) => (r >= 9 ? 9 : r >= 6 ? 6 : r >= 3 ? 3 : 0);

const TRANSFER = [
  { id: 'gpt-6.1-sol', note: 'model used to improve the Skill', fit: { 0: 0.474, 3: 0.512, 6: 0.639, 9: 0.681 }, pass: { 0: 30.0, 3: 34.2, 6: 45.8, 9: 52.5 } },
  { id: 'claude-opus-5-5', note: 'never seen by the loop', fit: { 0: 0.387, 3: 0.471, 6: 0.496, 9: 0.552 }, pass: { 0: 18.3, 3: 30.8, 6: 33.3, 9: 33.3 } },
  { id: 'gpt-6.1-terra', note: 'never seen by the loop', fit: { 0: 0.316, 3: 0.415, 6: 0.534, 9: 0.572 }, pass: { 0: 18.3, 3: 29.2, 6: 38.3, 9: 39.2 } },
  { id: 'gpt-6-astra', note: 'never seen by the loop', fit: { 0: 0.568, 9: 0.814 }, pass: { 0: 43.3, 9: 71.7 } },
];
const COMPARE = [
  { n: 'Particulate', s: 'feed-forward', v: { easy: 63.6, medium: 13.3, hard: 0.0, all: 22.5 } },
  { n: 'USDCraft', s: 'agent · gpt-6-astra', v: { easy: 100, medium: 80.0, hard: 50.0, all: 75.0 } },
  { n: 'Ours, seed S<sub>0</sub>', s: 'gpt-6.1-sol', v: { easy: 97.0, medium: 62.2, hard: 35.7, all: 62.5 }, c: 'ours' },
  { n: 'Ours, S<sub>9</sub>', s: 'gpt-6.1-sol', v: { easy: 100, medium: 86.7, hard: 73.8, all: 85.8 }, c: 'ours' },
  { n: 'Ours, seed S<sub>0</sub>', s: 'gpt-6-astra', v: { easy: 100, medium: 71.1, hard: 45.2, all: 70.0 }, c: 'ours' },
  { n: 'Ours, S<sub>9</sub>', s: 'gpt-6-astra', v: { easy: 100, medium: 100, hard: 78.6, all: 92.5 }, c: 'best' },
];

/* ============ illustrative pose errors of each Skill version ============ */
const D = (o) => ({ ...NO_DEFECT, ...o });
const HAND = {      // joint index -> defect, per champion. Modelled on the renders in the paper.
  tape_measure: {
    0: [D({ dp: [-0.16, -0.46, -0.02], rot: [0, 0, 1, 26], s: 1.32 }), D({ rot: [1, 0, 0, 14] })],
    3: [D({ dp: [-0.1, -0.52, -0.02], rot: [0, 0, 1, 12], s: 1.28 })],
    6: [undefined, D({ rot: [1, 0, 0, 9] })], 9: [],
  },
  cash_register: {
    0: [D({ dp: [0, -0.44, -0.03], s: 1.42 }), D({ rot: [1, 0, 0, -24] })],
    3: [D({ dp: [0, -0.52, -0.07], s: 1.42 })],
    6: [undefined, D({ pe: [0, 0.07, -0.05] })], 9: [],
  },
  microscope: {
    0: [D({ dp: [0, 0, -0.07] }), D({ pe: [0, 0, 0.09] }), D({ s: 0.7, dp: [0, 0, 0.04] }), D({ s: 0.62, dp: [0, 0, -0.07] })],
    3: [undefined, D({ pe: [0, 0, 0.09] }), D({ s: 1.24, dp: [0, 0, -0.035] })],
    6: [undefined, D({ ax: 34 })], 9: [],
  },
  computer_desk: {
    0: [D({ dp: [0, -0.21, 0], s: 1.22 }), D({ dp: [0, -0.18, -0.02], s: 1.3 })],
    3: [D({ dp: [0, -0.23, 0], s: 1.22 }), D({ dp: [0, -0.2, -0.03], s: 1.3 })],
    6: [undefined, D({ dp: [0, -0.045, 0] })], 9: [],
  },
};
function defectsFor(slug, version, nJoints) {
  if (version >= 9) return [];
  if (HAND[slug]) return HAND[slug][version] || [];
  const r = rng(hash(slug) ^ 0x51ed), out = [];
  for (let i = 0; i < nJoints; i++) {
    const fixed = [3, 6, 9, 9, 6][Math.floor(r() * 5)], kind = Math.floor(r() * 5), sg = r() < 0.5 ? -1 : 1, m = 0.7 + r() * 0.6;
    const fx = i === 0 ? Math.max(fixed, 6) : fixed;
    if (version >= fx) { out.push(undefined); continue; }
    const k = version === 0 ? 1 : 0.75;
    out.push([
      D({ dp: [0.16 * sg * m * k, -0.2 * m * k, 0.05 * k] }),
      D({ rot: [0, 0, 1, 34 * sg * m * k], dp: [0, 0, 0.05 * k] }),
      D({ s: 1 + 0.42 * m * k, dp: [0, -0.08 * k, 0] }),
      D({ ax: 48 * sg * m * k }),
      D({ pe: [0.1 * sg * m * k, 0, 0.1 * m * k], rot: [1, 0, 0, 12 * sg * k] }),
    ][kind]);
  }
  return out;
}

/* ================= top bar, reveal, counters ================= */
{
  const bar = $('#bar'); let last = 0;
  bar.classList.toggle('top', scrollY < 8);
  addEventListener('scroll', () => { const y = scrollY; bar.classList.toggle('top', y < 8); bar.classList.toggle('hide', y > innerHeight * 0.5 && y > last + 4); if (y < last - 4) bar.classList.remove('hide'); last = y; }, { passive: true });
  // phones: the section links open as a sheet under the bar
  const menu = $('#bar-menu'), setMenu = (open) => { bar.classList.toggle('open', open); menu.setAttribute('aria-expanded', open); menu.setAttribute('aria-label', open ? 'Close the section menu' : 'Open the section menu'); };
  menu.onclick = () => setMenu(!bar.classList.contains('open'));
  $$('.bar-nav a').forEach((a) => a.addEventListener('click', () => setMenu(false)));
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && bar.classList.contains('open')) { setMenu(false); menu.focus(); } });
  addEventListener('pointerdown', (e) => { if (bar.classList.contains('open') && !bar.contains(e.target)) setMenu(false); });
  addEventListener('scroll', () => { if (bar.classList.contains('hide')) setMenu(false); }, { passive: true });
  const links = $$('.bar-nav a'), secs = links.map((a) => $(a.getAttribute('href')));
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a, i) => { const on = secs[i] === e.target; a.classList.toggle('on', on); if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); }); }), { rootMargin: '-45% 0px -50% 0px' });
  secs.forEach((s) => s && io.observe(s));

  $$('.sec-head, .panel, .theater, .loops, .tm-rail, .wb-shelf, .arti-stats, .tiers, .bib').forEach((e) => e.classList.add('reveal'));
  const ro = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); ro.unobserve(e.target); } }), { rootMargin: '0px 0px -8% 0px' });
  $$('.reveal').forEach((e) => ro.observe(e));

  const fmt = (v, dec, sep) => { const s = v.toFixed(dec); return sep ? s.replace(/\B(?=(\d{3})+(?!\d))/g, ',') : s; };
  const co = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return; co.unobserve(e.target);
    const b = e.target, a = +b.dataset.count, z = +b.dataset.to, dec = +b.dataset.dec, suf = b.dataset.suf || '', t0 = performance.now(), dur = reduced ? 1 : 1500;
    const step = (now) => { const t = clamp((now - t0) / dur); b.textContent = fmt(lerp(a, z, ease(t)), dec, b.dataset.sep) + suf; if (t < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }), { threshold: 0.6 });
  $$('[data-count]').forEach((b) => co.observe(b));
}

/* ================= hero: scroll to assemble ================= */
(async () => {
  const hero = $('#top'), copy = $('#hero-copy'), steps = $$('#hero-steps li'), rail = $('.hero-rail'), fill = $('#hero-rail-fill'), cue = $('#scroll-cue');
  requestAnimationFrame(() => setTimeout(() => $('.h1-b').style.setProperty('--fill', '100%'), 300));
  const stage = new Stage($('#gl-hero'), { orbit: false, fov: 24, shadow: 0.26, glow: 0.13 });
  const slugs = ['slot_machine', 'microscope', 'cash_register'], xs = [-1.02, 0.0, 1.05], hs = [0.92, 0.82, 0.62];
  const wait = $('#hero-loading');
  let assets;
  try { assets = await Promise.all(slugs.map((s) => loadAsset(s))); } catch (_) { wait.textContent = 'The 3D models could not be loaded'; wait.classList.add('failed'); return; }
  wait.classList.add('off');
  const models = assets.map((a, i) => {
    const m = new Model(a, { seed: 3 + i, spread: 1.08 });
    const k = hs[i] / m.size.z; m.group.scale.setScalar(k); m.group.position.x = xs[i]; m.group.rotation.z = [0.5, -0.35, -0.6][i];
    m.t = 0; m.gizmos.visible = false; stage.add(m);
    // the root body starts loose as well: tilted about its own centre, rescaled, and lifted off the floor
    const g = m.rootMesh.geometry; if (!g.boundingBox) g.computeBoundingBox();
    m.rootC = g.boundingBox.getCenter(new THREE.Vector3());
    m.rootQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(...[[0.55, -0.4, 0.9], [-0.7, 0.45, -0.8], [0.5, 0.55, 0.7]][i]));
    return m;
  });
  const QI = new THREE.Quaternion(), RV = new THREE.Vector3();
  // Loose parts. Each part is capped in size and then spread in the view plane (x to the right, y up), so that no part
  // overlaps another part or an object, and all of them stay inside one box that the layout below keeps clear of the copy.
  const UNIT = Math.cos((24 * Math.PI) / 360) / (2 * 1.42);     // one world unit in view heights at pad 1
  const ext = { w: 1.1, h: 0.76, cx: 0.03, cy: 0.18 }, extObj = { w: 0.95, cx: 0 };
  {
    const AZ = 0.42, EL = 0.3, LO = -1.45, HI = 1.6, YHI = 2.02, CAP = 0.34;
    const right = new THREE.Vector3(Math.cos(AZ), 0, -Math.sin(AZ)), up = new THREE.Vector3(-Math.sin(EL) * Math.sin(AZ), Math.cos(EL), -Math.sin(EL) * Math.cos(AZ));
    const P = new THREE.Vector3(), parts = [], blocks = [], box = { x0: 9, x1: -9, y0: 9, y1: -9 }, obox = { x0: 9, x1: -9 };
    const grow = (x, y, r) => { box.x0 = Math.min(box.x0, x - r); box.x1 = Math.max(box.x1, x + r); box.y0 = Math.min(box.y0, y - r); box.y1 = Math.max(box.y1, y + r); };
    models.forEach((m) => {
      m.group.updateMatrixWorld(true); const k = m.group.scale.x, bb = m.meta.bbox;
      let x0 = 9, x1 = -9, y0 = 9, y1 = -9;                      // the assembled object as an obstacle
      for (let c = 0; c < 8; c++) { m.z.localToWorld(P.set(c & 1 ? bb.max[0] : bb.min[0], c & 2 ? bb.max[1] : bb.min[1], c & 4 ? bb.max[2] : bb.min[2])); const x = P.dot(right), y = P.dot(up); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
      blocks.push({ x0, x1, y0, y1 }); grow((x0 + x1) / 2, (y0 + y1) / 2, 0); grow(x0, y0, 0); grow(x1, y1, 0); obox.x0 = Math.min(obox.x0, x0); obox.x1 = Math.max(obox.x1, x1);
      m.links.forEach((L) => {
        const size = L.mesh.geometry.boundingBox.getSize(P), longest = Math.max(size.x, size.y, size.z) || 1;
        L.ss = Math.min(L.ss, CAP / (k * longest));
        const rad = 0.43 * k * L.ss * size.length() + 0.03, w = m.z.localToWorld(P.copy(L.center).add(L.sp));
        parts.push({ m, L, k, rad, x: w.dot(right), y: w.dot(up), sx: w.dot(right), sy: w.dot(up) });
      });
    });
    for (let it = 0; it < 240; it++) {
      for (let i = 0; i < parts.length; i++) {
        const a = parts[i];
        for (let j = i + 1; j < parts.length; j++) {
          const b = parts[j]; let dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1e-3; const o = a.rad + b.rad + 0.05 - d;
          if (o > 0) { dx /= d; dy /= d; a.x -= dx * o * 0.5; a.y -= dy * o * 0.5; b.x += dx * o * 0.5; b.y += dy * o * 0.5; }
        }
        for (const q of blocks) {                                 // leave the object through its nearest free side, never downwards
          const g = a.rad + 0.04; if (a.x < q.x0 - g || a.x > q.x1 + g || a.y > q.y1 + g || a.y < q.y0 - g) continue;
          const toTop = q.y1 + g - a.y, toL = a.x - (q.x0 - g), toR = q.x1 + g - a.x;
          if (toTop <= Math.min(toL, toR) * 1.6) a.y += toTop * 0.6; else if (toL < toR) a.x -= toL * 0.6; else a.x += toR * 0.6;
        }
        a.x = clamp(a.x, LO + a.rad, HI - a.rad); a.y = clamp(a.y, a.rad + 0.05, YHI - a.rad);
      }
    }
    parts.forEach((a) => {
      const inv = a.m.z.matrixWorld.clone().invert();
      a.L.sp.addScaledVector(P.copy(right).transformDirection(inv), (a.x - a.sx) / a.k).addScaledVector(P.copy(up).transformDirection(inv), (a.y - a.sy) / a.k);
      grow(a.x, a.y, a.rad);
    });
    models.forEach((m) => m.update());
    const cy = 0.42 * Math.cos(EL);                               // the camera target in the same coordinates
    ext.w = (box.x1 - box.x0) * UNIT + 0.04; ext.h = (box.y1 - box.y0) * UNIT + 0.04; ext.cx = ((box.x0 + box.x1) / 2) * UNIT; ext.cy = ((box.y0 + box.y1) / 2 - cy) * UNIT;
    extObj.w = (obox.x1 - obox.x0) * UNIT + 0.06; extObj.cx = ((obox.x0 + obox.x1) / 2) * UNIT;
  }
  let p = 0, mx = 0, my = 0;
  const onScroll = () => { p = clamp(scrollY / (hero.offsetHeight - innerHeight)); };
  addEventListener('scroll', onScroll, { passive: true }); onScroll();
  addEventListener('pointermove', (e) => { mx = e.clientX / innerWidth - 0.5; my = e.clientY / innerHeight - 0.5; }, { passive: true });
  let cx = 0, cy = 0, shown = -1;
  // Free area for the objects at scroll 0. On phones and portrait tablets the copy sits on top and the objects below it,
  // otherwise the objects take the space to the right of the lede. `ext` is the box around the loose parts and the objects,
  // in view heights at pad 1 (U is the shorter side of the view). On short landscape screens the step cards cover the left
  // half, so the assembled objects stay in the free area to their right.
  const stackMQ = matchMedia('(max-width: 760px) and (min-height: 501px), (max-width: 760px) and (orientation: portrait), (max-width: 900px) and (max-aspect-ratio: 1/1)');
  const lede = $('.lede', copy), sticky = $('.hero-sticky'), stepsEl = $('#hero-steps'), wrapEl = $('#film .wrap');
  let lay = { stack: false, pad: 1.42, shift: 0.17, lift: 0, pad1: 0.98, shift1: 0 };
  const layout = () => {
    const W = sticky.clientWidth, H = sticky.clientHeight, U = Math.min(W, H), stack = stackMQ.matches;
    if (!W || !H) return;
    if (stack) {
      const y0 = copy.offsetTop + copy.offsetHeight + 14, y1 = H - 56;
      const pad = clamp(Math.max((ext.w * U) / (W - 16), (ext.h * U) / Math.max(60, y1 - y0)), 1.3, 2.6);
      lay = { stack, pad, shift: (-ext.cx * U) / pad / W, lift: ((y0 + y1) / 2 + (ext.cy * U) / pad - H / 2) / H, pad1: 1.1, shift1: 0 };
    } else {
      const edge = wrapEl.offsetLeft + wrapEl.offsetWidth;
      const x0 = copy.offsetLeft + lede.offsetLeft + lede.offsetWidth + 28, x1 = Math.min(W - 20, edge + 0.02 * W);
      const pad = clamp(Math.max((ext.w * U) / Math.max(60, x1 - x0), (ext.h * U) / Math.max(60, H - 96)), 1.34, 2.6);
      lay = { stack, pad, shift: ((x0 + x1) / 2 - (ext.cx * U) / pad - W / 2) / W, lift: ((ext.cy * U) / pad + 8) / H, pad1: 0.98, shift1: 0 };
      if (H < 600) {
        const a0 = stepsEl.offsetLeft + stepsEl.offsetWidth + 20, a1 = W - 76;
        lay.pad1 = clamp((extObj.w * U) / Math.max(60, a1 - a0), 0.98, 2.2); lay.shift1 = ((a0 + a1) / 2 - (extObj.cx * U) / lay.pad1 - W / 2) / W;
      }
    }
  };
  layout(); addEventListener('resize', layout); new ResizeObserver(layout).observe(copy); new ResizeObserver(layout).observe(stepsEl); if (document.fonts && document.fonts.ready) document.fonts.ready.then(layout);
  stage.onFrame = (dt, time) => {
    const wide = !lay.stack;
    const out = smooth(p, 0.02, 0.1);
    copy.style.opacity = 1 - out; copy.style.transform = `translateY(calc(-46% - ${out * 40}px))`; copy.style.pointerEvents = out > 0.6 ? 'none' : '';
    if (!wide) copy.style.transform = `translateY(${-out * 40}px)`;
    cue.style.opacity = 1 - smooth(p, 0.0, 0.04);
    const step = p > 0.1 && p < 0.3 ? 0 : p >= 0.3 && p < 0.66 ? 1 : p >= 0.66 && p < 0.985 ? 2 : -1;
    if (step !== shown) { steps.forEach((s, i) => s.classList.toggle('on', i === step)); shown = step; rail.classList.toggle('on', step >= 0); }
    fill.style.height = clamp((p - 0.07) / 0.86) * 100 + '%';
    const t = clamp((p - 0.24) / 0.4), w = smooth(p, 0.67, 0.75);
    models.forEach((m, k) => {
      m.t = t; m.gizmos.visible = w > 0.02;
      const loose = 1 - smooth(t, 0, 0.55), R = m.rootMesh, sc = lerp(1, 0.82, loose);
      R.quaternion.copy(QI).slerp(m.rootQ, loose); R.scale.setScalar(sc);
      R.position.copy(m.rootC).sub(RV.copy(m.rootC).applyQuaternion(R.quaternion).multiplyScalar(sc));
      R.position.z += loose * m.size.z * (0.16 + (reduced ? 0 : Math.sin(time * 0.7 + k * 2.1) * 0.025));
      m.links.forEach((L) => {
        const mid = (L.lo + L.hi) / 2, half = (L.hi - L.lo) / 2;
        const target = reduced ? mid + half * 0.4 : L.joint.type === 'continuous' ? time * 1.3 + L.phase : mid + Math.sin(time * 1.25 + L.phase + k) * half * 0.9;
        const rest = clamp(0, L.lo, L.hi);
        L.q = lerp(rest, target, w);
      });
    });
    cx = lerp(cx, mx, 0.06); cy = lerp(cy, my, 0.06);
    const k0 = 1 - smooth(p, 0.03, 0.2), shift = lerp(lay.shift1, lay.shift, k0), lift = wide ? lay.lift * k0 : lerp(0.04, lay.lift, k0);
    const W = stage.canvas.clientWidth, H = stage.canvas.clientHeight;
    stage.frame(new THREE.Vector3(0, 0.42, 0), 1.42, { azim: 0.42 + p * 0.5 + cx * 0.22, elev: 0.3 + p * 0.1 - cy * 0.1, pad: lerp(lay.pad, lay.pad1, smooth(p, 0.05, 0.6)) });
    stage.camera.setViewOffset(W, H, -shift * W, -lift * H, W, H);
    return true;
  };
})();

/* ================= film ================= */
{
  const v = $('#demo'), screen = $('.screen'), track = $('#v-track'), fillEl = $('#v-fill'), marks = $('#v-marks'), timeEl = $('#v-time'), list = $('#chapters');
  const CH = [[0, 'Loose parts in'], [8.6, 'The task'], [17.6, 'Two nested loops'], [29.4, 'Runtime Assembly Loop'], [38.6, 'GT-based Evaluator'], [47.2, 'Self-Improvement Loop'], [57.4, 'Twelve rounds'], [67, 'Held-out objects'], [82.1, 'Transfer'], [89.9, 'ArtiAssembly']];
  const DUR = 103.5, mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  // A server without byte ranges (python3 -m http.server) lets the browser play the film but never seek in it.
  // The first seek then loads the file once as a whole, after which every position can be reached.
  let whole = null;
  const canSeek = (t) => { for (let i = 0; i < v.seekable.length; i++) if (t >= v.seekable.start(i) - 0.5 && t <= v.seekable.end(i) + 0.5) return true; return false; };
  const seek = async (t, play) => {
    if (v.readyState < 1) await new Promise((ok) => v.addEventListener('loadedmetadata', ok, { once: true }));
    if (!canSeek(t) && !whole) {
      screen.classList.add('buffering');
      whole = fetch(v.currentSrc).then((r) => r.blob()).then((b) => new Promise((ok) => { v.addEventListener('loadedmetadata', ok, { once: true }); v.src = URL.createObjectURL(b); }))
        .catch(() => {}).finally(() => screen.classList.remove('buffering'));
    }
    if (whole) await whole;
    v.currentTime = t; if (play) v.play();
  };
  const items = CH.map(([t, name], i) => {
    const li = el('li', '', `<b>${name}</b><span>${mmss(t)}</span><i></i>`); list.append(li);
    li.onclick = () => seek(t + 0.05, true);
    li.tabIndex = 0; li.setAttribute('role', 'button'); li.setAttribute('aria-label', `Play from chapter ${i + 1}, ${name}, at ${mmss(t)}`); li.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); li.onclick(); } };
    if (i) { const m = el('i'); m.style.left = (t / DUR) * 100 + '%'; marks.append(m); }
    return li;
  });
  const icon = (playing) => (playing ? '<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>' : '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>');
  const toggle = () => (v.paused ? v.play() : v.pause());
  $('#play').onclick = toggle; v.onclick = toggle; $('#v-toggle').onclick = toggle;
  v.onplay = () => { screen.classList.add('playing', 'paused-started'); $('#v-toggle').innerHTML = icon(true); $('#v-toggle').setAttribute('aria-label', 'Pause'); };
  v.onpause = () => { screen.classList.remove('playing'); $('#v-toggle').innerHTML = icon(false); $('#v-toggle').setAttribute('aria-label', 'Play'); };
  v.ontimeupdate = () => {
    const d = v.duration || DUR, t = v.currentTime; fillEl.style.width = (t / d) * 100 + '%'; timeEl.textContent = `${mmss(t)} / ${mmss(d)}`;
    track.setAttribute('aria-valuenow', Math.round(t)); track.setAttribute('aria-valuetext', `${mmss(t)} of ${mmss(d)}`);
    let cur = 0; CH.forEach(([s], i) => { if (t >= s) cur = i; });
    items.forEach((li, i) => { li.classList.toggle('on', i === cur); const a = CH[i][0], b = i + 1 < CH.length ? CH[i + 1][0] : d; $('i', li).style.width = clamp((t - a) / (b - a)) * 100 + '%'; });
  };
  track.onclick = (e) => { const b = track.getBoundingClientRect(); seek(((e.clientX - b.left) / b.width) * (v.duration || DUR)); };
  track.onkeydown = (e) => { const d = v.duration || DUR, k = { ArrowLeft: -5, ArrowRight: 5, Home: -d, End: d }[e.key]; if (k) { e.preventDefault(); seek(clamp(v.currentTime + k, 0, d - 0.1)); } };
  $('#v-mute').onclick = (e) => { v.muted = !v.muted; e.target.textContent = v.muted ? 'MUTED' : 'SOUND'; e.target.setAttribute('aria-label', v.muted ? 'Sound is muted. Turn it on' : 'Sound is on. Mute it'); };
  $('#v-full').onclick = () => (document.fullscreenElement ? document.exitFullscreen() : screen.requestFullscreen && screen.requestFullscreen());
  items[0].classList.add('on');
}

/* ================= loops diagram ================= */
{
  const svg = $('#loops-svg'), card = $('#loops-card');
  const N = {
    skill: { x: 10, y: 196, w: 178, t: 'The Skill', s: 'Markdown + helpers', k: 'The only thing that changes', h: 'The Skill', p: 'A Markdown procedure and a set of helper scripts that the Executor loads before it touches a part. The outer loop edits nothing else.', f: [['seed', 'hand-written S₀'], ['edited by', 'the Skill Evolver'], ['returned', 'last champion, S₉']], ico: 'M-9-11h13l5 5v17h-18zM4-11v5h5M-5-1h10M-5 4h10' },
    exec: { x: 338, y: 60, w: 140, t: 'Executor', s: 'LLM agent', k: 'Runtime Assembly Loop', h: 'Executor', p: 'An LLM agent that reads the Skill and assembles one object. It writes code against the SDK, looks at renders, and corrects itself step by step.', f: [['model', 'gpt-6.1-sol'], ['reasoning effort', 'low'], ['budget', '60 tool turns']], ico: 'M-9-4h18v13h-18zM-4 2v2M4 2v2M0-4v-6M-3-11h6' },
    sdk: { x: 435, y: 300, w: 182, t: 'SDK + engine', s: 'simulate and render', k: 'Runtime Assembly Loop', h: 'SDK and engine', p: 'A lightweight agent-native engine, Light-3js-phyx. One call places a part, sets a joint, or returns four aligned passes of the current assembly.', f: [['passes', 'RGB, edges, ids, depth'], ['parallel', '14 sessions, one workstation'], ['resolution', '512 px']], ico: 'M0-11l10 5v11l-10 5-10-5v-11zM-10-6l10 5 10-5M0-1v11' },
    val: { x: 198, y: 300, w: 207, t: 'Runtime Validator', s: 'script, no ground truth', k: 'Runtime Assembly Loop', h: 'Runtime Validator', p: 'A script that audits every intermediate assembly without ground truth. A failed check sends the Executor back before the error can propagate.', f: [['reads', 'structure, renders, sweeps'], ['ground truth', 'never'], ['also edited by', 'the Skill Evolver']], ico: 'M0-11l9 3v7c0 6-4 10-9 12-5-2-9-6-9-12v-7zM-4 0l3 3 5-6' },
    eval: { x: 648, y: 60, w: 200, t: 'GT-based Evaluator', s: 'script, 12 checks', k: 'Self-Improvement Loop', h: 'GT-based Evaluator', p: 'Scores every URDF by its motion. The score ignores naming, axis sign, and offset placement, and comes with one record per joint that says what went wrong.', f: [['checks', '12 scored'], ['pose tolerance', 'τ = max(2 cm, 0.04 d)'], ['axis tolerance', '10°']], ico: 'M-10 9h20M-7 9v-8M0 9v-18M7 9v-12' },
    evo: { x: 668, y: 196, w: 160, t: 'Skill Evolver', s: 'LLM agent', k: 'Self-Improvement Loop', h: 'Skill Evolver', p: 'A second agent that reads the clustered failures of a whole batch and rewrites the Skill. Every edit must fix its own failure cluster on three train objects first.', f: [['reasoning effort', 'highest'], ['may edit', 'Markdown, helpers, checks'], ['sees ground truth', 'never']], ico: 'M-10 10l3-9 11-11 6 6-11 11zM1-7l6 6' },
    gate: { x: 648, y: 332, w: 200, t: 'Regression strategy', s: 'two layers, z ≥ 1.5', k: 'Self-Improvement Loop', h: 'Two-layer regression strategy', p: 'First, the Skill Evolver tests each edit on train objects of its failure cluster. Then the candidate and the champion assemble the validation split, and the candidate becomes the champion only when its mean gain exceeds zero by 1.5 standard errors.', f: [['layer 1', '3 train objects per edit'], ['layer 2', '40 validation objects × 3'], ['accepted', '3 of 9 candidates']], ico: 'M-10 10v-16M10 10v-16M-10-6h20M-10 2h20M0-6v8' },
  };
  const H = 60, node = (id) => {
    const n = N[id];
    return `<g class="node" data-id="${id}" tabindex="0" role="button" aria-label="${n.t}" transform="translate(${n.x},${n.y})"><rect width="${n.w}" height="${H}" rx="12"/><path class="ico" transform="translate(27,30)" d="${n.ico}"/><text class="t" x="50" y="27">${n.t}</text><text class="s" x="50" y="44">${n.s}</text></g>`;
  };
  svg.innerHTML = `
    <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0l10 5-10 5z" fill="#eef0f4"/></marker>
    <marker id="ahh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0l10 5-10 5z" fill="#b9a2ff"/></marker></defs>
    <ellipse cx="408" cy="222" rx="168" ry="128" fill="rgba(143,176,255,.05)" stroke="#8fb0ff" stroke-width="1.2" stroke-dasharray="2 5"/>
    <text class="loop-label cool" x="408" y="206" text-anchor="middle">RUNTIME</text>
    <text class="loop-label cool" x="408" y="223" text-anchor="middle">ASSEMBLY LOOP</text>
    <text class="fl" x="408" y="243" text-anchor="middle">one object, up to 60 turns</text>
    <path class="flow" marker-end="url(#ah)" d="M462 124 C 508 152, 530 220, 524 295"/>
    <path class="flow" marker-end="url(#ah)" d="M431 330 L 410 330"/>
    <path class="flow" marker-end="url(#ah)" d="M300 296 C 293 220, 312 152, 356 124"/>
    <path id="p-in" class="flow outer" marker-end="url(#ahh)" d="M193 226 L 235 226"/>
    <path id="p-a" class="flow outer" marker-end="url(#ahh)" d="M482 90 L 643 90"/>
    <path id="p-b" class="flow outer" marker-end="url(#ahh)" d="M748 124 L 748 191"/>
    <path id="p-c" class="flow outer" marker-end="url(#ahh)" d="M748 260 L 748 327"/>
    <path id="p-d" class="flow outer" marker-end="url(#ahh)" d="M644 366 C 612 366, 620 440, 576 440 L 135 440 C 109 440, 99 420, 99 262"/>
    <text class="fl" x="562" y="80" text-anchor="middle">URDF</text>
    <text class="fl" x="760" y="162">diagnostics</text>
    <text class="fl" x="760" y="298">candidate S′</text>
    <text class="fl" x="340" y="430" text-anchor="middle">accepted → new champion</text>
    <text class="loop-label hot" x="10" y="72">SELF-IMPROVEMENT LOOP</text>
    <text class="fl" x="10" y="91">one batch, one edit</text>
    ${reduced ? '<!--' : ''}<circle r="6" fill="#b9a2ff" stroke="#eef0f4" stroke-width="1.5"><animateMotion dur="7s" repeatCount="indefinite" path="M482 90 L 748 90 L 748 362 L 644 366 C 612 366, 620 440, 576 440 L 135 440 C 109 440, 99 420, 99 262 L 99 226 L 235 226"/></circle>
    <circle r="5" fill="#8fb0ff" stroke="#eef0f4" stroke-width="1.5"><animateMotion dur="3.2s" repeatCount="indefinite" path="M462 124 C 508 152, 530 220, 524 295 L 524 330 L 300 330 L 300 296 C 293 220, 312 152, 356 124 L 462 124"/></circle>${reduced ? '-->' : ''}
    ${Object.keys(N).map(node).join('')}`;
  const order = ['skill', 'exec', 'sdk', 'val', 'eval', 'evo', 'gate']; let cur = 0, auto = true;
  const show = (id) => {
    const n = N[id]; $$('.node', svg).forEach((g) => { g.classList.toggle('on', g.dataset.id === id); g.setAttribute('aria-pressed', g.dataset.id === id); });
    card.innerHTML = `<span class="k1 ${n.k.startsWith('Self') ? 'outer' : n.k.startsWith('Runtime') ? 'inner' : ''}">${n.k}</span><h3>${n.h}</h3><p>${n.p}</p><ul>${n.f.map(([a, b]) => `<li><span>${a}</span><b>${b}</b></li>`).join('')}</ul>`;
  };
  $$('.node', svg).forEach((g) => { g.addEventListener('click', () => { auto = false; show(g.dataset.id); }); g.addEventListener('mouseenter', () => { auto = false; show(g.dataset.id); }); g.addEventListener('focus', () => { auto = false; show(g.dataset.id); }); });
  show('skill');
  setInterval(() => { if (auto && !reduced) { cur = (cur + 1) % order.length; show(order[cur]); } }, 3200);
}

/* ================= lineage / time machine ================= */
(async () => {
  const railEl = $('#tm-rail'), svg = $('#tm-svg');
  const OBJ = [['tape_measure', 'Tape measure', 0], ['cash_register', 'Cash register', 1], ['microscope', 'Microscope', 2], ['computer_desk', 'Computer desk', 3]];
  let r = 0, obj = 0, model = null, playing = null, token = 0;
  const stage = new Stage($('#gl-tm'), { orbit: true, fov: 26 });
  stage.enableDrag();
  const W = 440, Hh = 430, L = 40, R = 16, T = 20, B = 50, y0 = 0.38, y1 = 0.74;
  const X = (i) => L + (i / 11) * (W - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (Hh - T - B);
  const champPath = (upto) => { let d = `M${X(0)} ${Y(CHAMP_VAL[0])}`, c = 0; for (let i = 1; i <= upto; i++) { const n = championAt(i); if (n !== c) { d += ` L${X(i)} ${Y(CHAMP_VAL[c])} L${X(i)} ${Y(CHAMP_VAL[n])}`; c = n; } else d += ` L${X(i)} ${Y(CHAMP_VAL[c])}`; } return d; };
  let g = '';
  for (let v = 0.4; v <= 0.701; v += 0.1) g += `<line class="gridl" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`;
  ROUNDS.forEach((d, i) => { g += `<text x="${X(i)}" y="${Hh - 28}" text-anchor="middle">${i}</text>`; });
  g += `<text x="${(L + W - R) / 2}" y="${Hh - 8}" text-anchor="middle">round</text>`;
  g += `<path class="train" d="${ROUNDS.map((d, i) => `${i ? 'L' : 'M'}${X(i)} ${Y(d.train)}`).join(' ')}"/>`;
  g += `<path class="champ future" d="${champPath(11)}"/><path class="champ" id="tm-champ-line" d=""/>`;
  g += `<line class="cursor" id="tm-cursor" x1="0" x2="0" y1="${T}" y2="${Hh - B}"/>`;
  ROUNDS.forEach((d, i) => { if (d.val == null) return; const c = d.gate === 'accepted' ? '#46d79a' : d.gate === 'seed' ? '#eef0f4' : '#b9a2ff';  g += `<circle class="pt" aria-hidden="true" data-i="${i}" cx="${X(i)}" cy="${Y(d.val)}" r="5.5" fill="${c}"/>`; });
  [[3, 'S3'], [6, 'S6'], [9, 'S9']].forEach(([i, n]) => { g += `<text class="lbl" x="${X(i) + (i === 9 ? -6 : 8)}" y="${Y(CHAMP_VAL[i]) - 10}"${i === 9 ? ' text-anchor="end"' : ''}>${n} ${CHAMP_VAL[i].toFixed(3)}</text>`; });
  g += `<text class="lbl" x="${X(0) + 2}" y="${Y(CHAMP_VAL[0]) + 22}">S0 0.439</text>`;
  svg.innerHTML = g;
  // the champion line is drawn once and revealed by its dash length, so that it grows smoothly from round to round
  const champLine = $('#tm-champ-line'), LEN = ROUNDS.map((_, i) => { champLine.setAttribute('d', champPath(i)); return i ? champLine.getTotalLength() : 0; });
  champLine.setAttribute('d', champPath(11));
  const fade = (e) => { if (!reduced && e.animate) e.animate([{ opacity: 0, transform: 'translateY(5px)' }, { opacity: 1, transform: 'none' }], { duration: 520, easing: 'cubic-bezier(.2,.7,.2,1)' }); };
  // one continuous position along the run: the champion line, the cursor, the points and the rail all follow it
  const railFill = el('span', 'tm-rail-fill');
  const scrub = (f) => {
    const i = Math.min(10, Math.floor(f)), u = f - i;
    champLine.style.strokeDasharray = `${lerp(LEN[i], LEN[i + 1], u).toFixed(1)} 4000`;
    $('#tm-cursor').style.transform = `translateX(${X(f).toFixed(2)}px)`;
    $$('.pt', svg).forEach((c) => { c.style.opacity = +c.dataset.i <= f + 0.02 ? 1 : 0.28; });
    const b = $$('.rd', railEl), mid = (k) => b[k].offsetLeft + b[k].offsetWidth / 2;
    if (b.length === 12) { railFill.style.left = mid(0) + 'px'; railFill.style.width = lerp(mid(i), mid(i + 1), u) - mid(0) + 'px'; }
  };
  let fitShown = null, fitRaf = 0;
  const showFit = (v) => {
    const b = $('#tm-fit'), a = fitShown; fitShown = v; cancelAnimationFrame(fitRaf);
    if (a == null || reduced) { b.textContent = v.toFixed(2); return; }
    const t0 = performance.now(), step = (now) => { const t = clamp((now - t0) / 800); b.textContent = lerp(a, v, ease(t)).toFixed(2); if (t < 1) fitRaf = requestAnimationFrame(step); };
    fitRaf = requestAnimationFrame(step);
  };
  $$('.pt', svg).forEach((c) => (c.onclick = () => { stop(); select(+c.dataset.i); }));

  ROUNDS.forEach((d, i) => {
    const b = el('button', 'rd ' + d.gate, `<i>${d.gate === 'accepted' ? '✓' : d.gate === 'rejected' ? '✕' : d.gate === 'seed' ? 'S₀' : '–'}</i><span>${i === 0 ? 'seed' : 'round ' + i}</span>`);
    b.setAttribute('aria-label', i === 0 ? 'Seed Skill' : `Round ${i}, ${d.gate === 'unchanged' ? 'no edit' : d.gate}`);
    b.onclick = () => { stop(); select(i); }; railEl.append(b);
  });
  railEl.prepend(railFill);
  OBJ.forEach(([slug, label], i) => { const c = el('button', 'chip', label); c.onclick = () => setObj(i); $('#tm-objs').append(c); });

  const applyModel = (instant) => {
    const ch = championAt(r), col = { 0: 3, 3: 4, 6: 5, 9: 6 }[ch];
    $('#tm-real').src = `assets/img/cells/q${String(OBJ[obj][2] * 7 + col).padStart(2, '0')}.jpg`;
    $('#tm-real').alt = `Render from the paper of the ${OBJ[obj][1].toLowerCase()} assembled with S${ch}`;
    if (!model) return;
    model.setDefects(defectsFor(model.asset.slug, ch, model.links.length), instant); stage.dirty = true;
    const ev = evaluate(model); if (instant) fitShown = null; showFit(ev.fitness);
    const pf = $('#tm-pass'); pf.hidden = false; pf.textContent = ev.pass ? 'PASS' : 'FAIL'; pf.classList.toggle('pass', ev.pass);
  };
  async function setObj(i) {
    obj = i; $$('#tm-objs .chip').forEach((c, k) => { c.classList.toggle('on', k === i); c.setAttribute('aria-pressed', k === i); }); const my = ++token;
    const wait = $('#tm-loading'); wait.classList.remove('off', 'failed'); wait.textContent = 'loading parts…';
    let a; try { a = await loadAsset(OBJ[i][0]); } catch (_) { if (my === token) { wait.textContent = 'This model could not be loaded'; wait.classList.add('failed'); } return; }
    if (my !== token) return; wait.classList.add('off');
    if (model) stage.remove(model);
    model = stage.add(new Model(a)); model.gizmos.visible = $('#tm-axes').checked;
    stage.frameModel(model, { azim: -0.62, elev: 0.36, pad: 1.55 }); applyModel(true);
  }
  function select(i, quiet) {
    r = i; const d = ROUNDS[i], ch = championAt(i);
    $$('.rd', railEl).forEach((b, k) => { b.classList.toggle('on', k === i); b.setAttribute('aria-pressed', k === i); });
    if (!quiet) scrub(i);
    $$('.pt', svg).forEach((c) => c.setAttribute('r', +c.dataset.i === i ? 8 : 5.5));
    $('#tm-file').textContent = d.file; $('#tm-round').textContent = i ? `round ${i}` : 'seed';
    const changed = $('#tm-title').textContent !== d.title; $('#tm-title').textContent = d.title;
    $('#tm-diff').innerHTML = d.diff.map((l) => l.replace(/<add>/g, '<span class="add">').replace(/<\/add>/g, '</span>').replace(/<del>/g, '<span class="del">').replace(/<\/del>/g, '</span>').replace(/<cm>/g, '<span class="cm">').replace(/<\/cm>/g, '</span>')).map((l) => (l.startsWith('<span class="add"') || l.startsWith('<span class="del"') ? l : l + '\n')).join('');
    const st = $('#tm-stamp'); st.className = 'stamp ' + d.gate; st.textContent = d.gate === 'unchanged' ? 'NO EDIT' : d.gate.toUpperCase();
    void st.offsetWidth; st.classList.add('hit');
    if (changed) { fade($('#tm-title')); fade($('#tm-diff')); fade($('#tm-gate-note')); }
    const zf = $('#tm-zfill');
    if (d.z != null) {
      const a = 25, b = clamp(25 + d.z * 25, 0, 100); zf.style.left = Math.min(a, b) + '%'; zf.style.width = Math.abs(b - a) + '%'; zf.style.background = d.z >= 1.5 ? 'var(--ok)' : 'var(--bad)';
      $('#tm-z').textContent = `Δ ${d.d > 0 ? '+' : '−'}${Math.abs(d.d).toFixed(3)} · z = ${d.z > 0 ? '+' : '−'}${Math.abs(d.z).toFixed(2)}`;
      $('#tm-gate-note').textContent = d.gate === 'accepted' ? `Clear paired gain on the 40 validation objects. S${d.v} becomes the champion.` : `The gain is within noise. The champion stays S${ch}.`;
    } else { zf.style.width = '0%'; $('#tm-z').textContent = '—'; $('#tm-gate-note').textContent = d.gate === 'seed' ? 'The seed states only the input, the output, and the SDK.' : 'Nothing to test this round. The champion stays S' + ch + '.'; }
    $('#tm-champ').innerHTML = S(ch); $('#bar-rev').innerHTML = S(ch);
    applyModel(false);
  }
  // replay: the playhead rests on a round while its card is read, then glides to the next one
  const PER = 2.6, DWELL = 0.36; let raf = 0;
  const setPlaying = (on) => { playing = on; svg.classList.toggle('playing', on); railEl.classList.toggle('playing', on); $('#tm-play').textContent = on ? 'Pause' : 'Replay the run'; };
  function stop() { if (playing) { cancelAnimationFrame(raf); setPlaying(false); scrub(r); } }
  $('#tm-play').onclick = () => {
    if (playing) return stop();
    select(0, true); scrub(0); setPlaying(true);
    let ph = 0, last = performance.now();
    const tick = (now) => {
      ph += Math.min(0.1, (now - last) / 1000) / PER; last = now;
      if (ph >= 11) { scrub(11); if (r !== 11) select(11, true); cancelAnimationFrame(raf); setPlaying(false); return; }
      const i = Math.floor(ph); if (i !== r) select(i, true);
      scrub(i + smooth(ph - i, DWELL, 1));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  };
  $('#tm-axes').onchange = (e) => { if (model) { model.gizmos.visible = e.target.checked; stage.dirty = true; } };
  // gentle joint sweep so that wrong pivots and axes are visible
  stage.onFrame = (dt, time) => { if (!model) return false; model.links.forEach((Lk) => { const mid = (Lk.lo + Lk.hi) / 2, half = (Lk.hi - Lk.lo) / 2; Lk.q = Lk.joint.type === 'continuous' ? time * 0.9 : mid + Math.sin(time * 1.1 + Lk.phase) * half * 0.85; }); return !reduced; };
  await setObj(0); select(0); $('#bar-rev').innerHTML = S(9);   // the bar shows the final Skill until the reader scrubs the run
  // start the replay once when the section scrolls into view
  const once = new IntersectionObserver((es) => { if (es[0].isIntersecting) { once.disconnect(); if (!reduced) $('#tm-play').click(); } }, { threshold: 0.35 });
  once.observe($('.tm-grid'));
})();

/* ================= workbench ================= */
(async () => {
  const stage = new Stage($('#gl-wb'), { orbit: true, fov: 26, zoom: false });
  const shelf = $('#wb-shelf'), sliders = $('#wb-sliders'), checks = $('#wb-checks');
  const index = await fetch('assets/models/index.json').then((x) => x.json());
  let model = null, version = 9, style = 'color', sweep = false, tT = 1, tE = 0, token = 0, pendingEval = false, manualRun = false;
  const TYPES = { prismatic: 'slides', revolute: 'hinges', continuous: 'spins' };
  const fmtQ = (L) => (L.joint.type === 'prismatic' ? `${(L.q * 1000).toFixed(0)} mm` : `${((L.q * 180) / Math.PI).toFixed(0)}°`);

  index.forEach((it, i) => {
    const b = el('button', 'item', `<em>${String(i + 1).padStart(2, '0')}</em><b>${it.label}</b><span>${it.joints} joint${it.joints > 1 ? 's' : ''} · ${it.types.join(', ')}</span>`);
    b.setAttribute('aria-label', `${it.label}, ${it.joints} joint${it.joints > 1 ? 's' : ''}`); b.onclick = () => load(i); shelf.append(b);
  });
  [0, 3, 6, 9].forEach((v) => { const b = el('button', v === 9 ? 'on' : '', `S<sub>${v}</sub><small>${v === 0 ? 'seed' : v === 9 ? 'final' : 'accepted'}</small>`); b.dataset.v = v; b.onclick = () => setVersion(v); $('#wb-skill').append(b); });

  function renderEval(blank) {
    if (!model) return; const ev = evaluate(model);
    const head = '<tr><th>joint</th><th>scale</th><th>rest</th><th>axis</th><th>pivot</th><th>motion</th></tr>';
    checks.innerHTML = head + ev.rows.map((row) => `<tr><td><i style="background:${hex(row.L.color)}"></i>${row.L.joint.name.replace('joint_', 'j')}</td>` +
      ['scale', 'rest', 'axis', 'pivot', 'motion'].map((k) => `<td>${blank ? '<span class="ck blank"></span>' : `<span class="ck ${row.checks[k] ? '' : 'x'}">${row.checks[k] ? '✓' : '✕'}</span>`}</td>`).join('') + '</tr>').join('');
    const vd = $('#wb-verdict');
    if (blank) { vd.textContent = 'WAITING'; vd.className = 'verdict wait'; $('#wb-fit').textContent = '–'; $('#wb-fitbar').style.width = '0%'; return; }
    vd.textContent = ev.pass ? 'PASS' : 'FAIL'; vd.className = 'verdict' + (ev.pass ? '' : ' fail');
    $('#wb-fit').textContent = ev.fitness.toFixed(2); $('#wb-fitbar').style.width = ev.fitness * 100 + '%'; $('#wb-fitbar').style.background = ev.pass ? 'var(--ok)' : ev.fitness > 0.6 ? 'var(--sun)' : 'var(--bad)';
  }
  function buildSliders() {
    sliders.innerHTML = '';
    model.links.forEach((L) => {
      const row = el('div', 'sl', `<label><b><i style="background:${hex(L.color)}"></i>${L.joint.name.replace('_', ' ')} <span style="font-weight:400;color:var(--mute)">${TYPES[L.joint.type]}</span></b><em>${fmtQ(L)}</em></label><input type="range" min="0" max="1" step="0.001">`);
      const inp = $('input', row); inp.setAttribute('aria-label', `${L.joint.name.replace('_', ' ')}, ${TYPES[L.joint.type]}`); inp.value = (L.q - L.lo) / (L.hi - L.lo || 1);
      inp.oninput = () => { stopSweep(); model.setQ01(L, +inp.value); $('em', row).textContent = fmtQ(L); stage.dirty = true; };
      L.ui = { inp, out: $('em', row) }; sliders.append(row);
    });
  }
  const syncSlider = (L) => { if (L.ui) { L.ui.inp.value = (L.q - L.lo) / (L.hi - L.lo || 1); L.ui.out.textContent = fmtQ(L); } };
  function setVersion(v) {
    version = v; $$('#wb-skill button').forEach((b) => { b.classList.toggle('on', +b.dataset.v === v); b.setAttribute('aria-pressed', +b.dataset.v === v); });
    if (!model) return; model.setDefects(defectsFor(model.asset.slug, v, model.links.length)); stage.dirty = true; renderEval(tT < 1);
  }
  function stopSweep() { sweep = false; $('#wb-play').classList.remove('active'); $('#wb-play').setAttribute('aria-pressed', false); }
  async function load(i) {
    const it = index[i], my = ++token; $$('.item', shelf).forEach((b, k) => { b.classList.toggle('on', k === i); b.setAttribute('aria-pressed', k === i); });
    const wait = $('#wb-loading'); wait.classList.remove('off', 'failed'); wait.textContent = 'loading parts…';
    let a; try { a = await loadAsset(it.slug); } catch (_) { if (my === token) { wait.textContent = 'This model could not be loaded'; wait.classList.add('failed'); } return; }
    if (my !== token) return;
    if (model) stage.remove(model);
    model = stage.add(new Model(a, { style, seed: 11, spread: 0.52, lift: 0.6 }));   // loose parts stay low enough to remain inside the stage model.gizmos.visible = $('#wb-axes').checked;
    model.links.forEach((L) => { L.q = clamp(0, L.lo, L.hi); });
    model.setDefects(defectsFor(it.slug, version, model.links.length), true);
    manualRun = false; model.t = 0; tT = 1; pendingEval = true; model.explode = 0; tE = 0; $('#wb-explode').classList.remove('active');
    stage.frameModel(model, { azim: -0.6, elev: 0.34, pad: 1.5 * (stage.canvas.clientWidth < 800 ? clamp(1.32 / stage.camera.aspect, 1, 1.3) : 1) });   // a portrait stage needs more room for the loose parts
    $('#wb-name').textContent = it.label; $('#wb-sub').textContent = `${a.meta.links.length} links · ${model.links.length} joint${model.links.length > 1 ? 's' : ''} · ${a.meta.faces.toLocaleString('en-US')} faces`;
    { const h = $('#wb-hint'); h.textContent = HINT; h.classList.remove('hot'); h.style.opacity = 1; }
    buildSliders(); renderEval(!HAS_GL ? false : true); if (!HAS_GL) pendingEval = false; wait.classList.add('off');
  }
  const HINT = 'drag a part · drag the background to orbit', HINT2 = 'click a loose part to fit it yourself';
  const clearManual = () => model && model.links.forEach((L) => { L.manual = null; L.fitting = false; });
  $('#wb-scramble').onclick = () => { stopSweep(); clearManual(); tT = 0; renderEval(true); const h = $('#wb-hint'); h.textContent = HINT2; h.style.opacity = 1; h.classList.add('hot'); };
  $('#wb-assemble').onclick = () => {
    stopSweep(); if (!model) return;
    const done = model.links.every((L) => L.manual == null || L.manual >= 1);
    if (model.links.some((L) => L.manual != null) && !done) { model.links.forEach((L) => { if (L.manual == null) L.manual = 0; L.fitting = true; }); manualRun = true; }
    else { clearManual(); if (model.t > 0.98) model.t = 0; tT = 1; }
    pendingEval = true; renderEval(true); const h = $('#wb-hint'); h.textContent = HINT; h.classList.remove('hot');
  };
  $('#wb-explode').onclick = (e) => { tE = tE ? 0 : 1; e.currentTarget.classList.toggle('active', !!tE); e.currentTarget.setAttribute('aria-pressed', !!tE); };
  $('#wb-play').onclick = (e) => { sweep = !sweep; e.currentTarget.classList.toggle('active', sweep); e.currentTarget.setAttribute('aria-pressed', sweep); };
  $('#wb-axes').onchange = (e) => { if (model) { model.gizmos.visible = e.target.checked; stage.dirty = true; } };
  $$('#wb-style button').forEach((b) => (b.onclick = () => { style = b.dataset.style; $$('#wb-style button').forEach((x) => { x.classList.toggle('on', x === b); x.setAttribute('aria-pressed', x === b); }); if (model) { model.setStyle(style); stage.dirty = true; } }));
  stage.enableDrag((L, start) => {
    if (start) {
      stopSweep();
      if (model.t < 0.02 && tT === 0) {          // scrambled: fit this part by hand
        model.links.forEach((K) => { if (K.manual == null) K.manual = 0; });
        L.fitting = true; manualRun = true; pendingEval = true;
      } else $('#wb-hint').style.opacity = 0;
    }
    syncSlider(L);
  });
  stage.onFrame = (dt, time) => {
    if (!model) return false; let moving = false;
    if (model.t !== tT) { const sp = tT > model.t ? dt / 2.4 : -dt / 0.9; model.t = clamp(model.t + sp, 0, 1); if ((sp > 0 && model.t >= tT) || (sp < 0 && model.t <= tT)) model.t = tT; moving = true; }
    else if (pendingEval && tT === 1 && !manualRun) { pendingEval = false; renderEval(false); }
    if (manualRun && model.links.every((L) => L.manual != null && L.manual >= 1)) {
      manualRun = false; clearManual(); model.t = 1; tT = 1; pendingEval = false; renderEval(false);
      const h = $('#wb-hint'); h.textContent = HINT; h.classList.remove('hot');
    }
    model.gizmos.visible = $('#wb-axes').checked && (model.t > 0.97 || model.links.some((L) => L.manual != null && L.manual >= 1));
    if (model.explode !== tE) { model.explode = clamp(model.explode + (tE > model.explode ? 1 : -1) * dt / 0.5, 0, 1); if (Math.abs(model.explode - tE) < 0.02) model.explode = tE; moving = true; }
    if (sweep) { model.links.forEach((L) => { const mid = (L.lo + L.hi) / 2, half = (L.hi - L.lo) / 2; L.q = L.joint.type === 'continuous' ? ((time * 1.2 + Math.PI) % (Math.PI * 2)) - Math.PI : mid + Math.sin(time * 1.4 + L.phase) * half; syncSlider(L); }); moving = true; }
    return moving;
  };
  // assemble the first object when the bench scrolls into view
  let first = true;
  const io = new IntersectionObserver((es) => { if (es[0].isIntersecting && first) { first = false; io.disconnect(); load(3); } }, { threshold: 0.25 });
  io.observe($('.wb-stage'));
})();

/* ================= numbers ================= */
{
  let metric = 'fit', skill = 9;
  const bars = $('#nm-bars'), seg = $('#nm-skill');
  [3, 6, 9].forEach((v) => { const b = el('button', v === 9 ? 'on' : '', `S<sub>${v}</sub><small>vs. seed S<sub>0</sub></small>`); b.onclick = () => { skill = v; $$('button', seg).forEach((x) => { x.classList.toggle('on', x === b); x.setAttribute('aria-pressed', x === b); }); draw(); }; seg.append(b); });
  TRANSFER.forEach((m) => bars.append(el('div', 'brow', `<div class="nm">${m.id}<small>${m.note}</small></div><div class="btrack"><i class="b0"></i><i class="b1"></i><span class="na"></span></div><div class="val"><span></span><small></small></div>`)));
  function draw() {
    const max = metric === 'fit' ? 0.9 : 80;
    $$('.brow', bars).forEach((row, i) => {
      const m = TRANSFER[i][metric], a = m[0], b = m[skill], f = (x) => (metric === 'fit' ? x.toFixed(3) : x.toFixed(1) + '%');
      $('.b0', row).style.width = (a / max) * 100 + '%';
      $('.b1', row).style.width = b == null ? '0%' : (b / max) * 100 + '%'; $('.b1', row).style.opacity = b == null ? 0 : 1;
      $('.na', row).innerHTML = b == null ? `not run with ${S(skill)}` : '';
      $('.val span', row).textContent = b == null ? f(a) : f(b);
      $('.val small', row).style.color = b == null ? 'var(--mute)' : '';
      $('.val small', row).textContent = b == null ? 'seed' : metric === 'fit' ? `+${(b - a).toFixed(3)}` : `+${(b - a).toFixed(1)} pts`;
    });
  }
  $$('#nm-metric button').forEach((b) => (b.onclick = () => { metric = b.dataset.m; $$('#nm-metric button').forEach((x) => { x.classList.toggle('on', x === b); x.setAttribute('aria-pressed', x === b); }); draw(); }));
  draw();

  const cmp = $('#cmp'); let tier = 'all';
  COMPARE.forEach((c) => cmp.append(el('div', 'crow ' + (c.c || ''), `<div class="nm"><span>${c.n}</span><small>${c.s}</small></div><div class="ctrack"><i style="width:0"></i></div><div class="val"></div>`)));
  const TN = { all: 'all 40 test objects', easy: 'the 11 easy objects', medium: 'the 15 medium objects', hard: 'the 14 hard objects' };
  const drawC = () => {
    $$('.crow', cmp).forEach((row, i) => { const v = COMPARE[i].v[tier]; $('i', row).style.width = Math.max(v, 0.6) + '%'; $('.val', row).textContent = v.toFixed(1); });
    const gain = COMPARE[5].v[tier] - COMPARE[4].v[tier], lead = COMPARE[5].v[tier] - COMPARE[1].v[tier];
    $('#cmp-take').innerHTML = gain > 0
      ? `<b>+${gain.toFixed(1)}</b><span>points from self-improvement alone on ${TN[tier]}, with the same model and the same budget${lead > 0 ? `. That is ${lead.toFixed(1)} points ahead of USDCraft` : ''}.</span>`
      : `<b>${COMPARE[5].v[tier].toFixed(0)}%</b><span>of ${TN[tier]} are correct under both the seed Skill and S<sub>9</sub> with gpt-6-astra.</span>`;
  };
  $$('#cmp-tier button').forEach((b) => (b.onclick = () => { tier = b.dataset.t; $$('#cmp-tier button').forEach((x) => { x.classList.toggle('on', x === b); x.setAttribute('aria-pressed', x === b); }); drawC(); }));
  new IntersectionObserver((es, o) => { if (es[0].isIntersecting) { o.disconnect(); drawC(); } }, { threshold: 0.3 }).observe(cmp);

  const q = $('#q-grid');
  const cols = ['Ground truth', 'Particulate', 'USDCraft', 'S<sub>0</sub> seed', 'S<sub>3</sub>', 'S<sub>6</sub>', 'S<sub>9</sub> final'], rows = [['Tape measure', '2 prismatic joints'], ['Cash register', 'drawer and display'], ['Microscope', '4 joints, 3 types'], ['Computer desk', 'tray and drawer']];
  q.append(el('div', 'qh', ''));
  cols.forEach((c, i) => q.append(el('div', 'qh' + (i >= 3 ? ' ours' : ''), c)));
  rows.forEach(([n, s], rI) => {
    q.append(el('div', 'ql', `${n}<small>${s}</small>`));
    cols.forEach((c, cI) => q.append(el('div', 'qc' + (cI === 6 ? ' best' : ''), `<img loading="lazy" src="assets/img/cells/q${String(rI * 7 + cI).padStart(2, '0')}.jpg" alt="${n}, ${c.replace(/<[^>]+>/g, '')}">`)));
  });
}

/* ================= ArtiAssembly ================= */
{
  const tiers = [['T1', 90, 'tier 1 objects'], ['T2', 90, 'tier 2 objects'], ['T3', 75, 'tier 3 objects'], ['T4', 45, 'tier 4 objects, the most complex']];
  tiers.forEach(([n, c, s]) => { const t = el('div', 'tier', `<b>${n} · ${c}</b><span>${s}</span><i style="width:${(c / 90) * 100}%"></i>`); $('#tiers').append(t); });
  const wall = $('#arti-wall'), sec = $('#arti');
  if (!reduced) addEventListener('scroll', () => { const b = sec.getBoundingClientRect(); if (b.bottom < 0 || b.top > innerHeight) return; const k = (innerHeight - b.top) / (innerHeight + b.height); wall.style.transform = `translate3d(${(0.5 - k) * 9}%, 0, 0) scale(1.12)`; }, { passive: true });
}

/* ================= cite ================= */
$('#bib-copy').onclick = async (e) => { try { await navigator.clipboard.writeText($('#bib').textContent); e.target.textContent = 'copied'; setTimeout(() => (e.target.textContent = 'copy'), 1400); } catch (_) { /* clipboard unavailable */ } };

/* ================= pressed state of the static toggles ================= */
$$('.seg button').forEach((b) => { if (!b.hasAttribute('aria-pressed')) b.setAttribute('aria-pressed', b.classList.contains('on')); });
