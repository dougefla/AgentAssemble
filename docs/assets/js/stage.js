// Shared 3D engine of the AgentAssemble page.
// A Stage owns one canvas. A Model is one articulated object rebuilt from a GLB whose
// nodes are named link_<i> and a JSON file with the URDF joints (Z-up, metres).
// Every movable link sits in  pivot > scatter > error > mesh  so that the page can
// (1) drive the joint, (2) scatter and re-assemble the part, and (3) show the pose
// errors of an earlier Skill version, all about the link origin.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export { THREE };

export const INK = 0x07080b;
// root body first, then one colour per movable part. Slightly desaturated so that the parts sit calmly on the dark page.
export const PALETTE = [0x789fde, 0xeaa843, 0x7ccb88, 0xe6837a, 0xb09ae6, 0x55bdb7, 0xdb8fbc];
const edgeTint = (c, hex) => c.setHex(hex).multiplyScalar(0.2);   // edge lines are a deep shade of the part colour
const WHITE = new THREE.Color(0xffffff);
const C_OK = 0xf2f4f8, C_BAD = 0xff6b6b, ROUGH = 0.66;

// three.js needs WebGL 2. Without it every Stage shows a short message in place of its canvas.
export const HAS_GL = (() => { try { return !!document.createElement('canvas').getContext('webgl2'); } catch (_) { return false; } })();
if (!HAS_GL) document.documentElement.classList.add('no-gl');

const loader = new GLTFLoader();
const cache = new Map();

export function loadAsset(slug, base = 'assets/models/') {
  if (!cache.has(slug)) {
    cache.set(slug, Promise.all([
      fetch(base + slug + '.json').then((r) => r.json()),
      loader.loadAsync(base + slug + '.glb'),
    ]).then(([meta, gltf]) => {
      const geos = {}, edges = {};
      gltf.scene.updateMatrixWorld(true);
      gltf.scene.traverse((o) => {
        if (o.isMesh && /^link_\d+$/.test(o.name)) {
          const g = o.geometry.clone();
          g.applyMatrix4(o.matrixWorld);
          g.computeBoundingBox();
          geos[o.name] = g;
          edges[o.name] = new THREE.EdgesGeometry(g, 32);
        }
      });
      return { slug, meta, geos, edges };
    }));
  }
  return cache.get(slug);
}

export function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function hash(str) { let h = 2166136261; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const easeOutBack = (t) => { const c = 1.4; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); };

const Q_ID = new THREE.Quaternion();
const V = new THREE.Vector3(), V2 = new THREE.Vector3(), Qt = new THREE.Quaternion();
const Z = new THREE.Vector3(0, 0, 1);

export const NO_DEFECT = { dp: [0, 0, 0], rot: [0, 0, 1, 0], s: 1, ax: 0, pe: [0, 0, 0] };

export class Model {
  constructor(asset, { style = 'color', seed = 1, spread = 1, lift = 1 } = {}) {
    this.asset = asset; this.meta = asset.meta; this.style = style;
    const bb = this.meta.bbox;
    this.size = new THREE.Vector3(bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]);
    this.diag = this.size.length();
    this.group = new THREE.Group();            // Y-up wrapper
    this.group.rotation.x = -Math.PI / 2;
    this.z = new THREE.Group();                // native Z-up frame, centred on the ground
    this.z.position.set(-(bb.min[0] + bb.max[0]) / 2, -(bb.min[1] + bb.max[1]) / 2, -bb.min[2]);
    this.group.add(this.z);
    this.links = []; this.meshes = [];
    this.gizmos = new THREE.Group(); this.gizmos.visible = false; this.z.add(this.gizmos);
    this.t = 1;               // assembly progress, 0 scattered .. 1 assembled
    this.explode = 0;
    this.time = 0;
    const rand = rng(hash(asset.slug) + seed * 7919);

    const mk = (name, color, isRoot) => {
      const mat = new THREE.MeshStandardMaterial({ color, roughness: ROUGH, metalness: 0.0 });
      const mesh = new THREE.Mesh(asset.geos[name], mat);
      mesh.castShadow = true; mesh.receiveShadow = true;
      const segs = asset.edges[name].attributes.position.count / 2, base = clamp(2600 / segs, 0.035, 0.34);
      const line = new THREE.LineSegments(asset.edges[name], new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: base }));
      mesh.add(line);
      mesh.userData = { color, line, isRoot, base };
      return mesh;
    };
    const rootMesh = mk(this.meta.root, PALETTE[0], true);
    this.rootMesh = rootMesh; this.z.add(rootMesh);

    this.meta.joints.forEach((j, i) => {
      if (!asset.geos[j.child]) return;
      const color = PALETTE[(i + 1) % PALETTE.length];
      const mesh = mk(j.child, color, false);
      const origin = new THREE.Vector3().fromArray(j.origin);
      const pivot = new THREE.Group(), scat = new THREE.Group(), err = new THREE.Group();
      pivot.add(scat); scat.add(err); err.add(mesh); this.z.add(pivot);
      // scattered pose: a random rotation about the link origin and a rescale, as in the benchmark
      const gb = asset.geos[j.child].boundingBox; const gs = new THREE.Vector3(); gb.getSize(gs);
      const longest = Math.max(gs.x, gs.y, gs.z) || 1;
      const a = rand() * Math.PI * 2, r = this.diag * (0.75 + rand() * 0.55) * spread;
      const sp = new THREE.Vector3(Math.cos(a) * r, Math.sin(a) * r * 0.8, this.diag * (0.25 + rand() * 0.75) * spread * lift);
      const gc = new THREE.Vector3(); gb.getCenter(gc);
      sp.add(new THREE.Vector3(this.meta.bbox.min[0] + this.size.x / 2, this.meta.bbox.min[1] + this.size.y / 2, this.meta.bbox.min[2] + this.size.z / 2)).sub(gc);
      const sq = new THREE.Quaternion().setFromEuler(new THREE.Euler(rand() * 6.28, rand() * 6.28, rand() * 6.28));
      const ss = clamp((this.diag * 0.42) / longest, 0.6, 1.7);
      const tumble = new THREE.Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5).normalize();
      const L = {
        i, name: j.child, joint: j, mesh, pivot, scat, err, origin,
        axis: new THREE.Vector3().fromArray(j.axis).normalize(),
        q: 0, color, sp, sq, ss, tumble, phase: rand() * 6.28,
        defect: NO_DEFECT, defectFrom: NO_DEFECT, db: 1,
        lo: j.lower, hi: j.upper, center: gc.clone(),
      };
      if (j.type === 'continuous') { L.lo = -Math.PI; L.hi = Math.PI; }
      mesh.userData.link = L;
      this.links.push(L); this.meshes.push(mesh);
      L.gizmo = this._gizmo(L); this.gizmos.add(L.gizmo);
    });
    this.setStyle(style);
    this.update();
  }

  _gizmo(L) {
    const g = new THREE.Group();
    const len = this.diag * 0.42, mat = new THREE.LineDashedMaterial({ color: C_OK, dashSize: len * 0.07, gapSize: len * 0.045, depthTest: false, transparent: true });
    const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, -len), new THREE.Vector3(0, 0, len)]);
    const line = new THREE.Line(geo, mat); line.computeLineDistances(); g.add(line);
    const solid = new THREE.MeshBasicMaterial({ color: C_OK, depthTest: false, transparent: true });
    const dot = new THREE.Mesh(new THREE.SphereGeometry(this.diag * 0.014, 16, 12), solid); g.add(dot);
    if (L.joint.type === 'prismatic') {
      const cone = new THREE.Mesh(new THREE.ConeGeometry(this.diag * 0.022, this.diag * 0.07, 16), solid);
      cone.rotation.x = Math.PI / 2; cone.position.z = len; g.add(cone);
    } else {
      const arc = L.joint.type === 'continuous' ? Math.PI * 2 : Math.PI * 1.35;
      const ring = new THREE.Mesh(new THREE.TorusGeometry(this.diag * 0.085, this.diag * 0.0045, 8, 48, arc), solid); g.add(ring);
      if (L.joint.type !== 'continuous') {
        const tip = new THREE.Mesh(new THREE.ConeGeometry(this.diag * 0.016, this.diag * 0.045, 12), solid);
        tip.position.set(Math.cos(arc) * this.diag * 0.085, Math.sin(arc) * this.diag * 0.085, 0); tip.rotation.z = arc; g.add(tip);
      }
    }
    g.traverse((o) => { o.renderOrder = 20; });
    g.userData = { line: mat, solid };
    return g;
  }

  // colour: palette bodies with ink edges. manual (Blueprint): slate bodies with light edge lines.
  // xray: translucent, faintly glowing bodies with edges in the colour of the part.
  setStyle(style) {
    this.style = style;
    const all = [this.rootMesh, ...this.meshes];
    all.forEach((m) => {
      const mat = m.material, u = m.userData, lm = u.line.material;
      if (style === 'manual') {
        mat.color.setHex(u.isRoot ? 0x465273 : 0x61719b); mat.roughness = 1;
        lm.color.setHex(0xdfe8ff); lm.opacity = clamp(u.base * 3.2, 0.2, 0.9);
        u.em = 0; u.emI = 1;
      } else if (style === 'xray') {
        mat.color.setHex(u.color); mat.roughness = ROUGH;
        lm.color.setHex(u.color).lerp(WHITE, 0.45); lm.opacity = clamp(u.base * 2.6, 0.16, 0.8);
        u.em = u.color; u.emI = 0.4;
      } else {
        mat.color.setHex(u.color); mat.roughness = ROUGH;
        edgeTint(lm.color, u.color); lm.opacity = u.base;
        u.em = 0; u.emI = 1;
      }
      mat.emissive.setHex(u.em); mat.emissiveIntensity = u.emI;
      mat.transparent = style === 'xray'; mat.opacity = style === 'xray' ? 0.36 : 1; mat.depthWrite = style !== 'xray';
      m.castShadow = style !== 'xray';
      mat.needsUpdate = true;
    });
  }

  highlight(L, on) {
    const xray = this.style === 'xray';
    this.meshes.forEach((m) => { m.material.emissive.setHex(m.userData.em || 0); m.material.emissiveIntensity = m.userData.emI ?? 1; if (xray) m.material.opacity = 0.36; });
    if (!L || !on) return;
    const mat = L.mesh.material;       // the part under the pointer lights up in its own colour
    if (this.style === 'manual') { mat.emissive.setHex(0x8fb0ff); mat.emissiveIntensity = 0.5; }
    else if (xray) { mat.emissiveIntensity = 1; mat.opacity = 0.66; }
    else { mat.emissive.setHex(L.color); mat.emissiveIntensity = 0.3; }
  }

  setDefects(list, instant = false) {      // list[i] is the defect of joint i, or undefined
    this.links.forEach((L) => {
      const cur = this._current(L);
      L.defectFrom = instant ? (list[L.i] || NO_DEFECT) : cur;
      L.defect = list[L.i] || NO_DEFECT; L.db = instant ? 1 : 0;
    });
    this.update();
  }
  _current(L) {
    const a = L.defectFrom, b = L.defect, t = ease(clamp(L.db));
    const qa = new THREE.Quaternion().setFromAxisAngle(V.set(a.rot[0], a.rot[1], a.rot[2]).normalize(), a.rot[3] * Math.PI / 180);
    const qb = new THREE.Quaternion().setFromAxisAngle(V.set(b.rot[0], b.rot[1], b.rot[2]).normalize(), b.rot[3] * Math.PI / 180);
    qa.slerp(qb, t);
    const ang = 2 * Math.acos(clamp(qa.w, -1, 1)), sn = Math.sqrt(Math.max(1e-12, 1 - qa.w * qa.w));
    return {
      dp: a.dp.map((v, k) => lerp(v, b.dp[k], t)), rot: [qa.x / sn, qa.y / sn, qa.z / sn, ang * 180 / Math.PI],
      s: lerp(a.s, b.s, t), ax: lerp(a.ax, b.ax, t), pe: a.pe.map((v, k) => lerp(v, b.pe[k], t)), _q: qa,
    };
  }

  tick(dt) {
    this.time += dt; let moving = false;
    this.links.forEach((L) => {
      if (L.db < 1) { L.db = Math.min(1, L.db + dt / 0.9); moving = true; }
      if (L.fitting && L.manual != null && L.manual < 1) { L.manual = Math.min(1, L.manual + dt / 0.85); moving = true; }
      if (L.manual != null && L.manual < 1) moving = true;
    });
    return moving;
  }

  localT(L) {
    if (L.manual != null) return L.manual;
    const n = this.links.length, stag = 0.55;
    return clamp((this.t * (1 + stag) - (stag * L.i) / Math.max(1, n - 1 || 1)), 0, 1);
  }

  update() {
    const D = this.diag;
    this.links.forEach((L) => {
      const d = this._current(L);
      const axis = V2.copy(L.axis);
      if (d.ax) { const perp = Math.abs(axis.z) < 0.9 ? V.set(0, 0, 1) : V.set(1, 0, 0); perp.cross(axis).normalize(); axis.applyAxisAngle(perp, d.ax * Math.PI / 180); }
      const pe = V.set(d.pe[0] * D, d.pe[1] * D, d.pe[2] * D);
      L.pivot.position.copy(L.origin).add(pe);
      L.gizmo.position.copy(L.pivot.position);
      L.gizmo.quaternion.setFromUnitVectors(Z, axis);
      const bad = Math.abs(d.ax) > 9 || pe.length() > 0.035 * D;
      L.gizmo.userData.line.color.setHex(bad ? C_BAD : C_OK); L.gizmo.userData.solid.color.setHex(bad ? C_BAD : C_OK);
      const lt = this.localT(L), s = 1 - ease(lt);
      const q = L.q * (1 - s);
      if (L.joint.type === 'prismatic') { L.pivot.position.addScaledVector(axis, q); L.pivot.quaternion.identity(); }
      else L.pivot.quaternion.setFromAxisAngle(axis, q);
      L.mesh.position.copy(L.origin).add(pe).negate();
      // pose error of the Skill version, about the link origin
      L.err.position.set(d.dp[0] * D, d.dp[1] * D, d.dp[2] * D);
      L.err.quaternion.copy(d._q); L.err.scale.setScalar(d.s);
      // scatter, with a slow tumble while the part floats
      const bob = Math.sin(this.time * 0.7 + L.phase) * 0.035 * D * s;
      L.scat.position.copy(L.sp).multiplyScalar(s);
      L.scat.position.z += bob;
      // explode: push the part away from the object centre
      if (this.explode > 0) {
        V.copy(L.center).sub(V2.set(this.meta.bbox.min[0] + this.size.x / 2, this.meta.bbox.min[1] + this.size.y / 2, this.meta.bbox.min[2] + this.size.z / 2));
        if (V.lengthSq() < 1e-9) V.set(0, 0, 1);
        L.scat.position.addScaledVector(V.normalize(), this.explode * D * 0.42);
      }
      Qt.setFromAxisAngle(L.tumble, this.time * 0.22 + L.phase);
      L.scat.quaternion.copy(Q_ID).slerp(Qt.multiply(L.sq), s);
      const sc = lerp(1, L.ss, s); L.scat.scale.setScalar(sc);
      // the part turns and rescales about its own centre, so a loose part stays where the scatter put it
      V.copy(L.center).sub(L.origin); L.scat.position.add(V).sub(V.applyQuaternion(L.scat.quaternion).multiplyScalar(sc));
    });
  }

  setQ01(L, u) { L.q = lerp(L.lo, L.hi, u); }
  dispose() { this.group.removeFromParent(); this.meshes.concat(this.rootMesh).forEach((m) => { m.material.dispose(); m.userData.line.material.dispose(); }); }
}

export class Stage {
  constructor(canvas, { orbit = true, fov = 30, shadow = 0.34, glow = 0.15, zoom = false } = {}) {
    this.canvas = canvas;
    this.models = []; this.onFrame = null; this.visible = false; this.dirty = true; this.always = false;
    this.camera = new THREE.PerspectiveCamera(fov, 1, 0.01, 200);
    if (!HAS_GL) {
      this.dead = true; canvas.hidden = true;
      const note = document.createElement('p'); note.className = 'gl-fallback';
      note.innerHTML = '<b>3D view unavailable</b>This view needs WebGL 2. Turn on hardware acceleration or open the page in a current browser.';
      canvas.parentElement.append(note);
      return;
    }
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' }));
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.scene.add(new THREE.HemisphereLight(0xf3f6ff, 0x2e3244, 1.2));
    const key = (this.key = new THREE.DirectionalLight(0xfff6ea, 1.95));
    key.castShadow = true; key.shadow.mapSize.set(2048, 2048); key.shadow.bias = -0.0003; key.shadow.normalBias = 0.02; key.shadow.radius = 5;
    this.scene.add(key); this.scene.add(key.target);
    const fill = new THREE.DirectionalLight(0xa9c0ff, 0.55); fill.position.set(-3, 2, -2); this.scene.add(fill);
    // cool rim light from behind, which separates the silhouettes from the dark page
    const rim = (this.rim = new THREE.DirectionalLight(0xb7c6ff, 1.25)); this.scene.add(rim); this.scene.add(rim.target);
    // soft pool of light on the floor, drawn under the shadow
    const gc = document.createElement('canvas'); gc.width = gc.height = 256;
    const g2 = gc.getContext('2d'), gr = g2.createRadialGradient(128, 128, 0, 128, 128, 128);
    gr.addColorStop(0, 'rgba(176,194,255,1)'); gr.addColorStop(0.3, 'rgba(160,180,255,.55)'); gr.addColorStop(0.62, 'rgba(150,170,255,.16)'); gr.addColorStop(1, 'rgba(143,176,255,0)');
    g2.fillStyle = gr; g2.fillRect(0, 0, 256, 256);
    const gt = new THREE.CanvasTexture(gc); gt.colorSpace = THREE.SRGBColorSpace;
    this.glow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: gt, transparent: true, opacity: glow, depthWrite: false, toneMapped: false }));
    this.glow.rotation.x = -Math.PI / 2; this.glow.position.y = -0.0005; this.glow.renderOrder = -2; this.glow.visible = false; this.glowOn = glow > 0; this.scene.add(this.glow);   // shown once a model is framed
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.ShadowMaterial({ opacity: shadow }));
    this.ground.rotation.x = -Math.PI / 2; this.ground.receiveShadow = true; this.ground.renderOrder = -1; this.scene.add(this.ground);
    if (orbit) {
      const c = (this.controls = new OrbitControls(this.camera, canvas));
      c.enableDamping = true; c.dampingFactor = 0.09; c.enablePan = false; c.enableZoom = zoom;
      c.maxPolarAngle = Math.PI * 0.52; c.rotateSpeed = 0.7;
    }
    this.ray = new THREE.Raycaster(); this.ptr = new THREE.Vector2();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; }, { rootMargin: '120px' }).observe(canvas);
    this.resize();
    this.clock = new THREE.Clock();
    const loop = () => {
      requestAnimationFrame(loop);
      const dt = Math.min(0.05, this.clock.getDelta());
      if (!this.visible) return;
      let moving = this.always;
      if (this.onFrame && this.onFrame(dt, this.clock.elapsedTime)) moving = true;
      this.models.forEach((m) => { if (m.tick(dt) || m.t < 1) moving = true; if (moving || this.dirty) m.update(); });
      if (this.controls && this.controls.update()) moving = true;
      if (moving || this.dirty) { this.renderer.render(this.scene, this.camera); this.dirty = false; }
    };
    loop();
  }
  resize() {
    if (this.dead) return;
    const p = this.canvas.parentElement, w = p.clientWidth, h = p.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this.dirty = true;
  }
  add(model) { this.models.push(model); if (this.dead) return model; this.scene.add(model.group); this.dirty = true; return model; }
  remove(model) { this.models = this.models.filter((m) => m !== model); model.dispose(); this.dirty = true; }
  // place the camera so that a sphere (centre c, radius r) fills the view
  frame(c, r, { azim = 0.7, elev = 0.42, pad = 1.25 } = {}) {
    if (this.dead) return;
    const fov = (this.camera.fov * Math.PI) / 180, asp = this.camera.aspect;
    const half = Math.min(fov / 2, Math.atan(Math.tan(fov / 2) * asp));
    const d = (r / Math.sin(half)) * pad;
    this.camera.position.set(c.x + d * Math.cos(elev) * Math.sin(azim), c.y + d * Math.sin(elev), c.z + d * Math.cos(elev) * Math.cos(azim));
    this.camera.near = d / 60; this.camera.far = d * 30; this.camera.updateProjectionMatrix();
    this.camera.lookAt(c);
    if (this.controls) { this.controls.target.copy(c); this.controls.minDistance = d * 0.55; this.controls.maxDistance = d * 1.8; this.controls.update(); }
    this.key.position.set(c.x + r * 1.6, c.y + r * 4.2, c.z + r * 2.4); this.key.target.position.copy(c);
    this.rim.position.set(c.x - r * 2.4, c.y + r * 2.2, c.z - r * 3.2); this.rim.target.position.copy(c);
    this.glow.position.x = c.x; this.glow.position.z = c.z; this.glow.scale.setScalar(r * 4.4); this.glow.visible = this.glowOn;
    const sc = this.key.shadow.camera; sc.left = -r * 2.6; sc.right = r * 2.6; sc.top = r * 2.6; sc.bottom = -r * 2.6; sc.near = 0.01; sc.far = r * 12; sc.updateProjectionMatrix();
    this.dirty = true;
  }
  frameModel(m, opt) { this.frame(new THREE.Vector3(m.group.position.x, m.size.z * 0.5, m.group.position.z), m.diag * 0.56, opt); }
  pick(ev) {
    const b = this.canvas.getBoundingClientRect();
    this.ptr.set(((ev.clientX - b.left) / b.width) * 2 - 1, -((ev.clientY - b.top) / b.height) * 2 + 1);
    this.ray.setFromCamera(this.ptr, this.camera);
    const hits = this.ray.intersectObjects(this.models.flatMap((m) => m.meshes.concat(m.rootMesh)), false);
    return hits.length ? hits[0].object : null;
  }
  // drag a part to drive its joint
  enableDrag(onChange) {
    if (this.dead) return;
    const cv = this.canvas; let drag = null;
    const model = (mesh) => this.models.find((m) => m.meshes.includes(mesh));
    cv.addEventListener('pointermove', (ev) => {
      if (drag) {
        const L = drag.L, range = L.hi - L.lo, k = (ev.clientX - drag.x - (ev.clientY - drag.y)) / 260;
        L.q = clamp(drag.q + k * Math.min(range, Math.PI * 2), L.lo, L.hi); this.dirty = true; onChange && onChange(L); return;
      }
      const hit = this.pick(ev); const L = hit && hit.userData.link;
      this.models.forEach((m) => m.highlight(null)); if (L) model(hit).highlight(L, true);
      cv.style.cursor = L ? 'grab' : ''; this.dirty = true;
    });
    cv.addEventListener('pointerdown', (ev) => {
      const hit = this.pick(ev); const L = hit && hit.userData.link; if (!L) return;
      drag = { L, x: ev.clientX, y: ev.clientY, q: L.q }; if (this.controls) this.controls.enabled = false;
      cv.setPointerCapture(ev.pointerId); cv.style.cursor = 'grabbing'; onChange && onChange(L, true);
    });
    const end = () => { if (drag) { drag = null; if (this.controls) this.controls.enabled = true; cv.style.cursor = ''; } };
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', () => { if (!drag) { this.models.forEach((m) => m.highlight(null)); this.dirty = true; } });
  }
}

// Illustrative evaluator: the checks of the paper, applied to the pose error shown on screen.
export function evaluate(model) {
  const rows = model.links.map((L) => {
    const d = L.defect, D = 1;
    const dp = Math.hypot(...d.dp), pe = Math.hypot(...d.pe);
    const scale = Math.abs(d.s - 1) <= 0.05, rest = dp <= 0.04 && Math.abs(d.rot[3]) <= 8 && scale;
    const axis = Math.abs(d.ax) <= 10, pivot = L.joint.type === 'prismatic' ? true : pe <= 0.04;
    const motion = rest && axis && pivot;
    const checks = { scale, rest, axis, pivot, motion };
    const score = (scale + rest + axis + pivot + motion * 2) / 6;
    return { L, checks, score, drift: (dp + pe * 0.8 + Math.abs(d.ax) / 90 + Math.abs(d.s - 1) * 0.5 + Math.abs(d.rot[3]) / 400) / 0.04 * D };
  });
  const fitness = rows.length ? rows.reduce((a, r) => a + r.score, 0) / rows.length : 1;
  return { rows, fitness, pass: rows.every((r) => r.checks.motion) };
}
