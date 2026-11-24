// Interactive articulation viewer for the AgentAssemble project page.
// Stack mirrors the reference site: three.js 0.165 (via importmap) + GLTFLoader
// + OrbitControls. Each asset is a GLB whose nodes are named link_<i>, paired
// with a JSON describing the URDF joints (origin/axis/type/limits) in the model's
// native Z-up world frame. We rebuild the kinematic tree as THREE pivots so the
// joints can be driven live by sliders.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const MODELS_BASE = 'assets/models/';
const ROOT_COLOR = 0xcdc7bc;
const ACCENT = [0x5b8cff, 0x36d399, 0xffb454, 0x7c5cff, 0xff6b6b, 0x22d3ee];

const stage   = document.getElementById('gl-stage');
const canvas  = document.getElementById('gl');
const chipsEl = document.getElementById('model-chips');
const slidersEl = document.getElementById('joint-sliders');
const loadingEl = document.getElementById('gl-loading');
const titleEl = document.getElementById('gl-title');
const subEl   = document.getElementById('gl-sub');

// ---------- renderer / scene ----------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.04;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
camera.position.set(2.2, 1.7, 2.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.4;
controls.maxDistance = 12;
controls.maxPolarAngle = Math.PI * 0.92;

// lighting — soft studio
scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2f3a, 0.85));
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(3, 6, 4);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.5; key.shadow.camera.far = 30;
key.shadow.camera.left = -3; key.shadow.camera.right = 3;
key.shadow.camera.top = 3; key.shadow.camera.bottom = -3;
key.shadow.bias = -0.0004; key.shadow.normalBias = 0.02;
scene.add(key);
const fill = new THREE.DirectionalLight(0x9fb6ff, 0.5);
fill.position.set(-4, 2, -3);
scene.add(fill);

// ground (shadow catcher)
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.ShadowMaterial({ opacity: 0.28 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(20, 40, 0xb3bac8, 0xd3d8e0);
grid.material.opacity = 0.5; grid.material.transparent = true;
scene.add(grid);

const loader = new GLTFLoader();

let current = null;        // { root: Group, joints: [...], meta }
let autoRotate = false;
let playing = false;
let playT = 0;

// ---------- build a model from GLB + meta ----------
function disposeCurrent() {
  if (!current) return;
  scene.remove(current.root);
  current.root.traverse(o => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m.dispose());
  });
  current = null;
}

function buildModel(gltf, meta) {
  // collect link meshes by node name
  const meshes = {};
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse(o => {
    if (o.isMesh && /^link_\d+$/.test(o.name)) {
      o.geometry.applyMatrix4(o.matrixWorld);   // bake any node transform → world (Z-up) coords
      meshes[o.name] = o.geometry;
    }
  });

  // native Z-up assembly group; we colour & pivot here, then rotate the wrapper to Y-up
  const zup = new THREE.Group();
  let accentI = 0;
  const linkMesh = {};
  meta.links.forEach(link => {
    const geo = meshes[link];
    if (!geo) return;
    const isRoot = link === meta.root;
    const mat = new THREE.MeshStandardMaterial({
      color: isRoot ? ROOT_COLOR : ACCENT[accentI++ % ACCENT.length],
      roughness: 0.62, metalness: 0.04,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    linkMesh[link] = mesh;
  });

  // root mesh straight into the group
  if (linkMesh[meta.root]) zup.add(linkMesh[meta.root]);

  // one pivot per joint (all trees here are depth-1: parent === root)
  const joints = [];
  meta.joints.forEach(j => {
    const child = linkMesh[j.child];
    if (!child) return;
    const o = new THREE.Vector3().fromArray(j.origin);
    const pivot = new THREE.Group();
    pivot.position.copy(o);
    child.position.sub(o);            // keep child at its world-rest spot when pivot is unrotated
    pivot.add(child);
    zup.add(pivot);
    joints.push({
      spec: j,
      pivot,
      axis: new THREE.Vector3().fromArray(j.axis).normalize(),
      basePos: o.clone(),
      q: 0,
    });
  });

  // Z-up → Y-up wrapper
  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2;
  root.add(zup);
  scene.add(root);

  // frame: sit on ground, centre horizontally
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3(); box.getSize(size);
  const center = new THREE.Vector3(); box.getCenter(center);
  root.position.x -= center.x;
  root.position.z -= center.z;
  root.position.y -= box.min.y;       // bottom on the floor

  current = { root, joints, meta };
  applyAllJoints();

  // camera framing
  const radius = Math.max(size.x, size.y, size.z);
  const h = size.y;
  controls.target.set(0, h * 0.5, 0);
  const d = radius * 1.9 + 0.3;
  camera.position.set(d * 0.72, h * 0.55 + radius * 0.45, d * 0.86);
  controls.update();
}

// ---------- articulation ----------
function applyJoint(jt) {
  const { spec, pivot, axis, basePos, q } = jt;
  if (spec.type === 'prismatic') {
    pivot.position.copy(basePos).addScaledVector(axis, q);
    pivot.quaternion.identity();
  } else { // revolute / continuous
    pivot.position.copy(basePos);
    pivot.quaternion.setFromAxisAngle(axis, q);
  }
}
function applyAllJoints() { current && current.joints.forEach(applyJoint); }

function setJoint(i, value01) {
  const jt = current.joints[i];
  const { lower, upper } = jt.spec;
  jt.q = lower + (upper - lower) * value01;
  applyJoint(jt);
}

// ---------- UI ----------
function buildSliders(meta) {
  slidersEl.innerHTML = '';
  meta.joints.forEach((j, i) => {
    const rest01 = (upper => upper === j.lower ? 0 : (0 - j.lower) / (j.upper - j.lower))(j.upper);
    const isPrism = j.type === 'prismatic';
    const row = document.createElement('div');
    row.className = 'jrow';
    row.innerHTML = `
      <div class="jhead">
        <span class="jdot" style="background:#${ACCENT[i % ACCENT.length].toString(16).padStart(6,'0')}"></span>
        <span class="jname">${j.name}</span>
        <span class="jtype ${isPrism ? 'prism' : 'rev'}">${j.type}</span>
        <span class="jval" id="jval-${i}"></span>
      </div>
      <input type="range" min="0" max="1" step="0.001" value="${rest01}" id="jin-${i}" />`;
    slidersEl.appendChild(row);
    const input = row.querySelector('input');
    const valEl = row.querySelector('.jval');
    const update = () => {
      const v = parseFloat(input.value);
      setJoint(i, v);
      const q = j.lower + (j.upper - j.lower) * v;
      valEl.textContent = isPrism ? `${(q * 100).toFixed(0)} cm` : `${(q * 180 / Math.PI).toFixed(0)}°`;
    };
    input.addEventListener('input', () => { playing = false; syncPlayBtn(); update(); });
    update();
  });
}

let MODELS = [];
function buildChips(list) {
  chipsEl.innerHTML = '';
  list.forEach((m, i) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = m.label;
    const tag = document.createElement('span');
    tag.className = 'chip-dof';
    tag.textContent = `${m.joints}·${m.types.map(t => t[0].toUpperCase()).join('')}`;
    b.appendChild(tag);
    b.addEventListener('click', () => selectModel(i));
    chipsEl.appendChild(b);
  });
}

function highlightChip(i) {
  [...chipsEl.children].forEach((c, k) => c.classList.toggle('active', k === i));
}

async function selectModel(i) {
  const m = MODELS[i];
  highlightChip(i);
  loadingEl.style.display = 'flex';
  playing = false; syncPlayBtn();
  const meta = await fetch(`${MODELS_BASE}${m.slug}.json`).then(r => r.json());
  loader.load(`${MODELS_BASE}${m.slug}.glb`, (gltf) => {
    disposeCurrent();
    buildModel(gltf, meta);
    buildSliders(meta);
    titleEl.textContent = meta.label;
    const nRev = meta.joints.filter(j => j.type !== 'prismatic').length;
    const nPri = meta.joints.filter(j => j.type === 'prismatic').length;
    const parts = [];
    if (nRev) parts.push(`${nRev} revolute`);
    if (nPri) parts.push(`${nPri} prismatic`);
    subEl.textContent = `${meta.links.length} links · ${parts.join(' + ')} joint${meta.joints.length > 1 ? 's' : ''} · drag to orbit, scroll to zoom`;
    loadingEl.style.display = 'none';
  }, undefined, (err) => {
    loadingEl.textContent = 'Failed to load model';
    console.error('Load failed', m.slug, err);
  });
}

// ---------- buttons ----------
const btnReset = document.getElementById('btn-reset');
const btnPlay  = document.getElementById('btn-play');
const btnRot   = document.getElementById('btn-rotate');
function syncPlayBtn() { btnPlay && btnPlay.classList.toggle('active', playing); }

btnReset && btnReset.addEventListener('click', () => {
  playing = false; syncPlayBtn();
  slidersEl.querySelectorAll('input').forEach((inp, i) => {
    const j = current.meta.joints[i];
    const rest01 = j.upper === j.lower ? 0 : (0 - j.lower) / (j.upper - j.lower);
    inp.value = rest01; inp.dispatchEvent(new Event('input'));
  });
});
btnPlay && btnPlay.addEventListener('click', () => { playing = !playing; playT = 0; syncPlayBtn(); });
btnRot && btnRot.addEventListener('click', () => { autoRotate = !autoRotate; btnRot.classList.toggle('active', autoRotate); });

// ---------- resize / loop ----------
function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
resize();

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (autoRotate && current) current.root.rotation.y += dt * 0.45;
  if (playing && current) {
    playT += dt * 0.5;
    const s = 0.5 - 0.5 * Math.cos(playT * Math.PI);   // 0..1..0 ease
    slidersEl.querySelectorAll('input').forEach((inp, i) => {
      inp.value = s; setJoint(i, s);
      const j = current.meta.joints[i];
      const q = j.lower + (j.upper - j.lower) * s;
      const ve = document.getElementById('jval-' + i);
      if (ve) ve.textContent = j.type === 'prismatic' ? `${(q * 100).toFixed(0)} cm` : `${(q * 180 / Math.PI).toFixed(0)}°`;
    });
  }
  controls.update();
  renderer.render(scene, camera);
}

// ---------- boot ----------
fetch(`${MODELS_BASE}index.json`).then(r => r.json()).then(list => {
  MODELS = list;
  buildChips(list);
  selectModel(0);
  animate();
}).catch(e => { loadingEl.textContent = 'Could not load model index'; console.error(e); });
