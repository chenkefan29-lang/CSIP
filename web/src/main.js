// 入口：渲染器 / WebXR / 交互 / 数据绑定
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { Link } from './link.js';
import { Panel } from './panel.js';
import { Plant3D } from './plant3d.js';
import { Hud } from './hud.js';

const MQTT_URL = import.meta.env.VITE_MQTT_URL || `ws://${location.hostname || '127.0.0.1'}:9001`;

// ---------------- 渲染器 ----------------
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.xr.enabled = true;
// VR 性能：固定注视点渲染 + 适度降低帧缓冲分辨率，保证 72/90 fps
renderer.xr.setFoveation(1.0);
renderer.xr.setFramebufferScaleFactor(0.9);
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9cc3e0);
scene.fog = new THREE.Fog(0x9cc3e0, 60, 150);
scene.add(new THREE.HemisphereLight(0xdfefff, 0x3f4a3a, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(20, 40, 25);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -45, right: 45, top: 30, bottom: -30, near: 1, far: 120 });
scene.add(sun);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.05, 300);
camera.position.set(0, 26, 38);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(5, 2, 0);
orbit.maxPolarAngle = Math.PI * 0.48;
orbit.update();

// VR 用户的"身体"：传送时移动 rig，而不是相机
const rig = new THREE.Group();
rig.position.set(-16, 0, 7.5);
scene.add(rig);
renderer.xr.addEventListener('sessionstart', () => { rig.add(camera); camera.position.set(0, 0, 0); sun.shadow.mapSize.set(1024, 1024); });
renderer.xr.addEventListener('sessionend', () => { scene.add(camera); camera.position.set(0, 26, 38); orbit.update(); });

const plant = new Plant3D(scene);
const hud = new Hud(document.getElementById('hud'));

// ---------------- 数据链路 ----------------
const link = new Link(MQTT_URL);
let latest = null;
link.on('status', (s) => hud.setLink(s));
link.on('state', (s) => { latest = s; plant.update(s); hud.update(s, link.latencyStats()); refreshPanels(); });
link.on('ack', (a) => hud.toast(a.ok ? `命令已执行（往返 ${a.rtt.toFixed(1)} ms）` : `命令失败：${a.error}`, a.ok));
hud.onCommand = (cmd) => link.send(cmd);

// ---------------- VR 面板 ----------------
const infoPanel = new Panel(1.1, 0.8);
const menuPanel = new Panel(1.1, 1.0);
infoPanel.mesh.visible = false;
scene.add(infoPanel.mesh, menuPanel.mesh);
let selected = null;
let scenarioList = [];
fetch('/api/scenarios').then((r) => r.json()).then((l) => { scenarioList = l; refreshPanels(); }).catch(() => {});

const KLA_KEY = { 2: 'kla3', 3: 'kla4', 4: 'kla5' };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const set = (k, v) => ({ type: 'set', values: { [k]: v } });
// 调节以操作员设定值为基准：异常工况（鼓风机/回流泵故障）会压低实测值，直接在实测值上加减会把设定值拉低
const manual = (s, k, fallback) => s.manual?.[k] ?? fallback;

// 选中设备 → 信息面板内容（读数 + 可执行的调节按钮）
function infoContent(sel, s) {
  const d = sel.userData;
  const lines = [], buttons = [];
  if (d.kind === 'tank' || d.kind === 'blower') {
    const i = d.index;
    lines.push([`DO ${s.DO[i].toFixed(2)} mg/L    MLSS ${s.MLSS[i].toFixed(0)} mg/L`]);
    lines.push([`NH₄-N ${s.SNH[i].toFixed(2)}    NO₃-N ${s.SNO[i].toFixed(2)} mg/L`]);
    lines.push([`kLa ${s.kla[i].toFixed(0)} d⁻¹`, s.kla[i] < 10 && i >= 2 ? '#ff7a6b' : null]);
    if (i >= 2) {
      const k = KLA_KEY[i], base = manual(s, k, s.kla[i]);
      buttons.push({ label: 'kLa −20', cmd: set(k, clamp(base - 20, 0, 360)) });
      buttons.push({ label: 'kLa +20', cmd: set(k, clamp(base + 20, 0, 360)) });
      if (i === 4) buttons.push({ label: '恢复 DO 自动控制', cmd: set('kla5', null), color: '#2f6f4a' });
    } else {
      lines.push([`内回流 Qa ${s.Qa.toFixed(0)} m³/d`]);
      const qa = manual(s, 'Qa', s.Qa);
      buttons.push({ label: 'Qa −5000', cmd: set('Qa', clamp(qa - 5000, 0, 92230)) });
      buttons.push({ label: 'Qa +5000', cmd: set('Qa', clamp(qa + 5000, 0, 92230)) });
      buttons.push({ label: '恢复硝态氮自动控制', cmd: set('Qa', null), color: '#2f6f4a' });
    }
  } else if (d.kind === 'clarifier') {
    const e = s.eff;
    lines.push([`出水 TSS ${e.TSS.toFixed(1)}  COD ${e.COD.toFixed(1)}`, e.TSS > 30 ? '#ff7a6b' : null]);
    lines.push([`出水 TN ${e.TN.toFixed(1)}  NH₄ ${e.SNH.toFixed(2)}`, e.TN > 18 || e.SNH > 4 ? '#ff7a6b' : null]);
    lines.push([`外回流 Qr ${s.Qr.toFixed(0)}  排泥 Qw ${s.Qw.toFixed(0)} m³/d`]);
    const qr = manual(s, 'Qr', s.Qr), qw = manual(s, 'Qw', s.Qw);
    buttons.push({ label: 'Qr −2000', cmd: set('Qr', clamp(qr - 2000, 0, 36892)) });
    buttons.push({ label: 'Qr +2000', cmd: set('Qr', clamp(qr + 2000, 0, 36892)) });
    buttons.push({ label: 'Qw −50', cmd: set('Qw', clamp(qw - 50, 0, 1844)) });
    buttons.push({ label: 'Qw +50', cmd: set('Qw', clamp(qw + 50, 0, 1844)) });
  } else if (d.kind === 'pipe') {
    const q = { Qin: s.Qin, Qa: s.Qa, Qf: s.Qin + s.Qa + s.Qr, Qr: s.Qr, Qe: s.Qin - s.Qw, Qw: s.Qw }[d.key];
    lines.push([`流量 ${q.toFixed(0)} m³/d`]);
    if (d.key === 'Qin') lines.push([`进水 COD ${s.inf.COD.toFixed(0)}  NH₄ ${s.inf.SNH.toFixed(1)}  T ${s.Tin.toFixed(1)}℃`]);
  }
  return { title: d.name, lines, buttons };
}

function menuContent(s) {
  const lines = [];
  const alarms = s?.alarms || [];
  lines.push([`帧率 ${fps.toFixed(0)} fps    链路 ${link.client.connected ? '在线' : '离线'}`, fps < 72 ? '#ffb066' : '#8fe3a6']);
  const lat = link.latencyStats();
  if (lat) lines.push([`操作响应 均值 ${lat.mean.toFixed(1)} / P95 ${lat.p95.toFixed(1)} ms`, lat.p95 > 100 ? '#ff7a6b' : '#8fe3a6']);
  for (const a of (s?.scenarios || [])) lines.push([`工况：${a.name} ${(a.progress * 100).toFixed(0)}%`, '#ffb066']);
  for (const a of alarms.slice(0, 3)) lines.push([`⚠ ${a.message}`, a.level === 'critical' ? '#ff7a6b' : '#ffd166']);
  const buttons = [];
  for (const a of alarms.slice(0, 2)) buttons.push({ label: `执行预案 ${a.code}`, cmd: { type: 'plan', code: a.code }, color: '#8a3b2b' });
  for (const sc of scenarioList.slice(0, 6)) buttons.push({ label: sc.name, cmd: { type: 'scenario', key: sc.key } });
  buttons.push({ label: '清除工况/复位', cmd: { type: 'reset' }, color: '#2f6f4a' });
  return { title: '操作菜单', lines, buttons };
}

// 面板重绘限频：状态 10 Hz 推送，面板每 250 ms 最多重绘一次（Canvas 纹理上传较贵）
let lastPanelDraw = 0;
function refreshPanels(force = false) {
  const now = performance.now();
  if (!force && now - lastPanelDraw < 250) return;
  lastPanelDraw = now;
  hud.setScenarios(scenarioList);
  if (!latest) return;
  if (selected) infoPanel.draw(infoContent(selected, latest));
  if (renderer.xr.isPresenting) menuPanel.draw(menuContent(latest));
}

function select(obj) {
  selected = obj;
  if (!obj) { infoPanel.mesh.visible = false; hud.setSelected(null); return; }
  // 面板放在设备上方，水平朝向观察者
  const p = new THREE.Vector3();
  obj.getWorldPosition(p);
  const viewer = new THREE.Vector3();
  camera.getWorldPosition(viewer);
  const dir = viewer.clone().sub(p).setY(0).normalize();
  infoPanel.mesh.scale.setScalar(renderer.xr.isPresenting ? 2.2 : 6);
  infoPanel.mesh.position.copy(p).addScaledVector(dir, 2.5).setY(Math.max(p.y, 2) + 3.2);
  infoPanel.mesh.lookAt(viewer.x, infoPanel.mesh.position.y, viewer.z);
  infoPanel.mesh.visible = true;
  hud.setSelected(obj.userData.name);
  refreshPanels(true);
}

function sendCmd(cmd) {
  link.send(cmd);
  refreshPanels(true);
}

// ---------------- 射线拾取（鼠标与 VR 手柄共用） ----------------
const raycaster = new THREE.Raycaster();
raycaster.far = 120;

// 返回命中结果：面板按钮 > 设备 > 地面
function pick() {
  const panels = [infoPanel.mesh, menuPanel.mesh].filter((m) => m.visible);
  const hp = raycaster.intersectObjects(panels, false)[0];
  if (hp) return { kind: 'panel', point: hp.point, panel: hp.object.userData.panel, index: hp.object.userData.panel.hit(hp.uv) };
  const he = raycaster.intersectObjects(plant.interactive, false)[0];
  const hf = raycaster.intersectObjects(plant.floor, false)[0];
  if (he && (!hf || he.distance <= hf.distance + 0.01)) return { kind: 'equip', point: he.point, object: he.object };
  if (hf) return { kind: 'floor', point: hf.point };
  return null;
}

function activate(hit, allowTeleport) {
  if (!hit) return;
  if (hit.kind === 'panel') {
    const b = hit.panel.buttons[hit.index];
    if (b) sendCmd(b.cmd);
  } else if (hit.kind === 'equip') {
    select(hit.object);
  } else if (hit.kind === 'floor') {
    if (allowTeleport) rig.position.set(hit.point.x, 0, hit.point.z);
    else select(null);
  }
}

// 桌面：单击拾取（拖拽旋转视角不触发）
const pointer = new THREE.Vector2();
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  activate(pick(), false);
});
renderer.domElement.addEventListener('pointermove', (e) => {
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const h = pick();
  infoPanel.setHover(h?.kind === 'panel' && h.panel === infoPanel ? h.index : -1);
  renderer.domElement.style.cursor = h && h.kind !== 'floor' ? 'pointer' : 'default';
});

// VR：右手柄扳机 = 选择/按按钮/传送；左手腕挂操作菜单
const controllerFactory = new XRControllerModelFactory();
const tmpMat = new THREE.Matrix4();
const controllers = [0, 1].map((i) => {
  const c = renderer.xr.getController(i);
  const ray = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, -1)]),
    new THREE.LineBasicMaterial({ color: 0x7fd4ff }));
  ray.scale.z = 10;
  c.add(ray);
  c.userData.ray = ray;
  const grip = renderer.xr.getControllerGrip(i);
  grip.add(controllerFactory.createControllerModel(grip));
  rig.add(c, grip);
  c.addEventListener('connected', (e) => { c.userData.hand = e.data.handedness; if (e.data.handedness === 'left') attachMenu(grip); });
  c.addEventListener('selectstart', () => {
    const h = rayFrom(c);
    // 发送时刻记录在 Link 内部，从扳机按下到回执的时间即操作响应时延
    activate(h, true);
  });
  return c;
});

function attachMenu(grip) {
  grip.add(menuPanel.mesh);
  menuPanel.mesh.scale.setScalar(0.28);
  menuPanel.mesh.position.set(0.05, 0.12, -0.08);
  menuPanel.mesh.rotation.set(-Math.PI / 4, 0, 0);
}

function rayFrom(c) {
  tmpMat.identity().extractRotation(c.matrixWorld);
  raycaster.ray.origin.setFromMatrixPosition(c.matrixWorld);
  raycaster.ray.direction.set(0, 0, -1).applyMatrix4(tmpMat);
  return pick();
}

renderer.xr.addEventListener('sessionstart', () => { menuPanel.mesh.visible = true; select(null); refreshPanels(true); });
renderer.xr.addEventListener('sessionend', () => { menuPanel.mesh.visible = false; select(null); });
menuPanel.mesh.visible = false;

// ---------------- 主循环 ----------------
const clock = new THREE.Clock();
let fps = 0, frames = 0, fpsT0 = performance.now();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.1);
  plant.animate(dt, clock.elapsedTime);
  if (renderer.xr.isPresenting) {
    for (const c of controllers) {
      const h = rayFrom(c);
      c.userData.ray.scale.z = h?.point ? Math.max(0.1, raycaster.ray.origin.distanceTo(h.point)) : 10;
      for (const p of [infoPanel, menuPanel]) p.setHover(h?.kind === 'panel' && h.panel === p ? h.index : -1);
    }
  } else {
    orbit.update();
  }
  renderer.render(scene, camera);
  frames++;
  const now = performance.now();
  if (now - fpsT0 >= 1000) {
    fps = (frames * 1000) / (now - fpsT0);
    frames = 0; fpsT0 = now;
    hud.setFps(fps, renderer.info.render);
  }
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// 供验收脚本 / 控制台读取
window.csip = { link, plant, getFps: () => fps, send: sendCmd, select };
