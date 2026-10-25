// 污水处理厂三维数字孪生模型（程序化建模，几何与 BSM1 工艺一一对应）
// 进水 → 缺氧池 1、2 → 好氧池 3、4、5 → 二沉池（10 层）→ 出水，含内回流、外回流、排泥。
// 状态映射：水色 ← 氨氮/硝态氮，气泡密度 ← kLa，泥层高度 ← 二沉池固体分布，管道流速 ← 流量。
import * as THREE from 'three';

const TANK_W = 6, TANK_D = 8, TANK_H = 4.5, GAP = 0.8;
const TANK_X = [-16, -16 + (TANK_W + GAP), -16 + 2 * (TANK_W + GAP), -16 + 3 * (TANK_W + GAP), -16 + 4 * (TANK_W + GAP)];
const CLAR = { x: 26, z: 0, r: 6.5, h: 4.5 };

const mat = {
  concrete: new THREE.MeshStandardMaterial({ color: 0x9aa1a8, roughness: 0.9 }),
  ground: new THREE.MeshStandardMaterial({ color: 0x55604f, roughness: 1 }),
  rail: new THREE.MeshStandardMaterial({ color: 0xf2c230, roughness: 0.5, metalness: 0.4 }),
  pipe: new THREE.MeshStandardMaterial({ color: 0x3a6ea5, roughness: 0.4, metalness: 0.6 }),
  pipeRAS: new THREE.MeshStandardMaterial({ color: 0x8a5a2b, roughness: 0.4, metalness: 0.6 }),
  blower: new THREE.MeshStandardMaterial({ color: 0x2f8f6a, roughness: 0.5, metalness: 0.5 }),
};

function waterColor(snh, sno, anoxic) {
  // 氨氮高 → 偏黄褐；硝态氮高 → 偏青；缺氧池整体更暗
  const c = new THREE.Color(0x2a6b5c);
  c.lerp(new THREE.Color(0x8a6a2a), Math.min(1, snh / 20));
  c.lerp(new THREE.Color(0x2f8fa8), Math.min(0.5, sno / 30));
  if (anoxic) c.multiplyScalar(0.75);
  return c;
}

function label(text, scale = 1) {
  const cv = document.createElement('canvas');
  cv.width = 512; cv.height = 128;
  const g = cv.getContext('2d');
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false }));
  sprite.scale.set(4 * scale, 1 * scale, 1);
  sprite.renderOrder = 10;
  sprite.userData.draw = (t, sub, color = '#ffffff') => {
    g.clearRect(0, 0, 512, 128);
    g.fillStyle = 'rgba(12,20,28,0.78)';
    g.beginPath(); g.roundRect(4, 4, 504, 120, 18); g.fill();
    g.fillStyle = color; g.font = 'bold 44px sans-serif'; g.textAlign = 'center';
    g.fillText(t, 256, sub ? 54 : 78);
    if (sub) { g.font = '34px sans-serif'; g.fillStyle = '#cfe3f0'; g.fillText(sub, 256, 104); }
    sprite.material.map.needsUpdate = true;
  };
  sprite.userData.draw(text);
  return sprite;
}

function pipe(points, material, radius = 0.25) {
  const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)), false, 'catmullrom', 0.01);
  const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 64, radius, 10, false), material);
  m.userData.curve = curve;
  return m;
}

export class Plant3D {
  constructor(scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    scene.add(this.root);
    this.interactive = []; // 可被 VR 射线 / 鼠标选中的设备
    this.tanks = [];
    this.flows = [];
    this.buildSite();
    this.buildTanks();
    this.buildClarifier();
    this.buildPipes();
    this.buildBlowers();
  }

  buildSite() {
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(140, 90), mat.ground);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.root.add(ground);
    // 巡检走道（VR 用户在此行走）
    const walk = new THREE.Mesh(new THREE.BoxGeometry(60, 0.1, 2.2), new THREE.MeshStandardMaterial({ color: 0x6d6d6d }));
    walk.position.set(5, 0.05, TANK_D / 2 + 2.2);
    walk.receiveShadow = true;
    this.root.add(walk);
    this.floor = [ground, walk];
  }

  buildTanks() {
    const wallT = 0.35;
    for (let i = 0; i < 5; i++) {
      const g = new THREE.Group();
      g.position.set(TANK_X[i], 0, 0);
      const outer = new THREE.Shape();
      outer.moveTo(-TANK_W / 2, -TANK_D / 2); outer.lineTo(TANK_W / 2, -TANK_D / 2);
      outer.lineTo(TANK_W / 2, TANK_D / 2); outer.lineTo(-TANK_W / 2, TANK_D / 2); outer.closePath();
      const hole = new THREE.Path();
      const w = TANK_W / 2 - wallT, d = TANK_D / 2 - wallT;
      hole.moveTo(-w, -d); hole.lineTo(-w, d); hole.lineTo(w, d); hole.lineTo(w, -d); hole.closePath();
      outer.holes.push(hole);
      const wall = new THREE.Mesh(new THREE.ExtrudeGeometry(outer, { depth: TANK_H, bevelEnabled: false }), mat.concrete);
      wall.rotation.x = -Math.PI / 2;
      wall.castShadow = wall.receiveShadow = true;
      g.add(wall);

      const waterMat = new THREE.MeshStandardMaterial({ color: 0x2a6b5c, roughness: 0.15, metalness: 0.1, transparent: true, opacity: 0.92 });
      const water = new THREE.Mesh(new THREE.PlaneGeometry(2 * w, 2 * d, 24, 32), waterMat);
      water.rotation.x = -Math.PI / 2;
      water.position.y = TANK_H - 0.4;
      g.add(water);

      // 栏杆
      const rail = new THREE.Mesh(new THREE.BoxGeometry(TANK_W, 0.08, 0.08), mat.rail);
      rail.position.set(0, TANK_H + 1.0, TANK_D / 2);
      g.add(rail);

      // 曝气气泡（好氧池）/ 搅拌器（缺氧池）
      const bubbles = this.makeBubbles(2 * w, 2 * d, i >= 2 ? 600 : 0);
      g.add(bubbles);
      let mixer = null;
      if (i < 2) {
        mixer = new THREE.Mesh(new THREE.TorusGeometry(0.9, 0.12, 8, 24), mat.blower);
        mixer.position.y = TANK_H - 0.45;
        mixer.rotation.x = Math.PI / 2;
        g.add(mixer);
      }

      const tag = label(`${i < 2 ? '缺氧池' : '好氧池'} ${i + 1}`);
      tag.position.set(0, TANK_H + 2.3, 0);
      g.add(tag);

      // 透明点击包围盒，作为可交互对象
      const hit = new THREE.Mesh(new THREE.BoxGeometry(TANK_W, TANK_H, TANK_D), new THREE.MeshBasicMaterial({ visible: false }));
      hit.position.y = TANK_H / 2;
      hit.userData = { kind: 'tank', index: i, name: `${i < 2 ? '缺氧池' : '好氧池'} ${i + 1}` };
      g.add(hit);
      this.interactive.push(hit);

      this.root.add(g);
      this.tanks.push({ g, water, waterMat, bubbles, mixer, tag, anoxic: i < 2 });
    }
  }

  makeBubbles(w, d, n) {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(Math.max(n, 1) * 3);
    for (let k = 0; k < n; k++) {
      pos[3 * k] = (Math.random() - 0.5) * w;
      pos[3 * k + 1] = Math.random() * (TANK_H - 0.4);
      pos[3 * k + 2] = (Math.random() - 0.5) * d;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setDrawRange(0, n);
    const p = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xe6f6ff, size: 0.12, transparent: true, opacity: 0.85 }));
    p.userData = { n, max: n, w, d };
    return p;
  }

  buildClarifier() {
    const g = new THREE.Group();
    g.position.set(CLAR.x, 0, CLAR.z);
    const wall = new THREE.Mesh(new THREE.CylinderGeometry(CLAR.r, CLAR.r, CLAR.h, 48, 1, true), mat.concrete);
    wall.position.y = CLAR.h / 2;
    wall.material = mat.concrete.clone();
    wall.material.side = THREE.DoubleSide;
    wall.castShadow = true;
    g.add(wall);
    // 10 层固体浓度分层（半透明圆盘堆叠，颜色深浅 ← 浓度）
    this.layers = [];
    for (let j = 0; j < 10; j++) {
      const m = new THREE.MeshStandardMaterial({ color: 0x6b4a2b, transparent: true, opacity: 0.1, depthWrite: false });
      const disk = new THREE.Mesh(new THREE.CylinderGeometry(CLAR.r - 0.3, CLAR.r - 0.3, CLAR.h / 10 - 0.02, 40), m);
      disk.position.y = CLAR.h - (j + 0.5) * (CLAR.h / 10);
      g.add(disk);
      this.layers.push(m);
    }
    // 刮泥桥
    this.bridge = new THREE.Mesh(new THREE.BoxGeometry(2 * CLAR.r, 0.3, 0.8), mat.rail);
    this.bridge.position.y = CLAR.h + 0.4;
    g.add(this.bridge);
    this.clarTag = label('二沉池');
    this.clarTag.position.set(0, CLAR.h + 2.5, 0);
    g.add(this.clarTag);
    const hit = new THREE.Mesh(new THREE.CylinderGeometry(CLAR.r, CLAR.r, CLAR.h, 16), new THREE.MeshBasicMaterial({ visible: false }));
    hit.position.y = CLAR.h / 2;
    hit.userData = { kind: 'clarifier', name: '二沉池' };
    g.add(hit);
    this.interactive.push(hit);
    this.root.add(g);
  }

  buildPipes() {
    const y = TANK_H + 0.3;
    const x0 = TANK_X[0] - TANK_W / 2, x4 = TANK_X[4] + TANK_W / 2;
    const defs = [
      { key: 'Qin', name: '进水', m: mat.pipe, pts: [[x0 - 10, 0.6, -2], [x0 - 4, 0.6, -2], [x0 - 1, y, -2], [x0 + 0.5, y, -2]] },
      { key: 'Qa', name: '内回流', m: mat.pipe, pts: [[x4 - 1, 0.6, -TANK_D / 2 - 1.2], [0, 0.6, -TANK_D / 2 - 1.5], [x0 + 1, 0.6, -TANK_D / 2 - 1.2]] },
      { key: 'Qf', name: '进二沉池', m: mat.pipe, pts: [[x4 - 0.5, y, 0], [x4 + 2, y, 0], [CLAR.x - CLAR.r - 0.5, y, 0], [CLAR.x - 1, y, 0]] },
      { key: 'Qr', name: '外回流', m: mat.pipeRAS, pts: [[CLAR.x, 0.4, CLAR.r + 0.6], [CLAR.x - CLAR.r - 2, 0.4, TANK_D / 2 + 4.5], [x0 - 1.5, 0.4, TANK_D / 2 + 4.5], [x0 - 1.5, y, 2]] },
      { key: 'Qe', name: '出水', m: mat.pipe, pts: [[CLAR.x + CLAR.r, CLAR.h - 0.3, 0], [CLAR.x + CLAR.r + 6, CLAR.h - 0.3, 0], [CLAR.x + CLAR.r + 10, 0.6, 0]] },
      { key: 'Qw', name: '排泥', m: mat.pipeRAS, pts: [[CLAR.x, 0.4, -CLAR.r - 0.6], [CLAR.x + 4, 0.4, -CLAR.r - 4], [CLAR.x + 10, 0.4, -CLAR.r - 4]] },
    ];
    for (const d of defs) {
      const tube = pipe(d.pts, d.m, d.key === 'Qw' ? 0.18 : 0.28);
      tube.castShadow = true;
      tube.userData = { ...tube.userData, kind: 'pipe', key: d.key, name: d.name };
      this.root.add(tube);
      this.interactive.push(tube);
      // 流动粒子
      const n = 40;
      const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.12, 6, 6),
        new THREE.MeshBasicMaterial({ color: d.m === mat.pipeRAS ? 0xffb066 : 0x9fe0ff }), n);
      this.root.add(dots);
      this.flows.push({ key: d.key, curve: tube.userData.curve, dots, n, phase: Math.random(), rate: 0 });
    }
  }

  buildBlowers() {
    this.blowers = [];
    for (let i = 2; i < 5; i++) {
      const b = new THREE.Group();
      b.position.set(TANK_X[i], 0, -TANK_D / 2 - 4.5);
      const body = new THREE.Mesh(new THREE.BoxGeometry(2, 1.6, 1.4), mat.blower.clone());
      body.position.y = 0.8;
      body.castShadow = true;
      b.add(body);
      const fan = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.15, 6), mat.rail);
      fan.rotation.z = Math.PI / 2;
      fan.position.set(1.05, 0.8, 0);
      b.add(fan);
      const tag = label(`鼓风机 B${i + 1}`, 0.6);
      tag.position.set(0, 2.4, 0);
      b.add(tag);
      body.userData = { kind: 'blower', index: i, name: `鼓风机 B${i + 1}（好氧池 ${i + 1}）` };
      this.interactive.push(body);
      this.root.add(b);
      this.blowers.push({ body, fan, tag, kla: 0 });
    }
  }

  // ------------------------------------------------------------------
  update(s) {
    this.state = s;
    for (let i = 0; i < 5; i++) {
      const t = this.tanks[i];
      t.waterMat.color.copy(waterColor(s.SNH[i], s.SNO[i], t.anoxic));
      t.tag.userData.draw(`${t.anoxic ? '缺氧池' : '好氧池'} ${i + 1}`,
        `DO ${s.DO[i].toFixed(2)}  NH₄ ${s.SNH[i].toFixed(1)}`,
        !t.anoxic && s.DO[i] < 0.5 ? '#ff7a6b' : '#ffffff');
      if (!t.anoxic) {
        const n = Math.round(t.bubbles.userData.max * Math.min(1, s.kla[i] / 360));
        t.bubbles.geometry.setDrawRange(0, n);
      }
    }
    const xmax = 6000;
    for (let j = 0; j < 10; j++) {
      const x = s.settler[j];
      this.layers[j].opacity = Math.min(0.9, 0.04 + 0.86 * Math.min(1, x / xmax));
    }
    const eff = s.eff;
    this.clarTag.userData.draw('二沉池', `出水 SS ${eff.TSS.toFixed(1)}  TN ${eff.TN.toFixed(1)}`,
      eff.TSS > 30 ? '#ff7a6b' : '#ffffff');
    const Qf = s.Qin + s.Qa + s.Qr;
    const rates = { Qin: s.Qin, Qa: s.Qa, Qf, Qr: s.Qr, Qe: s.Qin - s.Qw, Qw: s.Qw * 10 };
    for (const f of this.flows) f.rate = rates[f.key] ?? 0;
    for (const b of this.blowers) {
      b.kla = s.kla[b.body.userData.index];
      b.body.material.color.set(b.kla < 10 ? 0x8f2f2f : 0x2f8f6a);
      b.tag.userData.draw(`鼓风机 B${b.body.userData.index + 1}`, `kLa ${b.kla.toFixed(0)} d⁻¹`, b.kla < 10 ? '#ff7a6b' : '#ffffff');
    }
  }

  animate(dt, time) {
    const m = new THREE.Matrix4();
    for (const f of this.flows) {
      // 粒子沿管道移动速度 ∝ 流量（20000 m³/d → 0.25 圈/秒）
      f.phase = (f.phase + dt * 0.25 * (f.rate / 20000)) % 1;
      for (let k = 0; k < f.n; k++) {
        const u = (f.phase + k / f.n) % 1;
        m.makeTranslation(f.curve.getPointAt(u));
        f.dots.setMatrixAt(k, m);
      }
      f.dots.instanceMatrix.needsUpdate = true;
      f.dots.visible = f.rate > 1;
    }
    for (const t of this.tanks) {
      const p = t.bubbles.geometry.attributes.position;
      const n = t.bubbles.geometry.drawRange.count;
      for (let k = 0; k < n; k++) {
        let y = p.getY(k) + dt * (1.2 + (k % 7) * 0.15);
        if (y > TANK_H - 0.4) y = 0.1;
        p.setY(k, y);
      }
      if (n) p.needsUpdate = true;
      if (t.mixer) t.mixer.rotation.z += dt * 1.5;
      const wp = t.water.geometry.attributes.position;
      const amp = t.anoxic ? 0.03 : 0.02 + 0.06 * (t.bubbles.geometry.drawRange.count / Math.max(1, t.bubbles.userData.max));
      for (let k = 0; k < wp.count; k++) {
        wp.setZ(k, amp * Math.sin(time * 3 + wp.getX(k) * 1.7 + wp.getY(k) * 1.3));
      }
      wp.needsUpdate = true;
    }
    this.bridge.rotation.y += dt * 0.05;
    for (const b of this.blowers) b.fan.rotation.x += dt * b.kla * 0.05;
  }
}
