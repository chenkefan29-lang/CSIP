// 三维空间中的可交互面板（Canvas 纹理）。VR 里没有 HTML，菜单与信息都用它。
// 按钮命中检测：射线与平面求交 → UV → 画布坐标 → 按钮矩形。
import * as THREE from 'three';

export class Panel {
  constructor(w = 1.0, h = 0.7, px = 1024) {
    this.cv = document.createElement('canvas');
    this.cv.width = px;
    this.cv.height = Math.round(px * h / w);
    this.g = this.cv.getContext('2d');
    this.tex = new THREE.CanvasTexture(this.cv);
    this.tex.colorSpace = THREE.SRGBColorSpace;
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({ map: this.tex, transparent: true, depthTest: false, side: THREE.DoubleSide }));
    this.mesh.renderOrder = 20;
    this.mesh.userData.panel = this;
    this.buttons = [];
    this.hover = -1;
  }

  // content: { title, lines: [[text, color]], buttons: [{label, cmd, color}] }
  draw(content) {
    const { g, cv } = this;
    const W = cv.width, H = cv.height, s = W / 1024;
    g.clearRect(0, 0, W, H);
    g.fillStyle = 'rgba(10,18,26,0.9)';
    g.beginPath(); g.roundRect(0, 0, W, H, 28 * s); g.fill();
    g.strokeStyle = '#3aa3d9'; g.lineWidth = 4 * s; g.stroke();
    g.fillStyle = '#ffffff'; g.font = `bold ${46 * s}px sans-serif`; g.textAlign = 'left';
    g.fillText(content.title || '', 36 * s, 70 * s);
    let y = 128 * s;
    g.font = `${34 * s}px sans-serif`;
    for (const [text, color] of content.lines || []) {
      g.fillStyle = color || '#d6e6f2';
      g.fillText(text, 36 * s, y);
      y += 46 * s;
    }
    this.buttons = [];
    const bs = content.buttons || [];
    const cols = 2, bw = (W - 36 * s * 3) / cols, bh = 78 * s;
    const top = H - Math.ceil(bs.length / cols) * (bh + 18 * s) - 18 * s;
    bs.forEach((b, i) => {
      const x = 36 * s + (i % cols) * (bw + 36 * s);
      const by = top + Math.floor(i / cols) * (bh + 18 * s);
      g.fillStyle = i === this.hover ? '#4fb6e8' : (b.color || '#24597a');
      g.beginPath(); g.roundRect(x, by, bw, bh, 16 * s); g.fill();
      g.fillStyle = '#ffffff'; g.font = `bold ${32 * s}px sans-serif`; g.textAlign = 'center';
      g.fillText(b.label, x + bw / 2, by + bh / 2 + 11 * s);
      g.textAlign = 'left';
      this.buttons.push({ ...b, rect: [x, by, bw, bh] });
    });
    this.content = content;
    this.tex.needsUpdate = true;
  }

  hit(uv) {
    const x = uv.x * this.cv.width, y = (1 - uv.y) * this.cv.height;
    return this.buttons.findIndex(({ rect: [bx, by, bw, bh] }) => x >= bx && x <= bx + bw && y >= by && y <= by + bh);
  }

  setHover(i) {
    if (i !== this.hover) { this.hover = i; if (this.content) this.draw(this.content); }
  }
}
