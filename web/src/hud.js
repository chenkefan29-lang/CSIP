// 桌面端 HTML 叠加层：关键指标、告警与应急预案、异常工况触发、性能统计。
// VR 模式下 HTML 不可见，相同信息由 main.js 中的三维面板呈现。
const LIMITS = { TN: 18, COD: 100, SNH: 4, TSS: 30, BOD5: 10 };

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

export class Hud {
  constructor(root) {
    this.root = root;
    this.onCommand = null;
    this.scenarioKeys = '';
    root.innerHTML = `
      <section class="card" aria-label="链路与性能">
        <div class="row"><span>MQTT</span><b id="h-link" role="status">连接中…</b></div>
        <div class="row"><span>帧率</span><b id="h-fps">–</b></div>
        <div class="row"><span>操作响应</span><b id="h-lat">–</b></div>
        <div class="row"><span>仿真时间</span><b id="h-t">–</b></div>
      </section>
      <section class="card" aria-label="出水水质">
        <h2>出水水质 <small>mg/L（限值）</small></h2>
        <div id="h-eff"></div>
        <div class="row"><span>进水 Q</span><b id="h-qin">–</b></div>
        <div class="row"><span>EQI / 能耗</span><b id="h-eqi">–</b></div>
      </section>
      <section class="card" aria-label="智能优化控制">
        <h2>强化学习设定值优化</h2>
        <button id="h-ai" aria-pressed="false" disabled>未训练</button>
        <div class="row"><span>当前设定</span><b id="h-ai-sp">–</b></div>
      </section>
      <section class="card" aria-label="混合模型修正">
        <h2>混合模型修正 <small id="h-hyb-st">机理 → 机理+GRU</small></h2>
        <div id="h-hyb"><p class="muted">未加载模型</p></div>
      </section>
      <section class="card" aria-label="告警与应急预案">
        <h2>告警与应急预案</h2>
        <div id="h-alarms" aria-live="polite"><p class="muted">无告警</p></div>
      </section>
      <section class="card" aria-label="异常工况">
        <h2>异常工况演练</h2>
        <div id="h-active" class="muted"></div>
        <div id="h-scen" class="btns"></div>
        <button id="h-reset" class="ok">清除工况 / 复位设定</button>
      </section>
      <p id="h-sel" class="muted">点击池体、鼓风机、二沉池或管道查看详情并调节</p>
      <div id="h-toast" role="status" aria-live="polite"></div>`;
    this.$ = (id) => root.querySelector('#' + id);
    this.$('h-reset').addEventListener('click', () => this.onCommand?.({ type: 'reset' }));
    this.alarmKey = '';
    this.$('h-ai').addEventListener('click', (ev) =>
      this.onCommand?.({ type: 'ai', enabled: ev.currentTarget.getAttribute('aria-pressed') !== 'true' }));
  }

  setLink(s) {
    const b = this.$('h-link');
    b.textContent = s === 'online' ? '在线' : s;
    b.className = s === 'online' ? 'good' : 'bad';
  }

  setFps(fps, info) {
    const b = this.$('h-fps');
    b.textContent = `${fps.toFixed(0)} fps · ${info.calls} draw`;
    b.className = fps >= 72 ? 'good' : 'warn';
  }

  setSelected(name) {
    this.$('h-sel').textContent = name ? `已选中：${name}` : '点击池体、鼓风机、二沉池或管道查看详情并调节';
  }

  setScenarios(list) {
    const key = list.map((s) => s.key).join();
    if (key === this.scenarioKeys) return;
    this.scenarioKeys = key;
    const box = this.$('h-scen');
    box.replaceChildren();
    for (const s of list) {
      const b = el('button', null, s.name);
      b.title = s.desc;
      b.addEventListener('click', () => this.onCommand?.({ type: 'scenario', key: s.key }));
      box.appendChild(b);
    }
  }

  update(s, lat) {
    const days = s.t, d = Math.floor(days), hh = Math.floor((days - d) * 24), mm = Math.floor(((days - d) * 24 - hh) * 60);
    this.$('h-t').textContent = `第 ${d} 天 ${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
    if (lat) {
      const b = this.$('h-lat');
      b.textContent = `均值 ${lat.mean.toFixed(1)} / P95 ${lat.p95.toFixed(1)} ms`;
      b.className = lat.p95 < 100 ? 'good' : 'bad';
    }
    const eff = this.$('h-eff');
    eff.replaceChildren(...Object.entries(LIMITS).map(([k, lim]) => {
      const r = el('div', 'row');
      const v = s.eff[k];
      r.append(el('span', null, `${k}（${lim}）`), el('b', v > lim ? 'bad' : 'good', v.toFixed(2)));
      return r;
    }));
    if (s.ai) {
      const b = this.$('h-ai');
      b.disabled = !s.ai.available;
      b.setAttribute('aria-pressed', String(s.ai.enabled));
      b.className = s.ai.enabled ? 'ok' : '';
      b.textContent = !s.ai.available ? '未训练' : s.ai.enabled ? 'AI 控制中（点击交还人工）' : '启用 AI 控制';
      const sp = s.ai.enabled && s.ai.sp ? s.ai.sp : { DO_sp: null };
      this.$('h-ai-sp').textContent = sp.DO_sp == null ? '人工 / 默认'
        : `DO ${sp.DO_sp.toFixed(2)} · NO ${sp.NO_sp.toFixed(2)} · KLa3/4 ${sp.kla34.toFixed(0)}`;
    }
    if (s.hybrid) {
      // 机理孪生值与混合模型修正值对照（修正值更接近现场仪表）
      const twin = { eff_SNH: s.eff.SNH, eff_SNO: s.eff.SNO, eff_TSS: s.eff.TSS, SNH5: s.SNH[4], SNO2: s.SNO[1] };
      this.$('h-hyb').replaceChildren(...Object.entries(twin).map(([k, v]) => {
        const r = el('div', 'row');
        r.append(el('span', null, k), el('b', null, `${v.toFixed(2)} → ${s.hybrid[k].toFixed(2)}`));
        return r;
      }));
      this.$('h-hyb-st').textContent = s.hybrid_ready ? '机理 → 机理+GRU' : '窗口填充中（前 6 h）';
    }
    this.$('h-qin').textContent = `${s.Qin.toFixed(0)} m³/d · ${s.Tin.toFixed(1)} ℃`;
    this.$('h-eqi').textContent = `${s.EQI.toFixed(0)} kg/d · ${s.energy.toFixed(0)} kWh/d`;

    this.$('h-active').textContent = s.scenarios.length
      ? '进行中：' + s.scenarios.map((a) => `${a.name} ${(a.progress * 100).toFixed(0)}%`).join('，') : '';

    // 告警列表只在内容变化时重建，避免每 100 ms 打断用户点击
    const ak = s.alarms.map((a) => a.code + a.level).join();
    if (ak === this.alarmKey) return;
    this.alarmKey = ak;
    const box = this.$('h-alarms');
    if (!s.alarms.length) { box.replaceChildren(el('p', 'muted', '无告警')); return; }
    box.replaceChildren(...s.alarms.map((a) => {
      const c = el('div', 'alarm ' + a.level);
      c.append(el('b', null, `${a.level === 'critical' ? '严重' : '警告'} · ${a.message}`));
      const ol = el('ol');
      for (const step of a.plan) ol.append(el('li', null, step));
      c.append(ol);
      if (a.actions && Object.keys(a.actions).length) {
        const btn = el('button', 'warn', '一键执行预案');
        btn.addEventListener('click', () => this.onCommand?.({ type: 'plan', code: a.code }));
        c.append(btn);
      }
      return c;
    }));
  }

  toast(msg, ok = true) {
    const t = this.$('h-toast');
    t.textContent = msg;
    t.className = ok ? 'show good' : 'show bad';
    clearTimeout(this._tt);
    this._tt = setTimeout(() => { t.className = ''; }, 2500);
  }
}
