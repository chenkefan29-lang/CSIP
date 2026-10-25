// 实时数据链路：浏览器通过 MQTT over WebSocket 订阅孪生状态、下发操作命令。
// 每条命令带 id 和发送时间戳，收到回执后计算端到端响应时延（验收：< 100 ms）。
import mqtt from 'mqtt';

const TOPIC_STATE = 'csip/wwtp/state';
const TOPIC_CMD = 'csip/wwtp/cmd';
const TOPIC_ACK = 'csip/wwtp/cmd/ack';

export class Link {
  constructor(url) {
    this.handlers = { state: [], ack: [], status: [] };
    this.pending = new Map();
    this.latencies = [];
    this.lastStateAt = 0;
    this.client = mqtt.connect(url, { reconnectPeriod: 2000, connectTimeout: 5000 });
    this.client.on('connect', () => {
      this.client.subscribe([TOPIC_STATE, TOPIC_ACK], { qos: 0 });
      this.emit('status', 'online');
    });
    this.client.on('close', () => this.emit('status', 'offline'));
    this.client.on('error', (e) => this.emit('status', 'error: ' + e.message));
    this.client.on('message', (topic, payload) => {
      let msg;
      try { msg = JSON.parse(payload.toString()); } catch { return; }
      if (topic === TOPIC_STATE) {
        this.lastStateAt = performance.now();
        this.emit('state', msg);
      } else if (topic === TOPIC_ACK) {
        const sent = this.pending.get(msg.id);
        if (sent === undefined) return;
        this.pending.delete(msg.id);
        const rtt = performance.now() - sent;
        this.latencies.push(rtt);
        if (this.latencies.length > 50) this.latencies.shift();
        this.emit('ack', { ...msg, rtt });
      }
    });
  }

  on(evt, fn) { this.handlers[evt].push(fn); }
  emit(evt, data) { for (const fn of this.handlers[evt]) fn(data); }

  send(cmd) {
    const id = Math.random().toString(36).slice(2, 10);
    this.pending.set(id, performance.now());
    this.client.publish(TOPIC_CMD, JSON.stringify({ ...cmd, id, ts: Date.now() }), { qos: 1 });
    return id;
  }

  latencyStats() {
    const a = [...this.latencies].sort((x, y) => x - y);
    if (!a.length) return null;
    return { n: a.length, mean: a.reduce((s, v) => s + v, 0) / a.length, p95: a[Math.floor(0.95 * (a.length - 1))] };
  }
}
