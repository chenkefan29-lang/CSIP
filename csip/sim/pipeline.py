"""实时数据管道：引擎观测值 → MQTT（实时推送）+ InfluxDB（时序存储）；
MQTT 命令 → 引擎（VR / Web 端操作）。

Topic 约定：
    csip/wwtp/state      引擎状态快照（JSON，retain）
    csip/wwtp/cmd        操作命令（JSON，含 id 和 ts 用于响应时延统计）
    csip/wwtp/cmd/ack    命令回执（JSON：id、ok、server_ms）
"""
import json
import os
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import ASYNCHRONOUS

ROOT = Path(__file__).resolve().parents[2]
TOPIC_STATE = "csip/wwtp/state"
TOPIC_CMD = "csip/wwtp/cmd"
TOPIC_ACK = "csip/wwtp/cmd/ack"


def load_env(path=ROOT / ".env"):
    """读取 KEY=VALUE 格式的 .env，不覆盖已存在的环境变量。"""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def to_point(obs, wall_ns, measurement="process", source="twin"):
    """把嵌套观测字典展开成 InfluxDB 点：DO_0..DO_4、eff_SNH 等。"""
    p = Point(measurement).tag("source", source).time(wall_ns, WritePrecision.NS)
    p.field("sim_t", float(obs["t"]))
    for k, v in obs.items():
        if k in ("t", "alarms", "scenarios"):
            continue
        if k == "ai":  # 智能体状态：开关记 0/1，尚未决策时没有设定值
            p.field("ai_enabled", int(bool(v.get("enabled"))))
            for sk, sv in (v.get("sp") or {}).items():
                p.field(f"ai_{sk}", float(sv))
            continue
        if isinstance(v, dict):
            for sk, sv in v.items():
                if sv is None:
                    continue
                p.field(f"{k}_{sk}", float(sv))
        elif isinstance(v, list):
            for i, sv in enumerate(v):
                p.field(f"{k}_{i}", float(sv))
        elif k == "weather":
            p.field(k, int(v))
        else:
            p.field(k, float(v))
    p.field("n_alarms", len(obs.get("alarms", [])))
    return p


class DataPipeline:
    def __init__(self, on_command=None, use_influx=True):
        load_env()
        self.on_command = on_command
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="csip-engine")
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_message
        self.mqtt.connect(os.getenv("MQTT_HOST", "127.0.0.1"), int(os.getenv("MQTT_PORT", 1883)))
        self.mqtt.loop_start()
        self.influx = None
        if use_influx:
            self.influx = InfluxDBClient(os.environ["INFLUX_URL"], os.environ["INFLUX_TOKEN"],
                                         org=os.environ["INFLUX_ORG"])
            self.write_api = self.influx.write_api(write_options=ASYNCHRONOUS)
            self.bucket = os.environ["INFLUX_BUCKET"]
        self.cmd_latency_ms = []

    def _on_connect(self, client, userdata, flags, rc, props=None):
        client.subscribe(TOPIC_CMD, qos=1)

    def _on_message(self, client, userdata, msg):
        t0 = time.perf_counter()
        try:
            cmd = json.loads(msg.payload)
            res = self.on_command(cmd) if self.on_command else {"ok": False, "error": "no handler"}
        except Exception as e:  # 坏命令不能让管道线程崩溃
            cmd, res = {}, {"ok": False, "error": str(e)}
        server_ms = (time.perf_counter() - t0) * 1000.0
        self.cmd_latency_ms.append(server_ms)
        ack = {"id": cmd.get("id"), "ts": cmd.get("ts"), "server_ms": round(server_ms, 3), **res}
        client.publish(TOPIC_ACK, json.dumps(ack), qos=1)

    def publish(self, obs, extra_points=()):
        self.mqtt.publish(TOPIC_STATE, json.dumps(obs), qos=0, retain=True)
        if self.influx:
            now = time.time_ns()
            self.write_api.write(self.bucket, record=[to_point(obs, now), *extra_points])

    def close(self):
        self.mqtt.loop_stop()
        self.mqtt.disconnect()
        if self.influx:
            self.write_api.close()
            self.influx.close()
