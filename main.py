"""FastAPI 主服务：实时状态 + 历史查询 + 操作命令 + 混合模型推理。

端点：
    GET  /api/status           最新一拍观测（含告警、活跃工况）
    GET  /api/history          InfluxDB 历史曲线（fields 逗号分隔，start 如 -1h）
    POST /api/command          操作命令（与 MQTT csip/wwtp/cmd 相同格式）
    GET  /api/scenarios        可触发的异常工况列表
    GET  /api/hybrid           混合模型信息与离线测试报告

数字孪生在后台线程运行，每拍发布到 MQTT + InfluxDB。
安全说明：服务默认只监听 127.0.0.1，没有身份认证；如需局域网访问（VR 头显），
请在 .env 设置 API_TOKEN，客户端在请求头 X-API-Token 中携带。
"""
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from influxdb_client import InfluxDBClient

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from csip.sim.pipeline import DataPipeline, load_env  # noqa: E402
from csip.sim.realtime import DigitalTwinEngine, RealtimeRunner  # noqa: E402
from csip.sim.scenarios import SCENARIOS  # noqa: E402
from csip.ai.online import load_online  # noqa: E402
from csip.ai.rl_agent import load_agent  # noqa: E402

load_env()
FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
RANGE_RE = re.compile(r"^-\d{1,4}[smhd]$")
STATE = {}


def check_token(x_api_token: str | None = Header(default=None)):
    token = os.getenv("API_TOKEN")
    if token and x_api_token != token:
        raise HTTPException(401, "invalid token")


@asynccontextmanager
async def lifespan(app):
    engine = DigitalTwinEngine(warmup_days=float(os.getenv("WARMUP_DAYS", 10)))
    hybrid = load_online(ROOT)
    agent = load_agent(ROOT)

    def handle(cmd):
        # ai：开关强化学习设定值优化；操作员手动改设定或复位时自动交还人工
        if cmd.get("type") == "ai":
            if agent is None:
                return {"ok": False, "error": "RL policy not trained"}
            agent.enabled, agent.next_t = bool(cmd.get("enabled")), 0.0
            return {"ok": True}
        if agent and cmd.get("type") in ("set", "reset"):
            agent.enabled = False
        return engine.command(cmd)

    def on_command(cmd):
        res = handle(cmd)
        if res.get("ok"):
            runner.refresh()  # 立即回显，客户端看到的效果不再等下一个发布周期
        return res

    pipe = DataPipeline(on_command=on_command)

    def on_step(obs):
        if agent:
            agent.maybe_act(engine, obs)
            obs["ai"] = {"available": True, "enabled": agent.enabled, "sp": agent.last_sp}
        else:
            obs["ai"] = {"available": False, "enabled": False, "sp": None}
        # 混合模型修正（若已训练），结果随状态一起发布到 MQTT / InfluxDB
        if hybrid:
            hybrid.update(obs, engine.t)
        try:
            pipe.publish(obs)
        except Exception as e:  # 发布失败（如 Influx 字段异常）不应中断仿真线程
            print("publish failed:", repr(e), flush=True)

    runner = RealtimeRunner(engine, on_step, speed=float(os.getenv("SIM_SPEED", 60)))
    runner.start()
    STATE.update(engine=engine, pipe=pipe, runner=runner, hybrid=hybrid, handle=handle)
    yield
    runner.stop()
    pipe.close()


app = FastAPI(title="CSIP 污水处理数字孪生", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/status")
def status():
    return STATE["engine"].last


@app.get("/api/scenarios")
def scenarios():
    return [{"key": s.key, "name": s.name, "desc": s.description} for s in SCENARIOS.values()]


@app.get("/api/hybrid")
def hybrid_info():
    rep = ROOT / "models" / "hybrid" / "report.json"
    return {"loaded": STATE.get("hybrid") is not None,
            "report": json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else None}


@app.get("/api/history", dependencies=[Depends(check_token)])
def history(fields: str = "eff_SNH,eff_SNO,DO_4", start: str = "-1h"):
    names = [f.strip() for f in fields.split(",") if f.strip()]
    if not names or len(names) > 12 or not all(FIELD_RE.match(f) for f in names):
        raise HTTPException(400, "invalid fields")
    if not RANGE_RE.match(start):
        raise HTTPException(400, "start must look like -1h / -30m / -2d")
    # 字段与时间范围都已经过白名单正则校验，再拼入 Flux
    cond = " or ".join(f'r._field == "{f}"' for f in names)
    q = (f'from(bucket:"{os.environ["INFLUX_BUCKET"]}") |> range(start:{start})'
         f' |> filter(fn:(r) => r._measurement == "process" and ({cond}))'
         f' |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)')
    with InfluxDBClient(os.environ["INFLUX_URL"], os.environ["INFLUX_TOKEN"],
                        org=os.environ["INFLUX_ORG"]) as c:
        tables = c.query_api().query(q)
    out = {}
    for t in tables:
        for rec in t.records:
            out.setdefault(rec.get_field(), []).append([rec.get_time().timestamp(), rec.get_value()])
    return out


@app.post("/api/command", dependencies=[Depends(check_token)])
def command(cmd: dict):
    res = STATE["handle"](cmd)
    if res.get("ok"):
        STATE["runner"].refresh()
    if not res.get("ok"):
        raise HTTPException(400, res.get("error", "unknown"))
    return res


web_dist = ROOT / "web" / "dist"
if web_dist.exists():
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", 8000)))
