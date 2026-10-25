"""实时仿真引擎：机理模型 + 控制器 + 异常工况 + 应急诊断。

每一拍仿真 1 分钟（控制周期）：
    1. 取当前 15 分钟进水样本，生成本拍的基础输入
    2. 控制器根据 DO5 / SNO2 计算 KLa 和 Qa，叠加人工/预案设定
    3. 异常工况在基础输入的副本上修改（不会累积放大）
    4. RK4 积分 1 分钟，计算出水和过程指标
引擎本身不依赖网络；发布由 pipeline.py 负责。
"""
import copy
import threading
import time

import numpy as np

from .asm1 import IDX
from .control import CONTROL_DT, DefaultController
from .influent import generate_influent
from .plant import BSM1Plant, PlantInputs
from .scenarios import SCENARIOS, ScenarioManager, diagnose

INF_DT = 1.0 / 96


class DigitalTwinEngine:
    def __init__(self, influent=None, labels=None, plant=None, warmup_days=20.0):
        if influent is None:
            influent, labels = generate_influent(8000, seed=7)
        self.inf = influent  # (n, 17)：BSM1 16 列 + 进水温度
        self.weather = labels["weather"] if labels else np.zeros(len(influent), int)
        self.plant = plant or BSM1Plant()
        self.ctl = DefaultController(B=1)
        self.scen = ScenarioManager()
        self.lock = threading.RLock()
        # 操作员手动设定（None 表示自动）
        self.manual = {"Qr": 18446.0, "Qw": 385.0, "kla3": 240.0, "kla4": 240.0,
                       "kla5": None, "Qa": None}
        self.t = 0.0
        self.x = self.plant.initial_state(1)
        self._warmup(warmup_days)
        self.last = self._observe(self._inputs())

    # ------------------------------------------------------------------
    def _sample(self, t):
        k = int(t / INF_DT) % len(self.inf)
        return self.inf[k], int(self.weather[k])

    def _inputs(self):
        row, _ = self._sample(self.t)
        m = self.manual
        kla, qa = self.ctl(self.x[:, IDX["SO"] + 52], self.x[:, IDX["SNO"] + 13])
        kla[0, 2], kla[0, 3] = m["kla3"], m["kla4"]
        if m["kla5"] is not None:
            kla[0, 4] = m["kla5"]
        if m["Qa"] is not None:
            qa = np.array([m["Qa"]])
        u = PlantInputs.constant(1, row[1:14], row[15], kla=kla[0], Qa=qa[0],
                                 Qr=m["Qr"], Qw=m["Qw"], Tin=row[16], Tair=row[16] + 2.0)
        self.scen.apply(u, self.t)
        return u

    def _warmup(self, days):
        """用默认控制运行若干天，让污泥和二沉池进入动态平衡。"""
        for _ in range(int(days * 96)):
            for _ in range(15):
                self.x = self.plant.step(self.x, self._inputs(), CONTROL_DT)
                self.t += CONTROL_DT
        self.t0_warm = self.t

    # ------------------------------------------------------------------
    def step(self, n=1):
        with self.lock:
            for _ in range(n):
                u = self._inputs()
                self.x = self.plant.step(self.x, u, CONTROL_DT)
                self.t += CONTROL_DT
            self.last = self._observe(u)
            return self.last

    def refresh(self):
        """不推进时间，按当前设定值重算一拍观测（操作命令后立即回显用）。"""
        with self.lock:
            self.last = self._observe(self._inputs())
            return self.last

    def _observe(self, u):
        Z, X, S, T = BSM1Plant.unpack(self.x)
        q = self.plant.quality(self.plant.effluent(self.x))
        aer, pump = self.plant.energy_rate(u)
        Qe = u.Qin - u.Qw
        _, weather = self._sample(self.t)
        obs = {
            "t": round(self.t - self.t0_warm, 6),
            "weather": weather,
            "Qin": float(u.Qin[0]), "Qa": float(u.Qa[0]), "Qr": float(u.Qr[0]),
            "Qw": float(u.Qw[0]), "Tin": float(u.Tin[0]), "T": float(T[0]),
            "kla": [round(float(v), 2) for v in u.kla[0]],
            "DO": [float(v) for v in Z[0, :, IDX["SO"]]],
            "SNO": [float(v) for v in Z[0, :, IDX["SNO"]]],
            "SNH": [float(v) for v in Z[0, :, IDX["SNH"]]],
            "SS": [float(v) for v in Z[0, :, IDX["SS"]]],
            "MLSS": [float(v) for v in 0.75 * Z[0, :, 2:7].sum(1)],
            "settler": [float(v) for v in X[0]],
            "eff": {k: float(v[0]) for k, v in q.items()},
            "EQI": float(self.plant.eqi_rate(q, Qe)[0]),
            "energy": float(aer[0] + pump[0]),
            "inf": {"SNH": float(u.Zin[0, IDX["SNH"]]),
                    "COD": float(u.Zin[0, :7].sum())},
        }
        flags = {"DO3": obs["DO"][2], "DO5": obs["DO"][4], "Qin": obs["Qin"], "T": obs["T"],
                 "Qr": obs["Qr"], "nitrif_drop": any(a.scenario.key == "toxic" for a in self.scen.active),
                 **obs["eff"]}
        alarms = diagnose(flags, BSM1Plant.LIMITS)
        obs["alarms"] = [a.__dict__ for a in alarms]
        obs["scenarios"] = self.scen.status(self.t)
        # 操作员设定值（未叠加异常工况）；前端调节按钮以此为基准，None 表示自动
        obs["manual"] = dict(self.manual)
        return obs

    # ------------------------------------------------------------------
    # 操作接口（VR / Web 端通过 MQTT 命令调用）
    # ------------------------------------------------------------------
    def command(self, cmd):
        """cmd: {"type": "...", ...}。返回执行结果字典，参数错误时返回 ok=False 而不抛异常。"""
        with self.lock:
            kind = cmd.get("type")
            if kind == "scenario":
                if cmd.get("key") not in SCENARIOS:
                    return {"ok": False, "error": f"unknown scenario {cmd.get('key')}"}
                self.scen.trigger(cmd["key"], self.t)
            elif kind == "clear":
                self.scen.clear(cmd.get("key"))
            elif kind == "set":
                # 先整体校验再写入，避免一条命令只生效一半
                values = cmd.get("values")
                if not isinstance(values, dict):
                    return {"ok": False, "error": "values must be an object"}
                parsed = {}
                for k, v in values.items():
                    if k not in ("DO_sp", "NO_sp") and k not in self.manual:
                        return {"ok": False, "error": f"unknown setpoint {k}"}
                    if v is None and k in ("DO_sp", "NO_sp"):
                        return {"ok": False, "error": f"{k} cannot be null"}
                    try:
                        parsed[k] = None if v is None else float(v)
                    except (TypeError, ValueError):
                        return {"ok": False, "error": f"invalid value for {k}: {v!r}"}
                    if parsed[k] is not None and not np.isfinite(parsed[k]):
                        return {"ok": False, "error": f"invalid value for {k}: {v!r}"}
                for k, v in parsed.items():
                    if k == "DO_sp":
                        self.ctl.do_sp[:] = v
                    elif k == "NO_sp":
                        self.ctl.no_sp[:] = v
                    else:
                        self.manual[k] = v
            elif kind == "plan":
                if not self.apply_plan(cmd.get("code")):
                    return {"ok": False, "error": f"no active alarm {cmd.get('code')}"}
            elif kind == "reset":
                self.manual.update({"Qr": 18446.0, "Qw": 385.0, "kla3": 240.0,
                                    "kla4": 240.0, "kla5": None, "Qa": None})
                self.ctl.do_sp[:] = 2.0
                self.ctl.no_sp[:] = 1.0
                self.scen.clear()
            else:
                return {"ok": False, "error": f"unknown command {kind}"}
            return {"ok": True}

    def apply_plan(self, code):
        """执行告警对应的应急预案自动动作；当前没有该告警时返回 False。"""
        found = False
        for a in self.last["alarms"]:
            if a["code"] != code:
                continue
            found = True
            for k, v in a["actions"].items():
                if k == "DO_sp":
                    self.ctl.do_sp[:] = v
                elif k == "kla_boost":
                    self.manual["kla3"] = min(360.0, self.manual["kla3"] * v)
                    self.manual["kla4"] = min(360.0, self.manual["kla4"] * v)
                else:
                    self.manual[k] = v
        return found


class RealtimeRunner:
    """按 speed 倍速推进引擎，每拍调用 on_step(obs)。"""

    def __init__(self, engine, on_step, speed=60.0, publish_hz=10.0):
        self.engine, self.on_step = engine, on_step
        self.speed, self.publish_hz = speed, publish_hz
        self.paused = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def refresh(self):
        """收到操作命令后立刻重算并发布一拍，避免等待下一个发布周期（1/publish_hz 秒）。"""
        self.on_step(self.engine.refresh())

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self):
        # 每次发布间隔推进 speed / publish_hz 分钟仿真时间
        period = 1.0 / self.publish_hz
        acc = 0.0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            if not self.paused:
                acc += self.speed / 60.0 * period
                n = int(acc)
                if n:
                    acc -= n
                    self.on_step(self.engine.step(n))
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
