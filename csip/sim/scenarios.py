"""异常工况模拟与应急预案。

每个工况在一段时间内修改工厂输入（进水、设备、动力学系数），
应急预案根据实时出水/过程指标自动识别异常并给出处置动作。
"""
from dataclasses import dataclass, field

import numpy as np

from .asm1 import IDX


@dataclass
class Scenario:
    key: str
    name: str
    description: str
    duration: float  # 天

    def apply(self, u, tau, b):
        """tau ∈ [0, 1]：工况进行的进度；b：作用的批次索引。"""
        raise NotImplementedError


class StormSurge(Scenario):
    def apply(self, u, tau, b):
        f = 1.0 + 1.8 * np.sin(np.pi * tau) ** 2
        u.Qin[b] *= f
        u.Zin[b] /= (0.4 + 0.6 * f)  # 雨水稀释
        u.Zin[b, [IDX["XI"], IDX["XS"]]] *= 1.4  # 初期冲刷颗粒物


class ToxicShock(Scenario):
    def apply(self, u, tau, b):
        u.muA_factor[b] *= 1.0 - 0.85 * np.exp(-3.0 * tau)


class AmmoniaLoad(Scenario):
    def apply(self, u, tau, b):
        u.Zin[b, IDX["SNH"]] *= 1.0 + 2.0 * np.sin(np.pi * tau)


class SludgeBulking(Scenario):
    def apply(self, u, tau, b):
        u.settle_factor[b] *= 1.0 - 0.55 * np.sin(np.pi * tau)


class BlowerFailure(Scenario):
    def apply(self, u, tau, b):
        u.kla[b, 2:] *= 0.15


class ColdSnap(Scenario):
    def apply(self, u, tau, b):
        u.Tin[b] -= 6.0 * np.sin(np.pi * tau)
        u.Tair[b] -= 12.0 * np.sin(np.pi * tau)


class PumpFailure(Scenario):
    def apply(self, u, tau, b):
        u.Qr[b] *= 0.2


SCENARIOS = {s.key: s for s in [
    StormSurge("storm", "暴雨冲击", "进水流量 3 倍峰值，污染物稀释，颗粒物初期冲刷", 1.0),
    ToxicShock("toxic", "毒性物质冲击", "工业废水排入，硝化菌活性骤降 85% 后逐渐恢复", 3.0),
    AmmoniaLoad("ammonia", "高氨氮负荷", "进水氨氮最高升至 3 倍", 1.5),
    SludgeBulking("bulking", "污泥膨胀", "丝状菌膨胀，沉降速度下降 55%", 4.0),
    BlowerFailure("blower", "鼓风机故障", "好氧池曝气能力降至 15%", 0.5),
    ColdSnap("cold", "寒潮低温", "进水降温 6 ℃，环境降温 12 ℃", 3.0),
    PumpFailure("pump", "回流泵故障", "外回流量降至 20%", 1.0),
]}


@dataclass
class ActiveScenario:
    scenario: Scenario
    start: float

    def tau(self, t):
        return (t - self.start) / self.scenario.duration


@dataclass
class ScenarioManager:
    """管理当前生效的异常工况（单个工厂，批次 0）。"""
    active: list = field(default_factory=list)

    def trigger(self, key, t):
        if key not in SCENARIOS:
            raise KeyError(key)
        self.active = [a for a in self.active if a.scenario.key != key]
        self.active.append(ActiveScenario(SCENARIOS[key], t))

    def clear(self, key=None):
        self.active = [a for a in self.active if key is not None and a.scenario.key != key]

    def apply(self, u, t, b=0):
        self.active = [a for a in self.active if a.tau(t) < 1.0]
        for a in self.active:
            a.scenario.apply(u, max(a.tau(t), 0.0), b)

    def status(self, t):
        return [{"key": a.scenario.key, "name": a.scenario.name,
                 "progress": round(min(max(a.tau(t), 0.0), 1.0), 3)} for a in self.active]


# ----------------------------------------------------------------------
# 应急预案：基于规则的诊断 + 处置建议（可一键执行）
# ----------------------------------------------------------------------
@dataclass
class Alarm:
    code: str
    level: str  # warning / critical
    message: str
    plan: list  # 处置步骤（文字）
    actions: dict  # 可自动执行的设定值调整


def diagnose(obs, limits):
    """
    obs：当前过程指标字典（出水 SNH/TN/TSS/COD，各池 DO，Qin，T，SNO2，Qr）。
    返回 Alarm 列表。
    """
    alarms = []
    if obs["Qin"] > 2.0 * 18446:
        alarms.append(Alarm("HYD_OVERLOAD", "critical", f"进水流量 {obs['Qin']:.0f} m³/d，超过设计值 2 倍",
                            ["启用调蓄池削峰", "降低内回流避免二沉池水力超负荷", "加强出水 TSS 监测"],
                            {"Qa": 0.5 * 55338}))
    if obs["DO5"] < 0.3 and obs["DO3"] < 0.3:
        alarms.append(Alarm("AERATION_LOSS", "critical", "好氧池溶解氧持续低于 0.3 mg/L，疑似曝气系统故障",
                            ["切换备用鼓风机", "检查曝气管路与阀门", "必要时减少进水"],
                            {"kla_boost": 1.5}))
    if obs["SNH"] > limits["SNH"]:
        lvl = "critical" if obs["SNH"] > 2 * limits["SNH"] else "warning"
        plan = ["提高好氧池 DO 设定值", "延长污泥龄（减少排泥）"]
        actions = {"DO_sp": 2.5, "Qw": 300.0}
        if obs["T"] < 12:
            plan.append("低温工况：投加保温或降低负荷")
        if obs.get("nitrif_drop"):
            plan = ["疑似毒性冲击：排查上游工业排口", "投加活性污泥或硝化菌剂", "暂停排泥保留生物量"]
            actions = {"Qw": 100.0, "DO_sp": 2.5}
        alarms.append(Alarm("NH4_EXCEED", lvl, f"出水氨氮 {obs['SNH']:.2f} mg/L 超标（限值 {limits['SNH']}）", plan, actions))
    if obs["TN"] > limits["TN"]:
        alarms.append(Alarm("TN_EXCEED", "warning", f"出水总氮 {obs['TN']:.2f} mg/L 超标（限值 {limits['TN']}）",
                            ["提高内回流强化反硝化", "缺氧池外加碳源"], {"Qa": 1.3 * 55338}))
    if obs["TSS"] > limits["TSS"]:
        alarms.append(Alarm("TSS_EXCEED", "critical", f"出水 SS {obs['TSS']:.1f} mg/L 超标，二沉池跑泥",
                            ["投加 PAC/PAM 改善沉降", "降低表面负荷", "加大回流排出泥层"],
                            {"Qr": 1.3 * 18446}))
    if obs["COD"] > limits["COD"]:
        alarms.append(Alarm("COD_EXCEED", "warning", f"出水 COD {obs['COD']:.1f} mg/L 超标",
                            ["增加曝气", "检查进水负荷"], {"DO_sp": 2.5}))
    if obs["Qr"] < 0.5 * 18446:
        alarms.append(Alarm("RAS_LOW", "warning", "外回流量过低，疑似回流泵故障",
                            ["启动备用回流泵", "检修故障泵"], {"Qr": 18446.0}))
    return alarms
