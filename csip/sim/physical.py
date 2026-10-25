"""虚拟"物理厂"：充当实测数据来源。

真实污水厂与机理模型之间总存在偏差。这里用一个参数与结构都和
数字孪生不同的工厂来模拟现场：
    - 动力学参数偏离 BSM1 默认值（硝化菌更慢、水解更快等）
    - 曝气器结垢：KLa 实际效率随时间缓慢下降
    - 未建模的日周期效应（进水中存在模型不知道的抑制物）
    - 传感器噪声、仪表滞后（氨氮/硝氮分析仪 5–10 分钟滞后）
数字孪生只能看到传感器读数，不能直接读物理厂状态。
"""
import numpy as np

from .asm1 import ASM1Params, IDX
from .plant import BSM1Plant, PlantConfig

# 物理厂的"真实"参数（对孪生未知）
TRUE_PARAMS = ASM1Params().scaled(muA=0.88, KNH=1.25, kh=1.15, bH=1.1, KOA=1.2, YH=0.98)
TRUE_CONFIG = PlantConfig(v0=440.0, rh=0.00062, heat_loss_rate=0.06)

SENSORS = {
    # 名称: (噪声标准差（相对）, 滞后时间常数 [min], 绝对噪声下限)
    "DO": (0.03, 1.0, 0.02),
    "SNO2": (0.05, 5.0, 0.05),
    "SNH5": (0.05, 5.0, 0.05),
    "eff_SNH": (0.05, 10.0, 0.05),
    "eff_SNO": (0.05, 10.0, 0.05),
    "eff_TSS": (0.08, 2.0, 0.3),
    "eff_COD": (0.06, 15.0, 1.0),
    "MLSS5": (0.04, 2.0, 20.0),
    "T": (0.0, 1.0, 0.1),
}


class PhysicalPlant:
    def __init__(self, seed=0):
        self.plant = BSM1Plant(TRUE_PARAMS, TRUE_CONFIG)
        self.rng = np.random.default_rng(seed)
        self.x = self.plant.initial_state(1)
        self.t = 0.0
        self.lagged = {}

    def kla_efficiency(self):
        """曝气器结垢：90 天从 100% 线性降到 85%，带周期性清洗恢复。"""
        return 1.0 - 0.15 * ((self.t % 90.0) / 90.0)

    def inhibition(self):
        """未建模的日周期抑制：凌晨工业排放，硝化速率下降最多 12%。"""
        h = (self.t % 1.0) * 24.0
        return 1.0 - 0.12 * np.exp(-0.5 * ((h - 3.0) / 1.5) ** 2)

    def step(self, u, dt):
        u.kla = u.kla * self.kla_efficiency()
        u.muA_factor = u.muA_factor * self.inhibition()
        self.x = self.plant.step(self.x, u, dt)
        self.t += dt

    def truth(self):
        Z, X, S, T = BSM1Plant.unpack(self.x)
        q = self.plant.quality(self.plant.effluent(self.x))
        return {
            "DO": Z[0, :, IDX["SO"]].copy(),
            "SNO2": float(Z[0, 1, IDX["SNO"]]),
            "SNH5": float(Z[0, 4, IDX["SNH"]]),
            "eff_SNH": float(q["SNH"][0]), "eff_SNO": float(q["SNO"][0]),
            "eff_TSS": float(q["TSS"][0]), "eff_COD": float(q["COD"][0]),
            "MLSS5": float(0.75 * Z[0, 4, 2:7].sum()), "T": float(T[0]),
        }

    def measure(self, dt):
        """带噪声和一阶滞后的传感器读数。"""
        out = {}
        for k, v in self.truth().items():
            rel, tau_min, floor = SENSORS[k]
            a = 1.0 - np.exp(-dt * 1440.0 / tau_min)
            prev = self.lagged.get(k, v)
            lag = prev + a * (np.asarray(v) - prev)
            self.lagged[k] = lag
            noise = self.rng.normal(0.0, 1.0, np.shape(lag)) * np.maximum(rel * np.abs(lag), floor)
            out[k] = np.maximum(lag + noise, 0.0)
        out["DO"] = out["DO"].tolist()
        return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in out.items()}
