"""BSM1 污水处理厂机理模型（多物理场耦合，批量向量化）。

工艺：2 个缺氧池 + 3 个好氧池 + 10 层 Takács 二沉池，含内回流、外回流和剩余污泥排放。
耦合的物理过程：
    - 水力：完全混合池串联的进出水质量平衡
    - 生化：ASM1 动力学（见 asm1.py）
    - 传质：曝气氧传递 KLa·(SOsat − SO)，KLa 和 SOsat 都随温度变化
    - 沉降：Takács 双指数沉速 + 一维固体通量模型
    - 热量：池体温度随进水温度和环境散热变化，温度再反馈给动力学

状态向量（每个工厂 146 维）：
    [0:65)    5 个反应池 × 13 个 ASM1 组分
    [65:75)   二沉池 10 层 TSS（0 为顶层）
    [75:145)  二沉池 10 层 × 7 个溶解态组分
    [145]     生化池温度 T [℃]
所有函数都接受形状为 (B, ...) 的批量输入，B 为并行工厂数，便于强化学习并行训练。
"""
from dataclasses import dataclass

import numpy as np

from .asm1 import (ASM1Kinetics, ASM1Params, IDX, PARTICULATE_IDX, SOLUBLE_IDX,
                   TSS_IDX, X_TO_TSS)

N_REACTOR, N_COMP, N_LAYER, N_SOL = 5, 13, 10, len(SOLUBLE_IDX)
SL_R = slice(0, 65)
SL_X = slice(65, 75)
SL_S = slice(75, 145)
I_T = 145
N_STATE = 146


@dataclass(frozen=True)
class PlantConfig:
    volumes: tuple = (1000.0, 1000.0, 1333.0, 1333.0, 1333.0)  # m3
    settler_area: float = 1500.0  # m2
    settler_height: float = 4.0  # m
    feed_layer: int = 4  # 从顶层数第 5 层（自底向上第 6 层）
    # Takács 沉降参数
    v0_max: float = 250.0
    v0: float = 474.0
    rh: float = 0.000576
    rp: float = 0.00286
    fns: float = 0.00228
    Xt: float = 3000.0
    # 热量平衡
    heat_loss_rate: float = 0.05  # 1/d，向环境散热
    kla_theta: float = 1.024


DEFAULT_CONFIG = PlantConfig()


def oxygen_saturation(T):
    """饱和溶解氧随温度变化，15 ℃ 时等于 BSM1 的 8 g/m3。"""
    def poly(t):
        return 14.652 - 0.41022 * t + 0.007991 * t ** 2 - 0.000077774 * t ** 3
    return 8.0 * poly(T) / poly(15.0)


@dataclass
class PlantInputs:
    """一个采样周期内保持不变的外部输入，所有数组首维为 B。"""
    Zin: np.ndarray       # (B, 13) 进水组分
    Qin: np.ndarray       # (B,)   进水流量 m3/d
    Tin: np.ndarray       # (B,)   进水温度 ℃
    kla: np.ndarray       # (B, 5) 各池 KLa（15 ℃ 标定值）1/d
    Qa: np.ndarray        # (B,)   内回流 m3/d
    Qr: np.ndarray        # (B,)   外回流 m3/d
    Qw: np.ndarray        # (B,)   剩余污泥 m3/d
    Tair: np.ndarray      # (B,)   环境温度 ℃
    muA_factor: np.ndarray  # (B,) 硝化菌活性系数（毒性冲击 < 1）
    settle_factor: np.ndarray  # (B,) 沉降性能系数（污泥膨胀 < 1）

    @classmethod
    def constant(cls, B, Zin, Qin, kla=(0, 0, 240, 240, 84), Qa=55338.0,
                 Qr=18446.0, Qw=385.0, Tin=15.0, Tair=15.0):
        f = lambda v: np.full(B, float(v))
        return cls(Zin=np.tile(np.asarray(Zin, float), (B, 1)), Qin=f(Qin), Tin=f(Tin),
                   kla=np.tile(np.asarray(kla, float), (B, 1)), Qa=f(Qa), Qr=f(Qr),
                   Qw=f(Qw), Tair=f(Tair), muA_factor=f(1.0), settle_factor=f(1.0))


class BSM1Plant:
    """BSM1 机理模型：导数计算、RK4 积分、出水和性能指标。"""

    def __init__(self, params: ASM1Params = ASM1Params(), config: PlantConfig = DEFAULT_CONFIG):
        self.params = params
        self.cfg = config
        self.kin = ASM1Kinetics(params)
        self.V = np.asarray(config.volumes)
        self.h = config.settler_height / N_LAYER

    # ------------------------------------------------------------------
    # 状态拆分
    # ------------------------------------------------------------------
    @staticmethod
    def unpack(x):
        B = x.shape[0]
        return (x[:, SL_R].reshape(B, N_REACTOR, N_COMP), x[:, SL_X],
                x[:, SL_S].reshape(B, N_LAYER, N_SOL), x[:, I_T])

    # ------------------------------------------------------------------
    # 二沉池
    # ------------------------------------------------------------------
    def settling_velocity(self, X, Xf, settle_factor):
        c = self.cfg
        Xmin = c.fns * Xf[:, None]
        d = np.maximum(X - Xmin, 0.0)
        vs = c.v0 * settle_factor[:, None] * (np.exp(-c.rh * d) - np.exp(-c.rp * d))
        return np.clip(vs, 0.0, c.v0_max * settle_factor[:, None])

    def settler_fluxes(self, X, Xf, settle_factor):
        """Js[:, i]：第 i 层沉降到第 i+1 层的固体通量，最底层为 0。"""
        c, m = self.cfg, self.cfg.feed_layer
        J = self.settling_velocity(X, Xf, settle_factor) * X
        lower = np.minimum(J[:, :-1], J[:, 1:])
        Js = np.zeros_like(X)
        # 进水层及以下：浓缩区
        Js[:, m:-1] = lower[:, m:]
        # 进水层以上：澄清区，只有下层超过阈值浓度时才受限
        Js[:, :m] = np.where(X[:, 1:m + 1] > c.Xt, lower[:, :m], J[:, :m])
        return Js

    # ------------------------------------------------------------------
    # 导数
    # ------------------------------------------------------------------
    def derivative(self, x, u: PlantInputs):
        c, m, h = self.cfg, self.cfg.feed_layer, self.h
        Z, X, S, T = self.unpack(x)
        Zp = np.maximum(Z, 0.0)
        Xp = np.maximum(X, 0.0)
        Sp = np.maximum(S, 0.0)
        B = x.shape[0]

        Qf = u.Qin + u.Qr                 # 二沉池进水
        Qu = u.Qr + u.Qw                  # 底流
        Qe = Qf - Qu                      # 出水
        Q = Qf + u.Qa                     # 生化池流量

        # 底流回流组成：溶解态取底层，颗粒态按反应池 5 出口比例
        Z5 = Zp[:, 4]
        Xf = X_TO_TSS * Z5[:, TSS_IDX].sum(1)
        frac = Z5[:, PARTICULATE_IDX] / np.maximum(Xf, 1e-9)[:, None]
        Zr = np.zeros((B, N_COMP))
        Zr[:, SOLUBLE_IDX] = Sp[:, -1]
        Zr[:, PARTICULATE_IDX] = frac * Xp[:, -1:]

        # ---- 反应池 ----
        Zfeed = (u.Qin[:, None] * u.Zin + u.Qa[:, None] * Z5 + u.Qr[:, None] * Zr) / Q[:, None]
        upstream = np.concatenate([Zfeed[:, None], Zp[:, :-1]], axis=1)
        dZ = (Q[:, None, None] / self.V[None, :, None]) * (upstream - Zp)
        Tb = np.broadcast_to(T[:, None], (B, N_REACTOR))
        dZ += self.kin.rates(Zp, Tb, muA_factor=u.muA_factor[:, None])
        kla_T = u.kla * c.kla_theta ** (Tb - 15.0)
        dZ[:, :, IDX["SO"]] += kla_T * (oxygen_saturation(Tb) - Zp[:, :, IDX["SO"]])

        # ---- 二沉池固体 ----
        A = c.settler_area
        vup, vdn = (Qe / A)[:, None], (Qu / A)[:, None]
        Js = self.settler_fluxes(Xp, Xf, u.settle_factor)
        Js_in = np.concatenate([np.zeros((B, 1)), Js[:, :-1]], axis=1)
        dX = (Js_in - Js) / h
        dX[:, :m] += vup * (Xp[:, 1:m + 1] - Xp[:, :m]) / h
        dX[:, m + 1:] += vdn * (Xp[:, m:-1] - Xp[:, m + 1:]) / h
        dX[:, m] += (Qf * Xf - (Qe + Qu) * Xp[:, m]) / (A * h)

        # ---- 二沉池溶解态 ----
        Sf = Z5[:, SOLUBLE_IDX]
        dS = np.zeros_like(Sp)
        dS[:, :m] = vup[:, :, None] * (Sp[:, 1:m + 1] - Sp[:, :m]) / h
        dS[:, m + 1:] = vdn[:, :, None] * (Sp[:, m:-1] - Sp[:, m + 1:]) / h
        dS[:, m] = ((Qf / A)[:, None] * Sf - (vup + vdn) * Sp[:, m]) / h

        # ---- 热量 ----
        dT = u.Qin / self.V.sum() * (u.Tin - T) + c.heat_loss_rate * (u.Tair - T)

        out = np.empty_like(x)
        out[:, SL_R] = dZ.reshape(B, -1)
        out[:, SL_X] = dX
        out[:, SL_S] = dS.reshape(B, -1)
        out[:, I_T] = dT
        return out

    # ------------------------------------------------------------------
    # 积分
    # ------------------------------------------------------------------
    def step(self, x, u: PlantInputs, duration, dt=0.5 / 1440):
        """用定步长 RK4 积分 duration 天，输入在此期间保持不变。"""
        n = max(1, int(round(duration / dt)))
        dt = duration / n
        for _ in range(n):
            k1 = self.derivative(x, u)
            k2 = self.derivative(x + 0.5 * dt * k1, u)
            k3 = self.derivative(x + 0.5 * dt * k2, u)
            k4 = self.derivative(x + dt * k3, u)
            x = np.maximum(x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
        return x

    def initial_state(self, B=1, T=15.0):
        """接近 BSM1 稳态的启动状态，避免长时间预热。"""
        z = np.array([30, 2.0, 1149, 82, 2552, 148, 449, 0.5, 5.0, 5.0, 1.0, 5.0, 4.5])
        x = np.zeros((B, N_STATE))
        x[:, SL_R] = np.tile(z, N_REACTOR)
        x[:, SL_X] = [12.5, 18.1, 29.5, 69.0, 356, 356, 356, 356, 356, 6394]
        x[:, SL_S] = np.tile(z[SOLUBLE_IDX], N_LAYER)
        x[:, I_T] = T
        return x

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def effluent(self, x):
        """出水 13 组分（颗粒态按二沉池顶层 TSS 和反应池 5 比例换算）。"""
        Z, X, S, _ = self.unpack(np.maximum(x, 0.0))
        Z5 = Z[:, 4]
        Xf = X_TO_TSS * Z5[:, TSS_IDX].sum(1)
        Ze = np.zeros((x.shape[0], N_COMP))
        Ze[:, SOLUBLE_IDX] = S[:, 0]
        Ze[:, PARTICULATE_IDX] = Z5[:, PARTICULATE_IDX] / np.maximum(Xf, 1e-9)[:, None] * X[:, :1]
        return Ze

    def quality(self, Ze):
        """出水综合指标：COD、BOD5、TKN、总氮、TSS 等，单位 g/m3。"""
        p, i = self.params, IDX
        cod = Ze[:, [i[n] for n in ("SI", "SS", "XI", "XS", "XBH", "XBA", "XP")]].sum(1)
        bod = 0.25 * (Ze[:, i["SS"]] + Ze[:, i["XS"]] + (1 - p.fP) * (Ze[:, i["XBH"]] + Ze[:, i["XBA"]]))
        tkn = (Ze[:, i["SNH"]] + Ze[:, i["SND"]] + Ze[:, i["XND"]]
               + p.iXB * (Ze[:, i["XBH"]] + Ze[:, i["XBA"]]) + p.iXP * (Ze[:, i["XP"]] + Ze[:, i["XI"]]))
        tss = X_TO_TSS * Ze[:, TSS_IDX].sum(1)
        return {"COD": cod, "BOD5": bod, "TKN": tkn, "TN": tkn + Ze[:, i["SNO"]],
                "SNH": Ze[:, i["SNH"]], "SNO": Ze[:, i["SNO"]], "TSS": tss}

    @staticmethod
    def eqi_rate(q, Qe):
        """出水水质指数 EQI 的瞬时值 [kg 污染单位/d]（BSM1 权重）。"""
        return (2 * q["TSS"] + q["COD"] + 30 * q["TKN"] + 10 * q["SNO"] + 2 * q["BOD5"]) * Qe / 1000.0

    def energy_rate(self, u: PlantInputs):
        """曝气能耗 + 泵送能耗 [kWh/d]（BSM1 2008 版定义）。"""
        aeration = 8.0 / (1.8 * 1000.0) * (u.kla * self.V).sum(1)
        pumping = 0.004 * u.Qa + 0.008 * u.Qr + 0.05 * u.Qw
        return aeration, pumping

    # 出水排放限值（BSM1）
    LIMITS = {"TN": 18.0, "COD": 100.0, "SNH": 4.0, "TSS": 30.0, "BOD5": 10.0}
