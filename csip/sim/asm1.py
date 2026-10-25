"""ASM1 生化反应动力学（批量向量化版本）。

状态变量顺序（13 个）：
    SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK
单位：g/m3（SALK 为 mol/m3），时间单位：天。

参数取 BSM1 默认值（15 ℃）。温度修正采用 BSM2 的 Arrhenius 形式，
T = 15 ℃ 时与 BSM1 完全一致。
"""
from dataclasses import dataclass, fields, replace

import numpy as np

STATE_NAMES = (
    "SI", "SS", "XI", "XS", "XBH", "XBA", "XP",
    "SO", "SNO", "SNH", "SND", "XND", "SALK",
)
IDX = {name: i for i, name in enumerate(STATE_NAMES)}

# 颗粒态组分（随污泥沉降）与溶解态组分
PARTICULATE_IDX = np.array([IDX[n] for n in ("XI", "XS", "XBH", "XBA", "XP", "XND")])
TSS_IDX = np.array([IDX[n] for n in ("XI", "XS", "XBH", "XBA", "XP")])
SOLUBLE_IDX = np.array([IDX[n] for n in ("SI", "SS", "SO", "SNO", "SNH", "SND", "SALK")])
X_TO_TSS = 0.75


@dataclass(frozen=True)
class ASM1Params:
    # 化学计量参数
    YH: float = 0.67
    YA: float = 0.24
    fP: float = 0.08
    iXB: float = 0.08
    iXP: float = 0.06
    # 动力学参数（15 ℃）
    muH: float = 4.0
    KS: float = 10.0
    KOH: float = 0.2
    KNO: float = 0.5
    bH: float = 0.3
    etaG: float = 0.8
    etaH: float = 0.8
    kh: float = 3.0
    KX: float = 0.1
    muA: float = 0.5
    KNH: float = 1.0
    bA: float = 0.05
    KOA: float = 0.4
    ka: float = 0.05

    def scaled(self, **factors):
        """返回按比例缩放部分参数后的新参数集，例如 scaled(muA=0.8)。"""
        return replace(self, **{k: getattr(self, k) * v for k, v in factors.items()})

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


# 10 ℃ 参考值，用于构造 Arrhenius 温度系数 theta = ln(k15/k10)/5
_K10 = {"muH": 3.0, "bH": 0.2, "muA": 0.3, "bA": 0.03, "kh": 2.5, "ka": 0.04}


def stoichiometry(p: ASM1Params) -> np.ndarray:
    """8 个过程 × 13 个组分的化学计量矩阵。"""
    S = np.zeros((8, 13))
    YH, YA, fP, iXB, iXP = p.YH, p.YA, p.fP, p.iXB, p.iXP
    i = IDX
    # 1 异养菌好氧生长
    S[0, i["SS"]] = -1 / YH
    S[0, i["XBH"]] = 1
    S[0, i["SO"]] = -(1 - YH) / YH
    S[0, i["SNH"]] = -iXB
    S[0, i["SALK"]] = -iXB / 14
    # 2 异养菌缺氧生长（反硝化）
    S[1, i["SS"]] = -1 / YH
    S[1, i["XBH"]] = 1
    S[1, i["SNO"]] = -(1 - YH) / (2.86 * YH)
    S[1, i["SNH"]] = -iXB
    S[1, i["SALK"]] = (1 - YH) / (14 * 2.86 * YH) - iXB / 14
    # 3 自养菌好氧生长（硝化）
    S[2, i["XBA"]] = 1
    S[2, i["SO"]] = -(4.57 - YA) / YA
    S[2, i["SNO"]] = 1 / YA
    S[2, i["SNH"]] = -iXB - 1 / YA
    S[2, i["SALK"]] = -iXB / 14 - 1 / (7 * YA)
    # 4、5 异养菌/自养菌衰减
    for row, bio in ((3, "XBH"), (4, "XBA")):
        S[row, i[bio]] = -1
        S[row, i["XS"]] = 1 - fP
        S[row, i["XP"]] = fP
        S[row, i["XND"]] = iXB - fP * iXP
    # 6 可溶性有机氮氨化
    S[5, i["SND"]] = -1
    S[5, i["SNH"]] = 1
    S[5, i["SALK"]] = 1 / 14
    # 7 颗粒有机物水解
    S[6, i["XS"]] = -1
    S[6, i["SS"]] = 1
    # 8 颗粒有机氮水解
    S[7, i["XND"]] = -1
    S[7, i["SND"]] = 1
    return S


class ASM1Kinetics:
    """预计算化学计量矩阵，提供批量反应速率计算。"""

    def __init__(self, params: ASM1Params = ASM1Params()):
        self.p = params
        self.S = stoichiometry(params)
        self.theta = {k: np.log(getattr(params, k) / v) / 5.0 for k, v in _K10.items()}

    def temp_factor(self, name, T):
        return np.exp(self.theta[name] * (T - 15.0))

    def rates(self, Z, T, muA_factor=1.0, muH_factor=1.0):
        """
        Z: (..., 13) 浓度；T: 与 Z[..., 0] 同形状的温度 [℃]。
        muA_factor / muH_factor: 毒性抑制等工况的增长速率修正（可广播）。
        返回 (..., 13) 的生化反应速率 [g/m3/d]。
        """
        p = self.p
        Z = np.maximum(Z, 0.0)
        ss, xs, xbh, xba = Z[..., 1], Z[..., 3], Z[..., 4], Z[..., 5]
        so, sno, snh, snd, xnd = Z[..., 7], Z[..., 8], Z[..., 9], Z[..., 10], Z[..., 11]

        muH = p.muH * self.temp_factor("muH", T) * muH_factor
        bH = p.bH * self.temp_factor("bH", T)
        muA = p.muA * self.temp_factor("muA", T) * muA_factor
        bA = p.bA * self.temp_factor("bA", T)
        kh = p.kh * self.temp_factor("kh", T)
        ka = p.ka * self.temp_factor("ka", T)

        sub = ss / (p.KS + ss)
        o_h = so / (p.KOH + so)
        i_oh = p.KOH / (p.KOH + so)
        no = sno / (p.KNO + sno)

        rho = np.empty(Z.shape[:-1] + (8,))
        rho[..., 0] = muH * sub * o_h * xbh
        rho[..., 1] = muH * p.etaG * sub * i_oh * no * xbh
        rho[..., 2] = muA * snh / (p.KNH + snh) * so / (p.KOA + so) * xba
        rho[..., 3] = bH * xbh
        rho[..., 4] = bA * xba
        rho[..., 5] = ka * snd * xbh
        xs_ratio = xs / np.maximum(xbh, 1e-9)
        rho[..., 6] = kh * xs_ratio / (p.KX + xs_ratio) * (o_h + p.etaH * i_oh * no) * xbh
        rho[..., 7] = rho[..., 6] * xnd / np.maximum(xs, 1e-9)
        return rho @ self.S
