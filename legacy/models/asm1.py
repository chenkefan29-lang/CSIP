from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# 状态变量顺序：SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK
STATE_NAMES = (
    "SI", "SS", "XI", "XS", "XBH", "XBA", "XP",
    "SO", "SNO", "SNH", "SND", "XND", "SALK",
)


@dataclass(frozen=True)
class ASM1Parameters:
    # 化学计量参数
    YH: float = 0.67
    YA: float = 0.24
    fP: float = 0.08
    iXB: float = 0.08
    iXP: float = 0.06

    # 动力学参数：BSM1 默认值，约对应 15 摄氏度
    muH: float = 4.0
    KS: float = 10.0
    KO_H: float = 0.2
    KNO: float = 0.5
    bH: float = 0.3
    etaG: float = 0.8
    etaH: float = 0.8
    kh: float = 3.0
    KX: float = 0.1
    muA: float = 0.5
    KNH: float = 1.0
    bA: float = 0.05
    KO_A: float = 0.4
    ka: float = 0.05


DEFAULT_PARAMETERS = ASM1Parameters()

# 教学演示用的代表性进水状态，单位为 g/m3（数值等同于 mg/L）。
# 后续接入 BSM1 数据时，会用数据文件中的 13 个状态量替换它。
DEFAULT_INFLUENT = np.array(
    [30.0, 69.5, 51.2, 202.32, 28.17, 0.0, 0.0,
     0.0, 0.0, 31.56, 6.95, 10.59, 7.0],
    dtype=float,
)

# 初始池内状态。生物量设为已有活性污泥的启动状态。
DEFAULT_INITIAL = np.array(
    [30.0, 5.0, 51.2, 100.0, 1500.0, 100.0, 50.0,
     2.0, 1.0, 20.0, 2.0, 10.0, 7.0],
    dtype=float,
)


def process_rates(state, p=DEFAULT_PARAMETERS):
    """根据当前 13 个状态量，计算 ASM1 的 8 个生化反应速率。"""
    y = np.maximum(np.asarray(state, dtype=float), 0.0)
    if y.shape != (13,):
        raise ValueError("state 必须按 STATE_NAMES 顺序包含 13 个数值")

    (
        _si, ss, _xi, xs, xbh, xba, _xp,
        so, sno, snh, snd, xnd, _salk,
    ) = y

    aerobic_carbon = so / (p.KO_H + so)
    anoxic_carbon = p.KO_H / (p.KO_H + so)
    nitrate_factor = sno / (p.KNO + sno)

    # 1. 异养菌好氧生长
    rho1 = p.muH * ss / (p.KS + ss) * aerobic_carbon * xbh

    # 2. 异养菌缺氧生长
    rho2 = (
        p.muH * p.etaG * ss / (p.KS + ss)
        * anoxic_carbon * nitrate_factor * xbh
    )

    # 3. 自养菌好氧生长（硝化）
    rho3 = (
        p.muA * snh / (p.KNH + snh)
        * so / (p.KO_A + so) * xba
    )

    # 4-5. 异养菌和自养菌衰减
    rho4 = p.bH * xbh
    rho5 = p.bA * xba

    # 6. 可溶性有机氮氨化
    rho6 = p.ka * snd * xbh

    # 7. 慢速可降解有机物水解
    biomass = max(xbh, 1e-12)
    xs_per_biomass = xs / biomass
    hydrolysis_factor = xs_per_biomass / (p.KX + xs_per_biomass)
    oxygen_nitrate_factor = (
        aerobic_carbon
        + p.etaH * anoxic_carbon * nitrate_factor
    )
    rho7 = p.kh * hydrolysis_factor * oxygen_nitrate_factor * xbh

    # 8. 颗粒态有机氮水解
    rho8 = rho7 * xnd / max(xs, 1e-12)

    return np.array(
        [rho1, rho2, rho3, rho4, rho5, rho6, rho7, rho8],
        dtype=float,
    )


def component_rates(rho, p=DEFAULT_PARAMETERS):
    """把 8 个过程速率换算成 13 个状态变量的生化反应速率。"""
    rho1, rho2, rho3, rho4, rho5, rho6, rho7, rho8 = rho
    decay = rho4 + rho5

    r_si = 0.0
    r_ss = -rho1 / p.YH - rho2 / p.YH + rho7
    r_xi = 0.0
    r_xs = (1.0 - p.fP) * decay - rho7
    r_xbh = rho1 + rho2 - rho4
    r_xba = rho3 - rho5
    r_xp = p.fP * decay
    r_so = -((1.0 - p.YH) / p.YH) * rho1
    r_so -= ((4.57 - p.YA) / p.YA) * rho3
    r_sno = -((1.0 - p.YH) / (2.86 * p.YH)) * rho2
    r_sno += rho3 / p.YA
    r_snh = -p.iXB * rho1 - p.iXB * rho2
    r_snh -= (p.iXB + 1.0 / p.YA) * rho3
    r_snh += rho6
    r_snd = -rho6 + rho8
    r_xnd = (p.iXB - p.fP * p.iXP) * decay - rho8
    r_salk = -(p.iXB / 14.0) * rho1
    r_salk += (
        (1.0 - p.YH) / (14.0 * 2.86 * p.YH)
        - p.iXB / 14.0
    ) * rho2
    r_salk -= (p.iXB / 14.0 + 1.0 / (7.0 * p.YA)) * rho3
    r_salk += rho6 / 14.0

    return np.array(
        [
            r_si, r_ss, r_xi, r_xs, r_xbh, r_xba, r_xp,
            r_so, r_sno, r_snh, r_snd, r_xnd, r_salk,
        ],
        dtype=float,
    )


def reactor_derivative(
    time_days,
    state,
    influent,
    q_over_v,
    kla_day,
    oxygen_saturation,
    p=DEFAULT_PARAMETERS,
):
    """单个完全混合池的进出水质量平衡 + ASM1 生化反应。"""
    state_for_rates = np.maximum(np.asarray(state, dtype=float), 0.0)
    rho = process_rates(state_for_rates, p)
    biological_change = component_rates(rho, p)

    # 水力停留造成的进出水质量平衡，q_over_v 单位为 1/day。
    hydraulic_change = q_over_v * (influent - state_for_rates)

    # 氧传递只作用于溶解氧 SO，kla_day 单位为 1/day。
    oxygen_change = np.zeros(13, dtype=float)
    oxygen_change[7] = kla_day * (
        oxygen_saturation - state_for_rates[7]
    )

    return hydraulic_change + biological_change + oxygen_change


def simulate_single_cstr(
    duration_days=5.0,
    points=121,
    q_over_v=1.0,
    kla_day=80.0,
    oxygen_saturation=8.0,
    initial=None,
    influent=None,
    parameters=DEFAULT_PARAMETERS,
):
    """运行单池 ASM1 动态仿真，返回时间和 13 个状态变量的曲线。"""
    if duration_days <= 0 or points < 2 or q_over_v <= 0:
        raise ValueError("duration_days、points 或 q_over_v 参数不合理")

    initial_state = np.array(
        DEFAULT_INITIAL if initial is None else initial,
        dtype=float,
    )
    influent_state = np.array(
        DEFAULT_INFLUENT if influent is None else influent,
        dtype=float,
    )

    if initial_state.shape != (13,) or influent_state.shape != (13,):
        raise ValueError("initial 和 influent 都必须包含 13 个状态量")
    if np.any(initial_state < 0) or np.any(influent_state < 0):
        raise ValueError("状态浓度不能为负数")

    times = np.linspace(0.0, duration_days, points)

    solution = solve_ivp(
        fun=lambda t, y: reactor_derivative(
            t,
            y,
            influent_state,
            q_over_v,
            kla_day,
            oxygen_saturation,
            parameters,
        ),
        t_span=(0.0, duration_days),
        y0=initial_state,
        method="BDF",
        t_eval=times,
        rtol=1e-6,
        atol=1e-8,
    )

    if not solution.success:
        raise RuntimeError(f"ASM1 积分失败：{solution.message}")

    concentrations = np.maximum(solution.y.T, 0.0)

    return {
        "time_days": solution.t,
        "states": {
            name: concentrations[:, index]
            for index, name in enumerate(STATE_NAMES)
        },
    }


if __name__ == "__main__":
    result = simulate_single_cstr()
    final_state = {
        name: round(float(values[-1]), 3)
        for name, values in result["states"].items()
    }
    print("运行天数:", round(float(result["time_days"][-1]), 2))
    print("最终状态:", final_state)