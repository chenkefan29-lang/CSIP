"""孪生-实测并行运行，生成混合建模数据集。

物理厂（PhysicalPlant）和机理孪生接收同样的进水与控制输入；控制器读取
物理厂的传感器（与现场一致）。每 15 分钟记录一次：
    feats  : 孪生预测值 + 外部输入 + 时间编码
    target : 传感器实测值
"""
import copy

import numpy as np

from ..sim.asm1 import IDX
from ..sim.control import CONTROL_DT, DefaultController
from ..sim.influent import generate_influent
from ..sim.physical import PhysicalPlant
from ..sim.plant import BSM1Plant, PlantInputs

TARGETS = ["SNH5", "SNO2", "eff_SNH", "eff_SNO", "eff_TSS", "eff_COD", "MLSS5"]
INPUTS = ["Qin", "inf_SNH", "inf_COD", "Tin", "kla5", "Qa", "hsin", "hcos"]


def twin_outputs(plant, x):
    Z, _, _, _ = BSM1Plant.unpack(x)
    q = plant.quality(plant.effluent(x))
    return np.stack([Z[:, 4, IDX["SNH"]], Z[:, 1, IDX["SNO"]], q["SNH"], q["SNO"],
                     q["TSS"], q["COD"], 0.75 * Z[:, 4, 2:7].sum(1)], axis=1)


def run_parallel(days, seed, muA_factors=(1.0,), warmup=5.0):
    """返回 dict：twin (n, B, 7)、meas (n, 7)、truth (n, 7)、inputs (n, 8)。

    B 个孪生共用输入，各自取不同 muA_factor（用于参数标定扫描）。
    """
    inf, _ = generate_influent(int((days + warmup) * 96) + 1, seed=seed)
    B = len(muA_factors)
    twin = BSM1Plant()
    xt = twin.initial_state(B)
    phys = PhysicalPlant(seed=seed + 1000)
    ctl = DefaultController()
    mf = np.asarray(muA_factors, float)
    rec = {"twin": [], "meas": [], "truth": [], "inputs": []}
    for k in range(len(inf) - 1):
        r = inf[k]
        for _ in range(15):
            m = phys.measure(CONTROL_DT)
            kla, qa = ctl(np.array([m["DO"][4]]), np.array([m["SNO2"]]))
            u = PlantInputs.constant(1, r[1:14], r[15], kla=kla[0], Qa=qa[0],
                                     Tin=r[16], Tair=r[16] + 2.0)
            phys.step(copy.deepcopy(u), CONTROL_DT)
            ub = PlantInputs.constant(B, r[1:14], r[15], kla=kla[0], Qa=qa[0],
                                      Tin=r[16], Tair=r[16] + 2.0)
            ub.muA_factor = mf.copy()
            xt = twin.step(xt, ub, CONTROL_DT)
        if k < warmup * 96:
            continue
        tr = phys.truth()
        h = 2 * np.pi * (phys.t % 1.0)
        rec["twin"].append(twin_outputs(twin, xt))
        rec["meas"].append([m[t] for t in TARGETS])
        rec["truth"].append([tr[t] for t in TARGETS])
        rec["inputs"].append([r[15], r[1 + IDX["SNH"]], r[1:8].sum(), r[16],
                              kla[0, 4], qa[0], np.sin(h), np.cos(h)])
    return {k: np.asarray(v, np.float32) for k, v in rec.items()}
