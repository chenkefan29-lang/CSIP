"""强化学习监督控制环境（批量向量化，B 个独立水厂并行）。

智能体每 15 分钟设定一次上层设定值，底层 PI 回路每分钟跟踪：
    a[0] DO_sp   第 5 池溶解氧设定  [0.5, 3.0] mg/L
    a[1] NO_sp   第 2 池硝态氮设定  [0.3, 2.5] mg/L
    a[2] kla34   第 3/4 池曝气强度  [100, 360] 1/d
动作取 [-1, 1]，线性映射到上述区间。

奖励 = −( EQI/EQI0 + 能耗/E0 + 10·超标程度 + 超标指标数 )，EQI0、E0 为默认控制下的典型值，
超标程度 = Σ max(0, (出水 − 限值)/限值)，覆盖 SNH、TN、TSS、COD、BOD5；
另对每个超标指标额外扣 1（只按幅度惩罚时策略会用大量小幅 TN 超标换 SNH）。
连续任务（不复位）：每个水厂沿各自的进水序列前进，到末尾回绕。
"""
import numpy as np

from ..sim.asm1 import IDX
from ..sim.control import CONTROL_DT, DefaultController
from ..sim.influent import generate_influent
from ..sim.plant import BSM1Plant, PlantInputs

LO = np.array([0.5, 0.3, 100.0])
HI = np.array([3.0, 2.5, 360.0])
DEFAULT_SP = np.array([2.0, 1.0, 240.0])
EQI0, E0 = 6000.0, 4000.0
LIMITS = BSM1Plant.LIMITS
N_OBS = 13


def to_setpoints(a):
    return LO + (np.clip(a, -1, 1) + 1) / 2 * (HI - LO)


def to_action(sp):
    return 2 * (np.asarray(sp) - LO) / (HI - LO) - 1


class WWTPEnv:
    def __init__(self, B=32, days=60, seed=0, warmup=5.0, w_count=1.0):
        self.B = B
        self.w_count = w_count
        self.plant = BSM1Plant()
        n = int(days * 96)
        infs = [generate_influent(n + 1, seed=seed + i)[0] for i in range(B)]
        self.inf = np.stack(infs)  # (B, n+1, 17)
        self.n = n
        self.k = np.random.default_rng(seed).integers(0, n, B)  # 各水厂起点错开
        self.ctl = DefaultController(B=B)
        self.x = self.plant.initial_state(B)
        self.sp = np.tile(DEFAULT_SP, (B, 1))
        for _ in range(int(warmup * 96)):  # 默认控制预热
            self._advance()

    def _inputs(self):
        r = self.inf[np.arange(self.B), self.k]
        Z = self.x[:, :65].reshape(self.B, 5, 13)
        kla, qa = self.ctl(Z[:, 4, IDX["SO"]], Z[:, 1, IDX["SNO"]])
        u = PlantInputs.constant(self.B, np.zeros(13), 0.0)
        u.Zin, u.Qin, u.Tin, u.Tair = r[:, 1:14], r[:, 15], r[:, 16], r[:, 16] + 2.0
        u.kla, u.Qa = kla, qa
        return u, r

    def _advance(self):
        for _ in range(15):
            u, r = self._inputs()
            self.x = self.plant.step(self.x, u, CONTROL_DT)
        self.k = (self.k + 1) % self.n
        return u, r

    def _obs(self, u, r, q):
        Z = self.x[:, :65].reshape(self.B, 5, 13)
        h = 2 * np.pi * (self.k % 96) / 96
        o = np.stack([Z[:, 4, IDX["SO"]] / 2, Z[:, 1, IDX["SNO"]], Z[:, 4, IDX["SNH"]] / 4,
                      q["SNH"] / 4, q["TN"] / 18, q["TSS"] / 30,
                      r[:, 15] / 18446 - 1, r[:, 1 + IDX["SNH"]] / 30, (r[:, 16] - 15) / 5,
                      np.sin(h), np.cos(h), to_action(self.sp)[:, 0], to_action(self.sp)[:, 2]], 1)
        return o.astype(np.float32)

    def reset(self):
        u, r = self._inputs()
        q = self.plant.quality(self.plant.effluent(self.x))
        return self._obs(u, r, q)

    def step(self, a):
        self.sp = to_setpoints(a)
        self.ctl.do_sp[:], self.ctl.no_sp[:], self.ctl.kla34[:] = self.sp.T
        u, r = self._advance()
        q = self.plant.quality(self.plant.effluent(self.x))
        eqi = self.plant.eqi_rate(q, u.Qin - u.Qw)
        aer, pump = self.plant.energy_rate(u)
        energy = aer + pump
        viol = sum(np.maximum(0.0, (q[k] - v) / v) for k, v in LIMITS.items())
        n_viol = sum((q[k] > v).astype(float) for k, v in LIMITS.items())
        rew = -(eqi / EQI0 + energy / E0 + 10.0 * viol + self.w_count * n_viol)
        info = {"EQI": eqi, "energy": energy, "viol": viol, "SNH": q["SNH"], "TN": q["TN"]}
        return self._obs(u, r, q), rew.astype(np.float32), info
