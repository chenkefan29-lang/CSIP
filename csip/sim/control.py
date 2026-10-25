"""BSM1 默认控制回路：DO 控制（第 5 池 KLa）+ 硝态氮控制（内回流 Qa）。

离散 PI 控制器，带抗积分饱和，支持批量（每个元素一个独立回路）。
控制周期 1 分钟（与 BSM1 DO 传感器采样周期一致）。
"""
import numpy as np

CONTROL_DT = 1.0 / 1440


class PI:
    def __init__(self, Kp, Ti, lo, hi, bias, B=1):
        self.Kp, self.Ti, self.lo, self.hi, self.bias = Kp, Ti, lo, hi, bias
        self.integ = np.zeros(B)

    def __call__(self, sp, pv, dt):
        e = np.asarray(sp, float) - pv
        u = self.bias + self.Kp * (e + self.integ / self.Ti)
        uc = np.clip(u, self.lo, self.hi)
        # 未饱和或误差使输出回到范围内时才积分（抗积分饱和）
        ok = (u == uc) | ((u > self.hi) & (e < 0)) | ((u < self.lo) & (e > 0))
        self.integ = self.integ + np.where(ok, e * dt, 0.0)
        return uc

    def reset(self, mask=None):
        if mask is None:
            self.integ[:] = 0.0
        else:
            self.integ[mask] = 0.0


class DefaultController:
    """DO5 → KLa5，SNO2 → Qa。KLa3、KLa4 固定 240 1/d（BSM1 默认）。

    do_sp / no_sp 可以是标量或长度 B 的数组（RL 智能体会逐批次改写设定值）。
    """

    def __init__(self, B=1, do_sp=2.0, no_sp=1.0, kla34=240.0):
        self.B = B
        self.do_sp = np.full(B, do_sp)
        self.no_sp = np.full(B, no_sp)
        self.kla34 = np.full(B, kla34)
        self.do_pi = PI(Kp=25.0, Ti=0.002, lo=0.0, hi=360.0, bias=120.0, B=B)
        self.no_pi = PI(Kp=15000.0, Ti=0.05, lo=0.0, hi=5 * 18446.0, bias=18446.0, B=B)

    def __call__(self, do5, sno2, dt=CONTROL_DT):
        kla5 = self.do_pi(self.do_sp, do5, dt)
        qa = self.no_pi(self.no_sp, sno2, dt)
        z = np.zeros(self.B)
        return np.stack([z, z, self.kla34, self.kla34, kla5], axis=1), qa

    def reset(self, mask=None):
        self.do_pi.reset(mask)
        self.no_pi.reset(mask)
