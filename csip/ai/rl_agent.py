"""PPO 策略网络与在线智能体（接入实时数字孪生引擎）。

智能体每 15 分钟仿真时间读取一次引擎状态，按 rl_env 的观测定义构造特征，
输出 DO_sp / NO_sp / kla34 三个上层设定值，交给底层 PI 回路跟踪。
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .rl_env import DEFAULT_SP, N_OBS, to_action, to_setpoints

DECISION_DT = 1.0 / 96


class Policy(nn.Module):
    def __init__(self, h=128, init_sp=DEFAULT_SP):
        super().__init__()
        # 初始策略输出 init_sp；随权重一起保存，加载旧权重时缺省为默认设定值
        self.register_buffer("a0", torch.tensor(np.arctanh(np.clip(to_action(init_sp), -0.95, 0.95)), dtype=torch.float32))
        self.pi = nn.Sequential(nn.Linear(N_OBS, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 3))
        self.v = nn.Sequential(nn.Linear(N_OBS, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.log_std = nn.Parameter(torch.full((3,), -1.0))
        nn.init.zeros_(self.pi[-1].weight)
        nn.init.zeros_(self.pi[-1].bias)

    def mean(self, o):
        return torch.tanh(self.pi(o) + self.a0)

    def dist(self, o):
        return torch.distributions.Normal(self.mean(o), self.log_std.exp())


def engine_obs(obs, t_abs, sp):
    """把引擎发布的状态字典转换成与 WWTPEnv._obs 一致的 13 维特征。"""
    e, h = obs["eff"], 2 * np.pi * (int(t_abs * 96) % 96) / 96
    a = to_action(sp)
    return np.array([obs["DO"][4] / 2, obs["SNO"][1], obs["SNH"][4] / 4,
                     e["SNH"] / 4, e["TN"] / 18, e["TSS"] / 30,
                     obs["Qin"] / 18446 - 1, obs["inf"]["SNH"] / 30, (obs["Tin"] - 15) / 5,
                     np.sin(h), np.cos(h), a[0], a[2]], np.float32)


class OnlineAgent:
    def __init__(self, path):
        self.policy = Policy()
        self.policy.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        self.policy.eval()
        self.enabled = False
        self.next_t = 0.0
        self.last_sp = None

    def maybe_act(self, engine, obs):
        """到决策时刻时改写引擎设定值；返回本次设定（未决策返回 None）。"""
        if not self.enabled or engine.t < self.next_t:
            return None
        self.next_t = engine.t + DECISION_DT
        sp_now = np.array([engine.ctl.do_sp[0], engine.ctl.no_sp[0], engine.manual["kla3"]])
        with torch.no_grad():
            a = self.policy.mean(torch.as_tensor(engine_obs(obs, engine.t, sp_now))[None]).numpy()[0]
        sp = to_setpoints(a)
        engine.command({"type": "set", "values": {"DO_sp": sp[0], "NO_sp": sp[1],
                                                  "kla3": sp[2], "kla4": sp[2]}})
        self.last_sp = {"DO_sp": round(float(sp[0]), 3), "NO_sp": round(float(sp[1]), 3),
                        "kla34": round(float(sp[2]), 1)}
        return self.last_sp


def load_agent(root):
    p = Path(root) / "models" / "rl" / "policy.pt"
    return OnlineAgent(p) if p.exists() else None
