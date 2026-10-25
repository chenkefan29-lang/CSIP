"""混合模型在线推理：把实时孪生观测修正为"实测"预测值。

与 dataset.py 的采样一致：每 15 分钟仿真时间取一次特征（孪生输出 7 + 外部输入 8），
维护长度 WINDOW 的滑动窗口；每次采样后用 GRU 计算残差，结果在下一次采样前保持不变。
窗口未填满时用最早样本补齐，启动后即可输出（前 6 h 精度略低）。
"""
from collections import deque
from pathlib import Path

import numpy as np

from .dataset import TARGETS
from .hybrid import WINDOW, HybridModel

SAMPLE_DT = 1.0 / 96  # 15 min


def obs_features(obs, t_abs):
    """实时观测字典 → 15 维特征，列顺序与 HybridModel.features 相同。"""
    e = obs["eff"]
    twin = [obs["SNH"][4], obs["SNO"][1], e["SNH"], e["SNO"], e["TSS"], e["COD"], obs["MLSS"][4]]
    h = 2 * np.pi * (t_abs % 1.0)
    inputs = [obs["Qin"], obs["inf"]["SNH"], obs["inf"]["COD"], obs["Tin"],
              obs["kla"][4], obs["Qa"], np.sin(h), np.cos(h)]
    return np.asarray(twin + inputs, np.float32)


class OnlineHybrid:
    def __init__(self, path, device="cpu"):
        # 单样本推理在 CPU 上约 1 ms，不必占用 GPU
        self.model = HybridModel.load(path, device=device)
        self.buf = deque(maxlen=WINDOW)
        self.next_t = None
        self.residual = np.zeros(len(TARGETS), np.float32)

    def reset(self):
        self.buf.clear()
        self.next_t = None

    def update(self, obs, t_abs):
        """在 obs 中加入 obs["hybrid"]（修正后的 7 个目标量）。"""
        f = obs_features(obs, t_abs)
        if self.next_t is None or t_abs >= self.next_t:
            self.buf.append(f)
            self.next_t = t_abs + SAMPLE_DT if self.next_t is None else self.next_t + SAMPLE_DT
            w = list(self.buf)
            w = [w[0]] * (WINDOW - len(w)) + w
            self.residual = self.model.predict_window(np.stack(w))
        y = np.maximum(f[:len(TARGETS)] + self.residual, 0.0)
        obs["hybrid"] = {k: round(float(v), 4) for k, v in zip(TARGETS, y)}
        obs["hybrid_ready"] = len(self.buf) >= WINDOW
        return obs


def load_online(root):
    path = Path(root) / "models" / "hybrid"
    return OnlineHybrid(path) if (path / "residual_gru.pt").exists() else None
