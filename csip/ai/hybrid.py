"""机理 + 数据驱动混合模型（残差学习）。

    y_hat(t) = y_twin(t) + scale * f_θ( window[t-L+1 .. t] )

window 每一步的特征：孪生输出（7）+ 外部输入（8），全部标准化。
f_θ 是单层 GRU + 线性头，学习"实测 − 孪生"的残差（已按目标量纲缩放）。
机理部分保证外推时的物理合理性，网络只负责修正系统性偏差。
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import INPUTS, TARGETS

WINDOW = 24  # 24 × 15 min = 6 h


class ResidualGRU(nn.Module):
    def __init__(self, n_in, n_out, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h[:, -1])


class HybridModel:
    def __init__(self, net=None, stats=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        self.stats = stats

    # ------------------------------------------------------------------
    @staticmethod
    def features(d):
        return np.concatenate([d["twin"][:, 0, :], d["inputs"]], axis=1)

    def _windows(self, d):
        f = (self.features(d) - self.stats["f_mu"]) / self.stats["f_sd"]
        idx = np.arange(WINDOW - 1, len(f))
        X = np.stack([f[i - WINDOW + 1:i + 1] for i in idx]).astype(np.float32)
        return X, idx

    def fit(self, train, val, epochs=60, lr=2e-3, batch=256, log=print):
        f = self.features(train)
        res = train["meas"] - train["twin"][:, 0, :]
        self.stats = {"f_mu": f.mean(0), "f_sd": f.std(0) + 1e-6,
                      "r_sd": res.std(0) + 1e-6}
        self.net = ResidualGRU(f.shape[1], len(TARGETS)).to(self.device)
        Xtr, itr = self._windows(train)
        Ytr = (res[itr] / self.stats["r_sd"]).astype(np.float32)
        Xtr, Ytr = torch.tensor(Xtr, device=self.device), torch.tensor(Ytr, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state = np.inf, None
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Xtr), device=self.device)
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                loss = nn.functional.smooth_l1_loss(self.net(Xtr[b]), Ytr[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            sched.step()
            m = evaluate(self, val)
            score = np.mean(list(m["hybrid"].values()))
            if score < best:
                best = score
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            if ep % 10 == 0 or ep == epochs - 1:
                log(f"epoch {ep:3d} loss {loss.item():.4f} val mean rel-MAE {100 * score:.2f}%")
        self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, d):
        """返回 (idx, y_twin, y_hybrid)，idx 为有完整窗口的时间步。"""
        self.net.eval()
        X, idx = self._windows(d)
        r = self.net(torch.tensor(X, device=self.device)).cpu().numpy() * self.stats["r_sd"]
        tw = d["twin"][idx, 0, :]
        y = tw + r
        y[:, :] = np.maximum(y, 0.0)
        return idx, tw, y

    @torch.no_grad()
    def predict_window(self, feats_window):
        """在线推理：feats_window (WINDOW, 15) 原始特征 → 残差 (7,)。"""
        self.net.eval()
        x = (np.asarray(feats_window, np.float32) - self.stats["f_mu"]) / self.stats["f_sd"]
        r = self.net(torch.tensor(x[None], dtype=torch.float32, device=self.device))
        return r.cpu().numpy()[0] * self.stats["r_sd"]

    # ------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "residual_gru.pt")
        np.savez(path / "stats.npz", **self.stats)
        (path / "meta.json").write_text(json.dumps(
            {"targets": TARGETS, "inputs": INPUTS, "window": WINDOW}, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path, device=None):
        path = Path(path)
        stats = dict(np.load(path / "stats.npz"))
        m = cls(stats=stats, device=device)
        m.net = ResidualGRU(len(stats["f_mu"]), len(TARGETS)).to(m.device)
        m.net.load_state_dict(torch.load(path / "residual_gru.pt", map_location=m.device))
        return m


def rel_mae(pred, meas):
    """相对平均绝对误差 Σ|ŷ−y| / Σ|y|（各目标分别计算）。"""
    return np.abs(pred - meas).sum(0) / np.abs(meas).sum(0)


def evaluate(model, d):
    idx, tw, y = model.predict(d)
    meas = d["meas"][idx]
    return {"twin": dict(zip(TARGETS, rel_mae(tw, meas).tolist())),
            "hybrid": dict(zip(TARGETS, rel_mae(y, meas).tolist()))}
