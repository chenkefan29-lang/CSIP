"""训练机理+GRU 残差混合模型，并在测试集上报告仿真与实测偏差（验收：< 10%）。

用法：.venv/Scripts/python.exe scripts/train_hybrid.py [--epochs 60]
输出：models/hybrid/（权重、标准化参数、meta.json、report.json）
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from csip.ai.dataset import TARGETS  # noqa: E402
from csip.ai.hybrid import HybridModel, evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
load = lambda s: dict(np.load(ROOT / "data" / f"hybrid_{s}.npz"))  # noqa: E731
train, val, test = load("train"), load("val"), load("test")

t0 = time.time()
model = HybridModel().fit(train, val, epochs=args.epochs)
print(f"训练耗时 {time.time() - t0:.0f}s，设备 {model.device}")

m = evaluate(model, test)
print(f"\n{'指标':<10}{'纯机理':>10}{'混合':>10}")
for k in TARGETS:
    print(f"{k:<10}{100 * m['twin'][k]:>9.2f}%{100 * m['hybrid'][k]:>9.2f}%")
mean_tw, mean_hy = np.mean(list(m["twin"].values())), np.mean(list(m["hybrid"].values()))
worst = max(m["hybrid"].values())
print(f"{'平均':<10}{100 * mean_tw:>9.2f}%{100 * mean_hy:>9.2f}%")
print(f"\n验收（各指标相对偏差 < 10%）：{'通过' if worst < 0.10 else '未通过'}，最大 {100 * worst:.2f}%")

out = ROOT / "models" / "hybrid"
model.save(out)
(out / "report.json").write_text(json.dumps({"test": m, "pass": bool(worst < 0.10),
                                             "epochs": args.epochs}, ensure_ascii=False, indent=1))
print(f"模型已保存到 {out}")
