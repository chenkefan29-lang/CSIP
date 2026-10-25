"""PPO 训练上层设定值优化智能体，并与 BSM1 默认控制对比。

用法：python scripts/train_rl.py [--iters 60] [--B 32]
输出：models/rl/policy.pt、models/rl/report.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csip.ai.rl_agent import Policy  # noqa: E402
from csip.ai.rl_env import DEFAULT_SP, WWTPEnv, to_action  # noqa: E402

def evaluate(policy, days=7, B=8, seed=10_000):
    """在未见过的进水序列上比较：policy=None 表示默认控制。"""
    env = WWTPEnv(B=B, days=days + 1, seed=seed, warmup=3)
    env.k[:] = 0
    o = env.reset()
    logs = []
    for _ in range(days * 96):
        if policy is None:
            a = np.tile(to_action(DEFAULT_SP), (B, 1))
        else:
            with torch.no_grad():
                a = policy.mean(torch.as_tensor(o)).numpy()
        o, r, info = env.step(a)
        logs.append({k: np.asarray(v) for k, v in info.items()} | {"r": r})
    g = lambda k: float(np.mean([x[k] for x in logs]))
    return {"reward": g("r"), "EQI": g("EQI"), "energy_kWh_d": g("energy"),
            "viol_rate": float(np.mean([x["viol"] > 0 for x in logs])),
            "SNH_viol_rate": float(np.mean([x["SNH"] > 4 for x in logs])),
            "TN_viol_rate": float(np.mean([x["TN"] > 18 for x in logs]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=96)  # 每轮每厂 1 天
    ap.add_argument("--init", type=float, nargs=3, default=list(DEFAULT_SP), help="初始设定值 DO NO kla34")
    ap.add_argument("--w-count", type=float, default=1.0, help="每个超标指标的额外惩罚")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    out = ROOT / "models" / "rl"
    out.mkdir(parents=True, exist_ok=True)

    env = WWTPEnv(B=args.B, days=90, seed=0, w_count=args.w_count)
    pol = Policy(init_sp=np.array(args.init))
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    gamma, lam, clip = 0.98, 0.95, 0.2
    o = torch.as_tensor(env.reset())
    t0 = time.time()
    for it in range(args.iters):
        O, A, LP, R, V = [], [], [], [], []
        for _ in range(args.T):
            with torch.no_grad():
                d = pol.dist(o)
                a = d.sample()
                O.append(o); A.append(a); LP.append(d.log_prob(a).sum(-1)); V.append(pol.v(o)[:, 0])
            o2, r, _ = env.step(a.numpy())
            R.append(torch.as_tensor(r))
            o = torch.as_tensor(o2)
        with torch.no_grad():
            v_last = pol.v(o)[:, 0]
        # GAE（连续任务，轮末用价值函数自举）
        adv = torch.zeros(args.T, args.B)
        gae = torch.zeros(args.B)
        for t in reversed(range(args.T)):
            nv = v_last if t == args.T - 1 else V[t + 1]
            delta = R[t] + gamma * nv - V[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        ret = adv + torch.stack(V)
        O, A, LP = torch.cat(O), torch.cat(A), torch.cat(LP)
        adv, ret = adv.flatten(), ret.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(10):
            for idx in torch.randperm(len(O)).split(256):
                d = pol.dist(O[idx])
                ratio = (d.log_prob(A[idx]).sum(-1) - LP[idx]).exp()
                pl = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
                vl = (pol.v(O[idx])[:, 0] - ret[idx]).pow(2).mean()
                loss = pl + 0.5 * vl - 1e-3 * d.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
        print(f"iter {it:3d}  reward {torch.stack(R).mean():.3f}  std {pol.log_std.exp().detach().numpy().round(3)}"
              f"  {time.time() - t0:.0f}s", flush=True)
        torch.save(pol.state_dict(), out / f"policy{args.tag}.pt")

    base, rl = evaluate(None), evaluate(pol)
    static = evaluate(Policy(init_sp=np.array(args.init)))  # 只用初始固定设定值，不学习
    rep = {"default": base, "static_init": static, "rl": rl, "args": vars(args),
           "EQI_change": rl["EQI"] / base["EQI"] - 1, "energy_change": rl["energy_kWh_d"] / base["energy_kWh_d"] - 1}
    (out / f"report{args.tag}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
