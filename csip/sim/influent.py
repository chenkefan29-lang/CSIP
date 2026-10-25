"""BSM1 风格的动态进水生成器（与 legacy 相同，增加温度变化）。

输出格式：16 列白空格分隔，无表头，时间步长 15 分钟（1/96 天）。
前 16 列与 BSM1 官方数据完全一致：
    time [d], S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO,
    S_NH, S_ND, X_ND, S_ALK, TSS, Q [m3/d]
另存 _labels.csv：天气标签（0 干 1 雨 2 暴雨），_meta.json：统计摘要。
用 load_influent(path) 按列名读回 dict。
"""
import argparse, json, os
import numpy as np

COLS = ["time","S_I","S_S","X_I","X_S","X_BH","X_BA","X_P","S_O","S_NO",
        "S_NH","S_ND","X_ND","S_ALK","TSS","Q"]
DT = 1/96
MEAN_CONC = {"S_I": 30, "S_S": 69.5, "X_I": 51.2, "X_S": 202.32, "X_BH": 28.17,
             "X_BA": 0, "X_P": 0, "S_O": 0, "S_NO": 0, "S_NH": 31.56,
             "S_ND": 6.95, "X_ND": 10.59, "S_ALK": 7}
Q_MEAN = 18446.0
PARTICULATE = ["X_I", "X_S", "X_BH"]

def _pg(h, peaks, floor):
    def raw(x):
        r = np.full_like(x, floor, dtype=float)
        for c, w, a in peaks:
            d = (x - c + 12) % 24 - 12
            r += a * np.exp(-0.5 * (d / w) ** 2)
        return r
    ref = np.linspace(0, 24, 96, endpoint=False)
    return raw(np.asarray(h, float)) / raw(ref).mean()

def _ar1(rng, n, phi, sigma):
    e = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x

def _diurnal(t, we):
    h = (t % 1) * 24
    qwd = _pg(h, [(10, 2.5, .9), (20, 2.5, .6)], .35)
    qwe = _pg(h, [(11.5, 3, .7), (20.5, 2.5, .6)], .40)
    cwd = _pg(h, [(11, 2.8, .7), (21, 2.8, .5)], .55)
    cwe = _pg(h, [(12.5, 3.2, .5), (21.5, 2.8, .5)], .60)
    return (np.where(we, qwe, qwd), np.where(we, cwe, cwd))

def _rain(rng, n, dt, r1, r2):
    q = np.zeros(n)
    f = np.zeros(n)
    lab = np.zeros(n, int)
    t = np.arange(n) * dt
    d = n * dt
    def add(st, k):
        dur, peak, ff = ((rng.uniform(.25, 1), rng.uniform(.25, 1.1) * Q_MEAN, rng.uniform(.3, .8))
                         if k == 1 else
                         (rng.uniform(.5, 1.5), rng.uniform(1.3, 2.4) * Q_MEAN, rng.uniform(.9, 1.8)))
        rise, tau = rng.uniform(.03, .08), dur / 3
        s = t - st
        sp = np.clip(s, 0, None)
        hyd = np.where(s < 0, 0, np.where(s < rise, sp / rise, np.exp(-np.clip(s - rise, 0, None) / tau)))
        hyd *= 1 + .15 * np.sin(2 * np.pi * s / rng.uniform(.1, .3))
        hyd = np.clip(hyd, 0, None)
        pulse = np.where(s < 0, 0, np.exp(-sp / .06) * (1 - np.exp(-sp / .01)))
        return peak * hyd, ff * pulse, hyd > .05
    for k, rate in ((1, r1), (2, r2)):
        if rate <= 0 or d < 2.1: continue
        cnt = rng.poisson(rate * (d - 2))
        if cnt > 0:
            for st in rng.uniform(2, d, cnt):
                qr, p, m = add(st, k)
                q += qr; f += p; lab[m] = np.maximum(lab[m], k)
    return q, f, lab

def generate_influent(n=8000, seed=42, rain_rate=.18, storm_rate=.03, start_wd=0, noise=1, Tin_mean=15, Tin_amp=5):
    """
    生成 BSM1 风格进水 + 温度。Tin_mean: 平均进水温度，Tin_amp: 季节波动振幅。
    返回：data [n, 16+1]（多加了温度列），labels dict（weather, q_dry, T_influent）。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    we = ((np.floor(t).astype(int) + start_wd) % 7) >= 5
    qf, cf = _diurnal(t, we)
    dr = 1 + .04 * np.sin(2 * np.pi * t / 45 + rng.uniform(0, 6.28))
    nz = 1 + noise * _ar1(rng, n, .85, .03)
    q_dry = Q_MEAN * qf * dr * nz
    q_rain, ff, lab = _rain(rng, n, DT, rain_rate, storm_rate)
    q = q_dry + q_rain
    dil = q_dry / q
    data = np.zeros((n, 17))
    data[:, 0] = t
    ci = {c: i for i, c in enumerate(COLS)}
    for nm in ["S_I", "S_S", "X_I", "X_S", "X_BH", "S_NH", "S_ND", "X_ND", "S_ALK"]:
        mean = MEAN_CONC[nm]
        nz2 = 1 + noise * _ar1(rng, n, .7, .05)
        if nm == "S_ALK":
            c_dry = mean * (1 + .05 * (cf - 1)) * nz2
        elif nm == "S_I":
            c_dry = mean * (1 + .25 * (cf - 1)) * nz2
        else:
            c_dry = mean * cf * nz2
        load_dry = c_dry * q_dry
        if nm in PARTICULATE:
            load_ff = mean * ff * Q_MEAN * 2
        elif nm == "S_S":
            load_ff = mean * ff * Q_MEAN * .5
        else:
            load_ff = 0
        data[:, ci[nm]] = np.clip((load_dry + load_ff) / q, 0, None)
    for nm in ["X_BA", "X_P", "S_O", "S_NO"]:
        data[:, ci[nm]] = 0
    data[:, ci["TSS"]] = .75 * data[:, [ci[x] for x in ("X_S","X_I","X_BH","X_BA","X_P")]].sum(1)
    data[:, ci["Q"]] = q
    # 温度：季节 + 工业热废水（雨水冲淡）
    T = Tin_mean + Tin_amp * np.sin(2 * np.pi * t / 365.25) + noise * _ar1(rng, n, .9, .8) * dil
    data[:, 16] = T
    labels = {"weather": lab, "q_dry": q_dry, "T_influent": T}
    return data, labels

def save_influent(data, labels, out_dir="data", name="Inf_synth", meta=None):
    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, f"{name}.txt")
    np.savetxt(txt, data[:, :16], fmt="%.8e", delimiter="\t")
    lab = os.path.join(out_dir, f"{name}_labels.csv")
    arr = np.column_stack([data[:, 0], labels["weather"], labels["q_dry"], labels["T_influent"]])
    np.savetxt(lab, arr, delimiter=",", header="time,weather,q_dry,T_influent", comments="", fmt=["%.8f", "%d", "%.4f", "%.4f"])
    if meta:
        with open(os.path.join(out_dir, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return txt, lab

def load_influent(path):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 16:
        raise ValueError(f"expected 16 columns, got {data.shape}")
    return {c: data[:, i] for i, c in enumerate(COLS)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rain-rate", type=float, default=.18)
    ap.add_argument("--storm-rate", type=float, default=.03)
    ap.add_argument("--out", default="data")
    ap.add_argument("--Tin-mean", type=float, default=15)
    ap.add_argument("--Tin-amp", type=float, default=5)
    a = ap.parse_args()
    d, lab = generate_influent(a.n, a.seed, a.rain_rate, a.storm_rate, Tin_mean=a.Tin_mean, Tin_amp=a.Tin_amp)
    ci = {c: i for i, c in enumerate(COLS)}
    def mean(k): return float(d[:, ci[k]].mean())
    cod = d[:, [ci[x] for x in ("S_I","S_S","X_I","X_S","X_BH")]].sum(1)
    stats = {"n_steps": len(d), "days": round(float(d[-1, 0]), 2), "Q_mean": mean("Q"), "Q_max": float(d[:, ci["Q"]].max()),
             "S_NH_mean": mean("S_NH"), "COD_mean": float(cod.mean()), "TSS_mean": mean("TSS"),
             "T_mean": float(lab["T_influent"].mean()), "T_min": float(lab["T_influent"].min()),
             "T_max": float(lab["T_influent"].max()), "rain_frac": float((lab["weather"] == 1).mean()),
             "storm_frac": float((lab["weather"] == 2).mean())}
    meta = {"seed": a.seed, "rain_rate": a.rain_rate, "storm_rate": a.storm_rate, "columns": COLS, **stats}
    txt, lab_path = save_influent(d, lab, a.out, "Inf_synth", meta)
    print("saved:", txt, lab_path)
    print(json.dumps(stats, indent=2))
