"""
Synthetic BSM1-style influent generator (ASM1 state variables).

Output format mimics the official BSM1 influent txt files
(Inf_dry_2006.txt etc.): whitespace-delimited, no header, 16 columns,
15-minute sampling (time in days).

Columns:
 0 time [d]   1 S_I   2 S_S   3 X_I   4 X_S   5 X_BH  6 X_BA  7 X_P
 8 S_O   9 S_NO  10 S_NH  11 S_ND  12 X_ND  13 S_ALK  14 TSS  15 Q [m3/d]
(concentrations in g/m3, S_ALK in mol HCO3-/m3)

Extra output (not part of the official format):
 *_labels.csv : weather label per step (0 dry, 1 rain, 2 storm) and dry-weather flow.

Usage:
    python influent_generator.py                      # 8000 steps -> data/
    python influent_generator.py --n 5000 --seed 7
    python influent_generator.py --plot

To switch to official files later, use load_influent("Inf_dry_2006.txt");
it returns the same column layout.
"""
import argparse
import json
import os

import numpy as np

COLUMNS = ["time", "S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P",
           "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK", "TSS", "Q"]

DT_DAY = 1.0 / 96.0  # 15 minutes

# BSM1 dry-weather average influent (steady-state file values)
MEAN_CONC = {
    "S_I": 30.0, "S_S": 69.5, "X_I": 51.2, "X_S": 202.32, "X_BH": 28.17,
    "X_BA": 0.0, "X_P": 0.0, "S_O": 0.0, "S_NO": 0.0, "S_NH": 31.56,
    "S_ND": 6.95, "X_ND": 10.59, "S_ALK": 7.0,
}
Q_MEAN = 18446.0  # m3/d

PARTICULATE = ["X_I", "X_S", "X_BH"]
DILUTED = ["S_I", "S_S", "X_I", "X_S", "X_BH", "S_NH", "S_ND", "X_ND", "S_ALK"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _periodic_gauss_mix(hours, peaks, floor):
    """Smooth 24 h periodic profile: floor + wrapped gaussians, mean 1 over a full day."""
    def raw(h):
        prof = np.full_like(h, floor, dtype=float)
        for center, width, amp in peaks:
            d = (h - center + 12.0) % 24.0 - 12.0
            prof += amp * np.exp(-0.5 * (d / width) ** 2)
        return prof
    ref = np.linspace(0, 24, 96, endpoint=False)
    return raw(np.asarray(hours, dtype=float)) / raw(ref).mean()


def _ar1(rng, n, phi, sigma):
    """Zero-mean AR(1) noise with stationary std = sigma."""
    e = rng.normal(0.0, sigma * np.sqrt(1 - phi ** 2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _diurnal(t_day, weekend_mask):
    """Flow and concentration diurnal factors (weekday/weekend differ)."""
    hours = (t_day % 1.0) * 24.0
    # weekday: morning peak ~10h, evening peak ~20h, night minimum ~4h
    flow_wd = _periodic_gauss_mix(hours, [(10.0, 2.5, 0.9), (20.0, 2.5, 0.6)], 0.35)
    flow_we = _periodic_gauss_mix(hours, [(11.5, 3.0, 0.7), (20.5, 2.5, 0.6)], 0.40)
    # concentration (load) lags flow by ~1 h
    conc_wd = _periodic_gauss_mix(hours, [(11.0, 2.8, 0.7), (21.0, 2.8, 0.5)], 0.55)
    conc_we = _periodic_gauss_mix(hours, [(12.5, 3.2, 0.5), (21.5, 2.8, 0.5)], 0.60)
    flow = np.where(weekend_mask, flow_we, flow_wd)
    conc = np.where(weekend_mask, conc_we, conc_wd)
    return flow, conc


def _rain_events(rng, n, dt, rate_rain_per_day, rate_storm_per_day):
    """
    Returns rain_flow [m3/d], first_flush pulse [-], weather label [0/1/2].
    Hydrograph: fast rise, exponential decay.
    """
    q_rain = np.zeros(n)
    ff = np.zeros(n)
    label = np.zeros(n, dtype=int)
    t = np.arange(n) * dt
    total_days = n * dt

    def add_event(start, kind):
        if kind == 1:   # rain
            dur = rng.uniform(0.25, 1.0)          # days
            peak = rng.uniform(0.25, 1.1) * Q_MEAN
            ff_amp = rng.uniform(0.3, 0.8)
        else:           # storm
            dur = rng.uniform(0.5, 1.5)
            peak = rng.uniform(1.3, 2.4) * Q_MEAN
            ff_amp = rng.uniform(0.9, 1.8)
        rise = rng.uniform(0.03, 0.08)            # days
        tau = dur / 3.0
        s = t - start
        sp0 = np.clip(s, 0.0, None)
        hyd = np.where(s < 0, 0.0,
                       np.where(s < rise, sp0 / rise, np.exp(-np.clip(s - rise, 0.0, None) / tau)))
        # pulsy rain: modulate decaying hydrograph (mild)
        hyd = hyd * (1.0 + 0.15 * np.sin(2 * np.pi * s / rng.uniform(0.1, 0.3)))
        hyd = np.clip(hyd, 0, None)
        sp = np.clip(s, 0.0, None)
        pulse = np.where(s < 0, 0.0, np.exp(-sp / 0.06) * (1 - np.exp(-sp / 0.01)))
        return peak * hyd, ff_amp * pulse, (hyd > 0.05)

    for kind, rate in ((1, rate_rain_per_day), (2, rate_storm_per_day)):
        if rate <= 0:
            continue
        # poisson arrivals, leave first 2 days dry (warm-up for simulators)
        k = rng.poisson(rate * max(total_days - 2.0, 0))
        starts = rng.uniform(2.0, total_days, k)
        for st in starts:
            qr, f, mask = add_event(st, kind)
            q_rain += qr
            ff += f
            label[mask] = np.maximum(label[mask], kind)
    return q_rain, ff, label


# ----------------------------------------------------------------------
# main generator
# ----------------------------------------------------------------------
def generate_influent(n_steps=8000, seed=42, rain_rate=0.18, storm_rate=0.03,
                      start_weekday=0, noise_scale=1.0):
    """
    Generate a synthetic BSM1-like influent.

    rain_rate / storm_rate: expected events per day (default ~1 rain / 5.5 d,
    ~1 storm / 33 d). start_weekday: 0=Monday.
    Returns (data[n, 16] ndarray, labels dict).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * DT_DAY
    dow = (np.floor(t).astype(int) + start_weekday) % 7
    weekend = dow >= 5

    flow_f, conc_f = _diurnal(t, weekend)

    # slow drift (e.g. seasonal/industrial) + short-term noise
    drift_q = 1.0 + 0.04 * np.sin(2 * np.pi * t / 45.0 + rng.uniform(0, 6.28))
    q_noise = 1.0 + noise_scale * _ar1(rng, n_steps, 0.85, 0.03)
    q_dry = Q_MEAN * flow_f * drift_q * q_noise

    # rain / storm
    q_rain, ff, label = _rain_events(rng, n_steps, DT_DAY, rain_rate, storm_rate)
    q_tot = q_dry + q_rain
    dilution = q_dry / q_tot

    data = np.zeros((n_steps, 16))
    data[:, 0] = t
    cidx = {name: i for i, name in enumerate(COLUMNS)}

    for name in ["S_I", "S_S", "X_I", "X_S", "X_BH", "S_NH", "S_ND", "X_ND", "S_ALK"]:
        mean = MEAN_CONC[name]
        nz = 1.0 + noise_scale * _ar1(rng, n_steps, 0.7, 0.05)
        if name == "S_ALK":
            c_dry = mean * (1.0 + 0.05 * (conc_f - 1.0)) * nz
        elif name == "S_I":
            c_dry = mean * (1.0 + 0.25 * (conc_f - 1.0)) * nz   # inert: damped
        else:
            c_dry = mean * conc_f * nz
        load_dry = c_dry * q_dry
        if name in PARTICULATE:
            load_ff = mean * ff * Q_MEAN * 2.0                   # first flush wash-off
        elif name == "S_S":
            load_ff = mean * ff * Q_MEAN * 0.5
        else:
            load_ff = 0.0
        conc = (load_dry + load_ff) / q_tot
        data[:, cidx[name]] = np.clip(conc, 0.0, None)

    # components that are zero in the BSM1 influent
    for name in ["X_BA", "X_P", "S_O", "S_NO"]:
        data[:, cidx[name]] = 0.0

    # TSS consistent with ASM1 particulates (BSM1: 0.75 * sum of X)
    xs = ["X_S", "X_I", "X_BH", "X_BA", "X_P"]
    data[:, cidx["TSS"]] = 0.75 * sum(data[:, cidx[x]] for x in xs)
    data[:, cidx["Q"]] = q_tot

    labels = {"weather": label, "q_dry": q_dry}
    return data, labels


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def save_influent(data, labels, out_dir="data", name="Inf_synth", meta=None):
    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, f"{name}.txt")
    np.savetxt(txt, data, fmt="%.8e", delimiter="\t")   # official-like: no header
    lab = os.path.join(out_dir, f"{name}_labels.csv")
    arr = np.column_stack([data[:, 0], labels["weather"], labels["q_dry"]])
    np.savetxt(lab, arr, delimiter=",", header="time,weather,q_dry",
               comments="", fmt=["%.8f", "%d", "%.4f"])
    if meta:
        with open(os.path.join(out_dir, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return txt, lab


def load_influent(path):
    """Load a BSM1-format txt (official or synthetic). Returns ndarray [n,16]."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 16:
        raise ValueError(f"Expected 16 columns, got shape {data.shape}")
    return data


def sanity_check(data):
    ci = {n: i for i, n in enumerate(COLUMNS)}
    dt = np.diff(data[:, 0])
    assert np.allclose(dt, DT_DAY, atol=1e-6), "time step is not 15 min"
    assert (data[:, 1:] >= 0).all(), "negative values"
    tss = 0.75 * data[:, [ci[x] for x in ("X_S", "X_I", "X_BH", "X_BA", "X_P")]].sum(1)
    assert np.allclose(tss, data[:, ci["TSS"]], rtol=1e-6), "TSS inconsistent"
    return {
        "n_steps": len(data),
        "days": round(float(data[-1, 0]), 2),
        "Q_mean": float(data[:, ci["Q"]].mean()),
        "Q_max": float(data[:, ci["Q"]].max()),
        "S_NH_mean": float(data[:, ci["S_NH"]].mean()),
        "COD_mean": float(data[:, [ci[x] for x in ("S_I", "S_S", "X_I", "X_S", "X_BH")]].sum(1).mean()),
        "TSS_mean": float(data[:, ci["TSS"]].mean()),
    }


def plot_influent(data, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ci = {n: i for i, n in enumerate(COLUMNS)}
    t = data[:, 0]
    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(t, data[:, ci["Q"]], lw=0.7)
    ax[0].set_ylabel("Q [m3/d]")
    ax[1].plot(t, data[:, ci["S_NH"]], lw=0.7, color="C1")
    ax[1].set_ylabel("S_NH [g/m3]")
    cod = data[:, [ci[x] for x in ("S_I", "S_S", "X_I", "X_S", "X_BH")]].sum(1)
    ax[2].plot(t, cod, lw=0.7, color="C2")
    ax[2].set_ylabel("COD [g/m3]")
    ax[3].plot(t, data[:, ci["TSS"]], lw=0.7, color="C3")
    ax[3].set_ylabel("TSS [g/m3]")
    ax[3].set_xlabel("time [d]")
    w = labels["weather"]
    for a in ax:
        a.fill_between(t, 0, 1, where=(w == 1), color="tab:blue", alpha=0.12,
                       transform=a.get_xaxis_transform())
        a.fill_between(t, 0, 1, where=(w == 2), color="tab:red", alpha=0.18,
                       transform=a.get_xaxis_transform())
    fig.suptitle("Synthetic BSM1-style influent (blue: rain, red: storm)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="number of 15-min steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rain-rate", type=float, default=0.18, help="rain events / day")
    ap.add_argument("--storm-rate", type=float, default=0.03, help="storm events / day")
    ap.add_argument("--out", default="data")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    d, lab = generate_influent(a.n, a.seed, a.rain_rate, a.storm_rate)
    stats = sanity_check(d)
    stats["rain_fraction"] = float((lab["weather"] == 1).mean())
    stats["storm_fraction"] = float((lab["weather"] == 2).mean())
    meta = {"seed": a.seed, "rain_rate": a.rain_rate, "storm_rate": a.storm_rate,
            "columns": COLUMNS, **stats}
    txt, lab_path = save_influent(d, lab, a.out, "Inf_synth", meta)
    print("saved:", txt, lab_path)
    print(json.dumps(stats, indent=2))
    if a.plot:
        p = os.path.join(a.out, "Inf_synth.png")
        plot_influent(d, lab, p)
        print("plot:", p)
