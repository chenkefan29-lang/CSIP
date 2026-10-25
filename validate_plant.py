"""
BSM1 + ASM1 + 10-layer Takacs settler simulator with a Gym-style interface.
Single self-contained module (NumPy only, plus SciPy for the optional LSODA reference
solver). Nothing runs on import and there is no built-in test or demo.

Plant layout (COST/IWA Benchmark Simulation Model No. 1):

  Qin -> [R1 anoxic]->[R2 anoxic]->[R3 aerobic]->[R4 aerobic]->[R5 aerobic]--+
           ^  ^                                                              |
           |  +-------------- internal recycle Qa <--------------------------+
           |                                                                 v
           +------- sludge recycle Qr <-- [10-layer Takacs settler] --> effluent Qe
                                                  |
                                                  +--> waste sludge Qw

Typical use:

    from bsm1_simulator import BSM1Env, load_influent
    env = BSM1Env(influent="data/Inf_synth.txt", controlled=("KLa5", "Q_a"))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.default_action())

Influent files use the BSM1 txt format (16 columns, 15-min sampling):
    time[d] S_I S_S X_I X_S X_BH X_BA X_P S_O S_NO S_NH S_ND X_ND S_ALK TSS Q[m3/d]

Optional helper (not called automatically): compute_steady_state(save_path=...) spins the
plant up on the mean BSM1 influent (~90 s) and caches the result, which BSM1Env then uses
as the initial plant state.

Gym interface (Gymnasium convention):
    obs, info = env.reset(seed=None, options={"start": int})
    obs, reward, terminated, truncated, info = env.step(action)

Controllable variables (physical units, bounds from the BSM1 definition):
    KLa3, KLa4, KLa5 [1/d]  aeration in reactors 3-5
    Q_a  [m3/d]             internal recycle R5 -> R1
    Q_r  [m3/d]             sludge recycle
    Q_w  [m3/d]             waste sludge flow
    Q_carb [m3/d]           external carbon dosage into R1 (400 kg COD/m3)
`controlled` selects which of them the agent sets; the others stay at BSM1 defaults.
With normalize_action=True (default) the agent works in [-1, 1].

Reward (weights configurable via reward_weights):
    r = -(w_eq*EQI + w_energy*(AE+PE) + w_carbon*carbon)/scale - w_violation*n_violations
with BSM1 definitions of EQI [kg pollution units/d] and AE, PE [kWh/d].

Modelling notes: BSM1 15 degC ASM1 parameters, constant volumes, instantaneous hydraulics,
settler feed layer configurable (default 6th from top), zero-order hold of the influent over
each 15-min step. Solvers: "imex" (default, fast: RK4 transport + local Rosenbrock-2
reactions) or "lsoda" (slow reference).
"""
import os
from dataclasses import dataclass, field

import numpy as np

DT_DAY = 1.0 / 96.0  # one influent sample = 15 min

HERE = os.path.dirname(os.path.abspath(__file__))
STEADY_STATE_FILE = os.path.join(HERE, "data", "bsm1_steady_state.npz")

# BSM1 dry-weather mean influent (used only by compute_steady_state)
MEAN_INFLUENT = {
    "S_I": 30.0, "S_S": 69.5, "X_I": 51.2, "X_S": 202.32, "X_BH": 28.17,
    "X_BA": 0.0, "X_P": 0.0, "S_O": 0.0, "S_NO": 0.0, "S_NH": 31.56,
    "S_ND": 6.95, "X_ND": 10.59, "S_ALK": 7.0,
}
Q_MEAN = 18446.0  # m3/d


def load_influent(path):
    """Load a BSM1-format txt file (official or synthetic). Returns array [n, 16]."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 16:
        raise ValueError(f"Expected 16 columns, got shape {data.shape}")
    return data


COMP = ["S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P",
        "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK"]
SOL = np.array([0, 1, 7, 8, 9, 10, 12])        # soluble components
PART = np.array([2, 3, 4, 5, 6, 11])           # particulate components
TSS_IDX = np.array([3, 2, 4, 5, 6])            # X_S, X_I, X_BH, X_BA, X_P
I = {n: i for i, n in enumerate(COMP)}

N_R = 5
N_L = 10
N_SOL = len(SOL)
IDX_R = slice(0, N_R * 13)
IDX_S = slice(N_R * 13, N_R * 13 + N_L * N_SOL)
IDX_X = slice(N_R * 13 + N_L * N_SOL, N_R * 13 + N_L * N_SOL + N_L)
N_STATE = N_R * 13 + N_L * N_SOL + N_L

# default BSM1 operating point
KLA_DEFAULT = np.array([0.0, 0.0, 240.0, 240.0, 84.0])
QA_DEFAULT = 55338.0
QR_DEFAULT = 18446.0
QW_DEFAULT = 385.0
CARBON_CONC = 400000.0   # g COD / m3 of external carbon source


@dataclass
class ASM1Params:
    mu_H: float = 4.0
    K_S: float = 10.0
    K_OH: float = 0.2
    K_NO: float = 0.5
    b_H: float = 0.3
    eta_g: float = 0.8
    eta_h: float = 0.8
    k_h: float = 3.0
    K_X: float = 0.1
    mu_A: float = 0.5
    K_NH: float = 1.0
    b_A: float = 0.05
    K_OA: float = 0.4
    k_a: float = 0.05
    Y_A: float = 0.24
    Y_H: float = 0.67
    f_P: float = 0.08
    i_XB: float = 0.08
    i_XP: float = 0.06


@dataclass
class SettlerParams:
    area: float = 1500.0       # m2
    height: float = 4.0        # m
    feed_layer: int = 5        # 0-based from top (5 -> 6th layer from top)
    v0_max: float = 250.0      # m/d
    v0: float = 474.0          # m/d
    r_h: float = 0.000576      # m3/g
    r_p: float = 0.00286       # m3/g
    f_ns: float = 0.00228
    X_t: float = 3000.0        # g/m3 threshold


@dataclass
class PlantParams:
    volumes: np.ndarray = field(
        default_factory=lambda: np.array([1000.0, 1000.0, 1333.0, 1333.0, 1333.0]))
    SO_sat: float = 8.0
    asm: ASM1Params = field(default_factory=ASM1Params)
    settler: SettlerParams = field(default_factory=SettlerParams)


# ----------------------------------------------------------------------
# initial state (approximate BSM1 steady state; refined by spin-up)
# ----------------------------------------------------------------------
def default_initial_state():
    r1 = [30, 2.81, 1149, 82.1, 2552, 148, 449, 0.0043, 5.37, 7.92, 1.22, 5.28, 4.93]
    r5 = [30, 0.889, 1149, 64.9, 2559, 149, 452, 0.491, 10.4, 1.73, 0.688, 3.88, 4.13]
    R = np.array([r1, r1, r5, r5, r5], dtype=float)
    S = np.tile(R[4, SOL], (N_L, 1))
    X = np.array([12.5, 18.1, 29.5, 68.2, 356.1, 356.1, 356.1, 356.1, 356.1, 6393.0])
    return np.concatenate([R.ravel(), S.ravel(), X])


class BSM1Plant:
    def __init__(self, params=None, dt_int=1.0 / 1440.0):
        self.p = params or PlantParams()
        self.dt_int = dt_int
        self.y = default_initial_state()
        self._f = self.p.settler.feed_layer
        self._idx9 = np.arange(N_L - 1)

    # ---------------- state helpers ----------------
    def set_state(self, y):
        self.y = np.array(y, dtype=float).copy()

    def get_state(self):
        return self.y.copy()

    @staticmethod
    def unpack(y):
        R = y[IDX_R].reshape(N_R, 13)
        S = y[IDX_S].reshape(N_L, N_SOL)
        X = y[IDX_X]
        return R, S, X

    # ---------------- model ----------------
    def _reactions(self, R):
        a = self.p.asm
        eps = 1e-9
        SS = R[:, 1]; XS = R[:, 3]; XBH = R[:, 4]; XBA = R[:, 5]
        SO = R[:, 7]; SNO = R[:, 8]; SNH = R[:, 9]; SND = R[:, 10]; XND = R[:, 11]

        f_ss = SS / (a.K_S + SS)
        f_o = SO / (a.K_OH + SO)
        f_ox = a.K_OH / (a.K_OH + SO)
        f_no = SNO / (a.K_NO + SNO)

        r1 = a.mu_H * f_ss * f_o * XBH
        r2 = a.mu_H * f_ss * f_ox * f_no * a.eta_g * XBH
        r3 = a.mu_A * (SNH / (a.K_NH + SNH)) * (SO / (a.K_OA + SO)) * XBA
        r4 = a.b_H * XBH
        r5 = a.b_A * XBA
        r6 = a.k_a * SND * XBH
        xr = XS / (XBH + eps)
        r7 = a.k_h * (xr / (a.K_X + xr)) * (f_o + a.eta_h * f_ox * f_no) * XBH
        r8 = r7 * XND / (XS + eps)

        YH, YA, fP, iXB, iXP = a.Y_H, a.Y_A, a.f_P, a.i_XB, a.i_XP
        out = np.zeros_like(R)
        out[:, 1] = -(r1 + r2) / YH + r7
        out[:, 3] = (1 - fP) * (r4 + r5) - r7
        out[:, 4] = r1 + r2 - r4
        out[:, 5] = r3 - r5
        out[:, 6] = fP * (r4 + r5)
        out[:, 7] = -(1 - YH) / YH * r1 - (4.57 - YA) / YA * r3
        out[:, 8] = -(1 - YH) / (2.86 * YH) * r2 + r3 / YA
        out[:, 9] = -iXB * (r1 + r2) - (iXB + 1.0 / YA) * r3 + r6
        out[:, 10] = -r6 + r8
        out[:, 11] = (iXB - fP * iXP) * (r4 + r5) - r8
        out[:, 12] = (-iXB / 14.0 * r1
                      + ((1 - YH) / (14.0 * 2.86 * YH) - iXB / 14.0) * r2
                      - (iXB / 14.0 + 1.0 / (7.0 * YA)) * r3
                      + r6 / 14.0)
        return out

    def _takacs(self, X, Xmin):
        s = self.p.settler
        xe = np.maximum(X - Xmin, 0.0)
        v = s.v0 * (np.exp(-s.r_h * xe) - np.exp(-s.r_p * xe))
        return np.clip(v, 0.0, s.v0_max)

    def _rhs(self, y, u, react=True):
        p = self.p
        st = p.settler
        f = self._f
        h = st.height / N_L
        A = st.area

        Qin, zin, kla, Qa, Qr, Qw, Qc = u
        R, S, X = self.unpack(y)
        Rp = np.maximum(R, 0.0)
        Xp = np.maximum(X, 0.0)

        z5 = Rp[4]
        Qf = Qin + Qr
        Qu = Qr + Qw
        Qe = Qf - Qu
        vup = Qe / A
        vdn = Qu / A
        Xf = 0.75 * z5[TSS_IDX].sum()
        Xf_s = max(Xf, 1e-6)

        # underflow composition (recycle stream)
        zu = np.empty(13)
        zu[SOL] = np.maximum(S[N_L - 1], 0.0)
        zu[PART] = z5[PART] * Xp[N_L - 1] / Xf_s

        # ---- reactors ----
        Q1 = Qin + Qa + Qr
        V = p.volumes
        inflow = np.empty((N_R, 13))
        inflow[0] = (Qin * zin + Qa * z5 + Qr * zu) / V[0]
        inflow[0, 1] += Qc * CARBON_CONC / V[0]
        inflow[1:] = Q1 * Rp[:-1] / V[1:, None]
        dR = inflow - (Q1 / V)[:, None] * Rp
        if react:
            dR += self._react_rates(Rp, kla)

        # ---- settler: TSS ----
        Xmin = st.f_ns * Xf
        vs = self._takacs(Xp, Xmin)
        Fs = vs * Xp
        Jn = np.minimum(Fs[:-1], Fs[1:])
        Jc = np.where(Xp[1:] <= st.X_t, Fs[:-1], Jn)
        Js = np.zeros(N_L)
        Js[:-1] = np.where(self._idx9 < f, Jc, Jn)
        Jin = np.concatenate(([0.0], Js[:-1]))

        adv = np.empty(N_L)
        adv[:f] = vup * (Xp[1:f + 1] - Xp[:f])
        adv[f] = Qf * Xf / A - (vup + vdn) * Xp[f]
        adv[f + 1:] = vdn * (Xp[f:N_L - 1] - Xp[f + 1:])
        dX = (adv + Jin - Js) / h

        # ---- settler: soluble components ----
        Sp = np.maximum(S, 0.0)
        advS = np.empty_like(Sp)
        advS[:f] = vup * (Sp[1:f + 1] - Sp[:f])
        advS[f] = Qf * z5[SOL] / A - (vup + vdn) * Sp[f]
        advS[f + 1:] = vdn * (Sp[f:N_L - 1] - Sp[f + 1:])
        dS = advS / h

        return np.concatenate([dR.ravel(), dS.ravel(), dX])

    # ---------------- stepping ----------------
    _GAMMA = 1.0 + 1.0 / np.sqrt(2.0)

    def _react_rates(self, Rp, kla):
        """ASM1 reaction rates + aeration for each reactor row. Rp: (n,13), kla: (n,)."""
        out = self._reactions(Rp)
        out[:, 7] += kla * (self.p.SO_sat - Rp[:, 7])
        return out

    def _ros2_reactions(self, R, kla, h):
        """Local (per reactor) L-stable Rosenbrock-2 step for the stiff reaction part."""
        g = self._GAMMA
        Rp = np.maximum(R, 0.0)
        f0 = self._react_rates(Rp, kla)
        delta = 1e-6 * np.maximum(Rp, 1.0)                       # (5,13)
        Rpert = np.repeat(Rp[None], 13, axis=0)                   # (13,5,13)
        ar = np.arange(13)
        Rpert[ar, :, ar] += delta.T
        fp = self._react_rates(Rpert.reshape(13 * N_R, 13), np.tile(kla, 13))
        fp = fp.reshape(13, N_R, 13)
        J = np.transpose(fp - f0[None], (1, 2, 0)) / delta[:, None, :]   # (5,13,13)
        Minv = np.linalg.inv(np.eye(13)[None] - g * h * J)
        k1 = np.einsum("rij,rj->ri", Minv, f0)
        f1 = self._react_rates(np.maximum(Rp + h * k1, 0.0), kla)
        k2 = np.einsum("rij,rj->ri", Minv, f1 - 2.0 * k1)
        return Rp + 1.5 * h * k1 + 0.5 * h * k2

    def _rk4_transport(self, y, u, h):
        k1 = self._rhs(y, u, react=False)
        k2 = self._rhs(y + 0.5 * h * k1, u, react=False)
        k3 = self._rhs(y + 0.5 * h * k2, u, react=False)
        k4 = self._rhs(y + h * k3, u, react=False)
        return np.maximum(y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)

    def advance(self, dt, Qin, zin, kla, Qa, Qr, Qw, Qc=0.0, solver="imex"):
        """
        Integrate for dt [d] with held inputs. zin: 13-vector in COMP order.

        solver="imex"  : fast. Strang splitting: explicit RK4 for transport/settler,
                         local Rosenbrock-2 for the stiff ASM1 reactions (default).
        solver="lsoda" : slow reference using scipy's stiff LSODA (tight tolerances).
        """
        kla = np.asarray(kla, dtype=float)
        u = (Qin, np.asarray(zin, dtype=float), kla, Qa, Qr, Qw, Qc)
        if solver == "lsoda":
            from scipy.integrate import solve_ivp
            sol = solve_ivp(lambda t, y: self._rhs(y, u), (0.0, dt), self.y,
                            method="LSODA", rtol=1e-7, atol=1e-8)
            self.y = np.maximum(sol.y[:, -1], 0.0)
        else:
            n = max(1, int(round(dt / self.dt_int)))
            h = dt / n
            y = self.y
            y = self._rk4_transport(y, u, 0.5 * h)
            for i in range(n):
                R = y[IDX_R].reshape(N_R, 13)
                y = y.copy()
                y[IDX_R] = self._ros2_reactions(R, kla, h).ravel()
                y = self._rk4_transport(y, u, h if i < n - 1 else 0.5 * h)
            self.y = y
        return self.measurements(Qin, Qr, Qw)

    # ---------------- outputs ----------------
    def effluent(self, y=None):
        y = self.y if y is None else y
        R, S, X = self.unpack(y)
        z5 = np.maximum(R[4], 0.0)
        Xf = max(0.75 * z5[TSS_IDX].sum(), 1e-6)
        ze = np.empty(13)
        ze[SOL] = np.maximum(S[0], 0.0)
        ze[PART] = z5[PART] * max(X[0], 0.0) / Xf
        return ze

    def underflow(self, y=None):
        y = self.y if y is None else y
        R, S, X = self.unpack(y)
        z5 = np.maximum(R[4], 0.0)
        Xf = max(0.75 * z5[TSS_IDX].sum(), 1e-6)
        zu = np.empty(13)
        zu[SOL] = np.maximum(S[-1], 0.0)
        zu[PART] = z5[PART] * max(X[-1], 0.0) / Xf
        return zu

    def effluent_quality(self, ze):
        a = self.p.asm
        tss = 0.75 * ze[TSS_IDX].sum()
        cod = ze[1] + ze[0] + ze[3] + ze[2] + ze[4] + ze[5] + ze[6]
        nkj = (ze[9] + ze[10] + ze[11] + a.i_XB * (ze[4] + ze[5])
               + a.i_XP * (ze[6] + ze[2]))
        bod5 = 0.25 * (ze[1] + ze[3] + (1 - a.f_P) * (ze[4] + ze[5]))
        ntot = ze[8] + nkj
        return {"TSS_e": tss, "COD_e": cod, "SNkj_e": nkj, "BOD5_e": bod5,
                "Ntot_e": ntot, "SNH_e": ze[9], "SNO_e": ze[8]}

    def measurements(self, Qin=None, Qr=QR_DEFAULT, Qw=QW_DEFAULT):
        """Dictionary of plant variables useful as sensors / modelling targets."""
        R, S, X = self.unpack(self.y)
        ze = self.effluent()
        m = self.effluent_quality(ze)
        m["Q_e"] = (Qin - Qw) if Qin is not None else np.nan
        for k in range(N_R):
            m[f"SO_{k+1}"] = R[k, I["S_O"]]
            m[f"SNO_{k+1}"] = R[k, I["S_NO"]]
            m[f"SNH_{k+1}"] = R[k, I["S_NH"]]
            m[f"SS_{k+1}"] = R[k, I["S_S"]]
        m["MLSS_5"] = 0.75 * R[4, TSS_IDX].sum()
        m["TSS_underflow"] = X[-1]
        m["TSS_blanket_top"] = X[N_L // 2]
        return m

    # ---------------- cost indices (BSM1 definitions, per-day rates) ----------------
    def cost_rates(self, Qin, kla, Qa, Qr, Qw, Qc, ze):
        """Instantaneous effluent-quality index [kg PU/d] and energy terms [kWh/d]."""
        q = self.effluent_quality(ze)
        Qe = Qin - Qw
        eqi = (2.0 * q["TSS_e"] + q["COD_e"] + 30.0 * q["SNkj_e"]
               + 10.0 * q["SNO_e"] + 2.0 * q["BOD5_e"]) * Qe / 1000.0
        ae = self.p.SO_sat / (1.8 * 1000.0) * float(np.sum(self.p.volumes * np.asarray(kla)))
        pe = 0.004 * Qa + 0.008 * Qr + 0.05 * Qw
        carbon = 3.0 * Qc * CARBON_CONC / 1000.0   # kg COD/d * 3 (BSM1 weighting)
        return {"EQI": eqi, "AE": ae, "PE": pe, "carbon_cost": carbon}

    EFFLUENT_LIMITS = {"Ntot_e": 18.0, "COD_e": 100.0, "SNH_e": 4.0,
                       "TSS_e": 30.0, "BOD5_e": 10.0}


# ----------------------------------------------------------------------
# steady state
# ----------------------------------------------------------------------
def influent_vector(row):
    """Row of a BSM1 influent file -> (Qin, 13-vector in COMP order)."""
    return float(row[15]), np.asarray(row[1:14], dtype=float)


def steady_state(zin, Qin, days=120.0, dt_int=1.0 / 1440.0, plant=None, verbose=False):
    plant = plant or BSM1Plant(dt_int=dt_int)
    chunk = 5.0
    done = 0.0
    while done < days:
        plant.advance(chunk, Qin, zin, KLA_DEFAULT, QA_DEFAULT, QR_DEFAULT, QW_DEFAULT)
        done += chunk
        if verbose:
            m = plant.measurements(Qin)
            print(f"  day {done:5.0f}: SNH_e={m['SNH_e']:.3f} SNO_e={m['SNO_e']:.3f} "
                  f"MLSS5={m['MLSS_5']:.0f}")
    return plant.get_state()


# ----------------------------------------------------------------------
# Gym-style environment
# ----------------------------------------------------------------------
ACTIONS = {
    #  name      default          low     high
    "KLa3":   (KLA_DEFAULT[2],    0.0,    360.0),
    "KLa4":   (KLA_DEFAULT[3],    0.0,    360.0),
    "KLa5":   (KLA_DEFAULT[4],    0.0,    360.0),
    "Q_a":    (QA_DEFAULT,        0.0,    92230.0),
    "Q_r":    (QR_DEFAULT,        0.0,    36892.0),
    "Q_w":    (QW_DEFAULT,        150.0,  1844.6),
    "Q_carb": (0.0,               0.0,    5.0),
}

OBS_NAMES = ([
    "Q_in", "SNH_in", "COD_in", "TSS_in",                   # influent sensors
    "SO_3", "SO_4", "SO_5", "SNO_2", "SNH_5", "SNO_5", "MLSS_5",   # in-plant sensors
    "SNH_e", "SNO_e", "Ntot_e", "COD_e", "TSS_e", "BOD5_e",        # effluent
    "tod_sin", "tod_cos",                                          # time of day
])


class Box:
    """Minimal stand-in for gym.spaces.Box."""
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape

    def sample(self, rng=None):
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(np.float32)


class BSM1Env:
    def __init__(self, influent="data/Inf_synth.txt", labels=None,
                 controlled=("KLa3", "KLa4", "KLa5", "Q_a"),
                 normalize_action=True, episode_steps=96 * 7, warmup_steps=96,
                 dt_int=1.0 / 1440.0, solver="imex", sensor_noise=0.0,
                 reward_weights=None, seed=None):
        if isinstance(influent, str):
            path = influent if os.path.isabs(influent) else os.path.join(HERE, influent)
            influent = load_influent(path)
        self.influent = np.asarray(influent, dtype=float)
        self.labels = labels
        self.n_data = len(self.influent)
        self.controlled = tuple(controlled)
        for c in self.controlled:
            if c not in ACTIONS:
                raise ValueError(f"unknown action {c}; choose from {list(ACTIONS)}")
        self.normalize_action = normalize_action
        self.episode_steps = int(episode_steps)
        self.warmup_steps = int(warmup_steps)
        self.solver = solver
        self.sensor_noise = sensor_noise
        self.w = {"eq": 1.0, "energy": 1.0, "carbon": 1.0, "violation": 0.5, "scale": 1e4}
        if reward_weights:
            self.w.update(reward_weights)
        self.rng = np.random.default_rng(seed)
        self.plant = BSM1Plant(dt_int=dt_int)
        self._y0 = self._load_steady_state()

        lo = np.array([ACTIONS[c][1] for c in self.controlled])
        hi = np.array([ACTIONS[c][2] for c in self.controlled])
        self._lo, self._hi = lo, hi
        if normalize_action:
            self.action_space = Box(-np.ones(len(lo)), np.ones(len(lo)))
        else:
            self.action_space = Box(lo, hi)
        self.observation_space = Box(-np.inf * np.ones(len(OBS_NAMES)),
                                     np.inf * np.ones(len(OBS_NAMES)))
        self.obs_names = list(OBS_NAMES)
        self.t = 0
        self.k = 0
        self._last_action = None

    # ------------------------------------------------------------------
    @staticmethod
    def _load_steady_state(path=STEADY_STATE_FILE):
        """Cached steady state if present, else the built-in approximate initial state.
        Nothing is simulated here; call compute_steady_state() yourself to create the cache."""
        if os.path.exists(path):
            return np.load(path)["y"]
        return default_initial_state()

    # ---------------- action helpers ----------------
    def default_physical(self):
        return {k: v[0] for k, v in ACTIONS.items()}

    def default_action(self):
        """Action (in the env's action space) that reproduces the BSM1 open-loop setting."""
        return self.to_action({c: ACTIONS[c][0] for c in self.controlled})

    def to_action(self, physical):
        v = np.array([physical[c] for c in self.controlled], dtype=float)
        if self.normalize_action:
            v = 2.0 * (v - self._lo) / (self._hi - self._lo) - 1.0
        return np.clip(v, self.action_space.low, self.action_space.high).astype(np.float32)

    def to_physical(self, action):
        a = np.clip(np.asarray(action, dtype=float), self.action_space.low,
                    self.action_space.high)
        if self.normalize_action:
            a = self._lo + (a + 1.0) * 0.5 * (self._hi - self._lo)
        phys = self.default_physical()
        for name, val in zip(self.controlled, a):
            phys[name] = float(val)
        return phys

    # ---------------- observation ----------------
    def _obs(self, meas, row):
        zin = row[1:14]
        cod_in = zin[0] + zin[1] + zin[2] + zin[3] + zin[4] + zin[5] + zin[6]
        tod = 2 * np.pi * (row[0] % 1.0)
        vals = {
            "Q_in": row[15], "SNH_in": zin[9], "COD_in": cod_in, "TSS_in": row[14],
            "SO_3": meas["SO_3"], "SO_4": meas["SO_4"], "SO_5": meas["SO_5"],
            "SNO_2": meas["SNO_2"], "SNH_5": meas["SNH_5"], "SNO_5": meas["SNO_5"],
            "MLSS_5": meas["MLSS_5"],
            "SNH_e": meas["SNH_e"], "SNO_e": meas["SNO_e"], "Ntot_e": meas["Ntot_e"],
            "COD_e": meas["COD_e"], "TSS_e": meas["TSS_e"], "BOD5_e": meas["BOD5_e"],
            "tod_sin": np.sin(tod), "tod_cos": np.cos(tod),
        }
        o = np.array([vals[n] for n in self.obs_names], dtype=float)
        if self.sensor_noise > 0:
            noisy = o[:-2]
            o[:-2] = noisy * (1.0 + self.sensor_noise * self.rng.normal(size=noisy.shape))
        return o.astype(np.float32), vals

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        max_start = max(self.n_data - self.episode_steps - 1, 0)
        start = options.get("start")
        if start is None:
            start = int(self.rng.integers(0, max_start + 1))
        start = int(np.clip(start, 0, max_start))
        self.start = start
        self.t = start
        self.k = 0
        self.plant.set_state(self._y0)

        # warm-up on the preceding influent with default (open-loop) settings
        w0 = max(0, start - self.warmup_steps)
        phys = self.default_physical()
        meas = self.plant.measurements(self.influent[start, 15])
        for i in range(w0, start):
            Qin, zin = influent_vector(self.influent[i])
            meas = self.plant.advance(
                DT_DAY, Qin, zin,
                np.array([0, 0, phys["KLa3"], phys["KLa4"], phys["KLa5"]]),
                phys["Q_a"], phys["Q_r"], phys["Q_w"], phys["Q_carb"], solver=self.solver)
        self._meas = meas
        self._last_action = self.default_action()
        obs, vals = self._obs(meas, self.influent[self.t])
        return obs, {"start": start, "obs_dict": vals}

    def step(self, action):
        phys = self.to_physical(action)
        row = self.influent[self.t]
        Qin, zin = influent_vector(row)
        kla = np.array([0.0, 0.0, phys["KLa3"], phys["KLa4"], phys["KLa5"]])
        meas = self.plant.advance(DT_DAY, Qin, zin, kla, phys["Q_a"], phys["Q_r"],
                                  phys["Q_w"], phys["Q_carb"], solver=self.solver)
        ze = self.plant.effluent()
        cost = self.plant.cost_rates(Qin, kla, phys["Q_a"], phys["Q_r"], phys["Q_w"],
                                     phys["Q_carb"], ze)
        viol = {k: float(meas[k] > lim) for k, lim in self.plant.EFFLUENT_LIMITS.items()}
        n_viol = sum(viol.values())
        w = self.w
        reward = -(w["eq"] * cost["EQI"] + w["energy"] * (cost["AE"] + cost["PE"])
                   + w["carbon"] * cost["carbon_cost"]) / w["scale"] - w["violation"] * n_viol

        self._meas = meas
        self._last_action = np.asarray(action, dtype=np.float32)
        self.t += 1
        self.k += 1
        truncated = (self.k >= self.episode_steps) or (self.t >= self.n_data)
        nxt = self.influent[min(self.t, self.n_data - 1)]
        obs, vals = self._obs(meas, nxt)
        info = {"physical_action": phys, "measurements": meas, "cost": cost,
                "violations": viol, "n_violations": n_viol, "obs_dict": vals,
                "time_day": float(row[0]), "t": self.t - 1}
        if self.labels is not None:
            info["weather"] = int(self.labels[self.t - 1])
        return obs, float(reward), False, bool(truncated), info


# ----------------------------------------------------------------------
# Baseline: BSM1-style PI control (SO in R5 -> KLa5, nitrate in R2 -> Q_a)
# ----------------------------------------------------------------------
class BaselinePI:
    """
    Discrete PI loops sampled at the env step (15 min), with output clamping and
    back-calculation anti-windup. Gains are reduced vs. the 1-min BSM1 tuning because
    of the slower sampling. Requires env.controlled to contain "KLa5" and "Q_a".
    """
    def __init__(self, so_set=2.0, sno_set=1.0,
                 kp_so=15.0, ti_so=0.03, kp_no=6000.0, ti_no=0.05, ts=DT_DAY):
        self.so_set, self.sno_set = so_set, sno_set
        self.kp_so, self.ti_so, self.kp_no, self.ti_no, self.ts = kp_so, ti_so, kp_no, ti_no, ts
        self.reset()

    def reset(self):
        self.i_so = ACTIONS["KLa5"][0]
        self.i_no = ACTIONS["Q_a"][0]

    def act(self, env, obs_dict):
        # PI in "positional + integrator state" form with clamping (anti-windup)
        e_so = self.so_set - obs_dict["SO_5"]
        self.i_so += self.kp_so * self.ts / self.ti_so * e_so
        self.i_so = np.clip(self.i_so, *ACTIONS["KLa5"][1:])
        kla5 = np.clip(self.i_so + self.kp_so * e_so, *ACTIONS["KLa5"][1:])

        e_no = self.sno_set - obs_dict["SNO_2"]
        self.i_no += self.kp_no * self.ts / self.ti_no * e_no
        self.i_no = np.clip(self.i_no, *ACTIONS["Q_a"][1:])
        qa = np.clip(self.i_no + self.kp_no * e_no, *ACTIONS["Q_a"][1:])

        phys = {c: ACTIONS[c][0] for c in env.controlled}
        phys["KLa5"], phys["Q_a"] = kla5, qa
        return env.to_action(phys)


def compute_steady_state(zin=None, Qin=Q_MEAN, days=100, save_path=STEADY_STATE_FILE):
    """Explicit helper (never called automatically): spin the plant up on a constant
    influent and cache the resulting state for BSM1Env."""
    if zin is None:
        zin = np.array([MEAN_INFLUENT[c] for c in COMP])
    y = steady_state(zin, Qin, days=days)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, y=y, zin=zin, Qin=Qin)
    return y