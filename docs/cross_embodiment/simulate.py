"""Cross-embodiment study of "weigh by joint torque", with a realistic error budget (standalone).

Question: how well does the estimator of `src/robotokens/sensing/payload.py` work on different arms, once
the error sources that do NOT average away are included?

Arms: OpenArm v1, UR5e, Franka FR3 (real MJCF models, `cross_embodiment_openarm_ur5e_fr3.xml` from MuJoCo
Menagerie). Ground-truth gravity comes from the unmodified physics model plus a point-mass payload at the
(randomly offset) grasp point. The estimator does what payload.py does: dither the load-bearing joints
(1 deg, 3 cycles of 2 s), subtract the arm's gravity, least-squares mass from tau_res ~ J_z m g at the
NOMINAL tool point, standard error from the cycle-to-cycle spread, compare with z = diff / combined SE.

Error budget (per arm, see SENSORS; the values are ASSUMPTIONS from the sensing principle, not measured):
  * per-measurement torque offset (thermal drift, hysteresis, cable drag): constant during one weighing,
    so the dither cannot remove it and the 3-cycle SE does not see it;
  * asymmetric Coulomb friction (dither cancels only the symmetric part) + viscous friction;
  * torque ripple / cogging that depends on the joint angle, with 0.2 deg pose repeatability;
  * white sensor noise;
  * grasp position on the can (+-8 mm sideways, +-10 mm along the tool), unknown to the estimator;
  * 3 % mass error in the arm model used for gravity compensation ("model" estimator).
Two estimators: "model" (subtract model gravity, what the robot code does today) and "tare" (subtract an
empty-gripper weighing at the same pose, a proposed improvement).
Two decision rules: "today" (z from the 3-cycle SE, as payload.py) and "floor" (the SE is combined with the
spread of 5 repeated weighings of one can, as bring-up stage 5 already measures: a proposed improvement).

Usage:  .venv/bin/python docs/cross_embodiment/simulate.py [--trials 300]
Writes: docs/figures/cross_embodiment_results.png and docs/cross_embodiment/results.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "third_party/mujoco_menagerie/cross_embodiment_openarm_ur5e_fr3.xml"
FIG = ROOT / "docs/figures"

G = 9.81
MIN_LEVER_M = 0.05
RATE_HZ, PERIOD_S, CYCLES, DITHER_DEG = 100, 2.0, 3, 1.0  # same defaults as configs/robotokens.yaml
CALIB_WEIGHINGS = 5  # repeated weighings of one can (bring-up stage 5) to get the repeatability floor
N_S = 40  # gravity samples per dither cycle (white noise is scaled to the 200 real samples)
MIN_Z = 3.0
BASE_KG = 0.100  # the lighter can
DELTAS_KG = (0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.250)  # heavier can = BASE + delta
MODEL_MASS_ERR = 0.03
GRASP_SD_M = (0.008, 0.008, 0.010)  # grasp point offset in the tool frame (x, y sideways; z along the tool)
POSE_SD_DEG = 0.2  # weigh-pose repeatability
FRICTION_ASYM = 0.25  # |Coulomb(+) - Coulomb(-)| / mean, per joint
VISCOUS = 0.3  # Nm / (rad/s)

# Assumed sensor characteristics (Nm) by sensing principle. Sources of the ORDER of magnitude only:
# Damiao QDD motors report torque from phase current; UR e-Series joints estimate torque from current through
# ~100:1 harmonic drives (large, load-dependent friction); Franka arms have link-side joint torque sensors.
SENSORS = {
    "OpenArm v1": dict(kind="motor current, low-ratio planetary", white=0.05, offset=0.04, coulomb=0.15, ripple=0.04),
    "UR5e": dict(kind="motor current, ~100:1 harmonic drive", white=0.15, offset=0.25, coulomb=1.5, ripple=0.20),
    "Franka FR3": dict(kind="link-side joint torque sensors", white=0.03, offset=0.05, coulomb=0.10, ripple=0.02),
}
ARMS = {
    "OpenArm v1": dict(site="left_tcp", joints=[f"openarm_left_joint{i}" for i in range(1, 8)]),
    "UR5e": dict(site="gripper_ur5e_left_pinch",
                 joints=["ur5e_left_" + n + "_joint" for n in
                         ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]),
    "Franka FR3": dict(site="gripper_fr3_left_pinch", joints=[f"fr3_left_fr3_joint{i}" for i in range(1, 8)]),
}
COLORS = {"OpenArm v1": "#2a6fdb", "UR5e": "#e08a1e", "Franka FR3": "#2a9d5b"}


def load_models():
    spec = mujoco.MjSpec.from_file(str(XML))
    spec.body("openarm_left_hand_tcp").add_site(name="left_tcp", size=[0.01, 0, 0], group=4)  # true TCP
    truth = spec.compile()
    est = spec.compile()  # the estimator's model, with a mass error
    est.body_mass[:] *= 1 + MODEL_MASS_ERR
    est.body_inertia[:] *= 1 + MODEL_MASS_ERR
    dt, de = mujoco.MjData(truth), mujoco.MjData(est)
    mujoco.mj_setConst(est, de)  # body_subtreemass is cached: refresh after editing masses
    for m, d in ((truth, dt), (est, de)):
        mujoco.mj_resetDataKeyframe(m, d, 0)
    return truth, dt, est, de


class Arm:
    """One embodiment: its joints, nominal weigh pose (scene keyframe) and fixed 'hardware' imperfections."""

    def __init__(self, name, truth, dt, est, de, rng):
        self.name, self.truth, self.dt, self.est, self.de = name, truth, dt, est, de
        cfg, self.s = ARMS[name], SENSORS[name]
        jt = [truth.joint(j) for j in cfg["joints"]]
        self.qadr = np.array([j.qposadr[0] for j in jt])
        self.dadr = np.array([j.dofadr[0] for j in jt])
        self.sid = truth.site(cfg["site"]).id
        self.body = truth.site_bodyid[self.sid]
        self.q0 = dt.qpos[self.qadr].copy()
        n = len(self.qadr)
        _, jz = self.model_gravity(self.q0)
        self.used = [i for i in range(n) if abs(jz[i]) > MIN_LEVER_M]
        self.amp = np.zeros(n)
        self.amp[self.used] = np.radians(DITHER_DEG)
        self.asym = rng.uniform(-FRICTION_ASYM, FRICTION_ASYM, n)  # fixed per joint (this robot)
        self.rip_k = rng.uniform(30, 80, n)  # ripple spatial frequency, rad^-1
        self.rip_phi = rng.uniform(0, 2 * np.pi, n)
        ph = 2 * np.pi * np.arange(N_S) / N_S
        self.sin, self.cos = np.sin(ph), np.cos(ph)
        self.white = self.s["white"] / math.sqrt(RATE_HZ * PERIOD_S / N_S)

    def model_gravity(self, q):
        """What the estimator knows: model gravity torque and J_z at the nominal tool point."""
        self.de.qpos[self.qadr] = q
        mujoco.mj_forward(self.est, self.de)
        jac = np.zeros((3, self.est.nv))
        mujoco.mj_jacSite(self.est, self.de, jac, None, self.sid)
        return self.de.qfrc_bias[self.dadr].copy(), jac[2, self.dadr].copy()

    def true_torques(self, q, m, grasp):
        """Ground truth: exact arm gravity + point mass m at the offset grasp point."""
        d = self.dt
        d.qpos[self.qadr] = q
        mujoco.mj_forward(self.truth, d)
        tau = d.qfrc_bias[self.dadr].copy()
        if m > 0:
            p = d.site_xpos[self.sid] + d.site_xmat[self.sid].reshape(3, 3) @ grasp
            jac = np.zeros((3, self.truth.nv))
            mujoco.mj_jac(self.truth, d, jac, None, p, self.body)
            tau += jac[2, self.dadr] * m * G
        return tau

    def weigh(self, m, rng):
        """One weighing: returns (per-cycle mean torques (CYCLES, n), the pose actually reached)."""
        n = len(self.qadr)
        q = self.q0 + rng.normal(0, np.radians(POSE_SD_DEG), n)
        grasp = rng.normal(0, GRASP_SD_M) if m > 0 else np.zeros(3)
        offset = rng.normal(0, self.s["offset"], n)  # this weighing's drift: constant over its cycles
        qs = q + self.amp * self.sin[:, None]
        dq = self.amp * (2 * np.pi / PERIOD_S) * self.cos[:, None]
        grav = np.array([self.true_torques(qk, m, grasp) for qk in qs])
        fric = self.s["coulomb"] * (np.sign(dq) + self.asym) * (self.amp > 0) + VISCOUS * dq
        ripple = self.s["ripple"] * np.sin(self.rip_k * qs + self.rip_phi)
        base = grav + fric + ripple + offset
        cyc = np.array([(base + rng.normal(0, self.white, base.shape)).mean(axis=0) for _ in range(CYCLES)])
        return cyc, q

    def estimate(self, cyc, q, reference):
        """payload.py recipe: per-cycle least squares at the nominal tool point; SE from the cycle spread."""
        _, jz = self.model_gravity(q)
        j = self.used
        ms = [float((c - reference)[j] @ jz[j] / (G * jz[j] @ jz[j])) for c in cyc]
        return np.mean(ms), np.std(ms, ddof=1) / math.sqrt(len(ms))


def run(trials: int, seed: int) -> dict:
    truth, dt, est, de = load_models()
    rng = np.random.default_rng(seed)
    out = {}
    for name in ARMS:
        arm = Arm(name, truth, dt, est, de, rng)
        rec = {"model": {}, "tare": {}}
        for k in ("model", "tare"):
            rec[k] = {"A": [], **{d: [] for d in DELTAS_KG}}
        sd_rep = []  # per trial: the floor a robot would measure with CALIB_WEIGHINGS repeated weighings
        for _ in range(trials):
            calib = [arm.estimate(*arm.weigh(BASE_KG, rng), arm.model_gravity(arm.q0)[0])[0]
                     for _ in range(CALIB_WEIGHINGS)]
            sd_rep.append(float(np.std(calib, ddof=1)))
            tare_cyc, _ = arm.weigh(0.0, rng)
            tare_ref = tare_cyc.mean(axis=0)
            for key, m in [("A", BASE_KG)] + [(d, BASE_KG + d) for d in DELTAS_KG]:
                cyc, q = arm.weigh(m, rng)
                rec["model"][key].append(arm.estimate(cyc, q, arm.model_gravity(q)[0]))
                rec["tare"][key].append(arm.estimate(cyc, q, tare_ref))
        out[name] = {k: {kk: np.array(v) for kk, v in d.items()} for k, d in rec.items()}
        out[name]["joints"] = len(arm.used)
        out[name]["sd_rep"] = np.array(sd_rep)
        print(f"{name}: {trials} trials done")
    return out


def metrics(out: dict) -> dict:
    res = {}
    for name, o in out.items():
        r = {"joints": o["joints"], "sweep": {}}
        A = o["model"]["A"]
        for d in DELTAS_KG:
            B = o["model"][d]  # decisions use the robot's own estimator (model gravity)
            diff, z = B[:, 0] - A[:, 0], (B[:, 0] - A[:, 0]) / np.hypot(A[:, 1], B[:, 1])
            zf = diff / np.sqrt(A[:, 1] ** 2 + B[:, 1] ** 2 + 2 * o["sd_rep"] ** 2)
            r["sweep"][d] = dict(correct=float((diff > 0).mean()), act_right=float((z >= MIN_Z).mean()),
                                 act_wrong=float((z <= -MIN_Z).mean()), diff_sd=float(diff.std() * 1e3),
                                 act_right_f=float((zf >= MIN_Z).mean()), act_wrong_f=float((zf <= -MIN_Z).mean()))
        for est in ("model", "tare"):
            for key, true in (("A", BASE_KG), (DELTAS_KG[-1], BASE_KG + DELTAS_KG[-1])):
                e = (o[est][key][:, 0] - true) * 1e3
                r[f"{est}_{int(round(true * 1e3))}"] = dict(bias=float(e.mean()), sd=float(e.std()),
                                                             rmse=float(np.sqrt((e ** 2).mean())))
        t = o["tare"][DELTAS_KG[-1]]  # does the robot's own SE cover the real error?
        r["coverage_2se"] = float((np.abs(t[:, 0] - (BASE_KG + DELTAS_KG[-1])) <= 2 * t[:, 1]).mean())
        r["median_se_g"] = float(np.median(o["model"][DELTAS_KG[-1]][:, 1]) * 1e3)
        r["sd_rep_g"] = float(np.median(o["sd_rep"]) * 1e3)
        res[name] = r
    return res


def resolution(r, level=0.95, key="act_right"):
    """Smallest mass difference (g) at which the robot acts correctly in >= level of trials."""
    for d in DELTAS_KG:
        if r["sweep"][d][key] >= level:
            return int(round(d * 1e3))
    return None


def plot(out: dict, res: dict, trials: int) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0), dpi=150, gridspec_kw={"width_ratios": [1, 1, 0.95]})
    xs = [d * 1e3 for d in DELTAS_KG]
    for a_, (kr, kw, ttl) in zip(ax[:2], (("act_right", "act_wrong",
                                           "(a) Decision rule today: z from the 3-cycle SE"),
                                          ("act_right_f", "act_wrong_f",
                                           "(b) Proposed: SE floored by 5 repeated weighings"))):
        for name, r in res.items():
            c = COLORS[name]
            a_.plot(xs, [r["sweep"][d][kr] * 100 for d in DELTAS_KG], "-o", color=c, lw=2.2, ms=5, label=name)
            a_.plot(xs, [r["sweep"][d][kw] * 100 for d in DELTAS_KG], ":x", color=c, lw=1.8, ms=6)
        a_.set_xscale("log")
        a_.set_xticks(xs, [f"{x:.0f}" for x in xs])
        a_.minorticks_off()
        a_.set_ylim(-2, 103)
        a_.set_xlabel("mass difference between the cans (g); lighter can 100 g")
        a_.set_ylabel("% of trials")
        a_.set_title(ttl, fontsize=10.5, loc="left", fontweight="bold")
        a_.grid(alpha=0.25)
        a_.axhspan(-2, 5, color="#f6dede", zorder=0)
    h = [plt.Line2D([], [], color=COLORS[n], lw=2.2, marker="o", label=n) for n in res]
    h += [plt.Line2D([], [], color="#555", lw=2.2, marker="o", label="acts, and is right"),
          plt.Line2D([], [], color="#555", lw=1.8, ls=":", marker="x", label="acts, and is WRONG")]
    ax[0].legend(handles=h, fontsize=8, loc="center right", frameon=False)
    ax[1].text(xs[0], 90, "below a curve the robot abstains:\nit re-measures, then refuses to guess", fontsize=8.5,
               color="#555", va="top")

    a1 = ax[2]
    names = list(res)
    pos, data, cols = [], [], []
    for i, name in enumerate(names):
        for j, (true, key) in enumerate(((BASE_KG, "A"), (BASE_KG + DELTAS_KG[-1], DELTAS_KG[-1]))):
            pos.append(i * 3 + j)
            data.append((out[name]["tare"][key][:, 0] - true) * 1e3)
            cols.append(COLORS[name])
    bp = a1.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#222"))
    for k, (b, c) in enumerate(zip(bp["boxes"], cols)):
        b.set_facecolor(c)
        b.set_alpha(0.45 if k % 2 == 0 else 0.85)
    a1.axhline(0, color="#d33", lw=1.5)
    a1.set_xticks([i * 3 + 0.5 for i in range(len(names))], names)
    ylim = max(abs(np.percentile(np.concatenate(data), [1, 99]))) * 1.6
    a1.set_ylim(-ylim, ylim)
    for i, name in enumerate(names):
        r = res[name]
        a1.text(i * 3 + 0.5, ylim * 0.9, f"RMSE {r['tare_100']['rmse']:.0f} / {r['tare_350']['rmse']:.0f} g",
                ha="center", fontsize=8.5, color="#333")
    a1.set_ylabel("estimate minus true mass (g)")
    a1.set_title("(c) Absolute mass with empty-gripper tare\nlight = 100 g can, dark = 350 g can", fontsize=10.5,
                 loc="left", fontweight="bold")
    a1.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Torque-based weighing on three arms · MuJoCo, {trials} simulated trials per point · "
                 "error budget assumed per sensing principle, not measured", fontsize=12, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "cross_embodiment_results.png")


def write_md(res: dict, trials: int) -> str:
    L = ["## Error budget (assumed per sensing principle, not measured)", "",
         "| Arm | Torque sensing | White noise | Offset per weighing | Coulomb friction (±25 % asym.) | Ripple |",
         "|---|---|---|---|---|---|"]
    for n, s in SENSORS.items():
        L.append(f"| {n} | {s['kind']} | {s['white']} N·m | {s['offset']} N·m | {s['coulomb']} N·m | {s['ripple']} N·m |")
    L += ["", f"All arms: grasp point ±{GRASP_SD_M[0] * 1e3:.0f} mm sideways / ±{GRASP_SD_M[2] * 1e3:.0f} mm along "
          f"the tool, weigh-pose repeatability {POSE_SD_DEG}°, arm-model mass error {MODEL_MASS_ERR * 100:.0f} %, "
          f"dither {DITHER_DEG}° × {CYCLES} cycles of {PERIOD_S:.0f} s. {trials} trials per point.", "",
          "## Telling the cans apart (robot's own estimator, model gravity)", "",
          "| Arm | Joints used | Δm to act correctly in 95 %: today / with floor | Acts WRONGLY at Δm 10 g: "
          "today / with floor | Acts WRONGLY at Δm 30 g: today / with floor | Δm 250 g: acts correctly |",
          "|---|---|---|---|---|---|"]
    for n, r in res.items():
        s10, s30, s250 = r["sweep"][0.010], r["sweep"][0.030], r["sweep"][0.250]
        r1, r2 = resolution(r), resolution(r, key="act_right_f")
        L.append(f"| {n} | {r['joints']} | {r1 if r1 else '> 250'} g / {r2 if r2 else '> 250'} g | "
                 f"{s10['act_wrong'] * 100:.0f} % / {s10['act_wrong_f'] * 100:.1f} % | "
                 f"{s30['act_wrong'] * 100:.0f} % / {s30['act_wrong_f'] * 100:.1f} % | "
                 f"{s250['act_right'] * 100:.0f} % / {s250['act_right_f'] * 100:.0f} % |")
    L += ["", "## Absolute mass", "",
          "| Arm | Model gravity only: error on 100 g / 350 g (mean ± sd) | With tare: RMSE on 100 g / 350 g "
          "| Robot's own 3-cycle SE (median) | True error within ±2 SE | Spread of 5 repeated weighings (median) |",
          "|---|---|---|---|---|---|"]
    for n, r in res.items():
        m1, m3, t1, t3 = r["model_100"], r["model_350"], r["tare_100"], r["tare_350"]
        L.append(f"| {n} | {m1['bias']:+.0f} ± {m1['sd']:.0f} g / {m3['bias']:+.0f} ± {m3['sd']:.0f} g | "
                 f"{t1['rmse']:.0f} g / {t3['rmse']:.0f} g | {r['median_se_g']:.0f} g | "
                 f"{r['coverage_2se'] * 100:.0f} % | {r['sd_rep_g']:.0f} g |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not XML.exists():
        raise SystemExit(f"{XML} not found (see README)")
    FIG.mkdir(parents=True, exist_ok=True)
    out = run(a.trials, a.seed)
    res = metrics(out)
    plot(out, res, a.trials)
    md = write_md(res, a.trials)
    (ROOT / "docs/cross_embodiment/results.md").write_text(
        "# Cross-embodiment weighing: simulated results\n\nGenerated by `simulate.py`. Simulation with an assumed "
        "error budget: plausible, not measured. Only the OpenArm has run on the real robot.\n\n" + md)
    print(md)


if __name__ == "__main__":
    main()
