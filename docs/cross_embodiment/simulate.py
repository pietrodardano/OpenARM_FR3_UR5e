"""Cross-embodiment check of the "weigh by joint torque" principle (standalone, no robotokens code touched).

Question: does the same estimator work on arms other than the OpenArm?

The scene is `cross_embodiment_openarm_ur5e_fr3.xml` from MuJoCo Menagerie (OpenArm v1, UR5e, Franka
FR3, all real MJCF models). For each arm we hold two look-alike cans of different mass at the tool point
and run the SAME recipe as `src/robotokens/sensing/payload.py`:

  1. dither the load-bearing joints with a small sine (friction flips sign and cancels over a cycle);
  2. subtract the arm's own gravity torques, either from the *unloaded model* ("model") or from a
     measurement of the empty gripper at the same pose with the same dither ("tare");
  3. least-squares mass from tau_res ~ J_z * m * g, with the cycle-to-cycle spread as standard error;
  4. compare the two cans with z = diff / combined SE.

"Measured" torques come from an independent, loaded copy of the physics model (payload body at the
tool point) plus Coulomb friction, torque noise and a 3 % mass error in the estimator's own model, so
the estimator is not graded against itself. The tare removes that model error; the cost is one extra
(empty) measurement per pose. This is a simulation of the principle, not hardware
data: only the OpenArm has been run on the real robot.

Usage:  .venv/bin/python docs/cross_embodiment/simulate.py [--trials 200]
Needs:  third_party/mujoco_menagerie/cross_embodiment_openarm_ur5e_fr3.xml (see README).
Writes: docs/figures/cross_embodiment_results.png and docs/cross_embodiment/results.md
(the scene image and the GIF come from animate.py)
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
COULOMB_NM, NOISE_NM, MODEL_MASS_ERR = 0.15, 0.08, 0.03
MASSES_KG = (0.100, 0.350)  # e.g. a nearly empty and a full can, same look
MIN_Z = 3.0

ARMS = {
    "OpenArm v1": dict(site="left_ee_control_point", joints=[f"openarm_left_joint{i}" for i in range(1, 8)]),
    "UR5e": dict(site="gripper_ur5e_left_pinch",
                 joints=["ur5e_left_" + n + "_joint" for n in
                         ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]),
    "Franka FR3": dict(site="gripper_fr3_left_pinch", joints=[f"fr3_left_fr3_joint{i}" for i in range(1, 8)]),
}


def build(payload_site: dict[str, str]) -> tuple[mujoco.MjModel, dict[str, int]]:
    """Scene with one massless-until-set payload body welded at each arm's tool point."""
    spec = mujoco.MjSpec.from_file(str(XML))
    bodies = {}
    for arm, site_name in payload_site.items():
        site = spec.site(site_name)
        body = site.parent
        child = body.add_body(name=f"payload_{arm.split()[0]}", pos=list(site.pos))
        child.mass, child.inertia, child.explicitinertial = 1e-6, [1e-9] * 3, 1
        bodies[arm] = child.name
    model = spec.compile()
    return model, {a: model.body(n).id for a, n in bodies.items()}


def gravity_and_jz(model, data, q, qadr, dadr, site):
    data.qpos[qadr] = q
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, model.site(site).id)
    return data.qfrc_bias[dadr].copy(), jacp[2, dadr].copy()


def run(trials: int, rng: np.random.Generator) -> dict:
    sites = {a: c["site"] for a, c in ARMS.items()}
    truth, payload_ids = build(sites)  # loaded copy ("the real world")
    est = mujoco.MjModel.from_xml_path(str(XML))  # unloaded, with a mass error ("the arm's model")
    est.body_mass *= 1 + MODEL_MASS_ERR
    est.body_inertia *= 1 + MODEL_MASS_ERR
    dt, de = mujoco.MjData(truth), mujoco.MjData(est)
    mujoco.mj_setConst(est, de)  # body_subtreemass is cached: refresh after editing masses
    mujoco.mj_resetDataKeyframe(truth, dt, 0)
    mujoco.mj_resetDataKeyframe(est, de, 0)
    n_per = int(PERIOD_S * RATE_HZ)
    phase = 2 * np.pi * np.arange(n_per) / n_per
    out = {}
    for arm, cfg in ARMS.items():
        jt = [truth.joint(j) for j in cfg["joints"]]
        qadr = np.array([j.qposadr[0] for j in jt])
        dadr = np.array([j.dofadr[0] for j in jt])
        q0 = dt.qpos[qadr].copy()
        _, jz = gravity_and_jz(est, de, q0, qadr, dadr, cfg["site"])
        used = [i for i in range(len(qadr)) if abs(jz[i]) > MIN_LEVER_M]
        amp = np.zeros(len(qadr))
        amp[used] = np.radians(DITHER_DEG)
        pid = payload_ids[arm]
        res = {}
        tau_est, jz_est = gravity_and_jz(est, de, q0, qadr, dadr, cfg["site"])
        fric = COULOMB_NM * np.sign(amp * np.cos(phase)[:, None])

        def torque_cycle(m_kg: float) -> np.ndarray:
            """Static gravity torque of the (loaded) arm along one dither cycle."""
            truth.body_mass[pid] = max(m_kg, 1e-6)
            truth.body_inertia[pid] = 1e-5
            mujoco.mj_setConst(truth, dt)
            tau = np.empty((n_per, len(qadr)))
            for k in range(n_per):
                dt.qpos[qadr] = q0 + amp * np.sin(phase[k])
                mujoco.mj_forward(truth, dt)
                tau[k] = dt.qfrc_bias[dadr]
            return tau

        def mass_of(resid: np.ndarray) -> float:
            return float(resid[used] @ jz_est[used] / (G * jz_est[used] @ jz_est[used]))

        def noisy_mean(tau: np.ndarray) -> np.ndarray:
            return (tau + fric + rng.normal(0, NOISE_NM, tau.shape)).mean(axis=0)

        tau_empty = torque_cycle(0.0)
        for m_true in MASSES_KG:
            tau_t = torque_cycle(m_true)
            rows = []
            for _ in range(trials):
                c_model, c_tare = [], []
                for _c in range(CYCLES):
                    meas = noisy_mean(tau_t)
                    c_model.append(mass_of(meas - tau_est))
                    c_tare.append(mass_of(meas - noisy_mean(tau_empty)))
                rows.append([np.mean(c_model), np.std(c_model, ddof=1) / math.sqrt(CYCLES),
                             np.mean(c_tare), np.std(c_tare, ddof=1) / math.sqrt(CYCLES)])
            res[m_true] = np.array(rows)
        out[arm] = dict(res=res, used=used, jz=jz, q0=q0, qadr=qadr)
    return out


def summarize(out: dict, col: int) -> list[dict]:
    """col 0 = model-gravity subtraction, col 2 = empty-gripper tare (mass at col, SE at col + 1)."""
    rows = []
    for arm, o in out.items():
        a, b = (o["res"][m][:, col:col + 2] for m in MASSES_KG)
        z = (b[:, 0] - a[:, 0]) / np.hypot(a[:, 1], b[:, 1])
        rows.append(dict(arm=arm, joints=len(o["used"]), m1=a[:, 0].mean(), m1sd=a[:, 0].std(),
                         m2=b[:, 0].mean(), m2sd=b[:, 0].std(), diff=(b[:, 0] - a[:, 0]).mean(),
                         diffsd=(b[:, 0] - a[:, 0]).std(), right=float((b[:, 0] > a[:, 0]).mean()),
                         conclusive=float((z >= MIN_Z).mean())))
    return rows


def plot_results(tare_rows: list[dict], trials: int) -> None:
    """Headline figure: the estimator with the empty-gripper tare (the full comparison is in results.md)."""
    colors = {"OpenArm v1": "#2a6fdb", "UR5e": "#e08a1e", "Franka FR3": "#2a9d5b"}
    lo, hi = (m * 1e3 for m in MASSES_KG)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160)

    a0 = ax[0]  # (a) estimated vs true mass
    a0.plot([0, 450], [0, 450], color="#aaa", lw=1.3, ls="--", zorder=0, label="perfect estimate")
    for k, r in enumerate(tare_rows):
        c = colors[r["arm"]]
        xs = np.array([lo, hi]) + (k - 1) * 12
        a0.errorbar(xs, [r["m1"] * 1e3, r["m2"] * 1e3], [r["m1sd"] * 1e3, r["m2sd"] * 1e3], fmt="o", ms=10,
                    color=c, capsize=4, lw=1.8, label=r["arm"], zorder=3)
    a0.set_xlim(20, 430)
    a0.set_ylim(0, 450)
    a0.set_xticks([lo, hi], [f"{lo:.0f} g can", f"{hi:.0f} g can"])
    a0.set_ylabel("estimated mass (g)")
    a0.set_title("(a) Each arm estimates both cans correctly", fontsize=10.5, loc="left", fontweight="bold")
    a0.grid(axis="y", alpha=0.25)
    a0.legend(frameon=False, fontsize=9, loc="upper left")

    a1 = ax[1]  # (b) difference between the cans
    true_d = hi - lo
    a1.axhline(true_d, color="#d33", lw=2, zorder=1, label=f"true difference ({true_d:.0f} g)")
    for k, r in enumerate(tare_rows):
        a1.errorbar(k, r["diff"] * 1e3, r["diffsd"] * 1e3, fmt="o", ms=11, color=colors[r["arm"]], capsize=5, lw=1.8,
                    zorder=3)
    a1.set_xlim(-0.6, 2.6)
    a1.set_ylim(true_d - 14, true_d + 14)
    a1.set_xticks(range(3), [r["arm"] for r in tare_rows])
    a1.set_ylabel("heavier can minus lighter can (g)")
    a1.set_title("(b) ...so the heavier can is always identified", fontsize=10.5, loc="left", fontweight="bold")
    a1.grid(axis="y", alpha=0.25)
    a1.legend(loc="upper left", frameon=False, fontsize=9)
    ok = min(r["right"] for r in tare_rows) * 100
    a1.text(0.5, 0.05, f"heavier can identified in {ok:.0f} % of {trials} trials per arm", transform=a1.transAxes,
            ha="center", fontsize=9, color="#444")

    fig.suptitle("One torque-weighing recipe on three arms  ·  MuJoCo simulation", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "cross_embodiment_results.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not XML.exists():
        raise SystemExit(f"{XML} not found: clone mujoco_menagerie into third_party/ (see README)")
    FIG.mkdir(parents=True, exist_ok=True)
    out = run(a.trials, np.random.default_rng(a.seed))
    model_rows, tare_rows = summarize(out, 0), summarize(out, 2)
    plot_results(tare_rows, a.trials)
    t1, t2, td = (f"{m * 1e3:.0f}" for m in (*MASSES_KG, MASSES_KG[1] - MASSES_KG[0]))
    lines = [f"| Arm | Joints with lever arm | Method | Est. can A (true {t1} g) | Est. can B (true {t2} g) | "
             f"Est. difference (true {td} g) | Heavier identified | Conclusive (z>=3) |", "|---|---|---|---|---|---|---|---|"]
    for name, rows in (("model gravity only", model_rows), ("with empty-gripper tare", tare_rows)):
        for r in rows:
            lines.append(f"| {r['arm']} | {r['joints']} | {name} | {r['m1'] * 1e3:.0f} ± {r['m1sd'] * 1e3:.0f} g | "
                         f"{r['m2'] * 1e3:.0f} ± {r['m2sd'] * 1e3:.0f} g | {r['diff'] * 1e3:.1f} ± {r['diffsd'] * 1e3:.1f} g | "
                         f"{r['right'] * 100:.0f} % | {r['conclusive'] * 100:.0f} % |")
    (ROOT / "docs/cross_embodiment/results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
