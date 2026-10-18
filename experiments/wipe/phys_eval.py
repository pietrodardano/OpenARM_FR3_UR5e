"""Final physics comparison on held-out dirt layouts (seeds 5-14), two scenarios.

    .venv/bin/python experiments/wipe/phys_eval.py

Writes outputs/eidon/phys_eval.json and phys_eval.png. The scrub baseline was tuned on seeds 0-4.
"""

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import phys_run
import run
from physim import PhysWipe

SEEDS = range(5, 15)
NAMES = ["zigzag", "scrub", "greedy", "random", "bc", "diffusion"]
LABEL = {"zigzag": "IK zigzag\n(hand-coded)", "scrub": "IK scrub\n(hand-coded, tuned)",
         "greedy": "IK to dirt\n(hand-coded)", "random": "random walk", "bc": "BC\n(Eidon)",
         "diffusion": "diffusion\n(Eidon)"}


def main() -> None:
    sim = PhysWipe()
    out = {}
    for scen in ("spills", "stubborn"):
        out[scen] = {}
        for name in NAMES:
            r = [phys_run.episode(sim, name, s, stubborn=scen == "stubborn") for s in SEEDS]
            t90 = [x["t90_s"] if x["t90_s"] is not None else float("inf") for x in r]
            c = np.array([x["cleaned"] for x in r])
            path = np.array([x["sponge_path_m"] for x in r])
            out[scen][name] = {
                "cleaned_mean": float(c.mean()), "cleaned_std": float(c.std()), "cleaned": c.tolist(),
                "reached90": int(np.isfinite(t90).sum()), "t90_median_s": float(np.median(t90)),
                "sponge_path_m": float(path.mean()), "clean_per_m": float((c / np.maximum(path, 1e-6)).mean()),
                "force_mean_N": float(np.mean([x["force_mean"] for x in r])),
                "force_std_N": float(np.mean([x["force_std"] for x in r])),
                "contact_frac": float(np.mean([x["frac_in_contact"] for x in r])),
                "slip_mm_max": float(max(x["slip_mm_max"] for x in r)),
                "tilt_deg_max": float(max(x["tilt_deg_max"] for x in r)),
                "lifted_cm_min": float(min(x["lifted_m"] for x in r) * 100),
                "torque_sat_frac": float(np.mean([x["torque_sat_frac"] for x in r])),
                "curve_mean": np.mean([x["curve"] for x in r], 0).tolist()}
            v = out[scen][name]
            print(f"[{scen}] {name:9s} clean {v['cleaned_mean'] * 100:5.1f}±{v['cleaned_std'] * 100:4.1f}%  "
                  f"90% {v['reached90']}/10 (med {v['t90_median_s']:.1f}s)  path {v['sponge_path_m']:.2f} m  "
                  f"F {v['force_mean_N']:.1f}±{v['force_std_N']:.1f} N  contact {v['contact_frac'] * 100:.0f}%  "
                  f"slip≤{v['slip_mm_max']:.0f} mm  tilt≤{v['tilt_deg_max']:.0f}°  sat {v['torque_sat_frac'] * 100:.1f}%", flush=True)
    (run.OUT / "phys_eval.json").write_text(json.dumps(out, indent=1))
    sim.close()

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {"zigzag": "#4c72b0", "scrub": "#8172b2", "greedy": "#937860", "random": "#aaaaaa",
              "bc": "#dd8452", "diffusion": "#c44e52"}
    for i, scen in enumerate(("spills", "stubborn")):
        a = ax[i]
        t = np.arange(len(out[scen]["zigzag"]["curve_mean"])) / phys_run.RATE_HZ
        for n in NAMES:
            a.plot(t, np.array(out[scen][n]["curve_mean"]) * 100, label=LABEL[n].replace("\n", " "), color=colors[n],
                   lw=2.5 if n in ("diffusion", "zigzag") else 1.3)
        a.set_title(f"{'Spills' if scen == 'spills' else 'Stubborn stains (3x wear)'} - 10 held-out layouts, real contact")
        a.set_xlabel("wiping time (s)")
        a.set_ylabel("% dirt removed (mean)")
        a.set_ylim(0, 100)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower right")
    a = ax[2]
    x = np.arange(len(NAMES))
    for j, scen in enumerate(("spills", "stubborn")):
        m = [out[scen][n]["cleaned_mean"] * 100 for n in NAMES]
        s = [out[scen][n]["cleaned_std"] * 100 for n in NAMES]
        a.bar(x + (j - 0.5) * 0.38, m, 0.38, yerr=s, capsize=3, label=scen,
              color=[colors[n] for n in NAMES], alpha=1.0 if j == 0 else 0.55, edgecolor="k", lw=0.5)
    a.set_xticks(x, [LABEL[n] for n in NAMES], fontsize=7)
    a.set_ylabel("% removed after 25 s (solid: spills, light: stubborn)")
    a.set_ylim(0, 105)
    a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run.OUT / "phys_eval.png", dpi=90)


if __name__ == "__main__":
    main()
