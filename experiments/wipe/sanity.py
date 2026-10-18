"""Did the policy learn anything? Stroke statistics + ablations.

1. Stroke stats (no goal, no sim): human data vs diffusion vs untrained diffusion vs BC.
2. Cleaning with a hand-coded scrub (goal + sine wiggle, human freq/amplitude) as the baseline
   the learned policy has to beat to justify itself.
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import run
from policy import ACT_H, OBS_H, Diffusion, Norm, load
from wipesim import RATE_HZ, WipeSim

HUMAN = run.segs  # (N, 96, 2), already retargeted


def stats(paths: np.ndarray) -> dict:
    """paths (N, T, 2). Dominant frequency (per robot step rate), RMS radius, reversal rate."""
    p = paths - paths.mean(1, keepdims=True)
    d = np.diff(paths, axis=1)
    sp = np.abs(np.fft.rfft(p * np.hanning(p.shape[1])[None, :, None], axis=1)) ** 2
    f = np.fft.rfftfreq(p.shape[1], 1 / RATE_HZ)
    peak = f[1:][np.argmax(sp[:, 1:].sum(2), 1)]
    cosang = np.sum(d[:, 1:] * d[:, :-1], 2) / (np.linalg.norm(d[:, 1:], axis=2) * np.linalg.norm(d[:, :-1], axis=2) + 1e-12)
    return {"freq_hz": float(np.median(peak)), "radius_cm": float(np.sqrt((p ** 2).sum(2).mean()) * 100),
            "speed_cm_s": float(np.linalg.norm(d, axis=2).mean() * RATE_HZ * 100),
            "reversals_per_s": float((cosang < -0.5).sum(1).mean() / (p.shape[1] / RATE_HZ)),
            "net_drift_cm": float(np.linalg.norm(paths[:, -1] - paths[:, 0], axis=1).mean() * 100)}


def rollout(model, n: int = 64, T: int = 96, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    hist = [torch.zeros(n, 2)] * OBS_H
    pos, out = torch.zeros(n, 2), [torch.zeros(n, 2)]
    while len(out) < T:
        chunk = model.sample(torch.stack(hist[-OBS_H:], 1), torch.zeros(n, 2))
        for d in chunk[:, :run.EXEC].unbind(1):
            pos = pos + d
            hist.append(d)
            out.append(pos)
    return torch.stack(out[:T], 1).numpy()


class Scrub:
    """Hand-coded: walk to the dirt + perpendicular sine at the human stroke freq/amplitude."""

    def __init__(self, hz: float, amp: float, fwd: float = 0.35) -> None:
        self.hz, self.amp, self.fwd, self.k = hz, amp, fwd, 0

    def __call__(self, sim, xy, rng):
        g = run.goal_to_dirt(sim, xy)
        n = np.linalg.norm(g)
        u = g / n if n > 1e-9 else np.array([1.0, 0.0])
        self.k += 1
        w = 2 * np.pi * self.hz / RATE_HZ
        wiggle = np.array([-u[1], u[0]]) * self.amp * w * np.cos(w * self.k)  # derivative of amp*sin
        return xy + self.fwd * u * run.STEP + wiggle

    def observe(self, d) -> None:
        pass


def main() -> None:
    rows = {"human (Eidon, retargeted)": stats(HUMAN)}
    rows["diffusion, goal=0"] = stats(rollout(load("diffusion")))
    ck = torch.load(run.OUT / "diffusion.pt")
    torch.manual_seed(1)
    rows["diffusion UNTRAINED, goal=0"] = stats(rollout(Diffusion(Norm(*ck["norm"])).eval()))
    rows["bc, goal=0"] = stats(rollout(load("bc")))
    print(f"{'':30s} {'freq Hz':>8s} {'radius cm':>10s} {'speed cm/s':>11s} {'rev/s':>6s} {'drift cm':>9s}")
    for k, s in rows.items():
        print(f"{k:30s} {s['freq_hz']:8.2f} {s['radius_cm']:10.1f} {s['speed_cm_s']:11.1f} {s['reversals_per_s']:6.2f} {s['net_drift_cm']:9.1f}")

    h = rows["human (Eidon, retargeted)"]
    best = None
    sim = WipeSim()
    print("\ncleaning, 20 seeds, 25 s:")
    for hz in (0.5, 1.0):
        for amp in (0.02, 0.04):
            run.CONTROLLERS["scrub"] = lambda hz=hz, amp=amp: Scrub(hz, amp)
            r = [run.episode(sim, "scrub", s) for s in range(20)]
            c = np.mean([x["cleaned"] for x in r])
            t = np.median([x["t90_s"] or np.inf for x in r])
            print(f"  hand-coded scrub {hz} Hz ±{amp * 100:.0f} cm   {c * 100:5.1f} %   t90 {t:.1f}s")
    for name in ("diffusion", "greedy", "zigzag"):
        r = [run.episode(sim, name, s) for s in range(20)]
        print(f"  {name:28s}   {np.mean([x['cleaned'] for x in r]) * 100:5.1f} %   t90 {np.median([x['t90_s'] or np.inf for x in r]):.1f}s")
    sim.close()


if __name__ == "__main__":
    main()
