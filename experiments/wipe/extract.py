"""Eidon `cleaning` IMU -> 2D wipe strokes.

The harness gives orientations only, so the hand position is rebuilt by chaining bones from the
shoulder: upper arm (slot 2/5), forearm (1/4), hand (0/3), each along its sensor's local +Y
(checked: +Y points down the hanging arm). The torso heading is removed with the chest sensor.
Wiping-like windows are kept: mostly horizontal, periodic hand motion at 0.5-3 Hz.

Output: outputs/eidon/wipe_segments.npz with `segs` (N, T, 2) in metres, horizontal plane,
24 Hz, each segment centred on its mean.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

OUT = Path("outputs/eidon")
HZ = 24.0
L_UPPER, L_FORE, L_HAND = 0.30, 0.26, 0.09  # adult bone lengths (m)
WIN = int(4 * HZ)  # 4 s windows
HOP = int(1 * HZ)
SLOTS = {"left": (2, 1, 0), "right": (5, 4, 3)}


def resample(r: pd.DataFrame, t: np.ndarray) -> dict:
    out = {}
    for s in range(7):
        x = r[r.slot == s].sort_values("time_ms")
        q = x[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(dtype=np.float64, copy=True)
        # make consecutive quats share a hemisphere before interpolating
        flip = np.concatenate([[1.0], np.sign(np.sum(q[1:] * q[:-1], 1) + 1e-12)])
        q *= np.cumprod(flip)[:, None]
        qi = np.stack([np.interp(t, x.time_ms.to_numpy(), q[:, k]) for k in range(4)], 1)
        out[s] = R.from_quat(qi / np.linalg.norm(qi, axis=1, keepdims=True))
    return out


def hand_path(rots: dict, side: str) -> np.ndarray:
    up, fo, ha = SLOTS[side]
    y = np.array([0.0, 1.0, 0.0])
    p = L_UPPER * rots[up].apply(y) + L_FORE * rots[fo].apply(y) + L_HAND * rots[ha].apply(y)
    # remove torso heading (yaw of the chest's local X, projected on the horizontal plane)
    cx = rots[6].apply([1.0, 0.0, 0.0])
    yaw = np.unwrap(np.arctan2(cx[:, 1], cx[:, 0]))
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]], 1)


def wipe_score(w: np.ndarray) -> tuple:
    """(is_wipe, dominant Hz). Horizontal, periodic, big enough, not walking around."""
    xy = w[:, :2] - w[:, :2].mean(0)
    h_amp = np.sqrt((xy ** 2).sum(1).mean())
    v_amp = w[:, 2].std()
    sp = np.abs(np.fft.rfft(xy * np.hanning(len(w))[:, None], axis=0)) ** 2
    f = np.fft.rfftfreq(len(w), 1 / HZ)
    band = (f >= 0.5) & (f <= 3.0)
    power = sp.sum(1)
    peak = f[band][np.argmax(power[band])]
    periodic = power[band].sum() / (power[1:].sum() + 1e-12)
    ok = 0.03 < h_amp < 0.30 and v_amp < 0.6 * h_amp and periodic > 0.6
    return ok, peak


def main() -> None:
    df = pd.read_parquet(OUT / "cleaning_imu_subset.parquet")
    segs, meta = [], []
    for rid, r in df.groupby("recording_id"):
        t = np.arange(r.time_ms.min(), r.time_ms.max(), 1000 / HZ)
        rots = resample(r, t)
        for side in ("left", "right"):
            p = hand_path(rots, side)
            for a in range(0, len(p) - WIN, HOP):
                w = p[a:a + WIN]
                ok, hz = wipe_score(w)
                if ok:
                    segs.append(w[:, :2] - w[:, :2].mean(0))
                    meta.append((rid, side, a / HZ, hz))
    segs = np.asarray(segs, np.float32)
    meta = pd.DataFrame(meta, columns=["recording_id", "side", "t_s", "hz"])
    np.savez(OUT / "wipe_segments.npz", segs=segs, rid=meta.recording_id.to_numpy(), t=meta.t_s.to_numpy(),
             hz=meta.hz.to_numpy(), side=meta.side.to_numpy())
    print(f"{len(segs)} wipe windows of {WIN / HZ:.0f} s from {meta.recording_id.nunique()} / "
          f"{df.recording_id.nunique()} recordings  ({len(segs) * WIN / HZ / 60:.1f} min)")
    print(meta.groupby("side").size().to_dict(), "median stroke Hz", float(meta.hz.median()))
    print("RMS radius (m):", float(np.sqrt((segs ** 2).sum(2).mean())))


if __name__ == "__main__":
    main()
