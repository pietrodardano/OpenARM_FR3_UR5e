"""A scripted stand-in for the real OpenArm, so the whole chain runs without the robot.

It emits what the real side would publish (see ``stream``): the 7 left-arm joint angles, the gripper
opening and the current task phase, at 50 Hz. The arm does a plausible lateral pick, lift, weigh and
put-back of two cans, then presents the second can. It is NOT the real controller and checks no
collisions; it only has to look like the real arm and walk through the same phases.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from src.cross_embodiment.motion import JointPath
from src.cross_embodiment.stream import PHASES, ReplaySource, StateMessage
from src.kinematics.arm_kinematics import ArmModel

OPEN, CLOSED = 0.044, 0.0
# Real phases are slower than the sim robots' (the real arm moves cautiously), so the sync gate has work to do.
DEFAULT_DURATIONS = {
    "INIT": 2.0, "PICK_1": 26.0, "LIFT_1": 8.0, "WEIGH_1": 10.0, "PLACE_BACK_1": 12.0,
    "PICK_2": 24.0, "LIFT_2": 8.0, "WEIGH_2": 10.0, "PLACE_BACK_2": 12.0, "COMPARE": 3.0, "ACT": 30.0,
}
START = np.array([0.0, -0.8, 0.0, 0.7, 0.0, 0.0, 0.0])
Z = 0.82
CAN = {1: (0.305, 0.20), 2: (0.30, 0.09)}
HANDOVER = (0.30, 0.18, 1.05)


def _ik_chain(arm: ArmModel, points: list[tuple[float, float, float]], seed: np.ndarray) -> list[np.ndarray]:
    qs, q = [], seed
    for p in points:
        result = arm.solve_ik(np.array(p), q, restarts=8, pos_tol_m=2e-3)
        if not result.converged:
            raise RuntimeError(f"mock OpenArm: pose {p} unreachable")
        qs.append(result.q)
        q = result.q
    return qs


def build_mock_script(model: "mujoco.MjModel", rate_hz: float = 50.0, durations: Optional[dict[str, float]] = None) -> ReplaySource:
    """Build the full scripted state stream (a ``ReplaySource``) for the real OpenArm."""
    durations = durations or DEFAULT_DURATIONS
    arm = ArmModel(model, [f"openarm_left_joint{i}" for i in range(1, 8)], "openarm_left_hand_tcp")
    cx = {i: CAN[i] for i in (1, 2)}
    pre = {i: (cx[i][0] - 0.10, cx[i][1], Z) for i in (1, 2)}
    grasp = {i: (cx[i][0], cx[i][1], Z) for i in (1, 2)}
    up = {i: (cx[i][0], cx[i][1], Z + 0.12) for i in (1, 2)}
    # (waypoints, gripper schedule: list of (fraction of phase, opening))
    plan = {
        "INIT": ([None], [(0.0, OPEN)]),
        "PICK_1": ([pre[1], grasp[1]], [(0.0, OPEN), (0.8, OPEN), (1.0, CLOSED)]),
        "LIFT_1": ([grasp[1], up[1]], [(0.0, CLOSED)]),
        "WEIGH_1": ([up[1], up[1]], [(0.0, CLOSED)]),
        "PLACE_BACK_1": ([up[1], grasp[1], pre[1]], [(0.0, CLOSED), (0.6, CLOSED), (0.7, OPEN)]),
        "PICK_2": ([pre[2], grasp[2]], [(0.0, OPEN), (0.8, OPEN), (1.0, CLOSED)]),
        "LIFT_2": ([grasp[2], up[2]], [(0.0, CLOSED)]),
        "WEIGH_2": ([up[2], up[2]], [(0.0, CLOSED)]),
        "PLACE_BACK_2": ([up[2], grasp[2], pre[2]], [(0.0, CLOSED), (0.6, CLOSED), (0.7, OPEN)]),
        "COMPARE": ([pre[2], pre[2]], [(0.0, OPEN)]),
        "ACT": ([pre[2], grasp[2], up[2], HANDOVER, HANDOVER], [(0.0, OPEN), (0.2, OPEN), (0.3, CLOSED), (0.9, CLOSED), (1.0, OPEN)]),
    }
    messages: list[StateMessage] = []
    t0, q_prev = 0.0, START.copy()
    for phase in PHASES[:-1]:
        waypoints, grip = plan[phase]
        if waypoints[0] is None:
            qs = [q_prev, q_prev]
        else:
            qs = [q_prev] + _ik_chain(arm, waypoints, q_prev)
        path = JointPath(np.array(qs), durations[phase])
        steps = int(round(durations[phase] * rate_hz))
        for k in range(steps):
            frac = (k + 1) / steps
            q, _ = path.sample(frac * durations[phase])
            opening = grip[0][1]
            for (f0, o0), (f1, o1) in zip(grip, grip[1:] + [(2.0, grip[-1][1])]):
                if f0 <= frac < f1 or (frac >= f0 and f1 == 2.0):
                    span = max(f1 - f0, 1e-9) if f1 != 2.0 else 1.0
                    opening = o0 + (o1 - o0) * min(1.0, (frac - f0) / span) if f1 != 2.0 else o0
            messages.append(StateMessage(t=round(t0 + frac * durations[phase], 4), q=[float(v) for v in q], gripper_m=float(opening), phase=phase))
        t0 += durations[phase]
        q_prev = path.points[-1]
    messages.append(StateMessage(t=round(t0 + 0.02, 4), q=[float(v) for v in q_prev], gripper_m=OPEN, phase="DONE"))
    return ReplaySource(messages)
