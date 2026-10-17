"""
Shared "go to zero" wrapper for every motion test script.

Zero pose = ``hardware.rest_qpos`` in configs/task_config.yaml (all joints 0, arms hanging),
grippers closed. The driver's ``stop()`` moves to the same pose before disabling, so after
:func:`go_to_zero` the stop is motion-free.

:func:`run_with_zero` gives every script the same lifecycle:

    start -> [confirm] go to zero -> test() -> go to zero -> stop (disable)

Ctrl+C anywhere in the test (including at a prompt) jumps to "go to zero". While the
return is running, further Ctrl+C presses are ignored: use the hardware e-stop to abort.
After a latched fault (collision, silent motor, ...) the robot holds its pose and is
NOT moved automatically.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import Callable, Dict

import numpy as np

from src.control.robot_interface import SIDES, RobotBackend, Side, arm_slice

MIN_JERK_PEAK_FACTOR = 1.875  # peak / average speed of a minimum-jerk profile


def zero_pose(cfg: dict) -> Dict[Side, np.ndarray]:
    return {s: np.asarray(cfg["hardware"]["rest_qpos"][s], dtype=np.float64) for s in SIDES}


def distance_to_zero_deg(robot: RobotBackend, cfg: dict) -> float:
    targets = robot.get_joint_targets()
    zero = zero_pose(cfg)
    return max(float(np.degrees(np.max(np.abs(targets[arm_slice(s)] - zero[s])))) for s in SIDES)


def go_to_zero(robot: RobotBackend, cfg: dict, speed_deg_s: float = 10.0, grippers: bool = True) -> None:
    """Minimum-jerk joint move to the zero pose at ``speed_deg_s`` peak (grippers close if ``grippers``)."""
    torque_mode = [s for s in robot.active_sides if robot.get_state().arms[s].gripper_torque_cmd is not None]
    if torque_mode:
        print(f"Gripper torque mode off for {torque_mode} (releases a grasped object)")
        robot.set_gripper_torque({s: None for s in torque_mode})
        robot.advance(0.1)
    dist = distance_to_zero_deg(robot, cfg)
    duration = max(3.0, MIN_JERK_PEAK_FACTOR * dist / speed_deg_s)
    print(f"\nGoing to zero{' (grippers closed)' if grippers else ''}: {dist:.0f} deg max, {duration:.1f} s")
    robot.move_joints(zero_pose(cfg), gripper={s: 0.0 for s in SIDES} if grippers else None, duration=duration)
    robot.advance(0.5)


@contextmanager
def ctrl_c_ignored(message: str):
    old = signal.signal(signal.SIGINT, lambda *_: print(message))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


def run_with_zero(
    robot: RobotBackend,
    cfg: dict,
    test: Callable[[], None],
    speed_deg_s: float = 10.0,
    zero_grippers_first: bool = True,
    confirm_first: Callable[[str], bool] = lambda _msg: True,
) -> int:
    """Runs ``test()`` between two "go to zero" moves. Returns a process exit code."""
    busy = "  [Ctrl+C ignored: returning to zero. Use the e-stop to abort.]"
    rc = 0
    robot.start()
    try:
        try:
            if not confirm_first(f"Go to zero first ({distance_to_zero_deg(robot, cfg):.0f} deg max)?"):
                return 0
            go_to_zero(robot, cfg, speed_deg_s, grippers=zero_grippers_first)
            test()
        except KeyboardInterrupt:
            print("\nCtrl+C: going to zero...")
            rc = 130
        except RuntimeError as exc:
            print(f"\nABORTED: {exc}")
            rc = 1
        fault = robot.get_state().fault
        if fault:
            print(f"FAULT latched ({fault}): holding the measured pose, not moving to zero.")
            return 1
        with ctrl_c_ignored(busy):
            go_to_zero(robot, cfg, speed_deg_s)
    finally:
        with ctrl_c_ignored(busy):
            print("Stopping (disable motors)...")
            robot.stop()
    return rc
