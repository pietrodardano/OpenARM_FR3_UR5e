"""The weigh-by-touch task for a simulated robot (FR3 or UR5e), as a per-tick generator.

Sequence (same phases as the real task, see ``stream.PHASES``): reference torque, pick can 1 laterally,
lift, weigh, put back, repeat for can 2, compare, then either sort the cans into bins by weight or
hand the heavier one to a simulated user and release it only after torque-sensed contact.

The sim robots may cheat: can positions come straight from the layout, with no vision, and a grasp is
held by a weld that is created when the gripper closes. Everything else is physical: the arm is
torque-controlled, the weld loads it with the can's weight, and the mass is estimated from the
measured joint torque with the loaded-minus-unloaded method.

Each generator yields once per control tick, after commanding the robot; the caller steps the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np

from src.cross_embodiment.contact_detector import ContactDetector
from src.cross_embodiment.mass_probe import MassEstimate, estimate_mass, heavier
from src.cross_embodiment.motion import GRASP_ROTATION, JointPath, MotionError, duration_for, plan_joint_move, plan_line, solve_pose
from src.cross_embodiment.stream import PHASES, StateMessage, phase_index
from src.cross_embodiment.world import DemoWorld

Tick = Generator[None, None, None]


class PhaseGate:
    """Keeps a sim robot in step with the real task: wait until the real task has entered a phase."""

    def __init__(self, enabled: bool = True, timeout_s: float = 300.0) -> None:
        self.enabled = enabled
        self.timeout_s = timeout_s
        self.latest_phase: Optional[str] = None
        self.timed_out = False

    def update(self, message: Optional[StateMessage]) -> None:
        if message is not None and message.phase is not None:
            self.latest_phase = message.phase

    def reached(self, phase: str) -> bool:
        if not self.enabled or self.timed_out:
            return True
        return self.latest_phase is not None and phase_index(self.latest_phase) >= phase_index(phase)


@dataclass
class TaskResult:
    robot: str
    estimates: dict[int, MassEstimate] = field(default_factory=dict)
    decision: Optional[int] = None  # 1 or 2 = heavier can, 0 = indistinguishable
    ending: str = ""
    released_on_contact: bool = False
    fallback_used: bool = False
    final_can_xy: dict[int, tuple[float, float]] = field(default_factory=dict)
    events: list[tuple[float, str, str]] = field(default_factory=list)
    error: Optional[str] = None
    done: bool = False


class WeighTask:
    def __init__(
        self,
        world: DemoWorld,
        robot_name: str,
        gate: Optional[PhaseGate] = None,
        torque_std_nm: float = 0.25,
        user_touches: bool = True,
        user_delay_s: float = 1.0,
        handover_timeout_s: float = 12.0,
    ) -> None:
        self.world, self.name = world, robot_name
        self.robot = world.robots[robot_name]
        self.station = world.layout.stations[robot_name]
        self.cfg = world.layout
        self.gate = gate or PhaseGate(enabled=False)
        self.torque_std_nm = torque_std_nm
        self.user_touches, self.user_delay_s, self.handover_timeout_s = user_touches, user_delay_s, handover_timeout_s
        self.dt = world.layout.control_dt_s
        self.z = world.layout.can_rest_z_m()
        self.q_cmd = self.robot.home_q.copy()
        self._oriented = False  # False until the first move out of the home pose into the grasp orientation
        self.result = TaskResult(robot=robot_name, ending=self.station.ending)

    # ---------------------------------------------------------------- helpers
    def _log(self, phase: str, text: str) -> None:
        self.result.events.append((round(self.world.time, 2), phase, text))

    def _command(self, q: np.ndarray, dq: Optional[np.ndarray] = None) -> None:
        self.q_cmd = q
        self.robot.command(q, dq, self.robot.gravity_torque())

    def _follow(self, path: JointPath) -> Tick:
        for q, dq in path.ticks(self.dt):
            self._command(q, dq)
            yield

    def _hold(self, seconds: float) -> Tick:
        for _ in range(int(round(seconds / self.dt))):
            self._command(self.q_cmd)
            yield

    def _settle(self, still_s: float = 0.3, timeout_s: float = 8.0, speed: float = 5e-4) -> Tick:
        quiet, waited = 0.0, 0.0
        while quiet < still_s and waited < timeout_s:
            self._command(self.q_cmd)
            yield
            waited += self.dt
            quiet = quiet + self.dt if np.abs(self.robot.dq).max() < speed else 0.0

    def _wait_phase(self, phase: str) -> Tick:
        waited = 0.0
        while not self.gate.reached(phase):
            self._command(self.q_cmd)
            yield
            waited += self.dt
            if waited > self.gate.timeout_s:
                self.gate.timed_out = True
                self._log(phase, "sync timeout: continuing without the real task")

    def _move_joint(self, q_to: np.ndarray, ignore_can: Optional[int] = None, check: bool = True) -> Tick:
        path = plan_joint_move(self.robot, self.q_cmd, q_to, self.cfg.grasp.max_joint_speed_rad_s, self.world.model, self.world.data, check, ignore_can)
        yield from self._follow(path)

    def _cur(self) -> np.ndarray:
        """Cartesian pinch position of the current joint command."""
        return self.robot.arm.fk(self.q_cmd)[0]

    def _transfer_to_pre(self, plan: dict) -> Tick:
        """Go to a can's pre-grasp point without sweeping through the table or the cans: up, across, down."""
        z_safe = self.z + self.cfg.grasp.lift_height_m
        pre = plan["p"]["pre"]
        cur = self._cur()
        if cur[2] < z_safe - 1e-3:
            top = np.array([cur[0], cur[1], z_safe])
            yield from self._move_line(cur, top, check=False)
            cur = top
        over = np.array([pre[0], pre[1], z_safe])
        yield from self._move_line(cur, over, check=True)
        yield from self._move_line(over, pre, check=True)

    def _move_line(self, start: np.ndarray, end: np.ndarray, ignore_can: Optional[int] = None, check: bool = True) -> Tick:
        path = plan_line(self.robot, self.world.model, self.world.data, start, end, self.q_cmd, self.cfg.grasp.max_joint_speed_rad_s, check, ignore_can)
        yield from self._follow(path)

    def _measure(self, seconds: float) -> Generator[None, None, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Mean and std of measured torque over a window, plus the mean model gravity torque at the actual pose."""
        taus, gravs = [], []
        for _ in range(int(round(seconds / self.dt))):
            self._command(self.q_cmd)
            yield
            taus.append(self.robot.measured_torque())
            gravs.append(self.robot.gravity_torque())
        taus_arr = np.array(taus)
        return taus_arr.mean(axis=0), taus_arr.std(axis=0), np.mean(gravs, axis=0)

    # ------------------------------------------------------------- poses
    def _poses(self, i: int) -> dict:
        cx, cy = self.station.cans_xy_m[i - 1]
        c, lift, z = self.cfg.grasp.approach_clearance_m, self.cfg.grasp.lift_height_m, self.z
        p = {
            "pre": np.array([cx - c, cy, z]),
            "grasp": np.array([cx, cy, z]),
            "up": np.array([cx, cy, z + lift]),
            "back": np.array([cx - c, cy, z + lift]),
            "hi": np.array([cx, cy, z + lift + 0.01]),
            "lo": np.array([cx, cy, z + lift - 0.01]),
        }
        r = self.robot
        q = {}
        q["pre"] = solve_pose(r, p["pre"], self.q_cmd)
        path_in = plan_line(r, self.world.model, self.world.data, p["pre"], p["grasp"], q["pre"], self.cfg.grasp.max_joint_speed_rad_s, True, i)
        path_up = plan_line(r, self.world.model, self.world.data, p["grasp"], p["up"], path_in.points[-1], self.cfg.grasp.max_joint_speed_rad_s, True, i)
        q["weigh"] = path_up.points[-1]
        q["hi"] = solve_pose(r, p["hi"], q["weigh"])
        q["lo"] = solve_pose(r, p["lo"], q["weigh"])
        q["back"] = solve_pose(r, p["back"], q["weigh"])
        return {"p": p, "q": q}

    # -------------------------------------------------------- weighing steps
    def _two_arrivals(self, plan: dict) -> Generator[None, None, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Torque at the weigh pose averaged over an arrival from below and one from above (cancels friction)."""
        q, p, grasp = plan["q"], plan["p"], self.cfg.grasp
        taus, gravs, stds = [], [], []
        for k in range(2):
            yield from self._settle()
            tau, std, grav = yield from self._measure(grasp.weigh_window_s)
            taus.append(tau), gravs.append(grav), stds.append(std)
            if k == 0:
                yield from self._move_line(p["up"], p["hi"], check=False)
                yield from self._move_line(p["hi"], p["up"], check=False)
        return np.mean(taus, axis=0), np.mean(stds, axis=0), np.mean(gravs, axis=0)

    def _reference(self, i: int, plan: dict) -> Tick:
        """Unloaded torque at this can's weigh pose, gripper closed and empty."""
        self.robot.gripper(True)
        p, q = plan["p"], plan["q"]
        if not self._oriented:
            yield from self._move_joint(q["lo"], check=True)  # first move out of the home pose
            self._oriented = True
        else:
            cur = self._cur()
            lo = p["lo"]
            if cur[2] < lo[2] - 1e-3:
                yield from self._move_line(cur, np.array([cur[0], cur[1], lo[2]]), check=False)
            yield from self._move_line(np.array([cur[0], cur[1], lo[2]]), lo, check=True)
        yield from self._move_line(p["lo"], p["up"], check=False)  # arrive from below, like the lift does
        tau, std, grav = yield from self._two_arrivals(plan)
        plan["ref"] = (tau, grav)
        self._log(f"PICK_{i}", "reference torque recorded at the weigh pose")

    def _grasp(self, i: int, plan: dict) -> Tick:
        """Retreat to the pre-grasp point, open, approach laterally, close and attach the can."""
        p, q, g = plan["p"], plan["q"], self.cfg.grasp
        yield from self._move_line(p["up"], p["back"], check=False)
        yield from self._move_line(p["back"], p["pre"], check=True)
        self.robot.gripper(False)
        yield from self._hold(0.8)
        yield from self._move_line(p["pre"], p["grasp"], ignore_can=i)
        yield from self._settle()
        self.robot.gripper(True)
        yield from self._hold(0.8)
        self.robot.attach(i)
        yield from self._hold(0.2)
        self._log(f"PICK_{i}", f"can {i} grasped laterally at {np.round(p['grasp'], 3).tolist()}")

    def _lift(self, i: int, plan: dict) -> Tick:
        yield from self._move_line(plan["p"]["grasp"], plan["p"]["up"], ignore_can=i)
        self._log(f"LIFT_{i}", "lifted")

    def _weigh(self, i: int, plan: dict) -> Generator[None, None, MassEstimate]:
        tau_ref, grav_ref = plan["ref"]
        tau, std, grav = yield from self._two_arrivals(plan)
        delta = (tau - grav) - (tau_ref - grav_ref)
        est = estimate_mass(delta, self.robot.jacobian_at_pinch(), self.torque_std_nm)
        self.result.estimates[i] = est
        self._log(f"WEIGH_{i}", f"mass estimate {est.mass_kg:.3f} +- {est.std_kg:.3f} kg")
        return est

    def _release(self) -> Tick:
        """Open the gripper first (the weld still holds the can, collisions off), then drop the weld.

        Dropping the weld while the can still overlaps the finger pads would push it away violently.
        """
        self.robot.gripper(False)
        yield from self._hold(0.8)
        self.robot.detach()
        yield from self._hold(0.3)

    def _place_back(self, i: int, plan: dict) -> Tick:
        p = plan["p"]
        yield from self._move_line(p["up"], p["grasp"], ignore_can=i)
        yield from self._settle()
        yield from self._release()
        yield from self._move_line(p["grasp"], p["pre"], ignore_can=i)
        self._log(f"PLACE_BACK_{i}", f"can {i} put back")

    # ------------------------------------------------------------- endings
    def _carry_to(self, i: int, plan: dict, dest: np.ndarray) -> Tick:
        """Grasp can i (from the pre-grasp point), lift, and carry at lifted height to above ``dest`` xy."""
        p, q = plan["p"], plan["q"]
        yield from self._transfer_to_pre(plan)
        self.robot.gripper(False)
        yield from self._hold(0.6)
        yield from self._move_line(p["pre"], p["grasp"], ignore_can=i)
        yield from self._settle()
        self.robot.gripper(True)
        yield from self._hold(0.8)
        self.robot.attach(i)
        yield from self._hold(0.2)
        yield from self._move_line(p["grasp"], p["up"], ignore_can=i)

    def _sort(self, order: list[int], plans: dict) -> Tick:
        st, z, c, lift = self.station, self.z, self.cfg.grasp.approach_clearance_m, self.cfg.grasp.lift_height_m
        for i, label, (bx, by) in ((order[0], "heavy", st.bin_heavy_xy_m), (order[1], "light", st.bin_light_xy_m)):
            plan = plans[i]
            yield from self._carry_to(i, plan, np.array([bx, by, z]))
            above = np.array([bx, by, z + lift])
            place = np.array([bx, by, z + 0.004])
            yield from self._move_line(plan["p"]["up"], above, ignore_can=i)
            yield from self._move_line(above, place, ignore_can=i)
            yield from self._settle()
            yield from self._release()
            retreat = np.array([bx - c, by, z + 0.004])
            yield from self._move_line(place, retreat, ignore_can=i)
            self._log("ACT", f"can {i} sorted into the {label} bin")
        self.result.done = True

    def _handover(self, heavy: int, plan: dict) -> Tick:
        st, lift = self.station, self.cfg.grasp.lift_height_m
        yield from self._carry_to(heavy, plan, np.array(st.handover_xyz_m))
        hand_pose = np.array(st.handover_xyz_m)
        yield from self._move_line(plan["p"]["up"], hand_pose, ignore_can=heavy)
        yield from self._settle()
        window = []
        for _ in range(int(round(1.0 / self.dt))):  # baseline torque while holding still at the hand-over pose
            self._command(self.q_cmd)
            yield
            window.append(self.robot.measured_torque())
        detector = ContactDetector.from_window(np.array(window))
        self._log("ACT", f"presenting can {heavy}; waiting for contact")
        far = self.world.user_hand_pos(self.name)
        near = self.robot.pinch_pose()[0] + np.array([0.07, 0.0, 0.0])
        t, contact, pulling = 0.0, False, False
        while t < self.handover_timeout_s and not contact:
            if self.user_touches and t >= self.user_delay_s:
                alpha = min(1.0, (t - self.user_delay_s) / 1.5)
                self.world.set_user_hand(self.name, far + alpha * (near - far))
                if alpha >= 1.0 and not pulling:
                    self.world.push_can(self.name, heavy, (4.0, 0.0, 2.0))  # the user pulls on the can
                    pulling = True
            self._command(self.q_cmd)
            yield
            t += self.dt
            contact = detector.update(self.robot.measured_torque())
        self.world.push_can(self.name, heavy, (0.0, 0.0, 0.0))
        if contact:
            self.result.released_on_contact = True
            self.world.user_takes_can(self.name, heavy)
            yield from self._release()
            self._log("ACT", f"contact felt (score {detector.score:.1f}); gripper released")
            hand_far = np.array([0.95, st.y0_m, 1.15])
            start = self.world.user_hand_pos(self.name)
            for k in range(int(1.5 / self.dt)):
                self.world.set_user_hand(self.name, start + (k + 1) / (1.5 / self.dt) * (hand_far - start))
                self._command(self.q_cmd)
                yield
            self.result.done = True
        else:
            # No contact: never open on a timeout. Put the can in the heavy bin instead.
            self.result.fallback_used = True
            self._log("ACT", "no contact felt; placing the can in the heavy bin instead")
            bx, by = st.bin_heavy_xy_m
            z, c = self.z, self.cfg.grasp.approach_clearance_m
            above, place = np.array([bx, by, z + lift]), np.array([bx, by, z + 0.004])
            yield from self._move_line(hand_pose, above, ignore_can=heavy)
            yield from self._move_line(above, place, ignore_can=heavy)
            yield from self._settle()
            yield from self._release()
            yield from self._move_line(place, np.array([bx - c, by, z + 0.004]), ignore_can=heavy)
            self.result.done = True

    # ---------------------------------------------------------------- driver
    def run(self) -> Generator[None, None, TaskResult]:
        try:
            plans: dict[int, dict] = {}
            yield from self._wait_phase("INIT")
            for i in (1, 2):
                yield from self._wait_phase(f"PICK_{i}")
                plans[i] = self._poses(i)
                yield from self._reference(i, plans[i])
                yield from self._grasp(i, plans[i])
                yield from self._wait_phase(f"LIFT_{i}")
                yield from self._lift(i, plans[i])
                yield from self._wait_phase(f"WEIGH_{i}")
                yield from self._weigh(i, plans[i])
                yield from self._wait_phase(f"PLACE_BACK_{i}")
                yield from self._place_back(i, plans[i])
            yield from self._wait_phase("COMPARE")
            decision = heavier(self.result.estimates[1], self.result.estimates[2])
            self.result.decision = decision
            self._log("COMPARE", {1: "can 1 is heavier", 2: "can 2 is heavier", 0: "indistinguishable"}[decision])
            yield from self._wait_phase("ACT")
            if decision == 0:
                self._log("ACT", "masses cannot be told apart; no sorting or hand-over")
                self.result.done = True
            else:
                light = 2 if decision == 1 else 1
                if self.station.ending == "sort":
                    yield from self._sort([decision, light], plans)
                else:
                    yield from self._handover(decision, plans[decision])
        except MotionError as exc:
            self.result.error = str(exc)
            self._log("ERROR", str(exc))
        for i in (1, 2):
            c = self.world.can_pos(self.name, i)
            self.result.final_can_xy[i] = (round(float(c[0]), 3), round(float(c[1]), 3))
        return self.result
