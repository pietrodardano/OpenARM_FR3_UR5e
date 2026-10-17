"""
Weigh-and-sort demo: vision finds two look-alike objects, touch tells them apart.

Sequence (``docs/demo_task.md``):

1. **DETECT**: poll the vision pipeline; exactly two objects (base-frame XY on the table).
   They look alike, so vision cannot say which one is heavier.
2. **Object 1**: open the gripper, APPROACH (above the pre-grasp pose, pre-grasp, grasp
   pose), GRASP, LIFT (vertical, then the fixed weigh pose), WEIGH from the joint torques.
   The robot stores the result in its belief about object 1. PLACE_BACK at the same spot,
   release, RETRACT to the approach pose.
3. **Object 2**: the same up to WEIGH, then COMPARE right away with the stored object 1,
   while still holding object 2. Below ``min_z`` the held object is measured once more
   (optional). If the call is still unclear, object 2 is put back and nothing is sorted.
4. **Act** on the instruction (:class:`~src.task.prompt.TaskIntent`):
   - **sort**: object 2 goes straight to the bin of its weight class (the prompt names the
     light or heavy bin; the other class takes the other bin), then object 1 is picked into
     the remaining bin.
   - **handover**: the object of the requested class (default: heavy) is offered at the
     up-forward pose. The gripper opens only once the joint torques show that the user took it
     (:mod:`src.task.contact`). No contact before the timeout: it goes to its bin. The other
     object goes to its own bin, before or after the handover depending on which one is held.

This module sequences skills only. It does no motor or camera I/O itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .contact import ContactConfig, ExternalContactDetector, wait_for_contact
from .interfaces import (
    ComparisonResult,
    ManipulationSkills,
    MassEstimate,
    ObjectDetection,
    ObjectDetector,
    PayloadEstimator,
    TcpTarget,
)
from .prompt import DemoMode, TaskIntent
from .weighing import combine_estimates, compare_estimates
from .workcell import WorkcellGeometry


class Phase(str, Enum):
    """States of the demo, logged in order in :attr:`TaskReport.phase_log`."""

    DETECT = "detect"
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    WEIGH = "weigh"
    PLACE_BACK = "place_back"
    RETRACT = "retract"
    COMPARE = "compare"
    SORT = "sort"
    HANDOVER = "handover"
    WAIT_CONTACT = "wait_contact"
    RELEASE = "release"
    HOME = "home"
    DONE = "done"
    INCONCLUSIVE = "inconclusive"
    ABORTED = "aborted"


class TaskAborted(RuntimeError):
    """A precondition of the task failed (wrong detections, failed grasp, ...)."""


@dataclass
class BinPose:
    """Release location on the table, base frame."""

    x: float
    y: float
    release_above_table_m: float


@dataclass
class DemoConfig:
    """Tunables of the demo (``demo`` section of task_config.yaml) plus the table height."""

    object_prompt: str
    expected_objects: int
    min_confidence: float
    table_height_base_m: float
    grasp_height_above_table_m: float
    grasp_pitch_rad: float
    approach_distance_m: float
    approach_clearance_m: float
    place_back_offset_m: float
    release_settle_s: float
    min_z: float
    remeasure_once: bool
    bins: Dict[str, BinPose]
    handover: Tuple[float, float, float, float]  # x, y, height above table (m), pitch (rad)
    contact: ContactConfig

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DemoConfig":
        """Builds the config from the full task_config.yaml (``demo`` + ``workcell``)."""
        d = cfg["demo"]
        if int(d.get("expected_objects", 2)) != 2:
            raise ValueError("the demo compares exactly two objects: demo.expected_objects must be 2")
        if len(d["bins"]) != 2:
            raise ValueError("the demo needs exactly two bins")
        h = d["handover"]
        return cls(
            object_prompt=str(d["object_prompt"]),
            expected_objects=2,
            min_confidence=float(d["min_confidence"]),
            table_height_base_m=WorkcellGeometry.from_config(cfg).table_top_z_m,
            grasp_height_above_table_m=float(d["grasp"]["height_above_table_m"]),
            grasp_pitch_rad=math.radians(float(d["grasp"].get("pitch_deg", 0.0))),
            approach_distance_m=float(d["approach_distance_m"]),
            approach_clearance_m=float(d["approach_clearance_m"]),
            place_back_offset_m=float(d.get("place_back_offset_m", 0.0)),
            release_settle_s=float(d.get("release_settle_s", 0.0)),
            min_z=float(d["weigh"]["min_z"]),
            remeasure_once=bool(d["weigh"].get("remeasure_once", True)),
            bins={name: BinPose(float(b["x"]), float(b["y"]), float(b["release_above_table_m"]))
                  for name, b in d["bins"].items()},
            handover=(float(h["x"]), float(h["y"]), float(h["above_table_m"]), math.radians(float(h.get("pitch_deg", 0.0)))),
            contact=ContactConfig.from_dict(d.get("contact", {})),
        )


@dataclass
class ObjectRecord:
    """The robot's belief about one object: where it is, how heavy it felt, its class."""

    name: str
    detection: ObjectDetection
    estimate: Optional[MassEstimate] = None
    weight_class: Optional[str] = None  # "heavy" / "light", after the comparison


@dataclass
class TaskReport:
    """Outcome of one demo run."""

    intent: TaskIntent
    outcome: Phase = Phase.ABORTED
    objects: List[ObjectRecord] = field(default_factory=list)
    comparison: Optional[ComparisonResult] = None
    placements: Dict[str, str] = field(default_factory=dict)  # object name -> bin name or "user"
    handover_confirmed: Optional[bool] = None
    phase_log: List[Tuple[Phase, str]] = field(default_factory=list)
    error: Optional[str] = None


class WeighAndSortTask:
    """Runs the weigh-and-sort demo on any backend that satisfies the interfaces."""

    def __init__(
        self,
        detector: ObjectDetector,
        skills: ManipulationSkills,
        estimator: PayloadEstimator,
        config: DemoConfig,
        intent: TaskIntent,
    ) -> None:
        """Wires the task to its components.

        Args:
            detector: Vision pipeline (or mock).
            skills: Robot manipulation primitives (or mock).
            estimator: Torque-based payload estimator (or mock).
            config: Demo tunables.
            intent: What to do with the objects once weighed (parsed from the prompt).
        """
        for cls, b in intent.bin_for.items():
            if b not in config.bins:
                raise ValueError(f"the {cls} object is sent to bin '{b}', which is not in demo.bins")
        self.detector = detector
        self.skills = skills
        self.estimator = estimator
        self.config = config
        self.intent = intent
        self.contact_detector = ExternalContactDetector(config.contact)
        self.report = TaskReport(intent=intent)
        self._held: Optional[ObjectRecord] = None  # object currently in the gripper

    # ------------------------------------------------------------- geometry
    def _log(self, phase: Phase, message: str = "") -> None:
        """Appends a phase transition to the report."""
        self.report.phase_log.append((phase, message))

    def _table_target(self, x: float, y: float, above_table_m: float, yaw: float = 0.0,
                      pitch: Optional[float] = None) -> TcpTarget:
        """TCP target at a height above the table plane (grasp pitch unless given)."""
        p = self.config.grasp_pitch_rad if pitch is None else pitch
        return TcpTarget(x=x, y=y, z=self.config.table_height_base_m + above_table_m, yaw=yaw, pitch=p)

    def _grasp_target(self, det: ObjectDetection) -> TcpTarget:
        """TCP target that grasps ``det`` at the configured height above the table."""
        return self._table_target(det.x, det.y, self.config.grasp_height_above_table_m, det.yaw)

    def _place_back_target(self, det: ObjectDetection) -> TcpTarget:
        """Where a held object goes back down: its grasp pose, ``place_back_offset_m`` higher."""
        return self._table_target(det.x, det.y, self.config.grasp_height_above_table_m + self.config.place_back_offset_m,
                                  det.yaw)

    def _bin_target(self, name: str) -> TcpTarget:
        """Release pose of bin ``name``."""
        b = self.config.bins[name]
        return self._table_target(b.x, b.y, b.release_above_table_m)

    def _above(self, target: TcpTarget) -> TcpTarget:
        """``target`` raised by the approach clearance."""
        return TcpTarget(target.x, target.y, target.z + self.config.approach_clearance_m, target.yaw, target.pitch)

    def _pre(self, target: TcpTarget) -> TcpTarget:
        """Pre-grasp / retreat pose: ``approach_distance_m`` back along the finger axis."""
        d = self.config.approach_distance_m
        c = math.cos(target.pitch)
        return TcpTarget(target.x - d * c * math.cos(target.yaw), target.y - d * c * math.sin(target.yaw),
                         target.z + d * math.sin(target.pitch), target.yaw, target.pitch)

    def _path(self, *targets: TcpTarget) -> None:
        """Straight-line moves through ``targets``, skipping repeated waypoints."""
        last: Optional[TcpTarget] = None
        for t in targets:
            if t != last:
                self.skills.move_to(t)
            last = t

    # --------------------------------------------------------------- skills
    def _pick(self, record: ObjectRecord) -> None:
        """Opens, approaches ``record`` (above pre-grasp, pre-grasp, grasp), grasps, lifts vertically."""
        target = self._grasp_target(record.detection)
        pre = self._pre(target)
        self._log(Phase.APPROACH, record.name)
        self.skills.release()  # open before approaching
        self._path(self._above(pre), pre, target)
        self._log(Phase.GRASP, record.name)
        if not self.skills.grasp():
            raise TaskAborted(f"grasp of {record.name} failed: gripper closed on nothing")
        self._held = record
        self._log(Phase.LIFT, record.name)
        self.skills.move_to(self._above(target))

    def _put_down(self, target: TcpTarget) -> None:
        """Lowers the held object onto ``target``, releases it, backs out to the approach pose."""
        self._path(self._above(target), target)
        if self.config.release_settle_s > 0:
            self.skills.wait(self.config.release_settle_s)
        self.skills.release()
        self._held = None
        self._log(Phase.RETRACT)
        pre = self._pre(target)
        self._path(pre, self._above(pre))

    def _weigh_held(self, record: ObjectRecord) -> MassEstimate:
        """Moves the held object to the weigh pose and measures it."""
        self.skills.lift_to_weigh_pose()
        self._log(Phase.WEIGH, record.name)
        return self.estimator.measure(record.name)

    def _to_bin(self, record: ObjectRecord) -> None:
        """Puts the held ``record`` into the bin of its weight class."""
        assert record.weight_class is not None
        b = self.intent.bin_for[record.weight_class]
        self._log(Phase.SORT, f"{record.name} ({record.weight_class}) -> {b} bin")
        self._put_down(self._bin_target(b))
        self.report.placements[record.name] = b

    def _handover(self, record: ObjectRecord) -> None:
        """Offers the held ``record``; releases only on felt external contact, else bins it."""
        hx, hy, h_above, h_pitch = self.config.handover
        offer = self._table_target(hx, hy, h_above, pitch=h_pitch)
        self._log(Phase.HANDOVER, f"{record.name} ({record.weight_class}) offered to the user")
        self.skills.move_to(offer)
        self._log(Phase.WAIT_CONTACT, f"timeout {self.config.contact.timeout_s:.0f} s")
        felt = wait_for_contact(self.skills.read_joint_torques, self.skills.wait, self.config.contact,
                                self.contact_detector)
        self.report.handover_confirmed = felt
        if felt:
            self._log(Phase.RELEASE, "external contact felt: releasing")
            self.skills.release()
            self._held = None
            self.report.placements[record.name] = "user"
            self._log(Phase.RETRACT)
            self.skills.move_to(self._pre(offer))
            return
        self._log(Phase.RELEASE, "no contact felt: gripper kept closed")
        self._to_bin(record)

    def _relocate(self, record: ObjectRecord, max_shift_m: float = 0.05) -> None:
        """Polls vision again and updates where ``record`` stands (it was put back by the robot).

        The detection closest to the believed position, within ``max_shift_m``, is taken;
        otherwise the belief is kept.
        """
        dets = [d for d in self.detector.detect(self.intent.object_prompt)
                if d.confidence >= self.config.min_confidence and d.frame == "base"]
        if not dets:
            return
        best = min(dets, key=lambda d: float(math.hypot(d.x - record.detection.x, d.y - record.detection.y)))
        shift = math.hypot(best.x - record.detection.x, best.y - record.detection.y)
        if shift <= max_shift_m:
            self._log(Phase.DETECT, f"re-located {record.name} ({shift * 1e3:.0f} mm from belief)")
            record.detection = best

    def _detect(self) -> List[ObjectRecord]:
        """Polls the detector and checks the detections against the demo contract."""
        self._log(Phase.DETECT, self.intent.object_prompt)
        dets = [d for d in self.detector.detect(self.intent.object_prompt) if d.confidence >= self.config.min_confidence]
        bad = [d for d in dets if d.frame != "base"]
        if bad:
            raise TaskAborted(f"detections must be in the robot base frame, got frame '{bad[0].frame}'")
        if len(dets) != self.config.expected_objects:
            raise TaskAborted(f"expected {self.config.expected_objects} objects, vision found {len(dets)}")
        return [ObjectRecord(name=f"object_{i + 1}", detection=d) for i, d in enumerate(dets)]

    # --------------------------------------------------------------------- run
    def run(self) -> TaskReport:
        """Runs the full demo. Never raises: failures end in ``Phase.ABORTED``."""
        try:
            self.skills.go_home()
            first, second = self._detect()
            self.report.objects = [first, second]

            # Object 1: weigh, remember, put back.
            self._pick(first)
            first.estimate = self._weigh_held(first)
            self._log(Phase.PLACE_BACK, f"{first.name}: {first.estimate.mass_kg * 1e3:.0f} g remembered")
            self._put_down(self._place_back_target(first.detection))

            # Object 2: weigh and compare while holding it.
            self._pick(second)
            second.estimate = self._weigh_held(second)
            cmp = compare_estimates(first.estimate, second.estimate, self.config.min_z)
            self._log(Phase.COMPARE, f"diff={cmp.diff_kg * 1e3:.0f} g z={cmp.z:.1f}")
            if not cmp.conclusive and self.config.remeasure_once:
                second.estimate = combine_estimates(second.estimate, self.estimator.measure(second.name))
                cmp = compare_estimates(first.estimate, second.estimate, self.config.min_z)
                self._log(Phase.COMPARE, f"re-measured {second.name}: diff={cmp.diff_kg * 1e3:.0f} g z={cmp.z:.1f}")
            self.report.comparison = cmp
            if not cmp.conclusive:
                self._log(Phase.INCONCLUSIVE, f"z={cmp.z:.1f} < {self.config.min_z}: putting {second.name} back")
                self._put_down(self._place_back_target(second.detection))
                self.skills.go_home()
                self.report.outcome = Phase.INCONCLUSIVE
                return self.report

            heavy_name = cmp.heavier.label
            for rec in (first, second):
                rec.weight_class = "heavy" if rec.name == heavy_name else "light"

            if self.intent.mode is DemoMode.SORT:
                self._to_bin(second)
                self._relocate(first)
                self._pick(first)
                self._to_bin(first)
            elif second.weight_class == self.intent.handover_class:
                self._handover(second)
                self._relocate(first)
                self._pick(first)
                self._to_bin(first)
            else:
                self._to_bin(second)
                self._relocate(first)
                self._pick(first)
                self._handover(first)

            self._log(Phase.HOME)
            self.skills.go_home()
            self._log(Phase.DONE)
            self.report.outcome = Phase.DONE
        except Exception as exc:  # noqa: BLE001 - any skill fault must end in a safe state
            self.report.error = f"{type(exc).__name__}: {exc}"
            self._log(Phase.ABORTED, self.report.error)
            if self._held is not None:  # do not carry it into the park / zero moves
                held, self._held = self._held, None
                try:
                    self._log(Phase.PLACE_BACK, f"{held.name}: putting it back after the abort")
                    self._put_down(self._place_back_target(held.detection))
                except Exception as exc2:  # noqa: BLE001
                    self._log(Phase.ABORTED, f"could not put {held.name} back: {exc2}")
            self.skills.abort()
            self.report.outcome = Phase.ABORTED
        return self.report
