"""The demo: vision finds two look-alike objects, touch tells them apart (FR2-FR6, SR6, NFR5).

1. DETECT: exactly two objects (base-frame XY). They look alike: vision cannot weigh.
2. Object 1: APPROACH (above pre-grasp, pre-grasp, grasp pose), GRASP, LIFT, WEIGH at the fixed
   weigh pose; the mass goes into the belief. PLACE_BACK where it stood, RETRACT via the approach pose.
3. Object 2: the same up to WEIGH, then COMPARE with the stored belief while still holding it;
   below ``min_z`` it is measured once more. Still unclear: put it back, go home (INCONCLUSIVE).
4. Act on the intent:
   - SORT: object 2 into the bin of its class, then object 1 into the other bin.
   - HANDOVER: the requested class is offered at the handover pose and released only on felt
     contact (else it goes to its bin); the other object goes to its bin.

Any exception ends in ABORTED: a held object is put back first, then the arm parks (SR6).
This module sequences skills only; it does no motor or camera I/O itself.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config
from ..robot.motion import Target
from ..sensing.contact import wait_for_contact
from ..sensing.payload import Comparison, combine, compare
from .belief import ObjectBelief, assign_classes
from .prompt import Mode, TaskIntent
from .skills import Detector, Skills


class Phase(str, Enum):
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
    """A precondition failed (wrong number of objects, empty grasp, ...)."""


@dataclasses.dataclass
class Report:
    intent: TaskIntent
    outcome: Phase = Phase.ABORTED
    objects: List[ObjectBelief] = dataclasses.field(default_factory=list)
    comparison: Optional[Comparison] = None
    placements: Dict[str, str] = dataclasses.field(default_factory=dict)  # object -> bin or "user"
    handover_confirmed: Optional[bool] = None
    phase_log: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        def plain(v: Any) -> Any:
            if isinstance(v, Enum):
                return v.value
            if dataclasses.is_dataclass(v):
                return {k: plain(x) for k, x in dataclasses.asdict(v).items()}
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            return v

        d = {f.name: plain(getattr(self, f.name)) for f in dataclasses.fields(self)}
        d["intent"]["mode"] = self.intent.mode.value
        return d


def save_report(report: Report, runs_dir: str | Path) -> Path:
    """Writes ``<runs_dir>/<timestamp>/report.json`` and returns its path."""
    out = Path(runs_dir) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=float))
    return path


class DemoTask:
    def __init__(self, detector: Detector, skills: Skills, cfg: Config, intent: TaskIntent) -> None:
        for cls, b in intent.bin_for.items():
            if b not in cfg.workcell.bins:
                raise ValueError(f"the {cls} object is sent to bin '{b}', which is not in workcell.bins")
        self.detector = detector
        self.skills = skills
        self.cfg = cfg
        self.t = cfg.task
        self.intent = intent
        self.report = Report(intent=intent)
        self._held: Optional[ObjectBelief] = None
        self._grasp_pitch = math.radians(self.t.grasp.pitch_deg)

    # ------------------------------------------------------------ geometry
    def _log(self, phase: Phase, msg: str = "") -> None:
        self.report.phase_log.append((phase.value, msg))

    def _at(self, x: float, y: float, above_table: float, pitch: Optional[float] = None) -> Target:
        return Target(x, y, self.cfg.table_z + above_table, 0.0, self._grasp_pitch if pitch is None else pitch)

    def _grasp_target(self, o: ObjectBelief) -> Target:
        return self._at(o.x, o.y, self.t.grasp.height_above_table_m)

    def _place_back_target(self, o: ObjectBelief) -> Target:
        return self._at(o.x, o.y, self.t.grasp.height_above_table_m + self.t.place_back_offset_m)

    def _bin_target(self, name: str) -> Target:
        b = self.cfg.workcell.bins[name]
        return self._at(b.x, b.y, self.t.bin_release_above_table_m)

    def _above(self, t: Target) -> Target:
        return dataclasses.replace(t, z=t.z + self.t.approach_clearance_m)

    def _pre(self, t: Target) -> Target:
        """Pre-grasp / retreat pose: ``approach_distance_m`` back along the finger axis."""
        d, c = self.t.approach_distance_m, math.cos(t.pitch)
        return dataclasses.replace(t, x=t.x - d * c * math.cos(t.yaw), y=t.y - d * c * math.sin(t.yaw),
                                   z=t.z + d * math.sin(t.pitch))

    def _path(self, *targets: Target) -> None:
        last = None
        for t in targets:
            if t != last:
                self.skills.move_to(t)
            last = t

    # -------------------------------------------------------------- steps
    def _detect(self) -> List[ObjectBelief]:
        self._log(Phase.DETECT, self.intent.object_prompt)
        dets = [d for d in self.detector.detect(self.intent.object_prompt) if d.confidence >= self.t.min_confidence]
        if len(dets) != 2:
            raise TaskAborted(f"expected 2 objects, vision found {len(dets)}")
        dets.sort(key=lambda d: d.y)
        return [ObjectBelief(f"object_{i + 1}", d.x, d.y, d.confidence) for i, d in enumerate(dets)]

    def _pick(self, o: ObjectBelief) -> None:
        target = self._grasp_target(o)
        pre = self._pre(target)
        self._log(Phase.APPROACH, o.name)
        self.skills.release()  # open before approaching
        self._path(self._above(pre), pre, target)
        self._log(Phase.GRASP, o.name)
        if not self.skills.grasp():
            raise TaskAborted(f"grasp of {o.name} failed: the gripper closed on nothing")
        self._held = o
        self._log(Phase.LIFT, o.name)
        self.skills.move_to(self._above(target))

    def _put_down(self, target: Target) -> None:
        self._path(self._above(target), target)
        if self.t.release_settle_s > 0:
            self.skills.wait(self.t.release_settle_s)
        self.skills.release()
        self._held = None
        self._log(Phase.RETRACT)
        pre = self._pre(target)
        self._path(pre, self._above(pre))

    def _weigh(self, o: ObjectBelief) -> None:
        self.skills.to_weigh_pose()
        self._log(Phase.WEIGH, o.name)
        o.mass = self.skills.measure(o.name)

    def _to_bin(self, o: ObjectBelief) -> None:
        b = self.intent.bin_for[o.weight_class]
        self._log(Phase.SORT, f"{o.name} ({o.weight_class}) -> {b} bin")
        self._put_down(self._bin_target(b))
        self.report.placements[o.name] = b

    def _handover(self, o: ObjectBelief) -> None:
        h = self.t.handover
        offer = self._at(h.x, h.y, h.above_table_m, math.radians(h.pitch_deg))
        self._log(Phase.HANDOVER, f"{o.name} ({o.weight_class}) offered")
        self.skills.move_to(offer)
        self._log(Phase.WAIT_CONTACT, f"timeout {self.t.contact.timeout_s:.0f} s")
        felt = wait_for_contact(self.skills.read_torques, self.skills.wait, self.t.contact)
        self.report.handover_confirmed = felt
        if felt:
            self._log(Phase.RELEASE, "contact felt: releasing")
            self.skills.release()
            self._held = None
            self.report.placements[o.name] = "user"
            self._log(Phase.RETRACT)
            self.skills.move_to(self._pre(offer))
        else:
            self._log(Phase.RELEASE, "no contact felt: keeping the grip")
            self._to_bin(o)

    def _compare(self, first: ObjectBelief, second: ObjectBelief) -> Comparison:
        cmp = compare(first.mass, second.mass, self.t.weigh.min_z)
        self._log(Phase.COMPARE, f"diff={cmp.diff_kg * 1e3:.0f} g z={cmp.z:.1f}")
        if not cmp.conclusive and self.t.weigh.remeasure_once:
            second.mass = combine(second.mass, self.skills.measure(second.name))
            cmp = compare(first.mass, second.mass, self.t.weigh.min_z)
            self._log(Phase.COMPARE, f"re-measured {second.name}: diff={cmp.diff_kg * 1e3:.0f} g z={cmp.z:.1f}")
        return cmp

    # ---------------------------------------------------------------- run
    def rehearse(self) -> List[ObjectBelief]:
        """Dry path for bring-up: home, then above and at each pre-grasp pose, above both bins and
        the handover pose, home again. Never grasps, never goes below the pre-grasp poses."""
        self.skills.release()  # gripper open before any motion
        self.skills.go_home()
        objects = self._detect()
        for o in objects:
            pre = self._pre(self._grasp_target(o))
            self._path(self._above(pre), pre, self._above(pre))
        for name in self.cfg.workcell.bins:
            self.skills.move_to(self._above(self._bin_target(name)))
        h = self.t.handover
        self.skills.move_to(self._at(h.x, h.y, h.above_table_m, math.radians(h.pitch_deg)))
        self.skills.go_home()
        return objects

    def run(self) -> Report:
        """Runs the demo once. Never raises: failures end in ABORTED."""
        try:
            self.skills.release()  # gripper open (torque mode off) before any motion
            self.skills.go_home()
            first, second = self.report.objects = self._detect()

            self._pick(first)
            self._weigh(first)
            self._log(Phase.PLACE_BACK, f"{first.name}: {first.mass.mass_kg * 1e3:.0f} g remembered")
            self._put_down(self._place_back_target(first))

            self._pick(second)
            self._weigh(second)
            cmp = self.report.comparison = self._compare(first, second)
            if not cmp.conclusive:
                self._log(Phase.INCONCLUSIVE, f"z={cmp.z:.1f} < {self.t.weigh.min_z}: putting {second.name} back")
                self._put_down(self._place_back_target(second))
                self.skills.go_home()
                self.report.outcome = Phase.INCONCLUSIVE
                return self.report
            assign_classes((first, second), cmp)

            if self.intent.mode is Mode.SORT or second.weight_class != self.intent.handover_class:
                self._to_bin(second)
                self._pick(first)
                (self._handover if self.intent.mode is Mode.HANDOVER else self._to_bin)(first)
            else:
                self._handover(second)
                self._pick(first)
                self._to_bin(first)

            self._log(Phase.HOME)
            self.skills.go_home()
            self._log(Phase.DONE)
            self.report.outcome = Phase.DONE
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - every failure ends in a safe state (SR6)
            self.report.error = f"{type(exc).__name__}: {exc}"
            self._log(Phase.ABORTED, self.report.error)
            if self._held is not None:
                held, self._held = self._held, None
                try:
                    self._log(Phase.PLACE_BACK, f"{held.name}: putting it back after the abort")
                    self._put_down(self._place_back_target(held))
                except Exception as exc2:  # noqa: BLE001
                    self._log(Phase.ABORTED, f"could not put {held.name} back: {exc2}")
            self.skills.abort()
            self.report.outcome = Phase.ABORTED
        return self.report
