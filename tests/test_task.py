"""FR3-FR6, SR6, NFR5: the demo state machine on scripted skills (no robot)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robotokens.sensing.payload import MassEstimate
from robotokens.task.demo import DemoTask, Phase, save_report
from robotokens.task.prompt import Mode, TaskIntent, parse_prompt
from robotokens.task.skills import Detection, ManualDetector


class FakeSkills:
    """Scripted skills: masses per grasp order, optional contact time, optional failure."""

    def __init__(self, masses, se=0.005, contact_after_s=None, fail_on=None, fail_with=RuntimeError):
        self.masses = list(masses)
        self.fail_with = fail_with
        self.se = se
        self.contact_after_s = contact_after_s
        self.fail_on = fail_on
        self.calls = []
        self.held = None
        self.grasps = 0
        self.t = 0.0
        self.t_arrive = 0.0

    def _call(self, name, *args):
        self.calls.append((name, *args))
        if name == self.fail_on:
            self.fail_on = None  # fail once
            raise self.fail_with(f"{name} failed")

    def go_home(self):
        self._call("go_home")

    def move_to(self, target):
        self._call("move_to", round(target.x, 3), round(target.y, 3), round(target.z, 3))
        self.t_arrive = self.t

    def grasp(self):
        self._call("grasp")
        self.held = self.grasps
        self.grasps += 1
        return True

    def release(self):
        self._call("release")
        self.held = None

    def to_weigh_pose(self):
        self._call("to_weigh_pose")

    def measure(self, label):
        self._call("measure", label)
        order = {"object_1": 0, "object_2": 1}[label]
        return MassEstimate(label=label, mass_kg=self.masses[order], se_kg=self.se)

    def read_torques(self):
        tau = np.zeros(7) + np.random.default_rng(len(self.calls)).normal(0, 0.01, 7)
        if self.contact_after_s is not None and self.t - self.t_arrive >= self.contact_after_s:
            tau[3] += 1.0
        return tau

    def wait(self, dt):
        self.t += dt

    def abort(self):
        self._call("abort")

    def names(self):
        return [c[0] for c in self.calls]


OBJECTS = [Detection(0.21, 0.13), Detection(0.21, 0.27)]


def run(cfg, prompt, skills, objects=OBJECTS):
    task = DemoTask(ManualDetector(objects), skills, cfg, parse_prompt(prompt))
    return task.run()


def release_xy(cfg, name):
    b = cfg.workcell.bins[name]
    return round(b.x, 3), round(b.y, 3), round(cfg.table_z + cfg.task.bin_release_above_table_m, 3)


def test_sort_light_left(cfg):
    sk = FakeSkills(masses=[0.36, 0.06])  # object_1 heavy
    rep = run(cfg, "put the lighter one in the left bin", sk)
    assert rep.outcome is Phase.DONE
    assert rep.placements == {"object_2": "left", "object_1": "right"}
    assert [o.weight_class for o in rep.objects] == ["heavy", "light"]
    assert sk.names().count("measure") == 2 and sk.names().count("grasp") == 3
    assert ("move_to", *release_xy(cfg, "left")) in sk.calls


def test_beliefs_store_masses(cfg):
    rep = run(cfg, "sort", FakeSkills(masses=[0.06, 0.36]))
    b1, b2 = rep.objects
    assert (b1.x, b1.y) == (0.21, 0.13)
    assert b1.mass.mass_kg == 0.06 and b2.mass.mass_kg == 0.36
    assert rep.comparison.heavier == "object_2"


def test_handover_releases_on_contact(cfg):
    sk = FakeSkills(masses=[0.06, 0.36], contact_after_s=2.0)  # heavy is held second
    rep = run(cfg, "give me the heavy can", sk)
    assert rep.outcome is Phase.DONE and rep.handover_confirmed
    assert rep.placements == {"object_2": "user", "object_1": "left"}


def test_handover_timeout_puts_object_in_its_bin(cfg):
    rep = run(cfg, "give me the heavy can", FakeSkills(masses=[0.06, 0.36]))
    assert rep.outcome is Phase.DONE and rep.handover_confirmed is False
    assert rep.placements["object_2"] == "right"


def test_handover_of_first_object(cfg):
    sk = FakeSkills(masses=[0.36, 0.06], contact_after_s=1.0)  # heavy is object_1, put back first
    rep = run(cfg, "give me the heavy can", sk)
    assert rep.placements == {"object_2": "left", "object_1": "user"}


def test_inconclusive_puts_back_and_goes_home(cfg):
    sk = FakeSkills(masses=[0.10, 0.101], se=0.01)
    rep = run(cfg, "sort", sk)
    assert rep.outcome is Phase.INCONCLUSIVE
    assert sk.names().count("measure") == 3  # re-measured once
    assert sk.names()[-1] == "go_home" and sk.held is None


def test_wrong_object_count_aborts_before_moving(cfg):
    sk = FakeSkills(masses=[0.1, 0.2])
    rep = run(cfg, "sort", sk, objects=OBJECTS[:1])
    assert rep.outcome is Phase.ABORTED and "expected 2" in rep.error
    assert "grasp" not in sk.names()


def test_failure_while_holding_puts_object_back_then_aborts(cfg):  # SR6
    sk = FakeSkills(masses=[0.36, 0.06], fail_on="to_weigh_pose")
    rep = run(cfg, "sort", sk)
    assert rep.outcome is Phase.ABORTED
    assert sk.names()[-1] == "abort"
    assert sk.held is None


def test_ctrl_c_while_holding_puts_object_back(cfg):  # SR6
    sk = FakeSkills(masses=[0.36, 0.06], fail_on="measure", fail_with=KeyboardInterrupt)
    rep = run(cfg, "sort", sk)
    assert rep.outcome is Phase.ABORTED and "KeyboardInterrupt" in rep.error
    assert sk.held is None and sk.names()[-1] == "abort"


def test_report_json(cfg, tmp_path):
    rep = run(cfg, "sort", FakeSkills(masses=[0.36, 0.06]))
    path = save_report(rep, tmp_path)
    data = json.loads(path.read_text())
    assert data["outcome"] == "done"
    assert data["intent"]["mode"] == "sort"
    assert data["objects"][0]["mass"]["mass_kg"] == 0.36
    assert data["comparison"]["heavier"] == "object_1"
    assert data["phase_log"][0][0] == "detect"


def test_unknown_bin_rejected(cfg):
    intent = TaskIntent(mode=Mode.SORT, bin_for={"light": "middle", "heavy": "right"},
                        handover_class="heavy", object_prompt="soda can", text="")
    with pytest.raises(ValueError, match="middle"):
        DemoTask(ManualDetector(OBJECTS), FakeSkills([0, 0]), cfg, intent)


def test_rehearse_moves_without_grasping(cfg):
    sk = FakeSkills(masses=[0, 0])
    objs = DemoTask(ManualDetector(OBJECTS), sk, cfg, parse_prompt("sort")).rehearse()
    assert len(objs) == 2
    assert set(sk.names()) == {"release", "go_home", "move_to"} and "grasp" not in sk.names()
    lowest = min(c[3] for c in sk.calls if c[0] == "move_to")
    assert lowest > cfg.table_z + cfg.task.grasp.height_above_table_m  # never at grasp height


@pytest.mark.parametrize("mode", ["run", "rehearse"])
def test_gripper_opened_before_any_motion(cfg, mode):  # FR3
    sk = FakeSkills(masses=[0.36, 0.06])
    getattr(DemoTask(ManualDetector(OBJECTS), sk, cfg, parse_prompt("sort")), mode)()
    assert sk.names()[0] == "release"
