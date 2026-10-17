"""
Unit tests for the weigh-and-sort demo: prompt parsing, state machine, weighing, contact
detection, and the OpenArm adapter / full physics run in the demo scene.
"""

import math
import os
import sys
import unittest

import numpy as np
import yaml

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.task import (
    ContactConfig,
    DemoConfig,
    DemoMode,
    ExternalContactDetector,
    MassEstimate,
    ObjectDetection,
    Phase,
    TcpTarget,
    WeighAndSortTask,
    compare_estimates,
    parse_prompt,
)
from src.task.mocks import MockManipulationSkills, MockObjectDetector, MockPayloadEstimator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(REPO, "configs", "task_config.yaml")
HAS_MODEL = os.path.exists(os.path.join(REPO, "third_party/openarm/openarm_mujoco/v1/openarm_bimanual.xml"))


def load_cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_task(prompt="put the lighter one in the left bin", masses=(0.35, 0.02), detections=None, contact_at_s=3.0,
              grasp_succeeds=True, noise_kg=0.01):
    cfg = load_cfg()
    cfg["demo"]["contact"]["timeout_s"] = 6.0
    if detections is None:
        detections = [ObjectDetection("soda can", 0.21, 0.13, confidence=0.8),
                      ObjectDetection("soda can", 0.21, 0.27, confidence=0.8)]
    skills = MockManipulationSkills(grasp_succeeds=grasp_succeeds, contact_at_s=contact_at_s)
    estimator = MockPayloadEstimator({f"object_{i + 1}": m for i, m in enumerate(masses)}, noise_kg=noise_kg)
    config = DemoConfig.from_config(cfg)
    task = WeighAndSortTask(MockObjectDetector(detections), skills, estimator, config, parse_prompt(prompt))
    return task, skills, estimator, cfg


class TestPrompt(unittest.TestCase):
    def test_light_bin(self):
        it = parse_prompt("Sort the two cans: put the lighter one in the left bin.")
        self.assertEqual(it.mode, DemoMode.SORT)
        self.assertEqual(it.bin_for, {"light": "left", "heavy": "right"})
        self.assertEqual(it.object_prompt, "soda can")

    def test_heavy_bin_by_color(self):
        it = parse_prompt("The full box goes in the red bin")
        self.assertEqual(it.bin_for, {"heavy": "left", "light": "right"})
        self.assertEqual(it.object_prompt, "cardboard box")

    def test_handover(self):
        it = parse_prompt("Give me the heavy can")
        self.assertEqual(it.mode, DemoMode.HANDOVER)
        self.assertEqual(it.handover_class, "heavy")
        it = parse_prompt("Hand me the empty one, the other goes in the right bin")
        self.assertEqual(it.handover_class, "light")

    def test_contradiction_raises(self):
        with self.assertRaises(ValueError):
            parse_prompt("put the light one in the left bin and the heavy one in the left bin")


class TestWeighing(unittest.TestCase):
    def test_compare_picks_heavier_and_confidence(self):
        a = MassEstimate("a", 0.30, 0.01)
        b = MassEstimate("b", 0.05, 0.01)
        cmp = compare_estimates(b, a, min_z=3.0)
        self.assertEqual(cmp.heavier.label, "a")
        self.assertAlmostEqual(cmp.diff_kg, 0.25)
        self.assertTrue(cmp.conclusive)
        self.assertFalse(compare_estimates(MassEstimate("a", 0.30, 0.1), MassEstimate("b", 0.29, 0.1), 3.0).conclusive)


class TestContactDetector(unittest.TestCase):
    def setUp(self):
        self.cfg = ContactConfig(k_sigma=4.0, min_delta_nm=0.25, hold_s=0.05, rate_hz=100.0)
        self.rng = np.random.default_rng(1)
        self.base = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, -0.5])

    def test_noise_does_not_trigger(self):
        det = ExternalContactDetector(self.cfg)
        det.set_baseline(self.base + self.rng.normal(0, 0.02, (100, 7)))
        self.assertFalse(any(det.update(self.base + self.rng.normal(0, 0.02, 7)) for _ in range(2000)))

    def test_sustained_shift_triggers_but_spike_does_not(self):
        det = ExternalContactDetector(self.cfg)
        det.set_baseline(self.base + self.rng.normal(0, 0.02, (100, 7)))
        spike = self.base.copy()
        spike[3] += 1.0
        self.assertFalse(det.update(spike))
        self.assertFalse(det.update(self.base))
        self.assertTrue([det.update(spike) for _ in range(5)][-1])


class TestWeighAndSortTask(unittest.TestCase):
    def _phases(self, report):
        return [p for p, _ in report.phase_log]

    def test_sequence_weighs_first_puts_back_then_compares_holding_second(self):
        task, skills, est, _ = make_task()
        report = task.run()
        self.assertEqual(report.outcome, Phase.DONE)
        self.assertEqual(est.measured, ["object_1", "object_2"])
        ph = self._phases(report)
        # object 1 is put back before object 2 is picked; the comparison comes right after weighing object 2
        self.assertLess(ph.index(Phase.PLACE_BACK), [i for i, p in enumerate(ph) if p is Phase.APPROACH][1])
        self.assertEqual(ph[ph.index(Phase.COMPARE) - 1], Phase.WEIGH)

    def test_sort_light_bin_from_prompt(self):
        task, skills, _, cfg = make_task(prompt="put the lighter one in the left bin", masses=(0.35, 0.02))
        report = task.run()
        self.assertEqual(report.placements, {"object_2": "left", "object_1": "right"})
        # object 2 (held) is binned first, then object 1
        self.assertAlmostEqual(skills.released_at[-2].y, cfg["demo"]["bins"]["left"]["y"])
        self.assertAlmostEqual(skills.released_at[-1].y, cfg["demo"]["bins"]["right"]["y"])

    def test_sort_heavy_bin_from_prompt(self):
        report = make_task(prompt="heavy one in the left bin", masses=(0.02, 0.35))[0].run()
        self.assertEqual(report.placements, {"object_2": "left", "object_1": "right"})

    def test_handover_held_heavy_object_after_contact(self):
        task, skills, _, _ = make_task(prompt="give me the heavy can", masses=(0.02, 0.35))
        report = task.run()
        self.assertTrue(report.handover_confirmed)
        self.assertEqual(report.placements, {"object_2": "user", "object_1": "left"})  # light -> default light bin
        ph = self._phases(report)
        self.assertLess(ph.index(Phase.WAIT_CONTACT), ph.index(Phase.RELEASE))

    def test_handover_first_object_when_it_is_the_heavy_one(self):
        report = make_task(prompt="give me the heavy can", masses=(0.35, 0.02))[0].run()
        self.assertEqual(report.placements, {"object_2": "left", "object_1": "user"})

    def test_handover_without_contact_goes_to_heavy_bin(self):
        task, skills, _, cfg = make_task(prompt="give me the heavy can", masses=(0.02, 0.35), contact_at_s=None)
        report = task.run()
        self.assertFalse(report.handover_confirmed)
        self.assertEqual(report.placements["object_2"], "right")  # heavy bin
        h = cfg["demo"]["handover"]
        self.assertTrue(all(abs(t.x - h["x"]) > 1e-9 for t in skills.released_at if t is not None))  # never opened at the offer pose

    def test_inconclusive_comparison_puts_back_and_does_not_sort(self):
        task, skills, est, _ = make_task(masses=(0.20, 0.20), noise_kg=0.05)
        report = task.run()
        self.assertEqual(report.outcome, Phase.INCONCLUSIVE)
        self.assertEqual(report.placements, {})
        self.assertEqual(est.measured, ["object_1", "object_2", "object_2"])  # held object re-measured once

    def test_wrong_detection_count_aborts(self):
        task, skills, _, _ = make_task(detections=[ObjectDetection("soda can", 0.21, 0.2)])
        report = task.run()
        self.assertEqual(report.outcome, Phase.ABORTED)
        self.assertIn("abort", skills.call_names())
        self.assertNotIn("grasp", skills.call_names())

    def test_failed_grasp_aborts(self):
        report = make_task(grasp_succeeds=False)[0].run()
        self.assertEqual(report.outcome, Phase.ABORTED)
        self.assertIn("grasp", report.error)


@unittest.skipUnless(HAS_MODEL, "official OpenArm sources not in third_party/openarm")
class TestOpenArmAdapterInSim(unittest.TestCase):
    """Adapter checks and full physics runs in the demo scene (real cans, real weighing)."""

    @classmethod
    def setUpClass(cls):
        from scripts.run_weigh_demo import ensure_demo_scene

        cls.cfg = load_cfg()
        cls.scene = str(ensure_demo_scene(cls.cfg))

    def _setup(self):
        from src.planning.openarm_kinematics import OpenArmKinematics
        from src.simulation.openarm_sim import OpenArmSim
        from src.simulation.sim_user import SimObjectDetector, SimulatedUser, SimUserBackend
        from src.task.openarm_adapter import OpenArmPayloadEstimator, OpenArmSkills
        from src.task.workcell import WorkcellGeometry

        cfg = load_cfg()
        cfg["robot"]["sim_scene_path"] = self.scene
        geo = WorkcellGeometry.from_config(cfg)
        sim = OpenArmSim.from_config(cfg)  # stepped
        h, u = cfg["demo"]["handover"], cfg["demo"]["sim"]["user"]
        user = SimulatedUser(sim, geo.to_world([h["x"], h["y"], geo.table_top_z_m + h["above_table_m"]]),
                             ["can_1", "can_2"], u["grasp_after_s"], u["pull_force_n"],
                             geo.to_world([0.60, 0.25, geo.table_top_z_m + 0.06]))
        robot = SimUserBackend(sim, user)
        kin = OpenArmKinematics(cfg["robot"]["mjcf_path"])
        skills = OpenArmSkills(robot, cfg, "left", kin, collision_scene=self.scene)
        det = SimObjectDetector(sim, ["can_1", "can_2"], geo.base_z_in_world_m, geo.table_top_world_m)
        return cfg, sim, skills, OpenArmPayloadEstimator(robot, kin, "left", skills.motion), det

    def test_unsafe_targets_rejected_before_moving(self):
        from src.task.openarm_adapter import UnsafeMotion

        cfg, sim, skills, _, _ = self._setup()
        sim.start()
        try:
            skills.go_home()
            before = sim.get_joint_targets().copy()
            for target in (TcpTarget(0.50, 0.45, 0.20), TcpTarget(0.40, 0.20, 0.05),  # out of reach / box
                           TcpTarget(0.24, 0.0, 0.21, 0.0, math.radians(45))):         # forearm into the torso
                with self.assertRaises(UnsafeMotion):
                    skills.move_to(target)
            np.testing.assert_allclose(sim.get_joint_targets(), before)
            with self.assertRaises(UnsafeMotion):  # straight from home down to zero sweeps through the table
                skills.check_joint_path(skills.home["left"], np.zeros(7))
        finally:
            sim.stop()

    def _run(self, prompt):
        cfg, sim, skills, est, det = self._setup()
        task = WeighAndSortTask(det, skills, est, DemoConfig.from_config(cfg), parse_prompt(prompt))
        sim.start()
        try:
            report = task.run()
            skills.park()
        finally:
            sim.stop()
        cans = {n: sim.data.xpos[sim.model.body(n).id].copy() for n in ("can_1", "can_2")}
        return cfg, report, cans

    def test_physics_sort_weighs_and_bins_both_cans(self):
        cfg, report, cans = self._run("put the lighter one in the left bin")
        self.assertEqual(report.outcome, Phase.DONE, report.error)
        m1, m2 = (r.estimate.mass_kg for r in report.objects)
        true1, true2 = cfg["demo"]["sim"]["masses_kg"]
        self.assertAlmostEqual(m1, true1, delta=0.03)
        self.assertAlmostEqual(m2, true2, delta=0.03)
        heavy, light = ("can_1", "can_2") if true1 > true2 else ("can_2", "can_1")
        for can, b in ((heavy, "right"), (light, "left")):
            bx, by = cfg["demo"]["bins"][b]["x"], cfg["demo"]["bins"][b]["y"]
            self.assertLess(np.hypot(cans[can][0] - bx, cans[can][1] - by), 0.04, f"{can} not in the {b} bin")

    def test_physics_handover_releases_on_simulated_user(self):
        cfg, report, cans = self._run("give me the heavy can")
        self.assertEqual(report.outcome, Phase.DONE, report.error)
        self.assertTrue(report.handover_confirmed)
        self.assertIn("user", report.placements.values())


if __name__ == "__main__":
    unittest.main()
