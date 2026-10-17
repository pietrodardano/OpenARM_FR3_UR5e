"""Cross-embodiment demo tests: simulated FR3/UR5e running the weigh-by-touch task next to a follower OpenArm."""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cross_embodiment.contact_detector import ContactDetector  # noqa: E402
from src.cross_embodiment.layout import SCENE_PATH, load_layout  # noqa: E402
from src.cross_embodiment.mass_probe import MassEstimate, estimate_mass, heavier  # noqa: E402
from src.cross_embodiment.stream import PHASES, ReplaySource, StateMessage, UdpStateReceiver, UdpStateSender  # noqa: E402

HAS_SCENE = SCENE_PATH.is_file()
needs_scene = unittest.skipUnless(HAS_SCENE, "scene not generated: run scripts/final_task/setup_cross_embodiment_scene.py")


class TestLayout(unittest.TestCase):
    def test_layout_loads_with_two_distinct_stations(self):
        layout = load_layout()
        self.assertEqual(set(layout.sim_robots), {"ur5e", "fr3"})
        ur5e, fr3 = layout.stations["ur5e"], layout.stations["fr3"]
        self.assertNotEqual(ur5e.cans_xy_m, fr3.cans_xy_m)  # independent positions
        self.assertEqual({ur5e.ending, fr3.ending}, {"sort", "handover"})  # both task endings are exercised
        self.assertGreater(layout.can.full_kg, layout.can.empty_kg)
        self.assertEqual(ur5e.true_mass_kg(ur5e.full_can, layout.can), layout.can.full_kg)


class TestMassProbe(unittest.TestCase):
    def test_recovers_mass_from_a_loaded_minus_unloaded_torque(self):
        rng = np.random.default_rng(1)
        jacobian = rng.normal(size=(3, 6))
        true_mass = 0.36
        delta = true_mass * 9.81 * jacobian.T @ np.array([0, 0, 1.0]) + rng.normal(0, 0.01, 6)
        estimate = estimate_mass(delta, jacobian, torque_std_nm=0.1)
        self.assertAlmostEqual(estimate.mass_kg, true_mass, delta=0.02)
        self.assertGreater(estimate.std_kg, 0.0)

    def test_heavier_reports_indistinguishable_inside_the_uncertainty(self):
        a, b = MassEstimate(0.360, 0.04, 0.0), MassEstimate(0.060, 0.04, 0.0)
        self.assertEqual(heavier(a, b), 1)
        self.assertEqual(heavier(b, a), 2)
        self.assertEqual(heavier(MassEstimate(0.20, 0.05, 0.0), MassEstimate(0.22, 0.05, 0.0)), 0)


class TestContactDetector(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.baseline = self.rng.normal(0, 0.02, (100, 6)) + np.array([5, -3, 1, 0.2, 0.1, 0])

    def test_noise_does_not_trigger(self):
        detector = ContactDetector.from_window(self.baseline)
        fired = any(detector.update(self.baseline.mean(axis=0) + self.rng.normal(0, 0.02, 6)) for _ in range(500))
        self.assertFalse(fired)

    def test_sustained_change_triggers_but_a_single_spike_does_not(self):
        detector = ContactDetector.from_window(self.baseline, persist=5)
        self.assertFalse(detector.update(self.baseline.mean(axis=0) + np.array([3, 0, 0, 0, 0, 0])))
        self.assertFalse(detector.update(self.baseline.mean(axis=0)))
        results = [detector.update(self.baseline.mean(axis=0) + np.array([0, 2.0, 0, 0, 0, 0])) for _ in range(6)]
        self.assertTrue(results[-1])
        self.assertFalse(any(results[:4]))


class TestStream(unittest.TestCase):
    def test_json_round_trip_and_validation(self):
        message = StateMessage(t=1.5, q=[0.1] * 7, gripper_m=0.02, phase="WEIGH_1")
        self.assertEqual(StateMessage.from_json(message.to_json()), message)
        with self.assertRaises(ValueError):
            StateMessage.from_json('{"t": 0, "q": [0,0,0], "phase": null}')
        with self.assertRaises(ValueError):
            StateMessage.from_json('{"t": 0, "q": [0,0,0,0,0,0,0], "phase": "NOT_A_PHASE"}')

    def test_replay_source_plays_by_time(self):
        source = ReplaySource([StateMessage(0.0, [0.0] * 7, phase="INIT"), StateMessage(1.0, [1.0] * 7, phase="PICK_1")])
        self.assertIsNone(ReplaySource([StateMessage(2.0, [0.0] * 7)]).poll(1.0))
        self.assertEqual(source.poll(0.5).phase, "INIT")
        self.assertEqual(source.poll(1.2).phase, "PICK_1")

    def test_udp_loopback_keeps_the_latest_and_ignores_garbage(self):
        receiver = UdpStateReceiver(port=0, host="127.0.0.1")
        sender = UdpStateSender("127.0.0.1", receiver.port)
        try:
            sender.send(StateMessage(1.0, [0.0] * 7, phase="INIT"))
            sender._sock.sendto(b"not json", sender._addr)
            sender.send(StateMessage(2.0, [0.5] * 7, 0.03, "PICK_1"))
            time.sleep(0.1)
            latest = receiver.poll(0.0)
            self.assertEqual(latest.phase, "PICK_1")
            self.assertEqual(receiver.dropped, 1)
        finally:
            sender.close()
            receiver.close()

    def test_phase_order_is_the_documented_one(self):
        self.assertEqual(PHASES[0], "INIT")
        self.assertLess(PHASES.index("WEIGH_1"), PHASES.index("PICK_2"))
        self.assertEqual(PHASES[-1], "DONE")


@needs_scene
class TestWorld(unittest.TestCase):
    def test_arms_and_parked_arms_hold_their_pose_under_gravity(self):
        from src.cross_embodiment.world import DemoWorld

        world = DemoWorld()
        parked_start = {n: p.data.qpos[p.qadr].copy() for n, p in world.parked.items()}
        for _ in range(100):
            for robot in world.robots.values():
                robot.command(robot.home_q, tau_ff=robot.gravity_torque())
            world.step(world.layout.control_dt_s)
        for robot in world.robots.values():
            self.assertLess(np.abs(robot.q - robot.home_q).max(), 1e-3)
        for name, parked in world.parked.items():
            self.assertLess(np.abs(parked.data.qpos[parked.qadr] - parked_start[name]).max(), 5e-3)

    def test_openarm_is_a_follower_of_the_stream(self):
        from src.cross_embodiment.runner import DemoRunner
        from src.cross_embodiment.task import TaskResult
        from src.cross_embodiment.world import DemoWorld

        class Idle:  # a task that just lets a few ticks pass, so the runner's stream path is what moves the OpenArm
            gate = None
            result = TaskResult(robot="idle")

            def run(self):
                for _ in range(5):
                    yield
                return self.result

        world = DemoWorld()
        q = [0.1, -0.4, 0.2, 0.9, 0.0, -0.1, 0.3]
        DemoRunner(world, {"idle": Idle()}, ReplaySource([StateMessage(0.0, q, 0.02, "INIT")])).run(max_sim_s=1.0)
        np.testing.assert_allclose(world.data.qpos[world._oa_q_idx], q, atol=1e-9)
        np.testing.assert_allclose(world.data.qpos[world._oa_finger_idx], 0.02, atol=1e-9)


@needs_scene
class TestFullTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.cross_embodiment.runner import DemoRunner
        from src.cross_embodiment.task import WeighTask
        from src.cross_embodiment.world import DemoWorld

        cls.world = DemoWorld()
        tasks = {n: WeighTask(cls.world, n) for n in cls.world.layout.sim_robots}
        cls.results = DemoRunner(cls.world, tasks).run(max_sim_s=300)

    def test_both_robots_finish_without_motion_errors(self):
        for name, result in self.results.items():
            self.assertIsNone(result.error, f"{name}: {result.error}")
            self.assertTrue(result.done, name)

    def test_mass_estimates_separate_full_from_empty(self):
        layout = self.world.layout
        for name, result in self.results.items():
            station = layout.stations[name]
            for i in (1, 2):
                true = station.true_mass_kg(i, layout.can)
                self.assertAlmostEqual(result.estimates[i].mass_kg, true, delta=0.05, msg=f"{name} can {i}")
            self.assertEqual(result.decision, station.full_can, name)

    def test_fr3_sorts_each_can_into_the_right_bin(self):
        layout, result = self.world.layout, self.results["fr3"]
        station = layout.stations["fr3"]
        light = 2 if station.full_can == 1 else 1
        heavy_xy, light_xy = station.bin_heavy_xy_m, station.bin_light_xy_m
        self.assertLess(np.hypot(*(np.array(result.final_can_xy[station.full_can]) - heavy_xy)), 0.04)
        self.assertLess(np.hypot(*(np.array(result.final_can_xy[light]) - light_xy)), 0.04)

    def test_ur5e_hands_over_only_after_felt_contact(self):
        result = self.results["ur5e"]
        self.assertTrue(result.released_on_contact)
        self.assertFalse(result.fallback_used)
        contact = [e for e in result.events if "contact felt" in e[2]][0]
        presenting = [e for e in result.events if "presenting" in e[2]][0]
        self.assertGreater(contact[0] - presenting[0], 1.5)  # not before the simulated user has arrived and pulled


@needs_scene
class TestHandoverWithoutContact(unittest.TestCase):
    def test_never_opens_on_a_timeout_and_uses_the_heavy_bin_instead(self):
        from src.cross_embodiment.runner import DemoRunner
        from src.cross_embodiment.task import WeighTask
        from src.cross_embodiment.world import DemoWorld

        world = DemoWorld()
        task = WeighTask(world, "ur5e", user_touches=False, handover_timeout_s=3.0)
        result = DemoRunner(world, {"ur5e": task}).run(max_sim_s=300)["ur5e"]
        self.assertIsNone(result.error)
        self.assertFalse(result.released_on_contact)
        self.assertTrue(result.fallback_used)
        heavy = world.layout.stations["ur5e"].bin_heavy_xy_m
        full = world.layout.stations["ur5e"].full_can
        self.assertLess(np.hypot(*(np.array(result.final_can_xy[full]) - heavy)), 0.04)


class _PhaseSchedule:
    """A fake real task: phases start at fixed times (for the sync test)."""

    def __init__(self, starts):
        self.starts = starts

    def poll(self, sim_time):
        phase = None
        for name, t in self.starts:
            if sim_time >= t:
                phase = name
        return StateMessage(sim_time, [0.0, -0.8, 0.0, 0.7, 0.0, 0.0, 0.0], 0.044, phase)


@needs_scene
class TestSync(unittest.TestCase):
    def _run(self, gated):
        from src.cross_embodiment.runner import DemoRunner
        from src.cross_embodiment.task import PhaseGate, WeighTask
        from src.cross_embodiment.world import DemoWorld

        world = DemoWorld()
        task = WeighTask(world, "fr3", gate=PhaseGate(enabled=gated))
        schedule = _PhaseSchedule([("INIT", 0.0), ("PICK_1", 2.0), ("LIFT_1", 40.0), ("WEIGH_1", 52.0)])
        runner = DemoRunner(world, {"fr3": task}, schedule, PhaseGate(enabled=gated))
        runner.run(max_sim_s=75.0)
        return {phase: t for t, phase, _ in task.result.events}

    def test_sim_robot_waits_for_the_real_phase(self):
        events = self._run(gated=True)
        self.assertGreaterEqual(events["LIFT_1"], 40.0)
        self.assertGreaterEqual(events["WEIGH_1"], 52.0)
        self.assertNotIn("PLACE_BACK_1", events)  # the real task never reached it

    def test_free_run_does_not_wait(self):
        events = self._run(gated=False)
        self.assertLess(events["LIFT_1"], 40.0)


if __name__ == "__main__":
    unittest.main()
