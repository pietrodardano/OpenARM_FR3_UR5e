#!/usr/bin/env python3
"""Run the "vision cannot weigh" demo: weigh two look-alike objects, then sort or hand over.

    python scripts/run_weigh_demo.py                                              # mocks only
    mjpython scripts/run_weigh_demo.py --backend sim --render --yes               # MuJoCo, macOS viewer
    python scripts/run_weigh_demo.py --backend sim --render --yes --prompt "Give me the heavy can"
    python scripts/run_weigh_demo.py --backend real --mirror                      # real left arm + twin

The instruction is free text (``--prompt``, default ``demo.prompt``), e.g. "put the lighter can
in the left bin" or "hand me the heavy one" (src/task/prompt.py).

Backends:

- ``mock``: no robot, no physics.
- ``sim``: ``OpenArmSim`` in the demo scene (``scripts/setup_weigh_demo_scene.py``, generated
  if missing): measured workcell, 72.5 cm table, two look-alike cans with different real masses,
  two bins. The cans are really gripped and lifted, the weight comes from the simulated joint
  torques, and in the handover a simulated user pulls on the offered can (``demo.sim.user``).
- ``fake``: the real driver on a simulated CAN bus (no objects).
- ``real``: ``OpenArmBimanual`` over openarm_can.

Object positions (``--detector``, default ``auto``):

- ``vision``: the RGB camera with GroundingDINO + SAM2 (src/vision/vision_detector.py), mapped
  to the base frame by ``configs/camera_extrinsics.yaml``. Default for ``--backend real``. Needs
  torch, GroundingDINO, SAM2 and the weights (requirements-vision.txt) on the machine running it.
  Before anything moves, vision is polled once and the positions are shown for confirmation.
- ``sim``: ground truth of the simulated cans (default for ``--backend sim``).
- ``manual``: ``demo.manual_objects`` or ``--objects "x,y[,yaw];x,y[,yaw]"`` (base frame, m).

Lifecycle with a robot backend: start -> table-safe route home -> task -> park beside the table
-> zero -> stop. The arm is never sent straight between zero and the table area, because the
hanging pose is below the 72.5 cm tabletop right behind its edge. If parking is not possible,
the arm holds its pose and the script asks before the driver's stop() (which moves to rest).

``--mirror`` opens a read-only MuJoCo twin of the demo scene that follows the robot's
*measured* state (the robot is the leader; src/simulation/real_mirror.py). On macOS, viewers
need ``mjpython``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.task import DemoConfig, Phase, TaskReport, WeighAndSortTask  # noqa: E402
from src.task.mocks import MockManipulationSkills, MockObjectDetector, MockPayloadEstimator  # noqa: E402
from src.task.prompt import parse_prompt  # noqa: E402
from src.task.workcell import WorkcellGeometry  # noqa: E402


def print_report(report: TaskReport) -> None:
    """Prints the phase log and the decision of one run."""
    it = report.intent
    print(f"instruction: \"{it.text}\" -> {it.mode.value}, bins {it.bin_for}"
          + (f", hand over the {it.handover_class} one" if it.mode.value == "handover" else ""))
    for phase, message in report.phase_log:
        print(f"  [{phase.value:>12}] {message}")
    for rec in report.objects:
        est = rec.estimate
        mass = f"{est.mass_kg * 1000:.0f} +- {est.mass_se_kg * 1000:.0f} g" if est else "not weighed"
        d = rec.detection
        print(f"  {rec.name}: '{d.label}' at ({d.x:.3f}, {d.y:.3f}) m -> {mass}"
              + (f" ({rec.weight_class})" if rec.weight_class else ""))
    if report.comparison:
        c = report.comparison
        print(f"  heavier: {c.heavier.label} by {c.diff_kg * 1000:.0f} g (z = {c.z:.1f}, "
              f"{'conclusive' if c.conclusive else 'inconclusive'})")
    if report.placements:
        print(f"  placements: {report.placements}")
    if report.handover_confirmed is not None:
        print(f"  handover contact felt: {report.handover_confirmed}")
    print(f"outcome: {report.outcome.value}" + (f" ({report.error})" if report.error else ""))


def parse_objects(text: str, label: str) -> List[Dict[str, Any]]:
    """Parses ``"x,y[,yaw];x,y[,yaw]"`` (base frame, m / rad) into manual object entries."""
    objects = []
    for item in text.split(";"):
        vals = [float(v) for v in item.split(",")]
        if len(vals) not in (2, 3):
            raise ValueError(f"bad object '{item}': expected x,y or x,y,yaw")
        objects.append({"label": label, "x": vals[0], "y": vals[1], "yaw": vals[2] if len(vals) == 3 else 0.0})
    return objects


def ensure_demo_scene(cfg: Dict[str, Any]) -> Path:
    """Generates the demo scene if it is missing or older than the config."""
    scene = PROJECT_ROOT / cfg["demo"]["sim"]["scene_path"]
    config_path = PROJECT_ROOT / "configs" / "task_config.yaml"
    if not scene.exists() or scene.stat().st_mtime < config_path.stat().st_mtime:
        from scripts.setup_weigh_demo_scene import generate_demo_scene

        generate_demo_scene(cfg, scene)
        print(f"Generated {scene.relative_to(PROJECT_ROOT)}")
    return scene


def main() -> int:
    """Parses arguments, builds the backend and runs the demo once."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", help="free-text instruction (default: demo.prompt)")
    parser.add_argument("--backend", choices=["mock", "sim", "fake", "real"], default="mock")
    parser.add_argument("--objects", help='object positions "x,y[,yaw];x,y[,yaw]" (implies --detector manual)')
    parser.add_argument("--detector", choices=["auto", "vision", "sim", "manual"], default="auto",
                        help="where object positions come from (auto: real -> vision, sim -> sim, else manual)")
    parser.add_argument("--render", action="store_true", help="MuJoCo viewer of the simulation (sim only)")
    parser.add_argument("--mirror", action="store_true", help="live read-only twin that follows the robot's measured state")
    parser.add_argument("--no-user", action="store_true", help="sim: nobody takes the offered object (timeout path)")
    parser.add_argument("--fast", action="store_true", help="sim: step physics as fast as possible (no viewer)")
    parser.add_argument("--yes", action="store_true", help="do not ask before moving")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "task_config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    demo_cfg = cfg["demo"]
    config = DemoConfig.from_config(cfg)
    intent = parse_prompt(args.prompt or demo_cfg["prompt"], bins=list(config.bins), default_object=config.object_prompt)
    objects = parse_objects(args.objects, intent.object_prompt) if args.objects else demo_cfg["manual_objects"]
    detector = MockObjectDetector.from_config(objects)  # replaced by the vision ObjectDetector once merged
    print(f"Instruction: \"{intent.text}\" -> {intent.mode.value}, bins {intent.bin_for}"
          + (f", hand over the {intent.handover_class} one" if intent.mode.value == "handover" else ""))

    if args.backend == "mock":
        mock_cfg = demo_cfg["mock"]
        report = WeighAndSortTask(detector, MockManipulationSkills.from_config(mock_cfg),
                                  MockPayloadEstimator.from_config(mock_cfg), config, intent).run()
        print_report(report)
        return 0 if report.outcome is Phase.DONE else 1

    from scripts.jog_joints import confirm, make_robot
    from scripts.safe_zero import go_to_zero
    from src.planning.openarm_kinematics import OpenArmKinematics
    from src.task.openarm_adapter import OpenArmPayloadEstimator, OpenArmSkills, UnsafeMotion

    geo = WorkcellGeometry.from_config(cfg)
    scene = ensure_demo_scene(cfg)  # also the collision model of the workcell, for every backend
    if args.backend == "sim":
        cfg["robot"]["sim_scene_path"] = str(scene.relative_to(PROJECT_ROOT))
    kin = OpenArmKinematics(cfg["robot"]["mjcf_path"])
    if args.backend == "sim" and args.fast:
        from src.simulation.openarm_sim import OpenArmSim

        robot: Any = OpenArmSim.from_config(cfg)  # stepped, deterministic
    else:
        robot = make_robot(cfg, args.backend, args.render)
    if args.backend == "sim" and not args.no_user:
        from src.simulation.sim_user import SimulatedUser, SimUserBackend

        h, u = demo_cfg["handover"], demo_cfg["sim"]["user"]
        robot = SimUserBackend(robot, SimulatedUser(
            robot, geo.to_world([h["x"], h["y"], geo.table_top_z_m + h["above_table_m"]]),
            [f"can_{i + 1}" for i in range(len(objects))], u.get("grasp_after_s"), u["pull_force_n"],
            geo.to_world([0.60, 0.25, geo.table_top_z_m + 0.06]), gripper_side=demo_cfg.get("side", "left")))
    source = "manual" if args.objects else args.detector
    if source == "auto":
        source = {"real": "vision", "sim": "sim"}.get(args.backend, "manual")
    if source == "sim":
        if args.backend != "sim":
            parser.error("--detector sim needs --backend sim")
        from src.simulation.sim_user import SimObjectDetector

        detector = SimObjectDetector(getattr(robot, "_robot", robot), [f"can_{i + 1}" for i in range(len(objects))],
                                     geo.base_z_in_world_m, geo.table_top_world_m, intent.object_prompt)
    elif source == "vision":
        from src.vision.vision_detector import build_vision_detector

        detector = build_vision_detector(cfg, str(PROJECT_ROOT))
        seen = detector.detect(intent.object_prompt)
        print(f"Vision sees {len(seen)} '{intent.object_prompt}' (base frame): "
              + "; ".join(f"({d.x:.3f}, {d.y:.3f}) conf {d.confidence:.2f}" for d in seen))
        if len(seen) != config.expected_objects or not confirm("Do these positions match the table?", args.yes):
            if len(seen) != config.expected_objects:
                print(f"Expected {config.expected_objects} objects: fix the scene or the prompt, or pass --objects.")
            detector.close()
            return 1
        objects = [{"label": d.label, "x": d.x, "y": d.y} for d in seen]
    print(f"Object positions from: {source}")
    skills = OpenArmSkills(robot, cfg, demo_cfg.get("side", "left"), kin, require_object=args.backend != "fake",
                           collision_scene=str(scene))
    estimator = OpenArmPayloadEstimator(robot, kin, skills.side, skills.motion)
    task = WeighAndSortTask(detector, skills, estimator, config, intent)

    mirror = None
    if args.mirror:
        from src.simulation.real_mirror import MirrorConfig, RealRobotMirror

        mcfg = MirrorConfig.from_config(cfg)
        mirror = RealRobotMirror(robot, mcfg, objects=objects)
        mirror.start()  # follows once robot.start() publishes a state

    report = None
    rc = 0
    robot.start()
    try:
        print("Objects: " + "; ".join(f"({o['x']:.2f}, {o['y']:.2f})" for o in objects))
        if confirm(f"Run the demo on the {skills.side} arm ({args.backend})?", args.yes):
            report = task.run()
    except KeyboardInterrupt:
        print("\nCtrl+C: parking beside the table...")
        skills.abort()
        rc = 130
    finally:
        try:
            if robot.get_state().fault:
                print(f"FAULT latched ({robot.get_state().fault}): holding the measured pose.")
                rc = 1
            else:
                skills.park()
                go_to_zero(robot, cfg, skills.motion.joint_speed_deg_s)
        except UnsafeMotion as exc:
            print(f"\nCannot park beside the table: {exc}\nThe arm is holding its pose. robot.stop() moves "
                  "straight to the hanging pose: clear the path (or use the e-stop) first.")
            if args.backend == "real":
                input("Press Enter when the straight path to rest is clear... ")
            rc = 1
        print("Stopping (disable motors)...")
        robot.stop()
        if mirror is not None:
            mirror.stop()
        if hasattr(detector, "close"):
            detector.close()
    if report is not None:
        print_report(report)
        if args.backend == "sim":
            from src.simulation.sim_user import can_positions_world

            names = [f"can_{i + 1}" for i in range(len(objects))]
            for n, p in zip(names, can_positions_world(getattr(robot, "_robot", robot), names)):
                print(f"  sim {n} ends at world {p.round(3).tolist()}")
        return rc or (0 if report.outcome is Phase.DONE else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
