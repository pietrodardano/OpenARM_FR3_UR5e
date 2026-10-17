#!/usr/bin/env python3
"""Run the cross-embodiment demo: simulated FR3 and UR5e weigh two cans next to the real OpenArm.

The rendered OpenArm is a follower of the real arm (its joints come from a state stream). The FR3 and
UR5e are simulation only: they know the true can positions (no vision), grasp laterally, estimate each
can's mass from simulated joint torque, then sort by weight (FR3) or hand the heavy can to a simulated
user and release on felt contact (UR5e). Each sim robot starts a phase only after the real task has
entered it, so they stay in step with the real arm but remain independent of it.

Sources for the real-arm state:  mock (scripted stand-in) | udp:PORT (live) | replay:FILE (recorded)
Examples:
  uv run python scripts/final_task/run_cross_embodiment_demo.py                      # headless, mock source
  uv run python scripts/final_task/run_cross_embodiment_demo.py --show               # with the viewer
  uv run python scripts/final_task/run_cross_embodiment_demo.py --source udp:5600 --show
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
VENV_MJPYTHON = PROJECT_ROOT / ".venv" / "bin" / "mjpython"


def _relaunch_for_viewer() -> None:
    """The passive viewer needs mjpython on macOS."""
    if sys.platform != "darwin" or os.environ.get("ROBOTTOKENS_MJPYTHON") == "1" or not VENV_MJPYTHON.is_file():
        return
    env = dict(os.environ, ROBOTTOKENS_MJPYTHON="1")
    os.execve(str(VENV_MJPYTHON), [str(VENV_MJPYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="mock", help="mock | udp:PORT | replay:FILE | none")
    parser.add_argument("--sync", choices=["follow", "free"], default="follow", help="follow: wait for the real task's phases; free: ignore them")
    parser.add_argument("--ur5e-ending", choices=["sort", "handover"], help="override the config")
    parser.add_argument("--fr3-ending", choices=["sort", "handover"], help="override the config")
    parser.add_argument("--no-touch", action="store_true", help="the simulated user never touches the can in the hand-over (tests the no-contact fallback)")
    parser.add_argument("--max-sim-s", type=float, default=400.0)
    parser.add_argument("--show", action="store_true", help="open the MuJoCo viewer and run in real time")
    parser.add_argument("--save-mock", metavar="FILE", help="write the scripted real-arm stream to FILE (JSON lines) and exit")
    args = parser.parse_args()
    if args.show:
        _relaunch_for_viewer()

    import mujoco

    from src.cross_embodiment.layout import SCENE_PATH, load_layout
    from src.cross_embodiment.mock_openarm import build_mock_script
    from src.cross_embodiment.runner import DemoRunner
    from src.cross_embodiment.stream import ReplaySource, UdpStateReceiver
    from src.cross_embodiment.task import PhaseGate, WeighTask
    from src.cross_embodiment.world import DemoWorld

    if not SCENE_PATH.is_file():
        raise SystemExit(f"Scene not found: {SCENE_PATH}\nGenerate it first: uv run python scripts/final_task/setup_cross_embodiment_scene.py")

    layout = load_layout()
    overrides = {"ur5e": args.ur5e_ending, "fr3": args.fr3_ending}
    stations = {n: dataclasses.replace(s, ending=overrides.get(n) or s.ending) for n, s in layout.stations.items()}
    layout = dataclasses.replace(layout, stations=stations)

    model_for_script = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    if args.save_mock:
        build_mock_script(model_for_script).save(args.save_mock)
        print(f"wrote the scripted stream to {args.save_mock}")
        return

    realtime = args.show
    if args.source == "mock":
        source = build_mock_script(model_for_script)
    elif args.source.startswith("replay:"):
        source = ReplaySource.from_file(args.source.split(":", 1)[1])
    elif args.source.startswith("udp"):
        port = int(args.source.split(":", 1)[1]) if ":" in args.source else 5600
        source = UdpStateReceiver(port)
        realtime = True  # a live stream arrives in real time
        print(f"listening for the real arm's state on UDP port {source.port}")
    elif args.source == "none":
        source = None
    else:
        raise SystemExit(f"unknown --source {args.source!r}")

    world = DemoWorld(layout)
    gate = PhaseGate(enabled=(args.sync == "follow" and source is not None))
    tasks = {name: WeighTask(world, name, user_touches=not args.no_touch, handover_timeout_s=12.0 if not args.no_touch else 4.0) for name in layout.sim_robots}
    runner = DemoRunner(world, tasks, source, gate)

    viewer, wall_start = None, time.perf_counter()
    on_tick = None
    if args.show:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(world.model, world.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = world.model.camera("cross_embodiment_overview").id

    if realtime:
        def on_tick(w):  # noqa: E306
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    raise KeyboardInterrupt
            delay = w.time - (time.perf_counter() - wall_start)
            if delay > 0:
                time.sleep(delay)

    try:
        results = runner.run(args.max_sim_s, on_tick)
    except KeyboardInterrupt:
        print("stopped")
        return
    finally:
        if hasattr(source, "close"):
            source.close()

    print(f"\nsimulated {world.time:.1f} s")
    if source is not None and getattr(runner, "last_message", None) is not None:
        print(f"last real-arm message: phase={runner.last_message.phase}")
    can = layout.can
    for name, result in results.items():
        station = layout.stations[name]
        print(f"\n== {name.upper()} ({station.ending}) {'done' if result.done else 'NOT finished'}{' ERROR: ' + result.error if result.error else ''}")
        for i in (1, 2):
            if i in result.estimates:
                e = result.estimates[i]
                true = station.true_mass_kg(i, can)
                print(f"   can {i}: estimated {e.mass_kg:.3f} +- {e.std_kg:.3f} kg   (true {true:.3f} kg, error {100 * (e.mass_kg - true) / true:+.0f}%)")
        verdict = {1: "can 1 is heavier", 2: "can 2 is heavier", 0: "indistinguishable", None: "-"}[result.decision]
        print(f"   decision: {verdict}")
        if station.ending == "handover":
            print(f"   hand-over: {'released after felt contact' if result.released_on_contact else ('no contact felt; heavy can put in the heavy bin' if result.fallback_used else 'not reached')}")
        for event in result.events:
            print(f"   [{event[0]:7.2f}s] {event[1]}: {event[2]}")
    if viewer is not None:
        print("\nclose the viewer window to exit")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
