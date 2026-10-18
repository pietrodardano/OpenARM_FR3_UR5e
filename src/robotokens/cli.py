"""``robotokens`` command line: bring-up tools, calibration and the demo.

    robotokens check                        read joints, no enable, no motion
    robotokens zero                         safe move to the rest pose, disable
    robotokens grip                         close on / open an object held between the fingers
    robotokens weigh --labels full empty    weigh objects handed to the gripper, compare
    robotokens detect [--save]              vision: objects on the table (base frame)
    robotokens calib capture|intrinsics|touch
    robotokens run [--fake] [--objects manual|vision] [--prompt "..."] [--dry-path] [--no-twin]

Every motion command starts from and returns to the rest pose (via the side route beside the
table). Ctrl+C stops the task safely (a held object is put back first); further Ctrl+C presses
are ignored while the arm returns: use the e-stop to abort that.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from .config import REPO_ROOT, Config, load_config

log = logging.getLogger("robotokens")


def confirm(msg: str, yes: bool) -> bool:
    return yes or input(f"{msg} [Enter = yes, n = no] ").strip().lower() in ("", "y", "yes")


@contextlib.contextmanager
def ctrl_c_ignored() -> Iterator[None]:
    old = signal.signal(signal.SIGINT, lambda *_: print("  [Ctrl+C ignored while returning: use the e-stop]"))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


def make_robot(cfg: Config, fake: bool):
    """(arm, model): the configured arm on CAN, or on a FakeBus with a payload plant."""
    from .robot import make_arm
    from .robot.model import ArmModel

    if not fake:
        return make_arm(cfg), ArmModel(cfg)
    from .robot.fake import FakePlant

    g = cfg.gripper
    block = cfg.fake.object_width_m / 2 / g.open_m * g.motor_open_rad  # motor angle where the fingers stop
    arm = make_arm(cfg, fake=True, tau_fn=FakePlant(cfg, block))
    arm.bus.gripper_block_at = block
    return arm, ArmModel(cfg)


def shutdown(arm, motion, yes: bool) -> int:
    """Park beside the table, then ``arm.stop()`` (straight to rest, disable). Returns an exit code."""
    from .robot.motion import UnsafeMotion

    rc = 0
    with ctrl_c_ignored():
        try:
            if arm.fault:
                print(f"FAULT latched ({arm.fault}): holding the measured pose, not moving.")
                rc = 1
            else:
                motion.park()
        except UnsafeMotion as exc:
            print(f"Cannot park beside the table: {exc}\nThe next move goes STRAIGHT to the hanging rest pose.")
            if not yes:
                input("Clear the path (or use the e-stop), then press Enter... ")
            rc = 1
        except RuntimeError as exc:
            print(f"park failed: {exc}")
            rc = 1
        print("Stopping (rest pose, disable motors)...")
        arm.stop(rest=not arm.fault)
    return rc


# ------------------------------------------------------------------ commands
def cmd_check(cfg: Config, args: argparse.Namespace) -> int:
    from .robot import make_arm
    from .robot.model import ArmModel

    model = ArmModel(cfg)
    st = make_arm(cfg).read_passive()
    tol = cfg.hardware.startup_limit_tolerance_rad
    ok = True
    print(f"{cfg.robot.side} arm on {cfg.hardware.can_interface}")
    for i in range(7):
        inside = model.lower[i] - tol <= st.q[i] <= model.upper[i] + tol
        ok &= inside
        print(f"  joint{i + 1}: {st.q[i]:+.3f} rad  limits [{model.lower[i]:+.3f}, {model.upper[i]:+.3f}]  "
              f"status {st.motor_errors[i]:#x}{'' if inside else '  <-- OUTSIDE LIMITS'}")
    print(f"  gripper: {st.gripper * 1e3:.1f} mm  status {st.motor_errors[7]:#x}")
    ok &= all(c < 0x8 for c in st.motor_errors)
    print(f"  TCP: {np.round(model.fk(st.q)[0], 3).tolist()} m")
    print("OK: ready to stream." if ok else "NOT OK: fix motor faults / zero calibration first.")
    return 0 if ok else 1


def cmd_zero(cfg: Config, args: argparse.Namespace) -> int:
    from .robot.motion import Motion

    arm, model = make_robot(cfg, args.fake)
    arm.start()
    return shutdown(arm, Motion(arm, model, cfg), args.yes)


def cmd_grip(cfg: Config, args: argparse.Namespace) -> int:
    from .robot import gripper
    from .robot.motion import Motion

    arm, model = make_robot(cfg, args.fake)
    arm.start()
    try:
        gripper.release(arm, cfg.gripper)
        if confirm("Hold an object between the fingers. Close?", args.yes):
            opening = gripper.grip(arm, cfg.gripper)
            st = arm.state()
            print(f"holding at {cfg.gripper.grasp_torque_nm:.2f} Nm: opening {opening * 1e3:.1f} mm "
                  f"(jaw ~{2 * opening * 1e3:.0f} mm), measured torque {st.gripper_tau:+.3f} Nm")
            confirm("Hold the object, then release?", args.yes)
            gripper.release(arm, cfg.gripper)
    except KeyboardInterrupt:
        gripper.release(arm, cfg.gripper)
    return shutdown(arm, Motion(arm, model, cfg), args.yes)


def cmd_weigh(cfg: Config, args: argparse.Namespace) -> int:
    """Bring-up of the payload estimate: each object is put in the gripper at the weigh pose."""
    from .robot import gripper
    from .sensing.payload import compare
    from .task.skills import RobotSkills

    arm, model = make_robot(cfg, args.fake)
    skills = RobotSkills(arm, model, cfg)
    arm.start()
    results = {}
    try:
        skills.go_home()
        skills.to_weigh_pose()
        for label in args.labels:
            gripper.release(arm, cfg.gripper)
            if not confirm(f"Put '{label}' between the fingers. Close?", args.yes):
                break
            gripper.grip(arm, cfg.gripper)
            if not confirm("Let go of it. Measure?", args.yes):
                break
            masses = [skills.measure(label) for _ in range(args.n)]
            m = np.array([e.mass_kg for e in masses])
            results[label] = masses[-1]
            print(f"{label}: {m.mean() * 1e3:.0f} g mean, {m.std(ddof=1) * 1e3 if len(m) > 1 else float('nan'):.0f} g "
                  f"spread over {len(m)} runs (per run SE ~{np.mean([e.se_kg for e in masses]) * 1e3:.0f} g)")
            confirm(f"Take '{label}' back (the gripper opens)?", args.yes)
            gripper.release(arm, cfg.gripper)
        if len(results) == 2:
            a, b = results.values()
            c = compare(a, b, cfg.task.weigh.min_z)
            print(f"heavier: {c.heavier} by {c.diff_kg * 1e3:.0f} g, z = {c.z:.1f} "
                  f"({'conclusive' if c.conclusive else 'NOT conclusive'} at min_z {cfg.task.weigh.min_z})")
    except KeyboardInterrupt:
        print("\nCtrl+C: releasing and returning to rest.")
        gripper.release(arm, cfg.gripper)
    return shutdown(arm, skills.motion, args.yes)


def cmd_detect(cfg: Config, args: argparse.Namespace) -> int:
    import cv2

    from .task.prompt import parse_prompt
    from .vision.detector import build_vision_detector

    prompt = args.object or parse_prompt(args.prompt or cfg.task.prompt, cfg.task.object_prompt).object_prompt
    det = build_vision_detector(cfg)
    try:
        objs = det.detect(prompt)
        print(f"'{prompt}' ({det.cam.source} camera pose): {len(objs)} found")
        for o in objs:
            print(f"  x {o.x:+.3f}  y {o.y:+.3f}  conf {o.confidence:.2f}")
        out = REPO_ROOT / "runs" / "detect"
        out.mkdir(parents=True, exist_ok=True)
        view = det.last_frame.copy()
        for h in det.last_hits:
            x0, y0, x1, y1 = (int(v) for v in h.box_xyxy)
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
            if h.centroid is not None:
                cv2.drawMarker(view, tuple(int(c) for c in h.centroid), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.imwrite(str(out / "detections.png"), view)
        if args.save:
            cv2.imwrite(str(out / "frame.png"), det.last_frame)
        print(f"annotated frame: {out / 'detections.png'}")
    finally:
        det.close()
    return 0


def parse_objects(text: str):
    from .task.skills import Detection

    pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";")]
    return [Detection(x, y) for x, y in pts]


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    from .task.demo import DemoTask, Phase, save_report
    from .task.prompt import parse_prompt
    from .task.skills import Detection, ManualDetector, RobotSkills

    intent = parse_prompt(args.prompt or cfg.task.prompt, default_object=cfg.task.object_prompt)
    print(f'"{intent.text}" -> {intent.mode.value}, bins {intent.bin_for}'
          + (f", hand over the {intent.handover_class} one" if intent.mode.value == "handover" else ""))

    source = args.objects or ("manual" if args.fake else "vision")
    if args.xy:
        detections = parse_objects(args.xy)
    elif source == "manual":
        detections = [Detection(o.x, o.y) for o in cfg.task.manual_objects]
    else:
        from .vision.detector import build_vision_detector

        vision = build_vision_detector(cfg)
        try:
            detections = vision.detect(intent.object_prompt)
        finally:
            vision.close()
        print(f"vision ({vision.cam.source} camera pose): "
              + "; ".join(f"({d.x:.3f}, {d.y:.3f}) conf {d.confidence:.2f}" for d in detections))
        if not confirm("Do these positions match the table?", args.yes):
            return 1

    arm, model = make_robot(cfg, args.fake)
    skills = RobotSkills(arm, model, cfg)
    task = DemoTask(ManualDetector(detections), skills, cfg, intent)
    twin = None
    if not args.no_twin:
        from .twin.mirror import Twin

        twin = Twin(cfg, arm, objects=[(d.x, d.y) for d in detections])
        twin.start()
    report = None
    rc = 0
    arm.start()
    try:
        if confirm(f"Run on the {cfg.robot.side} arm ({'fake' if args.fake else 'REAL'}"
                   f"{', dry path' if args.dry_path else ''})?", args.yes):
            if args.dry_path:
                task.rehearse()
            else:
                report = task.run()
    except KeyboardInterrupt:
        print("\nCtrl+C: parking.")
        with ctrl_c_ignored():
            skills.abort()
    finally:
        rc = shutdown(arm, skills.motion, args.yes)
        if twin is not None:
            twin.stop()
    if report is not None:
        path = save_report(report, REPO_ROOT / "runs")
        print(f"outcome: {report.outcome.value}" + (f" ({report.error})" if report.error else ""))
        for o in report.objects:
            m = f"{o.mass.mass_kg * 1e3:.0f} +- {o.mass.se_kg * 1e3:.0f} g" if o.mass else "not weighed"
            print(f"  {o.name} at ({o.x:.3f}, {o.y:.3f}): {m}, {o.weight_class or '-'} -> "
                  f"{report.placements.get(o.name, '-')}")
        print(f"report: {path}")
        rc = rc or (0 if report.outcome is Phase.DONE else 1)
    return rc


def cmd_calib(cfg: Config, args: argparse.Namespace) -> int:
    from .vision import calib_cli

    if args.what == "capture":
        return calib_cli.capture(cfg, Path(args.out), args.square_mm, args.marker_mm)
    if args.what == "intrinsics":
        return calib_cli.intrinsics(cfg, Path(args.images), args.square_mm, args.marker_mm)
    return calib_cli.touch(cfg, Path(args.load) if args.load else None, args.reclick)


# --------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robotokens", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/robotokens.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def motion_cmd(name: str, help_: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--fake", action="store_true", help="FakeBus instead of CAN (no hardware)")
        s.add_argument("--yes", action="store_true", help="do not ask before moving")
        return s

    sub.add_parser("check", help="read joints without enabling or moving")
    motion_cmd("zero", "safe move to the rest pose, then disable")
    motion_cmd("grip", "close on / open an object held between the fingers")
    s = motion_cmd("weigh", "weigh objects handed to the gripper at the weigh pose")
    s.add_argument("--labels", nargs="+", default=["object_a", "object_b"])
    s.add_argument("--n", type=int, default=3, help="measurements per object")
    s = sub.add_parser("detect", help="vision: objects on the table")
    s.add_argument("--prompt", help="instruction (object noun is used)")
    s.add_argument("--object", help="GroundingDINO prompt, e.g. 'soda can'")
    s.add_argument("--save", action="store_true", help="also save the raw frame (for tests/data)")
    s = motion_cmd("run", "the demo")
    s.add_argument("--prompt", help="free-text instruction (default: task.prompt)")
    s.add_argument("--objects", choices=["vision", "manual"], help="default: vision (real), manual (--fake)")
    s.add_argument("--xy", help='object positions "x,y;x,y" (base frame, m)')
    s.add_argument("--dry-path", action="store_true", help="approach poses only, no grasp")
    s.add_argument("--no-twin", action="store_true", help="no MuJoCo viewer")
    s = sub.add_parser("calib", help="camera calibration")
    s.add_argument("what", choices=["capture", "intrinsics", "touch"])
    s.add_argument("--out", default="runs/charuco", help="capture: photo directory")
    s.add_argument("--images", default="runs/charuco", help="intrinsics: photo directory")
    s.add_argument("--square-mm", type=float, default=28.0, help="measured ChArUco square size")
    s.add_argument("--marker-mm", type=float, default=20.0, help="measured ChArUco marker size")
    s.add_argument("--load", help="touch: re-solve a saved session directory (its saved clicks)")
    s.add_argument("--reclick", action="store_true", help="touch --load: click the frames again")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    commands = {"check": cmd_check, "zero": cmd_zero, "grip": cmd_grip, "weigh": cmd_weigh, "detect": cmd_detect,
                "run": cmd_run, "calib": cmd_calib}
    return commands[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
