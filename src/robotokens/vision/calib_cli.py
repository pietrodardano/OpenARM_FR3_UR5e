"""Interactive calibration commands (FR8): photo capture, ChArUco intrinsics, robot-touch pose.

``robotokens calib touch`` never commands motion: move the arm by hand (motors disabled) and
press Enter at each point; the TCP comes from the robot's forward kinematics.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import REPO_ROOT, Config, resolve
from .calib_tools import (MIN_TOUCH_POINTS, board_corners, calibrate_intrinsics, charuco_board, implausible,
                          planarity_mm, solve_pose, write_intrinsics)
from .calibration import CameraModel, load_intrinsics, nominal_pose, save_extrinsics
from .camera import Camera

WINDOW = "robotokens calibration"


def capture(cfg: Config, out: Path, square_mm: float, marker_mm: float) -> int:
    """Live preview with the ChArUco corner count; space keeps a frame, q quits."""
    out.mkdir(parents=True, exist_ok=True)
    board = charuco_board(square_mm, marker_mm)
    cam = Camera(cfg.camera)
    kept = len(list(out.glob("*.png")))
    try:
        while True:
            frame = cam.read()
            n = board_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), board)
            view = frame.copy()
            cv2.putText(view, f"corners {n}  kept {kept}  [space] keep  [q] done", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if n >= 6 else (0, 0, 255), 2)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and n >= 6:
                cv2.imwrite(str(out / f"board_{kept:03d}.png"), frame)
                kept += 1
    finally:
        cam.close()
        cv2.destroyAllWindows()
    print(f"{kept} photos in {out}")
    return 0


def intrinsics(cfg: Config, images: Path, square_mm: float, marker_mm: float) -> int:
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    K, dist, rms, size, used = calibrate_intrinsics((cv2.imread(str(p)) for p in paths),
                                                    charuco_board(square_mm, marker_mm))
    print(f"{used}/{len(paths)} photos used, reprojection error {rms:.3f} px (< 0.5 good, > 1 retake)")
    if size != (cfg.camera.width, cfg.camera.height):
        print(f"WARNING: photos are {size[0]}x{size[1]}, config camera mode is {cfg.camera.width}x{cfg.camera.height}")
    path = resolve(cfg.camera.intrinsics_path)
    write_intrinsics(path, K, dist, size, rms)
    print(f"wrote {path}")
    return 0


def _collect(cfg: Config, session: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    from ..robot import make_arm
    from ..robot.model import ArmModel

    arm, model, cam = make_arm(cfg), ArmModel(cfg), Camera(cfg.camera)
    print("The TCP is the midpoint between the two fingertip ENDS (open or closed). Move the arm BY HAND (motors off)\n"
          "to 12-20 points over the pick area at 2-3 heights (table, ~10 cm, ~20 cm up).\n"
          "In the live window: [space] sample (hold still), [u] undo, [f] finish, [q] abort.")
    points: List[np.ndarray] = []
    frames: List[np.ndarray] = []
    try:
        while True:
            frame = cam.read()
            p, _ = model.fk(arm.read_passive().q)  # live TCP from the joints (read-only)
            view = frame.copy()
            for line, text in enumerate([
                    f"samples {len(points)} (min {MIN_TOUCH_POINTS}, aim 12-20)   "
                    f"TCP x {p[0]:+.3f} y {p[1]:+.3f} z {p[2]:+.3f} m",
                    "[space] sample  [u] undo  [f] finish  [q] abort"]):
                cv2.putText(view, text, (12, 30 + 30 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
                cv2.putText(view, text, (12, 30 + 30 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                raise SystemExit("aborted")
            if key == ord("u") and points:
                points.pop()
                frames.pop()
                print(f"  undone ({len(points)} samples)")
            elif key == ord("f"):
                if len(points) >= MIN_TOUCH_POINTS:
                    break
                print(f"  need at least {MIN_TOUCH_POINTS} samples")
            elif key == ord(" "):
                p, _ = model.fk(arm.read_passive().q)
                points.append(p)
                frames.append(cam.read())
                print(f"  sample {len(points)}: TCP {np.round(p, 4).tolist()}")
    finally:
        cam.close()
        cv2.destroyAllWindows()
    session.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(session / f"frame_{i:02d}.png"), f)
    (session / "points.json").write_text(json.dumps([p.tolist() for p in points]))
    print(f"session saved to {session} (re-click later with --load {session})")
    return np.array(points), frames


def _load(session: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    points = np.array(json.loads((session / "points.json").read_text()))
    return points, [cv2.imread(str(session / f"frame_{i:02d}.png")) for i in range(len(points))]


def _click(frames: List[np.ndarray], hints: np.ndarray) -> List[Optional[Tuple[float, float]]]:
    """Click the TCP in each frame: the midpoint between the two fingertip ENDS (open or closed).

    ``hints``: where the current (nominal) camera model expects the TCP, drawn as a yellow circle.
    click = place, Enter = accept, s = skip.
    """
    clicks: List[Optional[Tuple[float, float]]] = []
    cv2.namedWindow(WINDOW)
    for i, frame in enumerate(frames):
        picked: List[Tuple[float, float]] = []
        hint = tuple(int(c) for c in hints[i])

        def on_mouse(event: int, x: int, y: int, *_: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                picked[:] = [(float(x), float(y))]

        cv2.setMouseCallback(WINDOW, on_mouse)
        while True:
            view = frame.copy()
            cv2.circle(view, hint, 40, (0, 255, 255), 1)  # nominal estimate: only a guide (cm off)
            if picked:
                cv2.drawMarker(view, tuple(int(c) for c in picked[0]), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            for line, text in enumerate([f"{i + 1}/{len(frames)}: click the MIDPOINT between the two fingertip ENDS",
                                         "yellow circle = rough guess   Enter = accept   s = skip (tips hidden)"]):
                cv2.putText(view, text, (12, 30 + 30 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
                cv2.putText(view, text, (12, 30 + 30 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                clicks.append(None)
                break
            if key in (10, 13) and picked:
                clicks.append(picked[0])
                break
    cv2.destroyAllWindows()
    return clicks


def touch(cfg: Config, load: Optional[Path], reclick: bool = False) -> int:
    session = load or REPO_ROOT / "runs" / datetime.datetime.now().strftime("touch-%Y%m%d-%H%M%S")
    points, frames = _load(load) if load else _collect(cfg, session)
    K, dist = load_intrinsics(cfg.camera.intrinsics_path, (cfg.camera.width, cfg.camera.height))
    pos_nom, R_nom = nominal_pose(cfg)
    nominal = CameraModel(K=K, dist=np.asarray(dist, float), R=R_nom, position=pos_nom,
                          image_size=(cfg.camera.width, cfg.camera.height), source="nominal")
    saved = session / "clicks.json"
    if load and saved.exists() and not reclick:
        clicks = json.loads(saved.read_text())
        print(f"using the saved clicks in {saved} (--reclick to click again)")
    else:
        clicks = _click(frames, nominal.project(points))
        saved.write_text(json.dumps(clicks))
    sel = [i for i, c in enumerate(clicks) if c is not None]
    pts, px = points[sel], np.array([clicks[i] for i in sel])
    pos = pos_nom  # the mount position is measured; the fit determines the rotation
    R, res, kept = solve_pose(pts, px, K, dist, pos, R_nom)
    axis = R[:, 2]
    thick = planarity_mm(pts[kept])
    print(f"points agreeing {kept.sum()}/{len(pts)}, residual {np.median(res):.1f} px median / {res.max():.1f} px max "
          f"(fingertip clicks: ~10 px is normal)")
    print(f"camera at {np.round(pos, 3).tolist()} m, pitch {np.degrees(np.arcsin(-axis[2])):.1f} deg down, "
          f"yaw {np.degrees(np.arctan2(axis[1], axis[0])):+.1f} deg")
    low = pts[kept][:, 2].min()
    print(f"lowest touch is {(low - cfg.table_z) * 1e3:+.0f} mm from the configured tabletop (z = {cfg.table_z:.3f} m)")
    meta = {"method": "robot_touch (rotation fit, measured mount position)", "session": str(session),
            "points_agreeing": int(kept.sum()), "median_residual_px": float(np.median(res)),
            "point_cloud_thickness_mm": thick}
    why = implausible(pos, R, pos_nom, R_nom, int(kept.sum()), len(pts))
    if why:
        save_extrinsics(session / "camera_extrinsics.REJECTED.yaml", R, pos, {**meta, "rejected": why})
        print(f"REJECTED: {why}. The camera model is unchanged. Re-click with --load {session}")
        return 1
    save_extrinsics(cfg.camera.extrinsics_path, R, pos, meta)
    print(f"wrote {resolve(cfg.camera.extrinsics_path)} (now overrides the nominal mount)")
    return 0
