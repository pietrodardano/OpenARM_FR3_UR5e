"""Tabletop pick-and-place scene with the real OpenArm v1 bimanual model.

Loads ``assets/scenes/tabletop.xml`` (upstream OpenArm v1 MJCF from
``third_party/openarm_mujoco`` plus table, objects and cameras), adds a TCP
site between each gripper's fingertips, and exposes the handful of operations
scripted pick-and-place needs: reset, arm/gripper commands, TCP and object
poses, and offscreen rendering.

World frame: origin at the pedestal base on the floor, +x towards the table,
+y to the robot's left, +z up. The table top is at z = 0.30.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import mujoco
import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = REPO_ROOT / "assets" / "scenes" / "tabletop.xml"
ROBOT_PATH = REPO_ROOT / "third_party" / "openarm_mujoco" / "v1" / "openarm_bimanual.xml"

Side = Literal["left", "right"]
SIDES: Tuple[Side, Side] = ("left", "right")

OBJECT_NAMES: Tuple[str, ...] = ("plate", "cube", "bottle", "mug", "apple", "pen")
TABLE_HEIGHT = 0.30

# Cameras defined in tabletop.xml. "head" is the robot's only vision sensor; the
# rest are debug views. Upstream's own cameras are removed in build_model().
HEAD_CAMERA = "head"
SCENE_CAMERAS: Tuple[str, ...] = (HEAD_CAMERA, "overhead", "front", "three_quarter")

# Right-arm joint i = MIRROR[i] * left-arm joint i gives the mirror-image pose.
MIRROR = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])

# Home: upper arms swung back, elbows bent, grippers ~19 cm above the table's
# front edge and clear of the head camera's view of the objects.
_HOME_LEFT = np.array([1.2, -0.2, 0.0, 2.4, 0.0, 0.0, 0.0])
HOME_ARM_QPOS: Dict[str, NDArray[np.float64]] = {"left": _HOME_LEFT, "right": MIRROR * _HOME_LEFT}

# The upstream ``<side>_ee_control_point`` site sits on link7 at the wrist flange,
# ~20 cm behind the fingertips. The TCP is ~2 cm in from the tips, between the pads.
EE_BODY = "openarm_{side}_link7"
TCP_OFFSET = (0.0, 0.0, 0.180)

# Prismatic parallel-jaw fingers: 0 = closed, 0.044 = open (~86 mm aperture).
# "Closed" is commanded slightly past contact so the servo keeps squeezing.
GRIPPER_OPEN = 0.044
GRIPPER_CLOSED = -0.005


def build_model() -> mujoco.MjModel:
    """Compiles the tabletop scene with ``<side>_tcp`` sites and only our cameras.

    Robot links get ideal gravity compensation (``gravcomp=1``), standing in for
    the gravity feedforward of the real arm's controller; without it the position
    servos alone let the TCP sag ~2 cm at home. Objects are not compensated, so
    payloads still load the arm.
    """
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    for camera in list(spec.cameras):
        if camera.name not in SCENE_CAMERAS:
            spec.delete(camera)
    for body in spec.bodies:
        if body.name.startswith("openarm_"):
            body.gravcomp = 1.0
    add_tcp_sites(spec)
    return spec.compile()


def build_robot_model() -> mujoco.MjModel:
    """Robot-only model (no table/objects) with TCP sites, e.g. for IK."""
    spec = mujoco.MjSpec.from_file(str(ROBOT_PATH))
    add_tcp_sites(spec)
    return spec.compile()


def add_tcp_sites(spec: mujoco.MjSpec) -> None:
    """Adds ``<side>_tcp`` sites between the fingertip pads."""
    for side in SIDES:
        spec.body(EE_BODY.format(side=side)).add_site(
            name=f"{side}_tcp",
            pos=list(TCP_OFFSET),
            size=[0.006, 0.006, 0.006],
            rgba=[1.0, 0.5, 0.0, 1.0],
            group=4,
        )


class TabletopScene:
    """Model/data pair for the tabletop scene plus name-based accessors."""

    def __init__(self) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)

        m = self.model
        self._arm_qpos_adr = {
            side: np.array([m.jnt_qposadr[m.joint(f"openarm_{side}_joint{i}").id] for i in range(1, 8)])
            for side in SIDES
        }
        self._arm_ctrl_ids = {
            side: np.array([m.actuator(f"{side}_joint{i}_ctrl").id for i in range(1, 8)]) for side in SIDES
        }
        self._finger_qpos_adr = {
            side: [m.jnt_qposadr[m.joint(f"openarm_{side}_finger_joint{i}").id] for i in (1, 2)] for side in SIDES
        }
        self._gripper_ctrl_id = {side: m.actuator(f"{side}_finger1_ctrl").id for side in SIDES}
        self._object_qpos_adr = {name: m.jnt_qposadr[m.joint(f"{name}_joint").id] for name in OBJECT_NAMES}

        self.reset()

    # ------------------------------------------------------------------ reset

    def reset(self, seed: Optional[int] = None, object_xy_noise: float = 0.0) -> None:
        """Arms to home with grippers open; objects to their XML poses.

        Args:
            seed: RNG seed for the object perturbation.
            object_xy_noise: Half-width (m) of a uniform xy jitter applied to every
                object except the plate. Objects can collide if this is large.
        """
        mujoco.mj_resetData(self.model, self.data)
        for side in SIDES:
            self.data.qpos[self._arm_qpos_adr[side]] = HOME_ARM_QPOS[side]
            self.data.ctrl[self._arm_ctrl_ids[side]] = HOME_ARM_QPOS[side]
            self.data.qpos[self._finger_qpos_adr[side]] = GRIPPER_OPEN
            self.data.ctrl[self._gripper_ctrl_id[side]] = GRIPPER_OPEN

        if object_xy_noise > 0.0:
            rng = np.random.default_rng(seed)
            for name in OBJECT_NAMES:
                if name == "plate":
                    continue
                adr = self._object_qpos_adr[name]
                self.data.qpos[adr : adr + 2] += rng.uniform(-object_xy_noise, object_xy_noise, size=2)

        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------------- commands

    def arm_qpos(self, side: Side) -> NDArray[np.float64]:
        """Current 7 arm joint positions."""
        return self.data.qpos[self._arm_qpos_adr[side]].copy()

    def set_arm_target(self, side: Side, q: NDArray[np.float64]) -> None:
        """Sets the 7 arm position-servo targets, clipped to actuator limits."""
        ids = self._arm_ctrl_ids[side]
        lo, hi = self.model.actuator_ctrlrange[ids].T
        self.data.ctrl[ids] = np.clip(q, lo, hi)

    def set_gripper(self, side: Side, open_fraction: float) -> None:
        """Gripper command: 1 = fully open, 0 = closed (squeezing)."""
        f = float(np.clip(open_fraction, 0.0, 1.0))
        self.data.ctrl[self._gripper_ctrl_id[side]] = GRIPPER_CLOSED + f * (GRIPPER_OPEN - GRIPPER_CLOSED)

    def gripper_opening(self, side: Side) -> float:
        """Current finger joint position (m per finger; 0 = closed)."""
        return float(self.data.qpos[self._finger_qpos_adr[side][0]])

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    # ------------------------------------------------------------------ state

    def tcp_pose(self, side: Side) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """TCP position (3,) and rotation matrix (3, 3) in the world frame."""
        site = self.data.site(f"{side}_tcp")
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    def object_pose(self, name: str) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Object body position (3,) and quaternion (w, x, y, z) in the world frame."""
        body = self.data.body(name)
        return body.xpos.copy(), body.xquat.copy()

    # -------------------------------------------------------------- rendering

    def render(self, camera: str = HEAD_CAMERA, width: int = 640, height: int = 480) -> NDArray[np.uint8]:
        """Offscreen RGB render from a named camera (needs MUJOCO_GL=egl or osmesa when headless)."""
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            renderer.update_scene(self.data, camera=camera)
            return renderer.render()
