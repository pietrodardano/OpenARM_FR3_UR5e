"""Typed, strict loader for ``configs/robotokens.yaml`` (NFR4).

Every YAML section maps to a frozen dataclass. Unknown and missing keys raise
:class:`ConfigError` with the dotted key path, so a typo never silently falls back to a
default. Lists become tuples.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "robotokens.yaml"

# The official OpenArm v1 model's shoulder axis sits this far above its root body
# (openarm_body_link0), which is the origin of the robot base frame.
ARM_ORIGIN_ABOVE_ROOT_M = 0.697

Vec = Tuple[float, ...]


class ConfigError(ValueError):
    """The config file does not match the schema."""


@dataclass(frozen=True)
class RobotConfig:
    side: Literal["left", "right"]
    model_path: str
    home_qpos: Vec
    rest_qpos: Vec
    side_route_qpos: Tuple[Vec, ...]


@dataclass(frozen=True)
class HardwareConfig:
    can_interface: str
    can_fd: bool
    motor_types: Tuple[str, ...]
    send_can_ids: Tuple[int, ...]
    recv_can_ids: Tuple[int, ...]
    kp: Vec
    kd: Vec
    control_rate_hz: float
    recv_timeout_us: int
    max_joint_velocity_rad_s: Vec
    joint_limit_margin_rad: float
    startup_limit_tolerance_rad: float
    max_tracking_error_rad: float
    tracking_error_time_s: float
    missing_reply_time_s: float
    rest_duration_s: float
    gravity_compensation_scale: float
    realtime_priority: int


@dataclass(frozen=True)
class GripperConfig:
    motor_type: str
    send_can_id: int
    recv_can_id: int
    kp: float
    kd: float
    motor_open_rad: float
    open_m: float
    max_velocity_m_s: float
    max_torque_nm: float
    torque_kd: float
    torque_rate_nm_s: float
    approach_speed_mm_s: float
    approach_max_torque_nm: float
    approach_ramp_nm_s: float
    grasp_torque_nm: float
    min_held_opening_m: float


@dataclass(frozen=True)
class BoxConfig:
    low: Vec
    high: Vec


@dataclass(frozen=True)
class MotionConfig:
    rate_hz: float
    peak_speed_m_s: float
    peak_rot_speed_rad_s: float
    min_move_s: float
    joint_speed_deg_s: float
    ik_iters: int
    max_ik_pos_error_m: float
    max_ik_rot_error_rad: float
    check_points: int
    table_margin_m: float
    collision_clearance_m: float
    workspace: BoxConfig


@dataclass(frozen=True)
class TorsoConfig:
    width_m: float
    height_m: float


@dataclass(frozen=True)
class TableConfig:
    top_height_m: float
    thickness_m: float
    length_m: float
    width_m: float
    center_x_m: float


@dataclass(frozen=True)
class BinConfig:
    x: float
    y: float
    inner_size_m: float
    wall_height_m: float


@dataclass(frozen=True)
class CameraMountConfig:
    forward_offset_m: float
    height_above_torso_m: float
    pitch_down_deg: float


@dataclass(frozen=True)
class WorkcellConfig:
    arm_mount_height_m: float
    torso: TorsoConfig
    table: TableConfig
    bins: Dict[str, BinConfig]
    camera: CameraMountConfig


@dataclass(frozen=True)
class CameraConfig:
    device_index: int
    width: int
    height: int
    rotate_180: bool
    color_correct: bool
    intrinsics_path: str
    extrinsics_path: str


@dataclass(frozen=True)
class XYRange:
    x: Vec
    y: Vec


@dataclass(frozen=True)
class VisionConfig:
    frames: int
    merge_radius_m: float
    min_frame_fraction: float
    object_top_height_m: Dict[str, float]
    default_top_height_m: float
    workspace: XYRange
    sam2_size: Literal["small", "large"]
    device: str
    box_threshold: float
    text_threshold: float
    infer_size: int
    tilt_deg: float

    def top_height(self, prompt: str) -> float:
        """Top-face height above the table of the object named in ``prompt``."""
        low = prompt.lower()
        return next((h for key, h in self.object_top_height_m.items() if key in low), self.default_top_height_m)


@dataclass(frozen=True)
class XY:
    x: float
    y: float


@dataclass(frozen=True)
class GraspConfig:
    pitch_deg: float
    height_above_table_m: float


@dataclass(frozen=True)
class PoseConfig:
    x: float
    y: float
    above_table_m: float
    pitch_deg: float


@dataclass(frozen=True)
class WeighConfig:
    pose: PoseConfig
    settle_s: float
    dither_deg: float
    dither_period_s: float
    dither_cycles: int
    min_z: float
    remeasure_once: bool


@dataclass(frozen=True)
class ContactConfig:
    baseline_s: float
    k_sigma: float
    min_delta_nm: float
    hold_s: float
    timeout_s: float
    rate_hz: float


@dataclass(frozen=True)
class TaskConfig:
    prompt: str
    object_prompt: str
    min_confidence: float
    manual_objects: Tuple[XY, ...]
    grasp: GraspConfig
    approach_distance_m: float
    approach_clearance_m: float
    place_back_offset_m: float
    release_settle_s: float
    bin_release_above_table_m: float
    handover: PoseConfig
    weigh: WeighConfig
    contact: ContactConfig


@dataclass(frozen=True)
class TwinConfig:
    rate_hz: float


@dataclass(frozen=True)
class FakeConfig:
    masses_kg: Vec
    object_width_m: float
    torque_noise_nm: float


@dataclass(frozen=True)
class Config:
    robot: RobotConfig
    hardware: HardwareConfig
    gripper: GripperConfig
    motion: MotionConfig
    workcell: WorkcellConfig
    camera: CameraConfig
    vision: VisionConfig
    task: TaskConfig
    twin: TwinConfig
    fake: FakeConfig

    @property
    def base_z_in_world(self) -> float:
        """Height of the robot base frame above the floor (m)."""
        return self.workcell.arm_mount_height_m - ARM_ORIGIN_ABOVE_ROOT_M

    @property
    def table_z(self) -> float:
        """Tabletop height in the robot base frame (m)."""
        return self.workcell.table.top_height_m - self.base_z_in_world


def resolve(path: str | Path) -> Path:
    """Absolute path; relative paths are taken from the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _convert(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin is Literal:
        if value not in typing.get_args(tp):
            raise ConfigError(f"{where}: {value!r} is not one of {typing.get_args(tp)}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        args = typing.get_args(tp)
        item = args[0]
        return tuple(_convert(item, v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        _, item = typing.get_args(tp)
        return {str(k): _convert(item, v, f"{where}.{k}") for k, v in value.items()}
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is bool or tp is str:
        if not isinstance(value, tp):
            raise ConfigError(f"{where}: expected {tp.__name__}, got {value!r}")
        return value
    raise TypeError(f"unsupported config type {tp} at {where}")


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    fields = [f.name for f in dataclasses.fields(cls)]
    prefix = f"{where}." if where else ""
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"unknown key(s): {', '.join(prefix + k for k in unknown)}")
    missing = [f for f in fields if f not in data]
    if missing:
        raise ConfigError(f"missing key(s): {', '.join(prefix + k for k in missing)}")
    return cls(**{f: _convert(hints[f], data[f], prefix + f) for f in fields})


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Loads and validates the config file."""
    with open(resolve(path), encoding="utf-8") as f:
        cfg: Config = _build(Config, yaml.safe_load(f), "")
    if set(cfg.workcell.bins) != {"left", "right"}:
        raise ConfigError("workcell.bins: exactly the bins 'left' and 'right' are supported")
    if len(cfg.task.manual_objects) != 2:
        raise ConfigError("task.manual_objects: the demo compares exactly two objects")
    for name in ("home_qpos", "rest_qpos"):
        if len(getattr(cfg.robot, name)) != 7:
            raise ConfigError(f"robot.{name}: needs 7 joint values")
    return cfg
