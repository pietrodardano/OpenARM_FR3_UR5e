"""
DaiDai Lab Egocentric Hand Tracking Listener & Coordinate Mapper.

This module listens to egocentric vision data (RGB-D + 3D hand landmarks),
filters and smooths human hand trajectories, and maps them from camera frame
to the robot's base coordinate frame.
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


@dataclass
class HandLandmark:
    """Represents a single 3D hand landmark in camera space."""
    name: str
    position: np.ndarray  # shape (3,), in meters
    confidence: float = 1.0


@dataclass
class HandPose:
    """
    Parsed 3D hand pose and grasp state for a single hand.
    Coordinates are in meters relative to the egocentric camera frame.
    """
    handedness: str  # "left" or "right"
    wrist_position: np.ndarray  # shape (3,), in camera frame [x, y, z]
    orientation_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # [qw, qx, qy, qz]
    pinch_distance: float = 0.08  # meters between thumb and index fingertip
    grasp_strength: float = 0.0  # normalized [0.0 = fully open, 1.0 = fully closed]
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    landmarks: Optional[List[HandLandmark]] = None

    @property
    def is_grasping(self) -> bool:
        """Determines if the hand is executing a pinch/grasp."""
        return self.grasp_strength > 0.5


@dataclass
class BimanualHandFrame:
    """Synchronized egocentric frame containing both left and right hands."""
    timestamp: float
    left_hand: Optional[HandPose] = None
    right_hand: Optional[HandPose] = None


class EgocentricToRobotMapper:
    """
    Maps 3D hand positions from the egocentric camera frame to the robot base frame.
    Includes workspace scaling, temporal filtering (EMA), and workspace boundary clipping.
    """

    def __init__(
        self,
        scale_factor: float = 1.2,
        alpha: float = 0.75,
        axis_permutation: Tuple[int, int, int] = (2, 0, 1),
        axis_signs: Tuple[float, float, float] = (1.0, -1.0, -1.0),
        left_offset: Optional[np.ndarray] = None,
        right_offset: Optional[np.ndarray] = None,
        workspace_bounds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            scale_factor: Motion amplification from human hand to robot workspace.
            alpha: Exponential moving average smoothing weight (0 < alpha <= 1).
            axis_permutation: Coordinate mapping indices from cam to rob [x_rob_idx, y_rob_idx, z_rob_idx].
            axis_signs: Sign multipliers for each axis.
            left_offset: Nominal robot resting position for left end-effector.
            right_offset: Nominal robot resting position for right end-effector.
            workspace_bounds: Safety clipping box dict with keys x_min, x_max, y_min, y_max, z_min, z_max.
        """
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.axis_permutation = axis_permutation
        self.axis_signs = axis_signs

        self.left_offset = (
            np.array(left_offset, dtype=np.float64)
            if left_offset is not None
            else np.array([0.35, 0.18, 0.55], dtype=np.float64)
        )
        self.right_offset = (
            np.array(right_offset, dtype=np.float64)
            if right_offset is not None
            else np.array([0.35, -0.18, 0.55], dtype=np.float64)
        )

        self.workspace_bounds = workspace_bounds or {
            "x_min": 0.15,
            "x_max": 0.55,
            "y_min": -0.45,
            "y_max": 0.45,
            "z_min": 0.38,
            "z_max": 0.75,
        }

        # Filtering state
        self._prev_left_filtered: Optional[np.ndarray] = None
        self._prev_right_filtered: Optional[np.ndarray] = None

        # Operator zero/calibration reference (first observed raw camera position)
        self._left_cam_origin: Optional[np.ndarray] = None
        self._right_cam_origin: Optional[np.ndarray] = None

    def reset_calibration(self) -> None:
        """Reset human operator zero reference position."""
        self._left_cam_origin = None
        self._right_cam_origin = None
        self._prev_left_filtered = None
        self._prev_right_filtered = None

    def _transform_point(
        self,
        raw_cam_pos: np.ndarray,
        cam_origin: Optional[np.ndarray],
        robot_nominal_offset: np.ndarray,
        prev_filtered: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Core coordinate transformation logic for a single hand point."""
        if cam_origin is None:
            cam_origin = raw_cam_pos.copy()

        # Compute relative delta in camera frame
        delta_cam = raw_cam_pos - cam_origin

        # Permute and scale to robot base frame
        delta_rob = np.zeros(3, dtype=np.float64)
        for i, (idx, sign) in enumerate(zip(self.axis_permutation, self.axis_signs)):
            delta_rob[i] = sign * delta_cam[idx] * self.scale_factor

        # Add to nominal robot resting position
        target_rob = robot_nominal_offset + delta_rob

        # Safety clipping
        target_rob[0] = np.clip(target_rob[0], self.workspace_bounds["x_min"], self.workspace_bounds["x_max"])
        target_rob[1] = np.clip(target_rob[1], self.workspace_bounds["y_min"], self.workspace_bounds["y_max"])
        target_rob[2] = np.clip(target_rob[2], self.workspace_bounds["z_min"], self.workspace_bounds["z_max"])

        # Exponential moving average filter
        if prev_filtered is None:
            filtered = target_rob.copy()
        else:
            filtered = self.alpha * target_rob + (1.0 - self.alpha) * prev_filtered

        return filtered, cam_origin, filtered

    def map_hand_pose(self, hand_pose: HandPose) -> Dict[str, Any]:
        """
        Maps a single HandPose to robot Cartesian space target.
        Returns dictionary with 'cartesian_pos', 'orientation_quat', 'gripper_target'.
        """
        is_left = hand_pose.handedness.lower() == "left"
        nominal_offset = self.left_offset if is_left else self.right_offset
        cam_origin = self._left_cam_origin if is_left else self._right_cam_origin
        prev_filt = self._prev_left_filtered if is_left else self._prev_right_filtered

        filtered_pos, updated_origin, updated_filt = self._transform_point(
            raw_cam_pos=hand_pose.wrist_position,
            cam_origin=cam_origin,
            robot_nominal_offset=nominal_offset,
            prev_filtered=prev_filt,
        )

        if is_left:
            self._left_cam_origin = updated_origin
            self._prev_left_filtered = updated_filt
        else:
            self._right_cam_origin = updated_origin
            self._prev_right_filtered = updated_filt

        # Gripper slider range in OpenArms MJCF: [-0.035, 0] (negative is open/closed)
        # grasp_strength: 0.0 (open) -> 1.0 (closed)
        gripper_target = -0.035 * hand_pose.grasp_strength

        return {
            "handedness": hand_pose.handedness,
            "cartesian_pos": filtered_pos,
            "orientation_quat": hand_pose.orientation_quat,
            "gripper_target": gripper_target,
            "grasp_strength": hand_pose.grasp_strength,
            "confidence": hand_pose.confidence,
        }


class HandTrackerListener:
    """Base interface for receiving DaiDai Lab egocentric hand tracking frames."""

    def connect(self) -> bool:
        """Initialize connection to stream."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """Close connection to stream."""
        raise NotImplementedError

    def get_latest_frame(self) -> Optional[BimanualHandFrame]:
        """Poll the latest synchronized bimanual frame."""
        raise NotImplementedError


class MockHandTrackerStream(HandTrackerListener):
    """
    Mock hand tracking generator simulating real-time human bimanual interaction.
    Outputs smooth reaching motions and periodic grasping for hackathon testing.
    """

    def __init__(self, fps: int = 30):
        self.fps = fps
        self.dt = 1.0 / fps
        self._is_connected = False
        self._start_time = time.time()

    def connect(self) -> bool:
        self._is_connected = True
        self._start_time = time.time()
        return True

    def disconnect(self) -> None:
        self._is_connected = False

    def get_latest_frame(self) -> Optional[BimanualHandFrame]:
        if not self._is_connected:
            return None

        t = time.time() - self._start_time

        # Simulate natural human reaching motion in camera frame (meters)
        # Left hand: reaches forward and slightly left
        left_cam_pos = np.array([
            -0.15 + 0.05 * math.sin(0.8 * t),        # X: left in camera
            0.10 + 0.03 * math.cos(1.2 * t),         # Y: down in camera
            0.45 + 0.08 * math.sin(0.5 * t),        # Z: forward depth
        ], dtype=np.float64)

        # Right hand: reaches forward toward central manipulation table
        right_cam_pos = np.array([
            0.15 + 0.06 * math.cos(0.9 * t),         # X: right in camera
            0.10 + 0.04 * math.sin(1.1 * t),         # Y: down in camera
            0.45 + 0.10 * math.sin(0.6 * t + 0.5),   # Z: forward depth
        ], dtype=np.float64)

        # Periodic pinch/grasp gestures: grasp every ~4 seconds
        left_grasp = 0.5 * (1.0 + math.sin(0.7 * t - 1.0))
        right_grasp = 0.5 * (1.0 + math.sin(0.7 * t + 1.0))

        # Clamp grasp to [0.0, 1.0]
        left_grasp = float(np.clip(left_grasp, 0.0, 1.0))
        right_grasp = float(np.clip(right_grasp, 0.0, 1.0))

        left_pose = HandPose(
            handedness="left",
            wrist_position=left_cam_pos,
            grasp_strength=left_grasp,
            confidence=0.95,
            timestamp=time.time(),
        )

        right_pose = HandPose(
            handedness="right",
            wrist_position=right_cam_pos,
            grasp_strength=right_grasp,
            confidence=0.98,
            timestamp=time.time(),
        )

        return BimanualHandFrame(
            timestamp=time.time(),
            left_hand=left_pose,
            right_hand=right_pose,
        )
