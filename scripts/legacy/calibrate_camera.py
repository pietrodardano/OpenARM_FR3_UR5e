#!/usr/bin/env python3
"""
Camera & Hand Tracker Zero-Calibration Script.

Samples neutral human resting hand poses from the DaiDai Lab egocentric stream,
computes the zero reference translation offset, and prints recommended values
for configs/task_config.yaml.
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.vision.hand_tracker_listener import MockHandTrackerStream


def calibrate(num_samples: int = 50, rate_hz: float = 30.0):
    print("=" * 60)
    print("  DaiDai Lab Hand Tracker Calibration Utility")
    print("=" * 60)
    print("Please hold your hands in the neutral resting position in front of you...")
    time.sleep(1.0)

    stream = MockHandTrackerStream(fps=int(rate_hz))
    stream.connect()

    left_samples = []
    right_samples = []

    try:
        for i in range(num_samples):
            frame = stream.get_latest_frame()
            if frame and frame.left_hand:
                left_samples.append(frame.left_hand.wrist_position)
            if frame and frame.right_hand:
                right_samples.append(frame.right_hand.wrist_position)
            time.sleep(1.0 / rate_hz)
            if (i + 1) % 10 == 0:
                print(f"Captured {i + 1}/{num_samples} frames...")

        left_mean = np.mean(left_samples, axis=0) if left_samples else np.zeros(3)
        right_mean = np.mean(right_samples, axis=0) if right_samples else np.zeros(3)

        print("\n--- Calibration Results (Camera Frame) ---")
        print(f"Left Hand Origin : [X: {left_mean[0]:.3f}, Y: {left_mean[1]:.3f}, Z: {left_mean[2]:.3f}] m")
        print(f"Right Hand Origin: [X: {right_mean[0]:.3f}, Y: {right_mean[1]:.3f}, Z: {right_mean[2]:.3f}] m")
        print("\nAdd these origin offsets to your configs/task_config.yaml if calibrating real hardware.")
    finally:
        stream.disconnect()


if __name__ == "__main__":
    calibrate()
