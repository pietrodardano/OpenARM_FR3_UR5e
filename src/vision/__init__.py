"""Vision module.

Object localization for the demo (camera.py, object_localizer.py, rect_face_pca.py,
table_calibration.py, vision_detector.py) and the earlier hand-tracking listener.
Import the object-localization modules directly: they need OpenCV / torch, which this
package import does not pull in.
"""

from .hand_tracker_listener import (
    HandLandmark,
    HandPose,
    HandTrackerListener,
    MockHandTrackerStream,
    EgocentricToRobotMapper,
)

__all__ = [
    "HandLandmark",
    "HandPose",
    "HandTrackerListener",
    "MockHandTrackerStream",
    "EgocentricToRobotMapper",
]
