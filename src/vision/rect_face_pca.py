"""Top-face geometry for rectangular objects, from a segmentation mask.

The camera looks at the table from ~20 degrees off vertical, so a rectangular
top face projects to a trapezoid and its sides are foreshortened along the tilt
direction. Measuring the face directly in image pixels therefore reports the
wrong aspect ratio and the wrong orientation.

A rectifying homography (built from the tilt angle and the camera's focal
length) maps the image to a virtual straight-down view. PCA runs in that
rectified space, so the side lengths, aspect ratio and angle describe the real
top face; results are mapped back to image coordinates for drawing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# PCA over every mask pixel is wasteful on large masks; this many points is
# far more than enough to pin down a planar face's axes.
_MAX_PCA_POINTS = 20_000


def rectify_homography(
    tilt_deg: float,
    image_width: int,
    image_height: int,
    focal_px: float | None = None,
) -> NDArray[np.float64]:
    """Homography mapping the tilted view to a virtual straight-down view.

    H = K @ Rx(tilt) @ K^-1 rotates the virtual camera about its horizontal
    axis, which rectifies any plane whose normal sits at `tilt_deg` to the
    optical axis -- the table, and every face parallel to it.

    `focal_px` defaults to a ~90 degree diagonal FOV, matching the HBV camera's
    wide lens. Run scripts/calibrate_intrinsics.py to get the real value.
    """
    if focal_px is None:
        diagonal = float(np.hypot(image_width, image_height))
        focal_px = diagonal / 2.0  # tan(45 deg) == 1 for a 90 deg diagonal FOV

    cx, cy = image_width / 2.0, image_height / 2.0
    K = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]])
    th = np.radians(tilt_deg)
    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(th), -np.sin(th)],
        [0.0, np.sin(th), np.cos(th)],
    ])
    return K @ rot_x @ np.linalg.inv(K)


def _apply_h(H: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply a homography to (N, 2) points."""
    homogeneous = np.column_stack((points, np.ones(len(points))))
    out = homogeneous @ H.T
    return out[:, :2] / out[:, 2:3]


@dataclass
class RectFace:
    """Geometry of one rectangular face.

    Lengths, aspect ratio and angle are measured in the rectified (straight
    down) view, so they describe the real face. Centroid, axes and corners are
    in image pixels so they can be drawn directly on the frame.
    """

    centroid: NDArray[np.float64]  # (2,) [x, y] in image pixels
    long_axis: NDArray[np.float64]  # (2,) unit vector in image space
    short_axis: NDArray[np.float64]  # (2,) unit vector in image space
    long_length: float  # rectified px, the longer side
    short_length: float  # rectified px, the shorter side
    angle_deg: float  # orientation of the long side in the rectified view
    area_px: int
    _corners: NDArray[np.float64]  # (4, 2) in image pixels

    @property
    def aspect_ratio(self) -> float:
        """Long side / short side. 1.0 means square (orientation is ambiguous)."""
        return self.long_length / max(self.short_length, 1e-6)

    @property
    def is_square(self) -> bool:
        """True when the sides are within 10%, so the long axis is not meaningful."""
        return self.aspect_ratio < 1.1

    def corners(self) -> NDArray[np.float64]:
        """The fitted rectangle's four corners in image pixels, as (4, 2)."""
        return self._corners


def rectangularity(mask: NDArray[np.bool_], homography: NDArray[np.float64] | None = None) -> float:
    """How well the mask fills its own bounding rectangle, in [0, 1].

    A flat top face rectifies to a clean rectangle and scores near 1.0. A mask
    that also covers a lateral side face is an L-shaped or hexagonal blob and
    scores lower, which is how the top face is picked out of SAM2's candidates.

    Both areas are measured on the outline in the same (rectified) space, so
    the homography's non-uniform area scaling cancels out of the ratio.
    """
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(outline) < 4:
        return 0.0
    if homography is not None:
        outline = _apply_h(homography, outline)

    outline32 = outline.astype(np.float32)
    poly_area = float(cv2.contourArea(outline32))
    (_, (w, h), _) = cv2.minAreaRect(outline32)
    box_area = float(w * h)
    if box_area <= 0:
        return 0.0
    return min(1.0, poly_area / box_area)


def _silhouette_polygon(mask: NDArray[np.bool_]) -> NDArray[np.float32] | None:
    """Convex silhouette of the mask, simplified to its significant corners."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    # Keep this tight: the height edges are short next to the face's own sides,
    # and a loose epsilon simplifies the silhouette's hexagon straight back
    # into a quad, deleting exactly the edges that encode the box's height.
    epsilon = max(2.0, 0.004 * cv2.arcLength(hull, True))
    return cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2).astype(np.float32)


def estimate_extrusion_height(
    polygon: NDArray[np.float32],
    extrusion_dir: tuple[float, float] = (0.0, 1.0),
    angle_tolerance_deg: float = 15.0,
) -> float:
    """Projected height of the box, in pixels, from its silhouette.

    The silhouette of a prism is conv(T, T + h*d), so the hull contains two
    edges that are exactly the offset h*d -- each joins a top-face corner to
    its bottom-face twin. Measuring them gives h directly, with no search.

    A rotated top face can itself have an edge nearly parallel to d, so a lone
    aligned edge is not trusted: the height edges are identified as the aligned
    *pair* with the closest matching lengths, which a top-face edge has no
    reason to form.
    """
    direction = np.asarray(extrusion_dir, dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-9)
    min_cos = np.cos(np.radians(angle_tolerance_deg))

    aligned = []
    for i in range(len(polygon)):
        edge = polygon[(i + 1) % len(polygon)] - polygon[i]
        length = float(np.linalg.norm(edge))
        if length < 4.0:
            continue
        if abs(float(np.dot(edge / length, direction))) >= min_cos:
            aligned.append(length)

    if not aligned:
        return 0.0
    if len(aligned) == 1:
        return aligned[0]

    aligned.sort()
    best = min(
        ((b - a, 0.5 * (a + b)) for a, b in zip(aligned, aligned[1:])),
        key=lambda pair: pair[0] / max(pair[1], 1e-6),
    )
    return best[1]


def extract_top_face(
    mask: NDArray[np.bool_],
    homography: NDArray[np.float64] | None = None,
    extrusion_dir: tuple[float, float] = (0.0, 1.0),
    min_height_px: float = 6.0,
) -> tuple[NDArray[np.bool_], float]:
    """Recover a box's top face from its whole-object silhouette.

    A box is a prism, so its silhouette S is the top face T swept along the
    projected height: S = T (+) [0, h*d]. Undoing that sweep recovers the face,
    S ∩ (S - h*d) = T, and doing it on the hull polygon rather than on pixels
    keeps the result an exact quad instead of a ragged eroded blob.

    Returns (top-face mask, h). When no side face is visible -- a flat object,
    or one seen straight down -- h is ~0 and the mask is returned unchanged.
    """
    import cv2

    polygon = _silhouette_polygon(mask)
    if polygon is None or len(polygon) < 4:
        return mask, 0.0

    height = estimate_extrusion_height(polygon, extrusion_dir)
    if height < min_height_px:
        return mask, 0.0

    direction = np.asarray(extrusion_dir, dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-9)
    shifted = (polygon - (height * direction).astype(np.float32)).astype(np.float32)

    area, intersection = cv2.intersectConvexConvex(polygon, shifted)
    if intersection is None or area <= 0:
        return mask, 0.0

    top = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(top, [np.int32(intersection.reshape(-1, 2))], 1)
    top_mask = top.astype(bool)

    # Guard against a degenerate cut, e.g. a bad height on a non-prism object.
    if top_mask.sum() < 0.15 * mask.sum():
        return mask, 0.0
    if rectangularity(top_mask, homography) < rectangularity(mask, homography):
        return mask, 0.0
    return top_mask, height


def _mask_points(mask: NDArray[np.bool_]) -> NDArray[np.float64] | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 10:
        return None
    if xs.size > _MAX_PCA_POINTS:
        idx = np.linspace(0, xs.size - 1, _MAX_PCA_POINTS).astype(np.intp)
        xs, ys = xs[idx], ys[idx]
    return np.column_stack((xs, ys)).astype(np.float64)


def fit_rect_face(
    mask: NDArray[np.bool_],
    homography: NDArray[np.float64] | None = None,
) -> RectFace | None:
    """PCA fit of a rectangular face to a boolean mask.

    With `homography`, the mask is rectified to a straight-down view before the
    fit, so the reported lengths and angle are the real ones rather than the
    tilt-foreshortened ones. Returns None for an empty or degenerate mask.
    """
    points_img = _mask_points(mask)
    if points_img is None:
        return None

    points = _apply_h(homography, points_img) if homography is not None else points_img
    centroid = points.mean(axis=0)
    centered = points - centroid

    # Principal axes: eigenvectors of the 2x2 covariance, largest variance first.
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes = eigvecs[:, np.argsort(eigvals)[::-1]].T

    # Extent comes from projecting the pixels onto each axis rather than from
    # the eigenvalues, so the lengths are the face's real side lengths instead
    # of a distribution width.
    projections = centered @ axes.T
    spans = projections.max(axis=0) - projections.min(axis=0)

    long_axis, short_axis = axes[0], axes[1]
    long_length, short_length = float(spans[0]), float(spans[1])
    if short_length > long_length:
        long_axis, short_axis = short_axis, long_axis
        long_length, short_length = short_length, long_length

    # Point the long axis into the upper half-plane so the reported angle is
    # stable frame to frame instead of flipping by 180 degrees.
    if long_axis[0] < 0 or (long_axis[0] == 0 and long_axis[1] < 0):
        long_axis = -long_axis
    angle_deg = float(np.degrees(np.arctan2(-long_axis[1], long_axis[0])))

    half_long = 0.5 * long_length * long_axis
    half_short = 0.5 * short_length * short_axis
    corners_rect = np.array([
        centroid - half_long - half_short,
        centroid + half_long - half_short,
        centroid + half_long + half_short,
        centroid - half_long + half_short,
    ])

    if homography is None:
        centroid_img, corners_img = centroid, corners_rect
        long_axis_img, short_axis_img = long_axis, short_axis
    else:
        h_inv = np.linalg.inv(homography)
        back = _apply_h(h_inv, np.vstack([
            centroid, centroid + half_long, centroid + half_short, corners_rect,
        ]))
        centroid_img = back[0]
        long_axis_img = _unit(back[1] - centroid_img)
        short_axis_img = _unit(back[2] - centroid_img)
        corners_img = back[3:]

    return RectFace(
        centroid=centroid_img,
        long_axis=long_axis_img,
        short_axis=short_axis_img,
        long_length=long_length,
        short_length=short_length,
        angle_deg=angle_deg,
        area_px=int(mask.sum()),
        _corners=corners_img,
    )


def _unit(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-9 else np.array([1.0, 0.0])


class CentroidTracker:
    """Keeps stable ids for faces across frames by nearest-centroid matching.

    Smooths centroid and side lengths with an EMA so downstream consumers see a
    steady pose rather than per-frame segmentation jitter.
    """

    def __init__(self, max_match_distance: float = 80.0, alpha: float = 0.5) -> None:
        self.max_match_distance = max_match_distance
        self.alpha = alpha
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def update(self, faces: list[RectFace]) -> dict[int, RectFace]:
        """Match faces to existing tracks; returns {track_id: smoothed face}."""
        unmatched = set(self._tracks)
        assigned: dict[int, RectFace] = {}

        for face in faces:
            best_id, best_dist = None, self.max_match_distance
            for tid in unmatched:
                dist = float(np.linalg.norm(self._tracks[tid]["face"].centroid - face.centroid))
                if dist < best_dist:
                    best_id, best_dist = tid, dist

            if best_id is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"face": face, "misses": 0}
                assigned[tid] = face
                continue

            unmatched.discard(best_id)
            prev = self._tracks[best_id]["face"]
            a = self.alpha
            smoothed = RectFace(
                centroid=a * face.centroid + (1 - a) * prev.centroid,
                long_axis=face.long_axis,
                short_axis=face.short_axis,
                long_length=a * face.long_length + (1 - a) * prev.long_length,
                short_length=a * face.short_length + (1 - a) * prev.short_length,
                angle_deg=face.angle_deg,
                area_px=face.area_px,
                _corners=face.corners(),
            )
            self._tracks[best_id] = {"face": smoothed, "misses": 0}
            assigned[best_id] = smoothed

        for tid in unmatched:
            self._tracks[tid]["misses"] += 1
            if self._tracks[tid]["misses"] > 10:
                del self._tracks[tid]

        return assigned
