# Camera Calibration

Everything is expressed in the frame the MJCF and the controllers use:
**+x forward, +y left, +z up, origin on the floor under the torso.**

## The short version

Reliability has two independent halves, and they fail for different reasons:

| Half | Fixed by | Can tape alone do it? |
|---|---|---|
| **Lens** — focal length, principal point, distortion | `calibrate_intrinsics.py` with a ChArUco board | **No.** Guessed from a tape square on a wide-angle lens, fovy came out 62° instead of 85° and the camera position 0.6 m off — while still reprojecting the four corners to 0.4 px. |
| **Pose** — where the camera is in the robot's frame | `calibrate_camera_touch.py` (best) or `calibrate_table_tape.py` | Yes, but its accuracy is capped by your tape measure, not by the maths. |

So: **do the board step once**, then pick a pose method. A tablet is an excellent
board, for a reason that is not obvious — see below.

## 1. Lens: ChArUco on a tablet screen

**The board's physical size does not affect the intrinsics at all.** `fx`, `fy`,
`cx`, `cy` and the distortion coefficients come out bit-identical whether the
squares are 14 mm or 60 mm; the size only scales the per-photo distance to the
board, which `calibrate_intrinsics.py` discards. A tablet board smaller than A4
is therefore not a compromise, and a few percent of error in the size you type
changes nothing.

A screen is in some ways *better* than paper: dead flat, no printer scaling, no
curl, self-lit. Generate one sized to your screen's exact pixel resolution, so
any viewer that shows it fullscreen without cropping displays it 1:1:

```bash
../.venv/bin/python scripts/charuco_for_screen.py --screen 2360x1640 --ppi 264
# or, if you don't know the PPI:
../.venv/bin/python scripts/charuco_for_screen.py --screen 2360x1640 --diagonal-in 10.86
```

It prints the exact square size in mm (`square_px * 25.4 / ppi`) and the grid
width to confirm with a ruler. Measure across *all* the squares and divide —
that divides your ruler error by the number of squares.

On the tablet:

- open the PNG in the photo viewer, **fullscreen, no pinch-zoom, no crop**;
- **lock screen rotation**, and turn off auto-lock so it never dims mid-session;
- turn off **auto-brightness and night mode / warm tint**; set brightness around
  70% — a blown-out screen loses marker contrast;
- keep room lights off the glass. Glare is the one thing that beats a screen;
  paper has no specular highlight.

Then take 20–30 photos and run:

```bash
# shoot the photos with live detection + coverage feedback
../.venv/bin/python scripts/capture_frames.py --out outputs/intrinsics_photos \
    --charuco --square-size 19.242 --marker-size 13.758

../.venv/bin/python scripts/calibrate_intrinsics.py --images outputs/intrinsics_photos \
    --square-size 19.242 --marker-size 13.758 --out configs/camera_intrinsics.yaml
```

Tilt the tablet 20–45° in different directions, vary the distance, and put the
board in the **corners** of the frame as well as the middle — frame coverage is
what pins down distortion. Keep it sharp; motion blur is the usual reason a
photo gets skipped. Aim for under ~0.5 px reprojection error.

Validated on synthetic photos of a 96 × 135 mm screen board (19.2 mm squares,
28 views): recovered `fx` to 0.2%, the principal point to ~2 px, `k1` −0.317 vs
−0.320 true, 0.31 px reprojection — and fovy 85.10° against 85.00° true.

## 2. Pose: let the robot define the geometry (recommended)

```bash
../.venv/bin/python scripts/calibrate_camera_touch.py --side right --session outputs/touch_session
```

Move the gripper to 12–20 points, press Enter at each, then click the tip in each
captured frame. The reference positions come from the robot's own forward
kinematics, so there is **no tape measure anywhere in the chain**, and whatever
bias the FK carries is shared with the controller that will execute the grasp —
the camera and the arm agree even where both are slightly off, which is what
actually lands the gripper on the object.

The script **never commands motion**. Move the arm yourself: by hand with the
motors disabled, in a gravity-compensated hold, or with `scripts/jog_joints.py`
in another terminal.

- Close the gripper fully and click **where the fingertips meet**, every time.
  The TCP sits on the gripper axis 90 mm past the hand base.
- Spread the points over the whole area you will pick from, at **two or three
  heights** (on the table, ~100 mm up, ~200 mm up). All-coplanar points are
  noticeably worse conditioned and the script warns when your cloud is flat.
- `--session DIR` saves the frames and FK points. Touching the robot is the
  expensive part; `--load-session DIR` re-clicks and re-solves with no robot or
  camera attached.
- A mis-clicked tip is detected and dropped automatically (`--max-residual-px`).
- It reports how far your lowest touches sit from `workcell.table.top_height_m`,
  which is a free check on both the tabletop height and your tip offset.

Measured: ~2.3 mm mean tabletop error across the workspace and ~15 mm camera
position error, with 20 points, 1.5 px clicks, 2 mm of touch error, and the
tablet-derived intrinsics above.

## 3. Pose without moving the robot: tape rectangle

```bash
../.venv/bin/python scripts/calibrate_table_tape.py --side 500 --depth 400 \
    --near-edge-from-table-rear 150
```

Tape a rectangle on the tabletop, as **large** as the camera's view allows,
enclosing the area you pick from; square up its diagonals; measure the two sides
outer-edge to outer-edge; measure how far its near edge sits forward of the
table's rear edge (`--near-edge-from-table-rear`) and any lateral offset of its
centre (`--center-y`). You click the four corners, it overlays a 100 mm
base-frame grid and lets you click points to read their coordinates — do that
check with a ruler before trusting it.

The camera must see the table at an angle; a fronto-parallel view is
ill-conditioned and carries no focal-length information.

**What limits this method is the tape measure, not the camera:**

| How well you know where the tape is, in the base frame | Resulting tabletop error |
|---|---|
| exactly (simulation only) | 3.0 mm mean |
| ±3 mm | 4.7 mm |
| ±5 mm | 6.8 mm |
| ±10 mm | 12.5 mm |

Accuracy also degrades as you extrapolate outside the taped area — roughly
1–3 cm at 300 mm beyond an edge — which is why the rectangle should enclose the
workspace rather than sit in the middle of it.

A ChArUco board laid on the table instead of tape does *not* help: its corners
are detected ~10× more precisely than you can click, but once realistic
hand-placement error (±2 mm, ±1°) is included it lands at ~5 mm — the same
regime, because the limit is knowing where the target is, not where its corners
are in the image. Four placements of the board across the taped rectangle get to
~2.9 mm. Only the robot-touch method removes the term entirely.

Useful flags: `--image photo.jpg` to work from a photo, `--device N` if the
camera is not at `vision.device_index` (that value is macOS-oriented; on Linux
the same camera usually sits at a different `/dev/video*` index), `--mode edges`
to click along the tape edges instead of the corners (worth it at 5+ clicks per
edge, and the way out when the corners are rounded or occluded), `--points` for
a headless run.

## Consuming the result

Both pose scripts write the same `configs/camera_extrinsics.yaml` — camera pose,
MuJoCo `pos`/`quat`/`fovy`, a pixel→tabletop homography, and a `warnings:` list
if anything about the run was weak.

```python
from src.vision.table_calibration import TableCalibration

calib = TableCalibration.load("configs/camera_extrinsics.yaml")
x, y = calib.pixel_to_table(u, v)      # metres, base frame
target = calib.pixel_to_base(u, v)     # (x, y, table_z)
```

The homography is valid only for points lying on the tabletop plane, and only
while neither the camera nor the table moves. Re-run after either is touched,
and feed the camera through the same 180° rotation the calibration recorded
(`rotate_180_applied`).

## Picking the right camera

Don't pass `--device`; the scripts find the HBV by the `vision.camera_device`
name in the config and print which index they landed on. Two reasons it is worth
not guessing here:

- **Indices mean different things to different backends.** On this machine V4L2
  index 0 is the laptop's own ASUS webcam and index 2 is the HBV, while indices
  1 and 3 are metadata-only nodes. When a V4L2 node cannot stream, OpenCV
  silently falls back to its FFMPEG backend — which numbers *only* capture
  devices, so the same integer selects a different camera. The scripts pin
  `CAP_V4L2` so index N always means `/dev/videoN`.
- **This module only offers 1280x720 in MJPG.** Ask for that size without
  requesting MJPG first and it quietly hands back 640x480 YUYV. Calibrating in
  one mode and running in another invalidates the intrinsics — the field of view
  is not even the same between its 4:3 and 16:9 modes. The scripts request MJPG,
  then verify and report the mode they actually got.

To see what is attached:

```bash
../.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); \
    from calibrate_table_tape import describe_cameras; print(describe_cameras())"
```

## Note on the environment

Run everything with the existing venv one level above the repo:

```bash
../.venv/bin/python scripts/<script>.py ...
```

`uv run` currently fails in this checkout (it tries to build `torch` and the
build errors out), and the **system** Python has OpenCV 4.6, which lacks
`cv2.aruco.CharucoDetector` — `calibrate_intrinsics.py` and the live detection in
`capture_frames.py` cannot run there. The venv above has OpenCV 5.0 and MuJoCo
3.16, and imports the project's `src/` modules fine.

The two OpenCV versions also differ in the planar pose solvers: OpenCV 5's
`SOLVEPNP_IPPE` returns success-with-NaN for a target that is not in the object
frame's z = 0 plane, which `calibrate_table_tape.py` handles explicitly.
