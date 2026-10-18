# OpenArm v1 Physical Workcell

This scene models the measured fixed structure around the official OpenArm v1
parallel-link bimanual MJCF. The generated scene retains all official arm and
gripper geometry, joints, limits, and actuators. It deliberately removes the
upstream root chassis/torso geoms, replacing them with the measured aluminum
profile below.

## World Frame

The MuJoCo world uses $+X$ forward, $+Y$ left, and $+Z$ upward. The ground is
at $z=0$.

| Component | Placement |
| --- | --- |
| OpenArm shoulder centers | $(0, 0, 1.330)\,\mathrm{m}$ |
| Square aluminum torso profile | $0.060\,\mathrm{m} \times 0.060\,\mathrm{m}$, from ground to $z=1.420\,\mathrm{m}$ |
| Tabletop surface | $z=0.725\,\mathrm{m}$ |
| Camera mount origin | $(0.070, 0, 1.600)\,\mathrm{m}$ |
| Camera optical axis | Forward, pitched $70^\circ$ downward from horizontal (near-vertical, eyeballed on the mount, not instrument-measured) |

The official model places `arm_origin` $0.697\,\mathrm{m}$ above
`openarm_body_link0`. The generated scene therefore positions that root at
$z=0.633\,\mathrm{m}$, giving the measured shoulder-center height.

The table is a collision-enabled $40\,\mathrm{mm}$ thick top, with its upper
surface at the measured $725\,\mathrm{mm}$ height. Its currently configured
$1.20\,\mathrm{m} \times 0.80\,\mathrm{m}$ footprint is provisional until
the physical tabletop is measured.

The table's rear edge is at $x=0.050\,\mathrm{m}$, leaving a
$20\,\mathrm{mm}$ collision clearance from the aluminum torso's forward face.
The pick-and-place plan explicitly raises laterally beside the table before it
moves over the tabletop, then descends vertically for the top grasp.

## Pick-and-Place Demonstration Props

The generated simulation includes a $40 \times 50 \times 100\,\mathrm{mm}$,
$50\,\mathrm{g}$ red grasp block on the tabletop and a green destination area.
It is positioned close to the base and between the official v1 parallel-link
finger collision meshes at a verified reachable **lateral** grasp posture; the
previous top-down grasp was geometrically unreachable. `left_pick_cube_grasp` is a
disabled MuJoCo weld constraint used only by the simulation demonstration after
a top-down approach reaches the cube. It is not a physical grasp model and must
never be used as an execution or safety assumption for hardware.

Run the task planner with `uv run python scripts/run_pick_place_demo.py`; add
`--show` to animate it. Direct OMPL planning uses MuJoCo contacts to reject
arm–table and left–right arm contact. The grasped cube follows the TCP
kinematically from its captured tabletop grasp transform—the object never
teleports to the TCP—and the parallel-link fingers visibly close before the
lift. This demonstrates planning and task geometry, not force-closure or
hardware-safe manipulation.

The task script relaunches itself under the repository `.venv` when started
from a system interpreter or editor. On macOS, `--show` automatically uses the
venv `mjpython` launcher required by MuJoCo's passive viewer.

For repeatability testing, pass `--trials N`. Each trial creates a fresh
MuJoCo environment and replans all six phases under a varied RRTConnect
extension range, planning budget, and interpolation count. `--seed` initializes
the OMPL process-wide random sequence once before the trial sequence.

## Viewer Lighting

The generated scene includes an elevated ambient headlight plus soft key and
fill directional lights. These are presentation settings only: they do not
change collision geometry, dynamics, or camera calibration.

## Generate and View

Generate the local scene after cloning the official OpenArm sources:

```text
uv run python scripts/setup_openarm_v1_scene.py
uv run python -m mujoco.viewer --mjcf=third_party/openarm/openarm_mujoco/v1/openarm_bimanual_physical_scene.xml
```

Select `physical_hbv_rgb_camera` in the MuJoCo viewer to inspect the fixed RGB
camera viewpoint. Its $70^\circ$ vertical field of view is a temporary visual
approximation: the documented HBV camera has no calibrated intrinsics, so it
must be replaced after a checkerboard/Charuco calibration.