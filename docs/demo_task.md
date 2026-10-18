# Final Demo: Vision Cannot Weigh

## 1. Claim

Two objects that look the same, such as two identical cans or two identical boxes, can
differ in what matters: one is full and one is empty. A camera cannot tell them apart,
however good the segmentation or the VLM. The robot has to **interact**: lift each
object and **feel** its weight through its own joint torques (proprioception). It needs
no force sensor and no scale.

The demo shows the full chain: perception finds *where* the objects are; interaction
finds *what* they are; the robot acts on that knowledge.

## 2. Setup

| Item | Value |
|---|---|
| Robot | OpenArm v1, parallel-link gripper. **Left arm only** (right arm broken; `hardware.active_sides: [left]`) |
| Workcell | Measured ([physical_workcell.md](physical_workcell.md)): 60 mm aluminium torso, shoulder axis at 1.330 m, **tabletop at 0.725 m**, rear table edge 5 cm in front of the torso |
| Camera | HBV HD CAMERA, **RGB only**, on the torso mount ([camera_specifications.md](camera_specifications.md)) |
| Objects | Two look-alike cans *or* two look-alike boxes: one full, one empty |
| Destinations | Two bins on the table (`left` = robot's +Y, `right`) and an up-forward handover pose |
| Instruction | Free text, e.g. "put the lighter can in the left bin" or "give me the heavy one" ([src/task/prompt.py](../src/task/prompt.py)) |

Grasp: **lateral**. The fingers squeeze the upright can's sides with a horizontal closing axis, as in the
real-arm grip tests (2026-10-17, [real_robot_deployment.md](real_robot_deployment.md)), at **1.0 Nm**.
At a 72.5 cm table the hand cannot stay horizontal (the IK reaches no lower than TCP z ≈ 0.30 m), so the
hand is **pitched 45° down** (`demo.grasp.pitch_deg`). The closing axis stays horizontal and the can
stays upright the whole time: grasp, carry, weigh pose, bins and handover all use the same pitch.

## 3. Frames and the vision contract

- **Robot base frame:** the official OpenArm model frame used by the real-robot controller:
  +X forward, +Y left, +Z up; the shoulder axis is at $z = 0.697$ m. Every pose in `demo:` is in this frame.
- **Table plane:** $z = z_{table} = 0.092$ m in the base frame, derived from the measured 0.725 m tabletop
  and the 1.330 m shoulder height ([src/task/workcell.py](../src/task/workcell.py)).
- **Vision output:** [`VisionObjectDetector`](../src/vision/vision_detector.py) implements the task's
  `ObjectDetector`. Per object it returns an [`ObjectDetection`](../src/task/interfaces.py): `label`,
  `x`, `y` (base frame, m), `confidence`, `frame="base"`. Pipeline per poll:
  1. Grab `vision.object_detection.frames` frames (rotated like the calibration, colour-corrected).
  2. [`ObjectLocalizer`](../src/vision/object_localizer.py): GroundingDINO boxes from the text prompt,
     SAM2 masks, PCA fit of each object's top face.
  3. RGB only, so depth comes from geometry. The top-face centroid's camera ray is intersected with the
     plane at table height + the object's top height (0.12 m for a can):
     with intrinsics $K$, camera-to-base rotation $R$ and camera centre $c$,
     $d = R K^{-1}[u, v, 1]^T$, $p = c + \lambda d$, $\lambda = (z_{plane} - c_z)/d_z$.
     Mapping the top face through the table homography instead would be off by about 3 cm (parallax).
  4. Fuse the frames: merge within 4 cm, keep objects seen in at least half the frames, drop anything
     outside `vision.object_detection.workspace`.
- **Calibration:** `configs/camera_extrinsics.yaml` (intrinsics, camera pose, table homography), written by
  `scripts/calibrate_camera_touch.py` (best) or `calibrate_table_tape.py`; see
  [camera_calibration.md](camera_calibration.md). Both now default the table plane to the **base-frame**
  height (0.092 m).

The vision pipeline only decides *where*. It must not and cannot decide *which one is full*.

## 4. Sequence

Implemented in [`WeighAndSortTask`](../src/task/weigh_and_sort.py); each step is a logged `Phase`.

1. **DETECT.** Poll vision for the object prompt. It must return exactly two objects with
   `confidence >= demo.min_confidence`. They look alike, so vision cannot tell which one is heavier.
2. **Object 1:**
   1. Open the gripper.
   2. **APPROACH:** above the pre-grasp pose, then the pre-grasp pose (6 cm back along the finger axis), then the grasp pose.
   3. **GRASP:** torque mode at 1 Nm.
   4. **LIFT:** 15 cm up, then the fixed **weigh pose** (`demo.weigh.pose`).
   5. **WEIGH:** joint torques give the mass; it is stored in the belief about object 1.
   6. **PLACE_BACK:** at the same spot, then release and **RETRACT** to the approach pose.
3. **Object 2:**
   1. The same up to WEIGH.
   2. **COMPARE** right away with the stored object 1, while still holding object 2:
      $\Delta m$, $z = \Delta m / \sqrt{se_1^2 + se_2^2}$.
   3. If $z <$ `min_z` (3), object 2 is measured once more.
   4. If it is still unclear: put it back, go home, **INCONCLUSIVE**. The robot never sorts on a guess.
4. **Act** on the instruction:
   - **Sort:** object 2 goes straight to the bin of its class (the prompt names the light or the
     heavy bin; the other class takes the other bin). Then vision is polled again to re-locate object 1,
     which the robot put back, and object 1 goes to the other bin.
   - **Handover** ("give me the heavy can"): the requested object is offered at the up-forward pose.
     - The robot opens the gripper **only when it feels the user take the object** in its joint torques.
     - If no contact comes before `contact.timeout_s`, the object goes to its class bin (heavy bin for a heavy object).
     - The other object goes to its own bin, before or after the handover depending on which one is in hand.
5. **HOME → DONE.**
   - Any failure leads to **ABORTED**. A held object is first put back where it came from, then the arm parks beside the table.

## 5. How the weight is measured ([src/estimation/payload.py](../src/estimation/payload.py))

The Damiao motors report output torque from phase current. A payload $m$ at the TCP adds

$$\Delta\tau = J_z(q)\, m\, g$$

where $J_z$ is the vertical row of the TCP position Jacobian. At elbow 90°, 300 g adds
about 1.2 Nm on joints 1 and 4 and about 0.56 Nm on joint 7. `payload.measure_load` works as follows:

- It dithers the load-bearing joints ±1° over whole sine cycles, so Coulomb friction averages out.
- It subtracts the model's arm gravity at the measured pose.
- It least-squares-fits $m$ on $J_z$, giving one mass per cycle, hence mean ± SE.

The absolute mass is rough. The **difference** between two objects measured the same
way is reliable, which is exactly what the demo needs.

## 6. Configuration

All tunables are in the `demo:` section of
[task_config.yaml](../configs/task_config.yaml): `mode`, `object_prompt`,
`min_confidence`, `grasp.*` (pitch, height, torque), `place_back_offset_m`, `release_settle_s`, `motion.*` (speeds, guard, side route),
`grasp_height_above_table_m`, `approach_clearance_m`, `grasp.{style, torque_nm}`,
`weigh.{min_z, retry_once}`, `bins.{heavy, light}`, `handover`, `contact.*`, and `mock.*`
for offline runs.

## 7. Reach, layout and motion safety

The arm's shoulder is 0.605 m above the tabletop. The usable table area of the 45° lateral grasp is about
x 0.20–0.32 m, y 0.06–0.36 m (robot base frame, origin at the arm root, table top at z = 0.092 m).
It is found by an IK scan plus a collision check:

- below y ≈ 0.05 m, the left forearm hits the torso;
- further out, the IK cannot follow.

So the default layout is a 2 × 2 grid: the cans in the back row (x 0.21, y 0.13 / 0.27), the bins in the
front row (x 0.32, y 0.12 / 0.26). The bins are about 4 cm apart and about 2.6 cm in front of the cans.

Every motion is checked before the arm moves ([OpenArmSkills](../src/task/openarm_adapter.py)); a failed
check raises `UnsafeMotion` without moving:

- **Reach:** the target is inside `controllers.ik.workspace`, and the IK follows every sampled point of the straight line.
- **Collision guard:**
  - Arm geoms keep 1 cm from the torso, table, bins and the parked right arm
    ([WorkcellCollision](../src/planning/workcell_collision.py), on the demo scene).
  - The arm bodies also stay 3.5 cm above the table.
- **Side route:** the hanging zero pose is *below* the tabletop, right behind its edge, so a straight
  joint move from zero to home (or the driver's `stop()` back to rest) would sweep the hand through the
  table. The demo routes zero ↔ home through `demo.motion.side_route_qpos`:
  1. the straight arm swings out to +Y beside the table;
  2. the elbow bends;
  3. the arm comes in to home.

  **The space at the robot's left, out to about 0.7 m, must be clear.**
- **Park before stop:** `run_weigh_demo.py` parks the arm beside the table before `go_to_zero` and `robot.stop()`.
  If parking is impossible, the arm holds and the script asks before stopping.

**Warning for the other real-robot scripts** (`compare_weights.py`, `cartesian_points.py`, …):
their start/stop wrapper `run_with_zero` moves straight between zero and the task. With the 72.5 cm
table this close, keep the table out of their way or run them without the table.

## 8. Run it

```bash
uv run python scripts/run_weigh_demo.py                                                  # mocks, no robot
mjpython scripts/run_weigh_demo.py --backend sim --render --yes                          # physics, sort
mjpython scripts/run_weigh_demo.py --backend sim --render --yes --prompt "Give me the heavy can"
uv run python scripts/run_weigh_demo.py --backend sim --fast --yes                       # headless, fast
uv run python scripts/run_weigh_demo.py --backend real --mirror                          # real arm + live twin
uv run python -m unittest tests.test_task -v
```

- **Sim (`--backend sim`):** the demo scene ([setup_weigh_demo_scene.py](../scripts/setup_weigh_demo_scene.py),
  regenerated when the config changes) has the measured workcell and two identical cans with different
  masses (`demo.sim.masses_kg`, 360 g and 60 g):
  - The cans are really gripped and lifted, and the weights come from the simulated joint torques.
  - A ground-truth detector stands in for vision.
  - In the handover, a simulated user pulls on the offered can after `demo.sim.user.grasp_after_s`.
    `--no-user` tests the timeout path.
- **Real (`--backend real`):** object positions come from vision (`--detector vision`, the default).
  - Before anything moves, vision is polled once and the base-frame positions are printed for you to confirm.
  - The task re-polls vision before it picks the object it put back.
  - `--objects "x,y;x,y"` or `--detector manual` uses fixed positions instead.
  - Vision alone, no robot: `python scripts/detect_objects.py --prompt "soda can"` prints the positions and saves an
    annotated frame to `outputs/detections.png`.
  - `--mirror` shows a read-only twin of the demo scene following the measured joints; the
  held can is drawn attached to the hand while the gripper squeezes. On macOS, viewers need `mjpython`.

## 9. Integration status

| Component | Where | Status |
|---|---|---|
| Task state machine, contracts, contact detector, mocks | `main`: [src/task/](../src/task/) | Done; mock + sim rehearsal pass |
| Real robot driver, Cartesian `IKController.move_l`, gripper torque mode, `payload.measure_load`, `OpenArmSim` | `main`, brought from `deploy/openarm-real` (files copied, branch not merged) | Proven on the real left arm |
| Robot adapter `OpenArmSkills` / `OpenArmPayloadEstimator` | `main`: [src/task/openarm_adapter.py](../src/task/openarm_adapter.py) | Done; sim-rehearsed, **not yet run on the real arm** |
| GroundingDINO + SAM2 localization, camera calibration, `VisionObjectDetector` | `main`: [src/vision/](../src/vision/), from `origin/demo-simulation` commit `6b5d670` (vision files only) | Integrated; geometry tested without torch. **Not yet run end to end with the camera + robot** |

From `deploy/openarm-real` we took `src/{control,hardware,estimation}`, the OpenArm
IK/kinematics, `OpenArmSim`, `docker/`, the deployment doc and its tests, and these scripts:
`setup_can.sh`, `check_robot`, `read_ee_pose`, `go_zero`, `safe_zero`, `jog_joints`,
`cartesian_points`, `ik_axis_test`, `grip_torque_test`, `compare_weights`, `run_control_demo`.
We left out `crush_test.py` (a one-off grip-limit characterization, results recorded) and
`gripper_test.py` (replaced by torque mode). Both remain on the branch. Do not merge the branch
wholesale; changes to these files belong on `main` now.

From `origin/demo-simulation` we took only the vision commit's files: `src/vision/{camera,
object_localizer, rect_face_pca, table_calibration}.py`, the calibration / capture / visualization
scripts, their tests and `docs/camera_calibration.md`. The sandbox, lever, jug scenes and that branch's
older copy of the controller were not taken. Its GroundingDINO / SAM2 dependencies are in
`requirements-vision.txt` (not in `pyproject.toml`: they need torch to build, which breaks `uv lock`).

Remaining checklist:

1. On the Linux workstation (README, "Full run on the Linux workstation"):
   1. `ROBOTOKENS_VISION=1 docker/dev.sh build`, then `scripts/download_vision_weights.sh`.
   2. Calibrate the camera: intrinsics, then `calibrate_camera_touch.py --side left`.
   3. Check `scripts/detect_objects.py` against a ruler.
2. Place the real cans and bins at the layout of §7 (or pass `--objects`). Check that the table's rear
   edge and height match `workcell.table` and that the robot's left side is clear for the side route.
3. First real run: `--backend real --mode sort` with two empty cans (low risk), then full / empty.
4. Calibrate `demo.contact.*` on the real arm: record holding-torque noise at the handover pose, then a real hand grasp.
