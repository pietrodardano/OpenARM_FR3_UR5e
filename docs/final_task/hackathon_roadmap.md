# Roadmap and Pitch Notes

> **Status: final task.** Describes the demo the project is building toward.

## Track 2: DaiDai Lab + OpenArms + RobotBlockSet (RBS)

The roadmap now follows the final demo ([demo_task.md](demo_task.md)), not the original hand-tracking teleop plan.
Work is split into parallel streams on separate branches ([branches_and_integration.md](branches_and_integration.md)).
Status is as last known; update it as streams land.

| Stream | Goal | Where | Status |
|---|---|---|---|
| Vision | Find two similar objects (SAM2 + Grounding DINO, optional VLM) and report their X-Y in the robot base frame from an RGB camera | Vision branch | In progress (not inspected) |
| Real control | Cartesian control of the left OpenArm on hardware, with a safety filter | `origin/deploy/openarm-real` | Reported working. Needs a per-arm enable gate before use: the driver enables both arms and will not start if the broken right arm is silent |
| Calibration | Joint zero, camera intrinsics, hand-eye (camera to robot base) | Robot and vision sides | Joint zero: tooling exists. Hand-eye and intrinsics: not started |
| Proprioception | Mass from joint torques; contact detection from torque change | Estimator not pushed; detector not started | Pending |
| Gripper | A grasp that holds a can reliably; force or torque sense if possible | Deploy branch is position-only | Open issue |
| Task and integration | State machine tying vision, control and proprioception together; bins and hand-over poses | `main` after integration | Not started |
| Demo | Full run: find, weigh, compare, then sort or hand over | All | Final |
| Sim lab | Reachability, controller tests and mass-estimation tests without the arm | `openarm-cartesian-sim-validated` (unmerged) | Simulation-validated, never run on the arm |

### Dependencies between streams

1. Right-arm gate in the deploy driver, then any hardware test.
2. Hand-eye calibration, then vision targets can move the arm.
3. Gripper holds a can, then weighing means anything.
4. Mass estimator, then compare and sort.
5. Contact detector, then the hand-over variant.

### Risks worth watching

- The gravity model is not exact, so the mass estimate must use a loaded-minus-unloaded torque difference at one pose.
- Two objects with the same state (both full or both empty) cannot be told apart; the demo must report that case.
- With the table top at 0.75 m and the shoulders at 1.33 m, top-down grasps are hard for this arm. Angled grasps or a higher table are likely needed.
- Merge conflicts across the branches are expected in `README.md`, `agents.md` and the config.

## 3-Minute Pitch Script

1. **The hook (30 s):**
   > "Look at these two cans. They look identical. One is full, one is empty. No camera can tell you which, however good the model. So what do you do? You pick them up."

2. **The idea (60 s):**
   > "Vision finds both cans and tells the robot where they sit. The table height is known, so the robot knows how high to reach. It grasps the first can, lifts it, and reads its own joint torques to feel the weight. Then the second. Vision gave us where; touch tells us what is inside."

3. **The demo (60 s):**
   > "Watch: the robot weighs both, then sorts them into the heavy and light bins. Or hands the full can to me. It will not let go until it feels my hand take the weight, even if my hand looks close to the camera."

4. **The point (30 s):**
   > "Vision alone cannot decide this. Interaction and proprioception are required, and a low-cost arm has both."

## OpenArm Integration Notes

- The OpenArm CAD and documentation repository is the mechanical source of truth for sim-to-real geometry.
- The hardware bridge is Python-native on the official `openarm_can` binding; ROS and ROS2 are not required.
- Calibrate each arm with `openarm-can-zero-position-calibration --robot-version v1` and pass the arm side and CAN port explicitly. Only the left arm (`left_arm`, `can1`) is in use.
