# Final Demo: Telling Full from Empty by Touch

> **Status: final task.** Describes the demo the project is building toward.

## The claim

Two objects that look the same (two cans, or two boxes) can differ in what is inside. **Vision alone cannot
tell which is full and which is empty.** The robot has to interact with each object and feel it: lifting
an object and reading the joint torques gives its mass, and mass separates full from empty.

What each part contributes:

| Part | Supplies | Does not supply |
|---|---|---|
| Vision (SAM2 + Grounding DINO, optional VLM, RGB camera only) | Which objects exist and their **X-Y pose** on the table | Whether an object is full or empty |
| Known workcell geometry | Table height, so the **Z** of every grasp | Object X-Y |
| Robot arm (left arm only) | Reaching, grasping, lifting, putting back, hand-over | Object identity |
| Proprioception (joint torques) | The object's **mass**, and external contact during the hand-over | Where objects are |

## Simulated robots alongside

In the rendered scene, a simulated FR3 and UR5e run this same task next to the real arm, in step with its phases but
on their own objects: see [cross_embodiment.md](cross_embodiment.md). They also serve as a test bed for the mass and
contact logic below.

## Sequence

1. **Detect.** Vision reports two objects with X-Y poses in the robot base frame. Z comes from the table height.
2. **Object 1.** Approach above the pose, descend to the known grasp height, close the gripper, check the grasp,
   lift to the weighing pose, hold still, record the torque window, estimate its mass. Put it back at the
   same pose, release and retract.
3. **Object 2.** Same procedure at its own pose.
4. **Compare.** The two mass estimates are compared. See "Decision logic" below.
5. **Act on the result**, one of two variants:
   - **Sort.** Place each object in a predefined bin chosen by weight (heavy bin, light bin).
   - **Hand over.** Take the heavy object (the full one) to an up-forward pose and offer it to the user. The
     gripper does not open until the arm **feels** an external contact in its joint torques. If the torques show
     no change, the user has not taken the object, even if they look close, so the gripper stays closed.

## State machine

```
INIT -> DETECT -> PICK(i) -> LIFT(i) -> WEIGH(i) -> PLACE_BACK(i) -> (next object) -> COMPARE
COMPARE -> SORT(i)           (variant 1)
COMPARE -> HANDOVER_PRESENT -> HANDOVER_WAIT_CONTACT -> RELEASE   (variant 2)
any state -> SAFE_HOLD       (on fault)
```

| State | Does | Leaves when | Failure handling |
|---|---|---|---|
| INIT | Checks the left arm is enabled and feedback is healthy; refuses the right arm | Healthy | Stay; report |
| DETECT | Takes vision detections for both objects, checks they are inside the reachable region | Two stable detections | Retry; if vision is lost, stop and report |
| PICK(i) | Approach, descend to known Z, close gripper | Gripper closed on an object (not fully closed) | Fully closed means missed: back off, re-detect, retry once, then SAFE_HOLD |
| LIFT(i) | Lift to the weighing pose | Pose reached and settled | A sudden torque drop means slip: put the object back, retry once |
| WEIGH(i) | Hold still, record the torque window | Window is steady | Unsteady window: wait and record again |
| PLACE_BACK(i) | Return the object to its saved pose, open, retract | Retracted | Controller fault: SAFE_HOLD |
| COMPARE | Compares the two masses | Decision made | If the masses are indistinguishable, see below |
| SORT(i) | Pick and place into the bin chosen by weight | Both placed | As PICK and PLACE |
| HANDOVER_PRESENT | Moves the heavy object to the up-forward pose | Pose reached | Fault: SAFE_HOLD |
| HANDOVER_WAIT_CONTACT | Holds, watches the torque change against a baseline | Sustained change seen | Timeout: do **not** drop it; put it in the heavy bin instead |
| RELEASE | Opens the gripper | Open | None |
| SAFE_HOLD | Holds the last measured pose, ignores new commands until cleared | Operator clears | None |

## Decision logic

- **Relative comparison is the robust part.** With two objects at the same grasp pose, the heavier one is
  decided from the difference of the two estimates. If the difference is smaller than the measurement
  uncertainty, report "indistinguishable" instead of guessing, and repeat the weighing. Two full cans or two
  empty cans can happen.
- **Absolute "full" or "empty" needs a threshold per object type** (for example a mass between the known empty
  and full values). Without a calibrated threshold, the demo can say heavier or lighter, not full or empty.

## How mass is estimated from torque (design constraints)

The estimator code is **not pushed yet**, so nothing here claims it is tested. The design constraints below
come from the physics and from simulation measurements:

- The static torque change from an object of mass $m$ at the gripper is approximately
  $\Delta\tau = J_v(q)^{\top}\,(m\,g\,\hat z)$, where $J_v$ is the TCP position Jacobian. Estimate $m$ by
  least squares over the joints, weighted by sensitivity.
- **Use the difference between the loaded and unloaded torque at the same pose.** A gravity model built from
  the official MJCF masses is not exact. In simulation, a 25% error in the model masses produced about 1 Nm of
  torque error at the ready pose, while a 0.35 kg mass difference at a 0.4 m lever adds roughly 1.4 Nm
  ($0.35 \cdot 9.81 \cdot 0.4$). Absolute model torques would drown the signal; the difference cancels the
  model error.
- Weigh with the arm **still** (joint velocities near zero) and average over a window.
- Grasp every object at the same pose relative to the TCP, so the lever arm does not change between objects.
- The gripper is position-controlled and exposes no force or torque. A loose grasp can slip or swing, so verify
  the grasp before trusting a measurement.

## How contact is felt in the hand-over

- Take a baseline torque window while holding at the hand-over pose, then watch the change in joint torque
  relative to it. A sustained change, in either direction (the user pulls, or takes some weight), is contact.
- Require the change to persist over several samples and exceed a threshold set from the noise of the baseline
  window. The gravity model is not needed for this, only the change against the baseline.
- Never open on a timeout. If no contact comes, the object goes to the heavy bin.

## Success criteria (to fill in with measured values)

| Criterion | Target |
|---|---|
| Object X-Y reaching error | to be measured with the hand-eye calibration |
| Mass-estimate repeatability on one object | to be measured |
| Smallest mass difference reliably separated | to be measured |
| Contact detection: false release rate, missed contact rate | to be measured |
| Full run: both objects found, weighed and sorted/handed over without intervention | demo success |

## Constraints to plan around

- **Reach versus table height.** The arm's shoulders sit at 1.330 m and the table top is at 0.750 m. A coarse
  simulation reachability check (16 grid points, five tool yaws, left arm, `openarm-cartesian-sim-validated`)
  found top-down grasp poses reachable at only 3 of 16 points at 0.75 m, against 9 of 16 at 0.95 m and
  11 of 16 at 1.05 m; a 45 degree forward tilt did better (4, 15 and 14 of 16). Plan angled or side grasps for a
  0.75 m table, or raise the table. Treat the counts as a trend, not a map.
- **One arm.** The right arm is disabled (known hardware problem); every step above uses the left arm.
- **Calibrations needed:** hand-eye (camera to robot base), camera intrinsics, joint zero (see the real-robot docs on the
  deploy branch). Calibration error moves the TCP: 0.01 rad of zero error per joint is about 7 to 8 mm at the
  TCP.
