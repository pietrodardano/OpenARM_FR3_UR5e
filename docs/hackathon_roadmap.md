# 30-Hour Hackathon Roadmap & Pitch Notes

## Track 2: OpenArms + RobotBlockSet (RBS) + MuJoCo

### Workstreams

| Workstream | Where | Status | Deliverable |
|---|---|---|---|
| Foundation & sim workcell | `main` | **Done** | Official OpenArm v1 scene, measured workcell, OMPL pick-and-place, cross-embodiment view |
| Real-robot control & weighing | `main` (from `deploy/openarm-real`) | **Done** | Cartesian `move_l`, gripper torque mode, torque-based load comparison tested on the real left arm |
| Vision: object finding | vision branch | **In progress, not merged** | GroundingDINO + SAM2 (+ VLM) → object XY on the table in the base frame |
| Demo state machine + robot adapter | `main` (`src/task/`) | **Done (mock + sim rehearsal)** | Detect → weigh × 2 → compare → sort / felt handover; `OpenArmSkills` with IK-checked moves |
| Physics sim of the full task | `main` | **Done** | 72.5 cm workcell, real can lifting, torque weighing, sort + felt handover (simulated user) |
| First real run | robot | Next | Cans/bins at the §7 layout, `--backend real --mirror` with manual object positions |
| Integration with vision | after vision merge | Waiting | Vision `ObjectDetector` replaces `demo.manual_objects`; calibrate contact thresholds |
| Rehearsal | `OpenArmSim` → real arm | Scheduled | Both modes end to end with cans and boxes |
| Demo video & pitch | — | Scheduled | Camera view with masks, robot weighing, result overlay |

The remaining checklist is in [demo_task.md §9](demo_task.md#9-integration-status).

---

## 3-Minute Pitch Script

1. **The hook (30 s):**
   > "Here are two cans. Same brand, same color, same dent. One is full, one is empty. Ask the best vision model you know which one is full: it can't. The information isn't in the pixels."

2. **The idea (45 s):**
   > "GroundingDINO and SAM2 tell our OpenArm *where* the cans are, from a single RGB camera and the known table height. Then the robot does what a person does: it picks each one up and *feels* it. No force sensor, no scale: just the torques its own joint motors already report, with a small dithering motion that cancels friction."

3. **The live demo (75 s):**
   > "Object one: grasp, lift, weigh, put back. Object two: the same. The robot now knows that object one is about 330 grams heavier, with a confidence of more than three sigma. It sorts them into the 'full' and 'empty' bins.
   > Second mode: it offers the full can to me. I reach in, and it doesn't let go while my hand is just near it. When I actually take the weight, it feels the change in its joints and opens the gripper."

4. **The impact (30 s):**
   > "Perception tells you where things are. Interaction tells you what they are. On low-cost open hardware, in plain Python, with the official OpenArm SDK."

---

## Demo Day Checklist

- Objects: two upright cans (grip at ≈1 Nm) or two boxes; one full, one empty.
- Table at 72.5 cm, rear edge 5 cm from the torso (as in `workcell.table`); robot's left side clear for the side route.
- Bins placed at `demo.bins.*`; handover pose clear of people except the volunteer.
- Hardware e-stop within reach. Rehearse once in `OpenArmSim` first.
