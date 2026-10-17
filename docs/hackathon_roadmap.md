# 30-Hour Hackathon Roadmap & Pitch Notes

## Track 2: DaiDai Lab + OpenArms + RobotBlockSet (RBS)

### 30-Hour Sprint Timeline

| Phase | Milestone | Hours | Status | Key Deliverable |
|---|---|---|---|---|
| **Phase 1** | Foundation & Architecture | 0 – 6 | **COMPLETED** | Repo scaffold, `agents.md`, bimanual MJCF, RBS wrapper, mock stream, unit tests |
| **Phase 2** | Live Perception Stream | 6 – 12 | Next Up | Connect DaiDai Lab live camera / WebSocket stream, wrist pose + 21-joint landmark ingestion |
| **Phase 3** | RBS OMPL & Dynamic Obstacles | 12 – 18 | Scheduled | Real-time C-space collision checking with table obstacles and moving targets |
| **Phase 4** | Bimanual Coordination & Grasping | 18 – 24 | Scheduled | Dual-arm handover task, pick-and-place with synchronous pinch detection |
| **Phase 5** | Physical OpenArms Sim2Real | 24 – 28 | Scheduled | Transfer joint trajectories to OpenArms hardware via serial/Dynamixel bridge |
| **Phase 6** | Demo Video & Presentation | 28 – 30 | Scheduled | Record side-by-side demo: Operator egocentric view + MuJoCo sim + OpenArms arm motion |

---

## 3-Minute Hackathon Pitch Script

1. **The Hook (30s):**
   > "Teleoperating dual-arm robotic systems traditionally requires cumbersome VR controllers, expensive motion capture suits, or rigid master-slave pantographs. We present **RoboTokens**: an intuitive, direct egocentric teleoperation and bimanual planning pipeline powered by DaiDai Lab's wearable vision, JSI's RobotBlockSet (RBS), and MuJoCo."

2. **The Innovation (60s):**
   > "Using only a wearable egocentric camera feed, our pipeline tracks the operator's 3D hand poses and gestures at 30Hz, maps them through an adaptive kinematic transformer, and uses RBS's OMPL motion planner to compute safe, collision-free joint trajectories in real time. We dynamically prevent the dual arms from colliding with each other or the table."

3. **The Live Demo (60s):**
   > "Watch as the human reaches forward: the bimanual OpenArms robot tracks every subtle gesture in MuJoCo with sub-centimeter accuracy, smooth minimum-jerk interpolation, and instant grasping upon index-thumb pinch."

4. **The Impact (30s):**
   > "Low-cost hardware, zero heavy infrastructure, open-source robotics stack."
