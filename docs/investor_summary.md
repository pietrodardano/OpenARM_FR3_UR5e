# RoboTokens: Vision is not enough

*Estimating physical properties for robust robot manipulation. Project summary for investors and partners.*

## The problem: physical AI needs physics

"Physical AI" is AI that acts in the physical world, so it has to deal with **physics**: mass, stiffness, friction, fill level. These properties are not visible. They are *latent*: a robot learns them only by **interacting** with the object, by pushing, lifting or squeezing it and measuring the response. The robotics literature calls this [interactive perception](https://arxiv.org/pdf/1604.03670): interaction creates sensory signals that passive observation does not provide.

A camera gives shape, colour and position. It cannot observe mass. Two cans that look the same, one full and one empty, are the same image to a vision-only system. Today's vision-language models try to guess such properties from appearance. [Gao et al. (ICRA 2024)](https://arxiv.org/abs/2309.02561) show that off-the-shelf VLMs lack physical concepts such as material and fragility, and improve only by learning priors from how objects usually look. A prior is not a measurement.

This matters whenever the decision depends on physics:

- **Grasp and lift:** grip force and arm effort depend on mass. Too little drops the object. Too much crushes it.
- **Sorting and verification:** look-alike items differ in content or weight (full or empty, genuine or defective, one item or two).
- **Human handover:** the robot must know when a person has taken the object. Vision cannot sense that.
- **Robustness under visual ambiguity:** with glare, occlusion or look-alikes, a vision-only system guesses. A system that can also *measure* can check its guess.

## Our thesis

**Vision tells the robot what to pick. Interaction tells it what it picked.**

RoboTokens adds a physical-property estimation layer to a vision-guided arm. It lifts the object and reads the joint torques the actuators already report. That gives a mass estimate with a quantified uncertainty, with no extra hardware. The decision rule is statistical: act when the evidence is conclusive, re-measure when it is not, and abstain if it still is not.

## What we built

A working end-to-end system on a real **OpenArm v1** arm with one RGB camera.

1. **See.** Open-vocabulary detection and segmentation (GroundingDINO + SAM2) find the objects a free-text instruction names.
2. **Interact.** The arm lifts each object and estimates mass from joint torques (dithered least squares, friction cancelled over whole cycles). The result is a belief: mass ± standard error.
3. **Decide.** A test on the mass difference (`z = Δm / SE ≥ 3`) triggers action, a re-measurement, or abstention.
4. **Act.** The robot sorts by weight, or hands an object to a person and releases it only when it *feels* the person take it.
5. **Audit.** A read-only MuJoCo digital twin follows the robot, and every run writes a report with the intent, beliefs, decision and phase log.

**Evidence so far**

| Claim | Support | Level |
|---|---|---|
| The pipeline runs end to end | OpenArm hardware, first system tests, offline test suite | hardware, no benchmarks yet |
| The estimator transfers across arms | MuJoCo with an assumed, realistic error budget: full vs. empty resolved on OpenArm, UR5e and Franka FR3 in 300 of 300 trials; about 20–30 g resolution on OpenArm and FR3, about 100 g on UR5e | simulation |
| The robot abstains when evidence is weak | implemented and unit-tested | software |

## Market context

Physical AI is a fast-growing category, and published forecasts differ widely because analysts define the market differently. We treat these numbers as directional, not as our addressable market.

| Source | Scope | Figure |
|---|---|---|
| [MarketsandMarkets](https://www.marketsandmarkets.com/Market-Reports/embodied-ai-market-83867232.html) | Embodied AI | USD 4.44 B (2025) to USD 23.06 B (2030), 39 % CAGR |
| [Technavio](https://www.technavio.com/report/physical-ai-and-robotics-as-a-service-market-industry-analysis) | Physical AI and robotics-as-a-service | +USD 21.24 B over 2025 to 2030, 30.8 % CAGR |
| [Christian & Timbers](https://www.christianandtimbers.com/insights/the-investment-cycle-for-physical-ai-through-2030), citing ABI Research and Mordor Intelligence | Robotics overall | about USD 50 B (2025) to about USD 111 B (2030), and USD 73.6 B to USD 185 B |

Industry direction: NVIDIA describes physical AI as ["AI that understands the laws of physics, AI that can work among us"](https://qz.com/ai-next-wave-robots-nvidia-jensen-huang-blackwell-rubin-1851515953), and at GTC 2026 said ["every industrial company will become a robotics company"](https://finance.yahoo.com/sectors/technology/articles/every-industrial-company-become-robotics-203111417.html). Those are vendor statements, not independent evidence.

## Business cases

Each case below pairs a documented practice or market with the gap RoboTokens addresses. We did not find independent data on the size of the gap for robot arms, and we say so.

### 1. Warehouse and e-commerce picking: weight as a verification signal

- **What exists:** mis-picks carry direct costs (re-picking, return and replacement shipping) and larger indirect costs (customer churn, chargebacks, inventory distortion), as summarised by [Racklify](https://racklify.com/encyclopedia/mis-pick-the-costly-error-hiding-in-your-warehouse/). Weight checks are already used as quality control: [pick-by-weight trucks](https://logisticsmatters.co.uk/?p=15449) compare the weight added after each pick with the expected weight, [Rapyuta Robotics](https://www.rapyuta-robotics.com/2025/12/12/pa-amrs-with-weight-inspection-in-piece-picking/) describes weight inspection on picking robots, and a [patent application](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12109700) skips a pick when the measured weight falls outside stored ranges.
- **The gap:** these checks rely on added scales or load cells on the vehicle or station. A manipulator that weighs what it grasps, as part of the grasp, adds the check without new hardware.
- **Not shown:** we found no published dollar cost per robotic mis-pick, and we have no accuracy benchmark on real parcels yet.

### 2. Laboratory and pharma handling: gravimetric verification

- **What exists:** gravimetric (weight-based) verification is standard practice. [BellatRx's filling monobloc](https://www.packworld.com/home/article/21203392/pharmaceutical-liquid-filling-monobloc-with-dualweight-verification) measures tare and gross weight and subtracts one from the other to verify fill volume, at 60 vials per minute. A [pharmacy-admixture patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7930066) describes a robot that weighs a vial and compares it with the expected weight, and a [filling-machine patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9493254) uses an empty-vial tare.
- **The link to our work:** the tare-and-subtract principle is the same one our simulation uses to remove the arm's own weight.
- **The gap, stated honestly:** dedicated balances are far more accurate than arm-based estimates, so we do not position RoboTokens as a replacement. The opportunity is flexible cells and mobile or low-volume setups, where a screening check inside the normal pick-and-place cycle is useful and a balance is not available.

### 3. Service and assistive robots: handover and aging populations

- **What exists:** one [assistive-robotics market report](https://www.giiresearch.com/report/vmr2073881-global-assistive-robotics-market-size-share-trends.html) puts the segment at about USD 13.4 B in 2026, growing at 22.2 % per year to about USD 66.6 B by 2034, driven by population aging. Estimates for [broad service robotics](https://www.globenewswire.com/news-release/2019/05/24/1843065/0/en/Global-Service-Robots-Market-Outlook-and-Forecasts-2019-2024-Aging-Population-Mass-Digitization-and-Human-Robotics-Interaction-Accelerates-Growth.html) vary widely by source.
- **The gap:** a robot that hands things to a person must release at the moment of contact. Our handover releases on felt torque, not on a camera guess.
- **Not shown:** we found no market sizing specific to robot-to-human handover, and the handover has been tested only in our own trials.

### 4. Robot learning and foundation models: measured physical labels

- **What exists:** vision-language models learn physical concepts from appearance ([Gao et al.](https://arxiv.org/abs/2309.02561)). Their annotations are crowdsourced or automated, not measured.
- **The gap:** every lift RoboTokens performs produces a measured mass with an uncertainty. A fleet of arms could generate physically grounded labels at no extra sensor cost. This is a research direction, not a product we have built.

## Why we are credible

- **It runs on a real robot,** and the first system tests passed.
- **Safety is built in.** The servo loop clips and rate-limits every target, latches a fault on motor errors, lost communication or tracking error, and checks every Cartesian move for reachability and collisions before the arm moves. On any exception the arm puts down what it holds, parks and disables its motors.
- **Honest uncertainty.** Every mass is a belief with a standard error, and the system acts on the difference between objects, which stays reliable when absolute values are offset.
- **Engineering discipline.** Each requirement has an ID and a test. The suite runs offline with no robot, GPU or model weights.

## Current limits

- Initial testing only. No published accuracy or throughput benchmarks yet.
- Absolute mass from the current robot code is offset by arm-model error (gravity model, cable drag, current offsets). The relative comparison is the reliable output. An empty-gripper tare brings the error to about 11–14 g on the OpenArm in simulation. It is a proposed improvement, not yet in the robot code.
- The robot's current uncertainty estimate is overconfident: in simulation it sometimes acts on mass differences below about 20 g and gets them wrong. A floor from repeated weighings removes this, at the cost of abstaining more often. Proposed, not yet in the robot code.
- Narrow scope: two objects, one arm, a fixed weigh pose, a rule-based instruction parser.
- Cross-embodiment evidence is simulation only. Only the OpenArm has run on hardware.

## Roadmap

1. **Validate:** repeatability and accuracy trials across object types and masses, published with the data.
2. **Implement the tare** in the robot code and test it on hardware.
3. **Widen sensing:** from mass to fill level, stiffness and slip, all of them properties that interaction reveals.
4. **Scale the task:** more than two objects, ranking and thresholds, clutter.
5. **Natural interface:** replace the rule-based parser with a language model.
6. **Portability:** other arms on hardware.
7. **Pilot:** a paid pilot in one of the cases above, starting with the one where a partner already has a weight-check step to replace or complement.


---

*Repository: see [README.md](../README.md), [requirements.md](requirements.md) and [hardware_bringup.md](hardware_bringup.md).*
