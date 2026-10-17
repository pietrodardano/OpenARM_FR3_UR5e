# Cross-Embodiment Visualization

The comparison scene renders three CAD-backed systems:

- OpenArm v1 bimanual system: $7+7$ arm degrees of freedom.
- Mirrored UR5e pair: $6+6$ arm degrees of freedom.
- Mirrored Franka FR3 pair: $7+7$ arm degrees of freedom.

UR5e and FR3 meshes and kinematic models come from MuJoCo Menagerie. The scene
is visual-only: no comparison robot has an execution bridge, collision plan, or
hardware safety claim.

## Layout

Each comparison pair is mounted on its own $60\,\mathrm{mm}$ square aluminum
torso. As on OpenArm, each arm base sits flush on a torso side face
($\pm 30\,\mathrm{mm}$) at the OpenArm shoulder height of $1.330\,\mathrm{m}$,
rotated $\mp 90^\circ$ about $X$ so the first joint axis points sideways and the
arms face $+x$ toward the table. In the frontal view, the UR5e station is on
OpenArm's left and the FR3 station is on OpenArm's right. Each arm starts from
an IK-solved home pose (in `HOME_QPOS`) that puts the TCP about $0.40\,\mathrm{m}$
ahead of the torso at $z=0.95\,\mathrm{m}$ with the tool axis along $+x$.

Both UR5e and FR3 carry the same Robotiq 2F-85 gripper (MuJoCo Menagerie
`robotiq_2f85`) on their flange `attachment_site`, so the end effector is
identical across embodiments. The `pinch` site of each gripper
(`gripper_<arm>_pinch`) is the TCP. Grippers are open in the `home` keyframe and
their actuators are included (`ctrl` $0$ = open, $255$ = closed).

## Transfer Rule

Do not transfer joint trajectories between embodiments. A demonstrated OpenArm
TCP trajectory is expressed in canonical task-frame coordinates and each target
robot computes its own feasible configuration:

$$T_{TCP}^{robot}(t) = {^\mathrm{base}T_\mathrm{task}}^{robot} \; T_{TCP}^{task}(t)$$

The cyan spheres in the render scene are canonical left/right target TCP
markers. Their positions are illustrative until task-frame calibration is
implemented.

## Generate and Render

```text
uv run python scripts/setup_openarm_v1_scene.py
uv run python scripts/setup_cross_embodiment_scene.py
uv run python scripts/view_cross_embodiment_scene.py
```

The viewer script uses MuJoCo's passive viewer, including the macOS `mjpython`
launcher requirement. This avoids the unreliable module-based viewer launch on
that platform.

This scene demonstrates the cross-embodiment layout only. It does not validate
OpenArm motion transfer to UR5e or FR3 hardware.