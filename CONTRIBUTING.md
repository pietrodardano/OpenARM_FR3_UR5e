# Contributing

Thank you for helping improve OpenARM FR3 UR5e. Contributions are especially useful when they make contact-aware, cross-embodiment manipulation more reproducible, safer, or better documented.

## Before you begin

- Read the [README](README.md), [functional and safety requirements](docs/requirements.md), and [hardware bring-up guide](docs/hardware_bringup.md).
- Use a focused issue before beginning substantial work, especially for new embodiments, controllers, or task behavior.
- State the evidence tier for every result: **offline test**, **simulation**, or **physical hardware**. Do not present simulation assumptions as measured hardware performance.

## Local development

The default development path does not need a robot, a GPU, model weights, or a CAN interface:

```bash
uv sync --no-default-groups --group dev
uv run --no-default-groups --group dev pytest
```

See [setup.sh](setup.sh) for the optional OpenArm and vision environment. Keep downloaded weights, local runs, and generated artifacts out of commits.

## Pull requests

Keep pull requests small and explain:

1. What changed and why.
2. The embodiment and evidence tier affected.
3. Tests run and their results.
4. Any documentation, configuration, or migration impact.
5. Any potential hardware or safety impact.

Add or update offline tests whenever practical. Update the README or the relevant documentation when a user-visible behavior, supported embodiment, or evidence level changes.

## Physical robot safety

Do not execute newly contributed motion, grasping, torque, or contact code on hardware without maintainer review and the staged checks in [hardware bring-up](docs/hardware_bringup.md). Keep an e-stop accessible and treat simulation success as insufficient evidence for a safe physical deployment.

## Reporting issues

Use the repository issue forms. Include environment details, the relevant robot/model, a minimal reproduction, and logs or configuration with secrets removed. For vulnerabilities, use [SECURITY.md](SECURITY.md) instead of a public issue.