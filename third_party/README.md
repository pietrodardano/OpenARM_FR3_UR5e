# Runtime model assets

This directory intentionally contains only the MuJoCo runtime assets required by this repository. Large mesh binaries are stored with Git LFS; install it before cloning or fetching models:

```bash
git lfs install
git lfs pull
```

## Included assets

| Source | Included content | Purpose |
|---|---|---|
| `enactic/openarm_mujoco` | OpenArm v1 bimanual MJCF and meshes | OpenArm configuration, twin, and hardware model |
| `google-deepmind/mujoco_menagerie` — `franka_fr3` | FR3 mesh assets | Cross-embodiment simulation |
| `google-deepmind/mujoco_menagerie` — `universal_robots_ur5e` | UR5e mesh assets | Cross-embodiment simulation |
| `google-deepmind/mujoco_menagerie` — `robotiq_2f85` | Gripper mesh assets | Cross-embodiment simulation |

The custom `mujoco_menagerie/cross_embodiment_openarm_ur5e_fr3.xml` is maintained in this repository. Its asset paths are repository-relative so it loads on any checkout.

## Deliberately excluded

The full upstream repositories, source packages, examples, tests, documentation, alternative robot versions, and build artifacts are excluded. They are not needed at runtime and would substantially increase clone size.

## Attribution and licenses

The corresponding upstream license texts are preserved in [LICENSES](LICENSES). The assets retain their upstream licenses; they are not relicensed by this repository.