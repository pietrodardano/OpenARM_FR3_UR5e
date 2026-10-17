"""MJCF prop library for the OpenArm manipulation sandbox.

Every ``add_*`` function appends bodies / geoms to an ``ET`` worldbody (and
materials to the ``asset`` element) and returns nothing. Names are prefixed with
the ``name`` argument so a scene can hold many instances.

Props:

=============  ==================================================================
dispenser      16x16x25 cm jug, box spigot, spring-return push-down lever (hinge joint)
cups           hollow tumbler / tall cup / mug, built from wall segments
blocks         wood / rubber / metal / foam presets (density + friction)
lidbox         box with a hinged lid
deformable     ``cloth`` and ``soft_sack`` (MuJoCo flexcomp)
=============  ==================================================================
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np

Vec3 = Sequence[float]


def _fmt(v: object) -> str:
    if isinstance(v, (tuple, list, np.ndarray)):
        return " ".join(f"{float(x):.6g}" for x in v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _sub(parent: ET.Element, tag: str, **attrs: object) -> ET.Element:
    return ET.SubElement(parent, tag, {k: _fmt(v) for k, v in attrs.items() if v is not None})


def _material(asset: ET.Element, name: str, rgba: Sequence[float], reflectance: float = 0.1) -> str:
    if asset.find(f"./material[@name='{name}']") is None:
        _sub(asset, "material", name=name, rgba=rgba, reflectance=reflectance)
    return name


# --------------------------------------------------------------------------- table


def add_table(
    worldbody: ET.Element,
    asset: ET.Element,
    top_z: float,
    center_x: float = 0.55,
    length: float = 0.80,
    width: float = 1.00,
    thickness: float = 0.04,
    name: str = "sandbox_table",
) -> None:
    """Tabletop with its upper surface at ``top_z`` plus four legs to the floor."""
    mat = _material(asset, f"{name}_mat", (0.55, 0.40, 0.26, 1.0), 0.08)
    _sub(worldbody, "geom", name=f"{name}_top", type="box", pos=(center_x, 0, top_z - thickness / 2),
         size=(length / 2, width / 2, thickness / 2), material=mat, friction=(0.9, 0.01, 0.001))
    leg_h = top_z - thickness
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        _sub(worldbody, "geom", name=f"{name}_leg{i}", type="box",
             pos=(center_x + sx * (length / 2 - 0.04), sy * (width / 2 - 0.04), leg_h / 2),
             size=(0.02, 0.02, leg_h / 2), material=mat)


# --------------------------------------------------------------------------- hollow rigid: cups


FLOOR_T = 0.010  # thick base: keeps the cup bottom-heavy and its contact with the table robust


def _wall_ring(body: ET.Element, prefix: str, radius: float, height: float, thickness: float,
               n_seg: int, z0: float, mass: float, rgba: Sequence[float], friction: float = 0.8,
               mat: Optional[str] = None) -> None:
    """Closed floor disc + ``n_seg`` box wall segments (a convex-friendly hollow cylinder)."""
    floor_t = FLOOR_T
    # Half the mass sits in the base, so cups stay upright when bumped.
    per_geom = 0.5 * mass / n_seg
    _sub(body, "geom", name=f"{prefix}_floor", type="cylinder", pos=(0, 0, z0 + floor_t / 2),
         size=(radius, floor_t / 2), mass=0.5 * mass, rgba=rgba if mat is None else None, material=mat,
         friction=(friction, 0.005, 0.0001))
    half_chord = (radius * math.tan(math.pi / n_seg)) * 1.03
    for k in range(n_seg):
        th = 2 * math.pi * k / n_seg
        r_mid = radius - thickness / 2
        _sub(body, "geom", name=f"{prefix}_wall{k}", type="box",
             pos=(r_mid * math.cos(th), r_mid * math.sin(th), z0 + height / 2),
             euler=(0, 0, th), size=(thickness / 2, half_chord, height / 2), mass=per_geom,
             rgba=rgba if mat is None else None, material=mat, friction=(friction, 0.005, 0.0001))


CUP_KINDS: Dict[str, Dict[str, float]] = {
    # radius (inner+wall), height, wall thickness, mass [kg]. Gripper span is ~8 cm: keep r <= 0.035.
    "tumbler": dict(radius=0.032, height=0.10, thickness=0.005, mass=0.15),
    "tall": dict(radius=0.028, height=0.14, thickness=0.005, mass=0.16),
    "mug": dict(radius=0.035, height=0.09, thickness=0.006, mass=0.30),
}


def add_cup(worldbody: ET.Element, asset: ET.Element, name: str, pos: Vec3, kind: str = "tumbler",
            rgba: Sequence[float] = (0.9, 0.9, 0.95, 1.0)) -> None:
    """Free-floating hollow cup standing on ``pos`` (the centre of its base)."""
    p = CUP_KINDS[kind]
    body = _sub(worldbody, "body", name=name, pos=pos)
    _sub(body, "freejoint", name=f"{name}_free")
    _wall_ring(body, name, p["radius"], p["height"], p["thickness"], 14, 0.0, p["mass"], rgba)
    if kind == "mug":  # a three-segment D handle on +x
        r = p["radius"]
        for i, (c, s) in enumerate((((r + 0.012, 0, 0.065), (0.012, 0.004, 0.004)),
                                    ((r + 0.022, 0, 0.045), (0.004, 0.004, 0.016)),
                                    ((r + 0.012, 0, 0.025), (0.012, 0.004, 0.004)))):
            _sub(body, "geom", name=f"{name}_handle{i}", type="box", pos=c, size=s, mass=0.01, rgba=rgba)
    _sub(body, "site", name=f"{name}_rim", pos=(0, 0, p["height"]), size=0.004, rgba=(1, 0, 0, 0.6), group=4)


# --------------------------------------------------------------------------- jar dispenser


def add_jar_dispenser(worldbody: ET.Element, asset: ET.Element, name: str, pos: Vec3,
                      facing: float = math.pi) -> Dict[str, Vec3]:
    """Jug dispenser: 16 x 16 x 25 cm body, a box spigot, and a spring-return push-down lever.

    Fixed to the table at ``pos`` (centre of the jug's footprint). ``facing`` is the yaw of
    the spigot (default pi: it points at the robot, -x).

    Spigot: sticks out 5.5 cm from the jug face, 3.5 cm wide, 4 cm tall, low on the jug so a
    10 cm tumbler fits under its nozzle (site ``<name>_nozzle``, its xy is returned).

    Lever (hinge joint ``<name>_lever``): 5.5 cm long, pivots on top of the spigot at the jug
    face and points towards the robot, rising 15 deg above horizontal. Pressing the tip down
    rotates it from 0 to +0.26 rad (lever flat on the spigot); a spring lifts it back to
    15 deg when released. Treat ``qpos(<name>_lever) > 0.15`` as "dispensing". The tip is
    site ``<name>_lever_tip``.
    """
    half, height = 0.08, 0.25
    sp_len, sp_half_w, sp_half_h = 0.055, 0.0175, 0.02
    sp_zc = 0.15  # spigot centre height above the table (bottom at 13 cm)
    lever_len, lever_ang = 0.055, math.radians(15.0)
    lever_half_t, lever_half_w = 0.003, 0.009

    body_mat = _material(asset, f"{name}_body_mat", (0.75, 0.88, 1.0, 0.45), 0.3)
    chrome = _material(asset, f"{name}_chrome", (0.8, 0.8, 0.82, 1.0), 0.5)
    lever_mat = _material(asset, f"{name}_lever_mat", (0.85, 0.15, 0.12, 1.0), 0.2)

    body = _sub(worldbody, "body", name=name, pos=pos, euler=(0, 0, facing))
    _sub(body, "geom", name=f"{name}_jug", type="box", pos=(0, 0, height / 2), size=(half, half, height / 2),
         material=body_mat)
    _sub(body, "geom", name=f"{name}_spigot", type="box", pos=(half + sp_len / 2, 0, sp_zc),
         size=(sp_len / 2, sp_half_w, sp_half_h), material=chrome)
    nozzle_x = half + sp_len - 0.012
    _sub(body, "geom", name=f"{name}_nozzle_geom", type="cylinder",
         fromto=(nozzle_x, 0, sp_zc - sp_half_h, nozzle_x, 0, sp_zc - sp_half_h - 0.01), size=0.007, material=chrome)
    _sub(body, "site", name=f"{name}_nozzle", pos=(nozzle_x, 0, sp_zc - sp_half_h - 0.01), size=0.005,
         rgba=(0, 0.6, 1, 0.6), group=4)

    # Pivot on the spigot's top face at the jug; the lever's underside just clears the spigot when flat.
    pivot_z = sp_zc + sp_half_h + lever_half_t + 0.001
    lever = _sub(body, "body", name=f"{name}_lever_body", pos=(half + 0.004, 0, pivot_z), euler=(0, -lever_ang, 0))
    _sub(lever, "joint", name=f"{name}_lever", type="hinge", axis=(0, 1, 0), range=(0.0, lever_ang),
         stiffness=0.8, springref=0.0, damping=0.01, armature=0.0002)
    _sub(lever, "geom", name=f"{name}_lever_arm", type="box", pos=(lever_len / 2, 0, 0),
         size=(lever_len / 2, lever_half_w, lever_half_t), mass=0.02, material=lever_mat, friction=(1.0, 0.005, 0.0001))
    _sub(lever, "site", name=f"{name}_lever_tip", pos=(lever_len, 0, 0), size=0.005, rgba=(0, 1, 0, 0.6), group=4)

    c, s_ = math.cos(facing), math.sin(facing)
    return dict(nozzle_xy=(pos[0] + c * nozzle_x, pos[1] + s_ * nozzle_x))


# --------------------------------------------------------------------------- rigid blocks


BLOCK_PRESETS: Dict[str, Dict[str, object]] = {
    "wood": dict(density=600, friction=0.5, rgba=(0.78, 0.6, 0.38, 1)),
    "rubber": dict(density=1100, friction=1.6, rgba=(0.15, 0.15, 0.15, 1)),
    "metal": dict(density=7800, friction=0.35, rgba=(0.7, 0.72, 0.76, 1)),
    "foam": dict(density=60, friction=1.0, rgba=(0.95, 0.85, 0.2, 1)),
}


def add_block(worldbody: ET.Element, name: str, pos: Vec3, preset: str = "wood",
              size: Vec3 = (0.025, 0.025, 0.025)) -> None:
    """Free box (half-extents ``size``) with a material preset (density / friction)."""
    p = BLOCK_PRESETS[preset]
    body = _sub(worldbody, "body", name=name, pos=pos)
    _sub(body, "freejoint", name=f"{name}_free")
    _sub(body, "geom", name=f"{name}_geom", type="box", size=size, density=p["density"], rgba=p["rgba"],
         friction=(p["friction"], 0.005, 0.0001))


def add_lidded_box(worldbody: ET.Element, asset: ET.Element, name: str, pos: Vec3) -> None:
    """Fixed 10 x 8 x 6 cm box with a hinged lid (hinge joint with friction + a small spring)."""
    mat = _material(asset, f"{name}_mat", (0.3, 0.55, 0.35, 1.0), 0.05)
    base = _sub(worldbody, "body", name=name, pos=pos)
    _sub(base, "geom", name=f"{name}_base", type="box", pos=(0, 0, 0.03), size=(0.05, 0.04, 0.03), material=mat)
    lid = _sub(base, "body", name=f"{name}_lid", pos=(-0.05, 0, 0.06))
    _sub(lid, "joint", name=f"{name}_hinge", type="hinge", axis=(0, 1, 0), range=(-2.0, 0.0), damping=0.05,
         frictionloss=0.01, stiffness=0.0)
    _sub(lid, "geom", name=f"{name}_lid_geom", type="box", pos=(0.05, 0, 0.005), size=(0.052, 0.042, 0.005), mass=0.05, material=mat)
    _sub(lid, "site", name=f"{name}_lid_knob", pos=(0.1, 0, 0.012), size=0.005, rgba=(0, 1, 0, 0.6), group=4)


# --------------------------------------------------------------------------- deformables


def add_cloth(worldbody: ET.Element, name: str, pos: Vec3, n: int = 10, spacing: float = 0.02) -> None:
    """Square cloth (flexcomp grid, ``n`` x ``n`` nodes) lying on the table."""
    fc = _sub(worldbody, "flexcomp", name=name, type="grid", count=(n, n, 1), spacing=(spacing, spacing, spacing),
              pos=pos, radius=0.004, rgba=(0.8, 0.2, 0.25, 1), dim=2, mass=0.04)
    _sub(fc, "contact", condim=3, solref=(0.01, 1), solimp=(0.95, 0.99, 0.0001), selfcollide="none", friction=0.8)
    _sub(fc, "edge", equality="true", damping=0.02)


def add_soft_sack(worldbody: ET.Element, name: str, pos: Vec3, n: int = 5, spacing: float = 0.022) -> None:
    """Squishy volumetric blob (flexcomp, 3-D, soft edge constraints) - a stand-in for a bag of filling."""
    fc = _sub(worldbody, "flexcomp", name=name, type="grid", count=(n, n, n), spacing=(spacing, spacing, spacing),
              pos=pos, radius=0.004, rgba=(0.35, 0.65, 0.85, 1), dim=3, mass=0.15)
    _sub(fc, "contact", condim=3, solref=(0.01, 1), solimp=(0.95, 0.99, 0.0001), selfcollide="none", friction=0.9)
    # Edge-length equality constraints (soft) keep it elastic. <elasticity> would force
    # integrator="discrete", which the OpenArm model's implicitfast setting rules out.
    _sub(fc, "edge", equality="true", damping=0.03, solref=(0.03, 1))


def camera_xyaxes(position: Vec3, target: Vec3) -> Tuple[float, ...]:
    """``xyaxes`` for a MuJoCo camera at ``position`` looking at ``target`` (z up)."""
    f = np.asarray(target, float) - np.asarray(position, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return tuple(x) + tuple(y)
