"""Passive (receive-only) CAN tap for live joint torques. Standalone: touches no operative code.

SocketCAN delivers a copy of every frame to every listening socket, so a second process can watch the
motor replies without disturbing the robot's own connection. This module:

  * opens a raw CAN socket and ONLY ever calls `recv` on it (the socket object is wrapped so that no send
    method is reachable; see `_ReceiveOnly`),
  * filters in the kernel to the motors' reply IDs (`hardware.recv_can_ids` and the gripper's, read from
    configs/robotokens.yaml),
  * decodes the Damiao MIT-mode state frame exactly as openarm_can does
    (third_party/openarm/openarm_can/src/openarm/damiao_motor/dm_motor_control.cpp, parse_motor_state_data),
  * keeps a bounded ring buffer in memory for the dashboard.

It never configures the interface (no `ip link`, no `can_up.sh`) and never opens an openarm_can connection.
Linux only (SocketCAN). Run it on the robot PC, where the interface lives.

Torque shown is the motor-reported torque, the same quantity the robot code reads as `ArmState.tau`
(`get_torque()`); the gripper value is in motor sign (the robot code flips its sign for display).
"""

from __future__ import annotations

import collections
import math
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

import yaml

# (pMax rad, vMax rad/s, tMax Nm) per motor type: copied from openarm_can dm_motor_constants.hpp
LIMITS: Dict[str, tuple[float, float, float]] = {
    "DM3507": (12.5, 50, 5), "DM4310": (12.5, 30, 10), "DM4310_48V": (12.5, 50, 10), "DM4340": (12.5, 10, 28),
    "DM4340_48V": (12.5, 10, 28), "DM6006": (12.5, 45, 20), "DM8006": (12.5, 45, 40), "DM8009": (12.5, 45, 54),
    "DM10010L": (12.5, 25, 200), "DM10010": (12.5, 20, 200), "DMH3510": (12.5, 280, 1), "DMH6215": (12.5, 45, 10),
    "DMG6220": (12.5, 45, 10),
}
ERROR_NAMES = {0x0: "disabled", 0x1: "enabled", 0x8: "overvoltage", 0x9: "undervoltage", 0xA: "overcurrent",
               0xB: "MOS overheat", 0xC: "coil overheat", 0xD: "communication lost", 0xE: "overload"}
FAULT_THRESHOLD = 0x8


def _u2d(x: int, lo: float, hi: float, bits: int) -> float:
    return x / ((1 << bits) - 1) * (hi - lo) + lo


def decode_state(data: bytes, motor_type: str) -> Optional[dict]:
    """One 8-byte Damiao MIT state frame -> q (rad), dq (rad/s), tau (Nm), temps, status nibble."""
    if len(data) < 8 or motor_type not in LIMITS:
        return None
    pmax, vmax, tmax = LIMITS[motor_type]
    q = (data[1] << 8) | data[2]
    dq = (data[3] << 4) | (data[4] >> 4)
    tau = ((data[4] & 0xF) << 8) | data[5]
    return {"status": data[0] >> 4, "q": _u2d(q, -pmax, pmax, 16), "dq": _u2d(dq, -vmax, vmax, 12),
            "tau": _u2d(tau, -tmax, tmax, 12), "t_mos": data[6], "t_rotor": data[7]}


def encode_state(motor_type: str, q: float, dq: float, tau: float, status: int = 1, motor_id: int = 1) -> bytes:
    """Inverse of `decode_state` (used by the tests and the synthetic feed)."""
    pmax, vmax, tmax = LIMITS[motor_type]

    def d2u(x, lo, hi, bits):
        return max(0, min((1 << bits) - 1, int(round((x - lo) / (hi - lo) * ((1 << bits) - 1)))))

    qu, dqu, tu = d2u(q, -pmax, pmax, 16), d2u(dq, -vmax, vmax, 12), d2u(tau, -tmax, tmax, 12)
    return bytes([(status << 4) | (motor_id & 0xF), qu >> 8, qu & 0xFF, dqu >> 4, ((dqu & 0xF) << 4) | (tu >> 8),
                  tu & 0xFF, 40, 40])


@dataclass
class Channel:
    name: str
    recv_id: int
    motor_type: str
    t_max: float = field(init=False)

    def __post_init__(self) -> None:
        self.t_max = LIMITS[self.motor_type][2]


def channels_from_config(config: Path) -> tuple[str, bool, List[Channel]]:
    """Interface name, CAN-FD flag and the 8 reply channels, read from the project config (read-only)."""
    cfg = yaml.safe_load(config.read_text())
    hw, gr = cfg["hardware"], cfg["gripper"]
    ids = [int(x) for x in hw["recv_can_ids"]]
    ch = [Channel(f"joint{i + 1}", ids[i], hw["motor_types"][i]) for i in range(len(ids))]
    ch.append(Channel("gripper", int(gr["recv_can_id"]), gr["motor_type"]))
    return hw["can_interface"], bool(hw.get("can_fd", False)), ch


class _ReceiveOnly:
    """Wraps the raw socket so that nothing but `recv` is reachable: this tap cannot transmit."""

    def __init__(self, sock: socket.socket) -> None:
        self._s = sock

    def recv(self, n: int) -> bytes:
        return self._s.recv(n)

    def close(self) -> None:
        self._s.close()


# Linux CAN constants (not all exposed by Python's socket module on every version)
SOL_CAN_RAW, CAN_RAW_FILTER, CAN_RAW_RECV_OWN_MSGS, CAN_RAW_FD_FRAMES = 101, 1, 4, 5
CAN_SFF_MASK = 0x7FF


def open_receive_only(interface: str, recv_ids: Sequence[int], fd: bool) -> _ReceiveOnly:
    if not sys.platform.startswith("linux") or not hasattr(socket, "AF_CAN"):
        raise RuntimeError("the CAN tap needs Linux SocketCAN: run it on the robot PC")
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        s.setsockopt(SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, 0)
        if fd:  # the arm runs CAN-FD: without this the kernel does not deliver FD frames to this socket
            s.setsockopt(SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
        flt = b"".join(struct.pack("=II", int(i), CAN_SFF_MASK) for i in recv_ids)  # kernel-side ID filter
        s.setsockopt(SOL_CAN_RAW, CAN_RAW_FILTER, flt)
        s.bind((interface,))
        s.settimeout(0.5)
    except Exception:
        s.close()
        raise
    return _ReceiveOnly(s)


class Tap:
    """Background reader: decimates to `rate_hz` bins and keeps the last `keep_s` seconds."""

    def __init__(self, channels: List[Channel], rate_hz: float = 50.0, keep_s: float = 120.0) -> None:
        self.channels = channels
        self.names = [c.name for c in channels]
        self.dt = 1.0 / rate_hz
        n = int(keep_s * rate_hz)
        self.buf: Deque[tuple] = collections.deque(maxlen=n)  # (seq, t, [tau...], [status...])
        self.lock = threading.Lock()
        self.seq = 0
        self.frames = 0
        self.last_frame_t: Optional[float] = None
        self.last_q: Dict[str, float] = {}
        self.last_status: Dict[str, int] = {}
        self.error: Optional[str] = None
        self.source = "can"
        self._acc: Dict[str, List[float]] = {c.name: [] for c in channels}
        self._bin_t0 = time.time()
        self._stop = threading.Event()
        self._rate_t, self._rate_n, self.fps = time.time(), 0, 0.0

    # --- feeding -----------------------------------------------------------------------------
    def feed(self, name: str, tau: float, q: float, status: int, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self.frames += 1
        self._rate_n += 1
        self.last_frame_t = now
        self.last_q[name], self.last_status[name] = q, status
        self._acc[name].append(tau)
        if now - self._bin_t0 >= self.dt:
            taus = [(sum(v) / len(v)) if v else math.nan for v in self._acc.values()]
            with self.lock:
                self.seq += 1
                self.buf.append((self.seq, now, taus, [self.last_status.get(n, -1) for n in self.names]))
            self._acc = {n: [] for n in self.names}
            self._bin_t0 = now
        if now - self._rate_t >= 1.0:
            self.fps, self._rate_t, self._rate_n = self._rate_n / (now - self._rate_t), now, 0

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            rows = [r for r in self.buf if r[0] > since]
        now = time.time()
        stale = None if self.last_frame_t is None else now - self.last_frame_t
        return {"enabled": True, "source": self.source, "names": self.names, "t_max": [c.t_max for c in self.channels],
                "now": now, "seq": self.seq, "fps": round(self.fps, 1), "stale_s": stale, "error": self.error,
                "status": {n: self.last_status.get(n) for n in self.names},
                "status_text": {n: ERROR_NAMES.get(s, "unknown") for n, s in self.last_status.items()},
                "fault": any(s is not None and s >= FAULT_THRESHOLD for s in self.last_status.values()),
                "q": self.last_q,
                "t": [r[1] for r in rows], "seq_rows": [r[0] for r in rows], "tau": [r[2] for r in rows]}

    # --- running -----------------------------------------------------------------------------
    def start(self, interface: str, fd: bool) -> "Tap":
        by_id = {c.recv_id: c for c in self.channels}
        sock = open_receive_only(interface, list(by_id), fd)

        def loop() -> None:
            fd_size, classic_size = 72, 16
            while not self._stop.is_set():
                try:
                    raw = sock.recv(fd_size)
                except socket.timeout:
                    continue
                except OSError as e:
                    self.error = f"CAN socket: {e}"
                    time.sleep(0.5)
                    continue
                if len(raw) < 8:
                    continue
                can_id, length = struct.unpack_from("=IB", raw, 0)
                ch = by_id.get(can_id & CAN_SFF_MASK)
                if ch is None or length < 8:
                    continue
                s = decode_state(raw[8:8 + length], ch.motor_type)
                if s is not None:
                    self.feed(ch.name, s["tau"], s["q"], s["status"])

        threading.Thread(target=loop, name="can-tap", daemon=True).start()
        self.source = f"{interface} (receive-only)"
        return self

    def stop(self) -> None:
        self._stop.set()


def synthetic(channels: List[Channel]) -> Tap:
    """Clearly labelled fake feed for previewing the live panel (no CAN needed)."""
    tap = Tap(channels)
    tap.source = "SYNTHETIC (demo)"

    def loop() -> None:
        t0 = time.time()
        rng = __import__("random").Random(1)
        while True:
            t = time.time() - t0
            lifted = 6 < (t % 24) < 20
            for i, c in enumerate(channels):
                base = [-3.2, 14.0, 0.4, 5.5, 0.2, 1.1, 0.05, 0.0][i] if i < 8 else 0
                load = (1.0 if i in (1, 3) else 0.3) * (1.2 if lifted else 0) * (1 if (t % 24) < 12 else 3.2) if i < 7 else 0
                dither = 0.5 * math.sin(2 * math.pi * t / 2.0) if (10 < (t % 24) < 18 and i in (1, 3)) else 0
                grip = (1.0 if lifted else 0.0) if c.name == "gripper" else 0
                tau = base + load + dither + grip + rng.gauss(0, 0.04)
                tap.feed(c.name, tau, 0.0, 1)
            time.sleep(0.004)

    threading.Thread(target=loop, name="synthetic-tap", daemon=True).start()
    return tap
