"""State stream between the real side and the simulation (proposed contract, see docs/final_task/interfaces.md).

The real side publishes one JSON object per line over UDP (about 50 Hz)::

    {"t": 12.34, "q": [7 OpenArm left joint angles, rad], "gripper_m": 0.044, "phase": "WEIGH_1"}

``phase`` is the real task's current phase (``PHASES`` below), or null if unknown. The simulation uses
``q`` and ``gripper_m`` to move the rendered OpenArm (a follower), and ``phase`` to keep the sim robots
in step with the real task. Sources: a live UDP receiver, a recorded log, or a scripted mock.
"""

from __future__ import annotations

import json
import socket
from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Protocol

# Task phases in order. The sim robots start phase P only after the real task has entered P.
PHASES = ("INIT", "PICK_1", "LIFT_1", "WEIGH_1", "PLACE_BACK_1", "PICK_2", "LIFT_2", "WEIGH_2", "PLACE_BACK_2", "COMPARE", "ACT", "DONE")


def phase_index(phase: str) -> int:
    return PHASES.index(phase)


@dataclass
class StateMessage:
    t: float
    q: list[float]
    gripper_m: Optional[float] = None
    phase: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "StateMessage":
        raw = json.loads(text)
        phase = raw.get("phase")
        if phase is not None and phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        if len(raw["q"]) != 7:
            raise ValueError("q must hold the 7 left OpenArm joint angles")
        return cls(t=float(raw["t"]), q=[float(v) for v in raw["q"]], gripper_m=raw.get("gripper_m"), phase=phase)


class StateSource(Protocol):
    def poll(self, sim_time: float) -> Optional[StateMessage]:
        """Return the latest message available at ``sim_time`` (None if nothing has arrived yet)."""


class ReplaySource:
    """Plays a recorded list of messages by simulation time (message ``t`` is seconds from the start)."""

    def __init__(self, messages: Iterable[StateMessage]) -> None:
        self.messages = sorted(messages, key=lambda m: m.t)
        self._i = 0

    def poll(self, sim_time: float) -> Optional[StateMessage]:
        while self._i + 1 < len(self.messages) and self.messages[self._i + 1].t <= sim_time:
            self._i += 1
        return self.messages[self._i] if self.messages and self.messages[0].t <= sim_time else None

    @classmethod
    def from_file(cls, path: str) -> "ReplaySource":
        with open(path) as handle:
            return cls(StateMessage.from_json(line) for line in handle if line.strip())

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            for message in self.messages:
                handle.write(message.to_json() + "\n")

    @property
    def duration_s(self) -> float:
        return self.messages[-1].t if self.messages else 0.0


class UdpStateSender:
    """Used by the real side (and tests) to publish state messages."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5600) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)

    def send(self, message: StateMessage) -> None:
        self._sock.sendto(message.to_json().encode(), self._addr)

    def close(self) -> None:
        self._sock.close()


class UdpStateReceiver:
    """Non-blocking receiver; keeps only the newest valid message and ignores malformed ones."""

    def __init__(self, port: int = 5600, host: str = "0.0.0.0") -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self.port = self._sock.getsockname()[1]
        self.latest: Optional[StateMessage] = None
        self.dropped = 0

    def poll(self, sim_time: float) -> Optional[StateMessage]:
        while True:
            try:
                data, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                return self.latest
            try:
                self.latest = StateMessage.from_json(data.decode())
            except (ValueError, KeyError, json.JSONDecodeError):
                self.dropped += 1

    def close(self) -> None:
        self._sock.close()
