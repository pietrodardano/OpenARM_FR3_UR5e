"""Tick loop for the cross-embodiment demo.

Every control tick: read the real-arm state (if any), move the rendered OpenArm to it, advance each
sim robot's task by one tick, then step the physics. The sim robots run their own tasks independently
(they know the true object positions), but each waits for the real task to enter a phase before
starting the same phase, so the two sides stay in step.
"""

from __future__ import annotations

from typing import Callable, Optional

from src.cross_embodiment.stream import StateSource
from src.cross_embodiment.task import PhaseGate, TaskResult, WeighTask
from src.cross_embodiment.world import DemoWorld


class DemoRunner:
    def __init__(self, world: DemoWorld, tasks: dict[str, WeighTask], source: Optional[StateSource] = None, gate: Optional[PhaseGate] = None) -> None:
        self.world, self.tasks, self.source = world, tasks, source
        self.gate = gate or PhaseGate(enabled=False)
        for task in tasks.values():
            task.gate = self.gate
        self.last_message = None

    def run(self, max_sim_s: float = 400.0, on_tick: Optional[Callable[[DemoWorld], None]] = None) -> dict[str, TaskResult]:
        dt = self.world.layout.control_dt_s
        generators = {name: task.run() for name, task in self.tasks.items()}
        results: dict[str, TaskResult] = {}
        while generators and self.world.time < max_sim_s:
            if self.source is not None:
                message = self.source.poll(self.world.time)
                self.gate.update(message)
                if message is not None:
                    self.last_message = message
                    self.world.set_openarm(message.q, message.gripper_m)
            for name in list(generators):
                try:
                    next(generators[name])
                except StopIteration as stop:
                    results[name] = stop.value
                    del generators[name]
            self.world.step(dt)
            if on_tick is not None:
                on_tick(self.world)
        for name, task in self.tasks.items():
            results.setdefault(name, task.result)  # unfinished tasks report what they have
        return results
