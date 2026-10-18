"""FR6: release only on felt external contact."""

from __future__ import annotations

import numpy as np

from robotokens.sensing.contact import ContactDetector, wait_for_contact


class Clock:
    def __init__(self):
        self.t = 0.0

    def wait(self, dt):
        self.t += dt


def trace(clock, contact_at=None, delta=0.6, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([1.0, -2.0, 0.5, 3.0, 0.1, 0.2, 0.0])

    def read():
        tau = base + rng.normal(0, noise, 7)
        if contact_at is not None and clock.t >= contact_at:
            tau[3] += delta
        return tau

    return read


def test_detects_step_after_contact(cfg):
    clock = Clock()
    assert wait_for_contact(trace(clock, contact_at=3.0), clock.wait, cfg.task.contact)
    assert 3.0 <= clock.t < 3.0 + 0.5


def test_no_false_positive_under_noise_until_timeout(cfg):
    clock = Clock()
    assert not wait_for_contact(trace(clock), clock.wait, cfg.task.contact)
    assert clock.t >= cfg.task.contact.timeout_s


def test_short_spike_is_ignored(cfg):
    det = ContactDetector(cfg.task.contact)
    det.set_baseline(np.random.default_rng(1).normal(0, 0.01, (100, 7)))
    spike = np.zeros(7)
    spike[0] = 1.0
    assert not det.update(spike)       # one sample above threshold
    assert not det.update(np.zeros(7))  # streak broken
    for _ in range(int(cfg.task.contact.hold_s * cfg.task.contact.rate_hz)):
        latched = det.update(spike)
    assert latched
