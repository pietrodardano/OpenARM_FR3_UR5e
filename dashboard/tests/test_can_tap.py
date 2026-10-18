"""Offline tests for the passive CAN tap (decoder, config channels, receive-only guarantee)."""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import can_tap  # noqa: E402

ROOT = HERE.parent


def test_decode_matches_openarm_can_formula():
    # frame built by hand following dm_motor_control.cpp parse_motor_state_data for a DM4340 (tMax 28 Nm)
    data = bytes([0x13, 0x80, 0x00, 0x80, 0x08, 0x00, 40, 41])  # id 3, status ENABLED, tau_uint = 0x800
    s = can_tap.decode_state(data, "DM4340")
    assert s["status"] == 1
    assert math.isclose(s["tau"], 2048 / 4095 * 56 - 28, rel_tol=1e-9)
    assert math.isclose(s["q"], 0x8000 / 65535 * 25 - 12.5, rel_tol=1e-9)


def test_encode_decode_roundtrip_within_quantisation():
    for mt in ("DM8009", "DM4340", "DM4310"):
        for tau in (-3.2, 0.0, 0.7, 5.5):
            s = can_tap.decode_state(can_tap.encode_state(mt, 0.3, -0.2, tau), mt)
            step = 2 * can_tap.LIMITS[mt][2] / 4095
            assert abs(s["tau"] - tau) <= step
            assert abs(s["q"] - 0.3) < 1e-3


def test_fault_nibble_is_reported():
    data = can_tap.encode_state("DM4310", 0, 0, 0, status=0xE)
    assert can_tap.decode_state(data, "DM4310")["status"] == 0xE
    assert can_tap.ERROR_NAMES[0xE] == "overload"


def test_channels_follow_the_project_config():
    iface, fd, ch = can_tap.channels_from_config(ROOT / "configs/robotokens.yaml")
    assert [c.recv_id for c in ch] == [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]
    assert [c.name for c in ch][-1] == "gripper" and iface and isinstance(fd, bool)


def test_tap_never_exposes_a_send_path():
    class FakeSock:
        def recv(self, n): return b""
        def send(self, b): raise AssertionError("transmitted")
        def sendto(self, b, a=None): raise AssertionError("transmitted")
        def close(self): pass

    r = can_tap._ReceiveOnly(FakeSock())
    assert not hasattr(r, "send") and not hasattr(r, "sendto") and not hasattr(r, "sendall")
    src = (HERE / "can_tap.py").read_text()
    assert ".send(" not in src.replace("no send", "") and ".sendto(" not in src and ".write(" not in src


def test_tap_bins_and_snapshot():
    _, _, ch = can_tap.channels_from_config(ROOT / "configs/robotokens.yaml")
    tap = can_tap.Tap(ch, rate_hz=50)
    t = __import__("time").time()
    for k in range(200):
        for c in ch:
            tap.feed(c.name, 2.0, 0.0, 1, now=t + k * 0.004)
    snap = tap.snapshot()
    assert snap["names"][0] == "joint1" and len(snap["tau"]) >= 30 and abs(snap["tau"][-1][0] - 2.0) < 1e-9
    later = tap.snapshot(since=snap["seq"])
    assert later["t"] == []  # incremental
