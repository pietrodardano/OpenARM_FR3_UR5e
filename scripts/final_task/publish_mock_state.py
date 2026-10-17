#!/usr/bin/env python3
"""Publish the scripted real-OpenArm state stream over UDP (a reference for the real side's publisher).

Run this in one terminal and `run_cross_embodiment_demo.py --source udp:5600` in another. The real
side only has to send the same JSON lines (see src/cross_embodiment/stream.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--replay", metavar="FILE", help="play a recorded JSON-lines file instead of the scripted stream")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    args = parser.parse_args()

    import mujoco

    from src.cross_embodiment.layout import SCENE_PATH
    from src.cross_embodiment.mock_openarm import build_mock_script
    from src.cross_embodiment.stream import ReplaySource, UdpStateSender

    source = ReplaySource.from_file(args.replay) if args.replay else build_mock_script(mujoco.MjModel.from_xml_path(str(SCENE_PATH)))
    sender = UdpStateSender(args.host, args.port)
    start = time.perf_counter()
    print(f"publishing {len(source.messages)} messages ({source.duration_s:.0f} s) to {args.host}:{args.port}")
    try:
        for message in source.messages:
            delay = message.t / args.speed - (time.perf_counter() - start)
            if delay > 0:
                time.sleep(delay)
            sender.send(message)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
    print("done")


if __name__ == "__main__":
    main()
