"""Hardware module: thin adapters to physical devices (OpenArm motors over SocketCAN).

Only :mod:`src.hardware.openarm_bus` imports the official ``openarm_can`` SDK, and
only when a real bus is opened, so this package imports fine on machines
without the SDK (macOS laptops, CI).
"""

from .openarm_bimanual import OpenArmBimanual, OpenArmHardwareConfig
from .openarm_bus import ArmBusConfig, BusReading, FakeOpenArmBus, OpenArmBus

__all__ = [
    "ArmBusConfig",
    "BusReading",
    "FakeOpenArmBus",
    "OpenArmBimanual",
    "OpenArmBus",
    "OpenArmHardwareConfig",
]
