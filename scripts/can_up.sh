#!/usr/bin/env bash
# Brings up the OpenArm CAN-FD interfaces (1 Mbit/s nominal, 5 Mbit/s data).
#
#   scripts/can_up.sh            # can0 (right arm) and can1 (left arm)
#   scripts/can_up.sh can0       # a single interface
#
# Run on the host (uses sudo when not root).
# Equivalent to the official `openarm-can-configure-socketcan <if> -fd`.
set -euo pipefail

ifaces=("$@")
[[ ${#ifaces[@]} -eq 0 ]] && ifaces=(can0 can1)
sudo=""
[[ $(id -u) -ne 0 ]] && sudo="sudo"

for ifc in "${ifaces[@]}"; do
  if ! ip link show "$ifc" &>/dev/null; then
    echo "[can_up] $ifc not found. Available:"; ip -br link | grep -E '^can' || echo "  (no CAN interfaces)"
    exit 1
  fi
  $sudo ip link set "$ifc" down
  # restart-ms 0: a bus-off stays visible (the driver faults) instead of silently recovering.
  $sudo ip link set "$ifc" type can bitrate 1000000 dbitrate 5000000 fd on restart-ms 0
  # Short TX queue: a deep queue would deliver stale commands after a hiccup.
  $sudo ip link set "$ifc" txqueuelen 10
  $sudo ip link set "$ifc" up
  echo "[can_up] $ifc up: $(ip -details link show "$ifc" | grep -oE 'bitrate [0-9]+|dbitrate [0-9]+' | tr '\n' ' ')"
done
