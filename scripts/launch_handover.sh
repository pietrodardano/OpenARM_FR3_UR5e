#!/usr/bin/env bash
# Launch the RoboTokens demo on the real robot: weigh both cans, hand the heavy one to a person.
# Stops at the first failing step. The arm still asks for confirmation before it moves.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== 1/6 Offline tests"
uv run pytest

echo "== 2/6 CAN bus"
sudo scripts/can_up.sh

echo "== 3/6 Arm check (must print 'OK: ready to stream')"
uv run robotokens check

echo "== 4/6 Place the two cans upright, ~14 cm apart, roughly x 0.20-0.33 m, y 0.05-0.32 m, away from the bins."
read -r -p "Press Enter when the cans are in place (Ctrl+C to abort)... " _
uv run robotokens detect

echo "== 5/6 Dry run of the path (no grasp)"
uv run robotokens run --dry-path

echo "== 6/6 Demo: weigh both cans, hand the heavy one to a person"
echo "The arm asks for confirmation before it moves. Ctrl+C at any time puts a held can back and parks the arm."
uv run robotokens run
