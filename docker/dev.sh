#!/usr/bin/env bash
# Thin wrapper around docker compose for the RoboTokens deployment container.
#
#   docker/dev.sh build          build the image (re-run after changing deps)
#   docker/dev.sh shell          interactive shell in the container
#   docker/dev.sh run <cmd...>   run a one-off command, e.g. `docker/dev.sh run pytest`
#   docker/dev.sh lock           regenerate uv.lock after editing pyproject.toml
#   docker/dev.sh down           stop and remove the container
set -euo pipefail

cd "$(dirname "$0")"
./detect_user.sh
source .env
compose=(docker compose -f compose.yaml)

if [[ "${DOCKER_ROOTLESS}" == "1" && "${1:-shell}" =~ ^(shell|run)$ ]]; then
  echo "[dev.sh] WARNING: rootless Docker detected. The container will NOT see the host's" >&2
  echo "[dev.sh]          SocketCAN interfaces (can0/can1). Fine for sim/tests; use rootful" >&2
  echo "[dev.sh]          Docker on the robot PC. See docs/real_robot_deployment.md." >&2
fi

case "${1:-shell}" in
  build)
    "${compose[@]}" build
    ;;
  shell)
    # Allow the container to open windows on the host X server, if there is one.
    if [[ -n "${DISPLAY:-}" ]] && command -v xhost >/dev/null; then
      xhost +SI:localuser:"$(id -un)" >/dev/null
    fi
    "${compose[@]}" run --rm dev bash
    ;;
  run)
    shift
    "${compose[@]}" run --rm -T dev "$@"
    ;;
  lock)
    # Resolve with uv's own image (non-slim: has git for the openarm_can git dep).
    docker run --rm -u "${HOST_UID}:${HOST_GID}" --tmpfs /tmp:mode=1777 -e UV_CACHE_DIR=/tmp/uv-cache \
      -v "$(cd .. && pwd)":/w -w /w ghcr.io/astral-sh/uv:0.9.5-python3.11-bookworm uv lock
    ;;
  down)
    "${compose[@]}" down
    ;;
  *)
    sed -n '2,9p' "$0"
    exit 1
    ;;
esac
