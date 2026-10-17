#!/usr/bin/env bash
# Writes docker/.env with the UID/GID the container should run as, so files it
# creates in the bind-mounted repo are owned by the host user.
#
# Rootless Docker maps container root (0) to the host user, and any other
# container UID to an unprivileged sub-UID that cannot write the repo, so the
# container runs as root there. Rootful Docker gets the host UID/GID instead.
set -euo pipefail

cd "$(dirname "$0")"
if docker info -f '{{.SecurityOptions}}' 2>/dev/null | grep -q rootless; then
  uid=0 gid=0 rootless=1
else
  uid="$(id -u)" gid="$(id -g)" rootless=0
fi
printf 'HOST_UID=%s\nHOST_GID=%s\nDOCKER_ROOTLESS=%s\n' "$uid" "$gid" "$rootless" > .env
