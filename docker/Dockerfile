# RoboTokens real-robot deployment image: OpenArm v1 bimanual over SocketCAN,
# UVC camera, MuJoCo (official OpenArm model for IK / dry runs), GPU for policies.
#
# The repo is bind-mounted at /workspace at runtime; the Python environment
# lives at /opt/venv inside the image so no .venv is written to the host.
FROM ubuntu:24.04

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION=3.11
# Object vision (GroundingDINO + SAM2 + torch): opt-in, large. `ROBOTOKENS_VISION=1 docker/dev.sh build`.
ARG VISION=0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive

# - build-essential/cmake/libcli11-dev: build the official openarm_can Python binding
#   (uv git dep; its CMake tree also configures the CLI, which needs CLI11).
# - can-utils/iproute2: candump/cansend and `ip link` for SocketCAN bring-up.
# - v4l-utils: `v4l2-ctl --list-devices` for the UVC camera.
# - GL/EGL/OSMesa + X11 libs: MuJoCo viewer / offscreen rendering, OpenCV highgui.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git curl sudo gnupg software-properties-common \
        build-essential cmake pkg-config libcli11-dev \
        can-utils iproute2 v4l-utils \
        libglvnd0 libgl1 libegl1 libgles2 libosmesa6 libglib2.0-0 \
        libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 \
        libxkbcommon0 libxkbcommon-x11-0 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

# Official OpenArm CAN tools (openarm-can-cli, openarm-can-configure-socketcan,
# zero-position calibration) from the Enactic PPA. The Python binding itself is
# built from the pinned openarm_can tag by uv (see pyproject.toml, extra "hardware").
RUN add-apt-repository -y ppa:openarm/main && \
    apt-get update && apt-get install -y --no-install-recommends openarm-can-utils && \
    rm -rf /var/lib/apt/lists/*

# Let the NVIDIA runtime expose the host driver's EGL implementation.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

# Run as the host UID/GID so files created in the bind mount stay owned by the
# host user (see docker/detect_user.sh). Ubuntu 24.04 ships a default 'ubuntu'
# user at 1000:1000, which would collide with the common host UID.
RUN if [ "${UID}" != "0" ]; then \
        (userdel -r ubuntu 2>/dev/null || true) && \
        (getent group ${GID} >/dev/null || groupadd -g ${GID} ${USERNAME}) && \
        useradd -m -u ${UID} -g ${GID} -G video,dialout -s /bin/bash ${USERNAME} && \
        echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
    fi && \
    mkdir -p /opt/venv /opt/python /workspace && \
    chown -R ${UID}:${GID} /opt/venv /opt/python /workspace

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONPATH=/workspace \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

USER ${UID}:${GID}
WORKDIR /workspace

# Install dependencies only (not the project), so this layer is cached until
# pyproject.toml / uv.lock change. The project source comes from the bind mount.
COPY --chown=${UID}:${GID} pyproject.toml uv.lock requirements-vision.txt /tmp/deps/
RUN cd /tmp/deps && uv sync --frozen --no-install-project --extra hardware --extra dev --extra calib && rm -rf /tmp/deps/.venv

# Optional vision stack. torch first (CUDA wheels from TORCH_INDEX_URL), then GroundingDINO and SAM2
# built against it (--no-build-isolation). No CUDA toolkit in the image: both build their PyTorch
# paths only (src/vision/object_localizer.py routes GroundingDINO's deformable attention through
# its PyTorch fallback). Installed with `uv pip`, so do not run `uv sync` in the container afterwards
# (it would remove them); run scripts with `python`.
RUN if [ "${VISION}" = "1" ]; then \
        cd /tmp/deps && \
        uv pip install --python /opt/venv/bin/python torch torchvision --index-url ${TORCH_INDEX_URL} && \
        uv pip install --python /opt/venv/bin/python setuptools wheel && \
        SAM2_BUILD_CUDA=0 uv pip install --python /opt/venv/bin/python --no-build-isolation -r requirements-vision.txt && \
        (uv pip uninstall --python /opt/venv/bin/python opencv-python opencv-python-headless || true) && \
        uv pip install --python /opt/venv/bin/python --reinstall "opencv-contrib-python>=4.9.0" && \
        /opt/venv/bin/python -c "import cv2, torch, groundingdino, sam2; assert hasattr(cv2.aruco, 'CharucoDetector'); print('vision stack OK, torch', torch.__version__)"; \
    fi && rm -rf /tmp/deps

CMD ["bash"]
