FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# MuJoCo rendering: osmesa for software, egl for GPU. Override with MUJOCO_GL at runtime.
ENV MUJOCO_GL=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    ca-certificates \
    # OpenGL / MuJoCo rendering (osmesa=CPU, egl=GPU)
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libosmesa6 \
    libegl1 \
    libglfw3 \
    libglu1-mesa \
    libglib2.0-0 \
    # X11 (needed for GLFW viewer window)
    libx11-6 \
    libxext6 \
    libxrender1 \
    xvfb \
    x11vnc \
    x11-utils \
    novnc \
    websockify \
    # HID (hidapi dep)
    libhidapi-hidraw0 \
    libxtst6 \
    libxv1 \
    libegl1-mesa \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install VirtualGL for GPU-accelerated rendering through Xvfb (viewer service)
RUN wget -q -O /tmp/virtualgl.deb \
        "https://github.com/VirtualGL/virtualgl/releases/download/3.1/virtualgl_3.1_amd64.deb" \
    && dpkg -i /tmp/virtualgl.deb \
    && rm /tmp/virtualgl.deb

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps before copying full source for better layer caching
COPY pyproject.toml README.md ./
COPY molmo_spaces/__init__.py molmo_spaces/
RUN uv venv --python 3.11 .venv && \
    uv pip install -e ".[mujoco]"

# Upgrade jaxlib to the CUDA 12 build so JAX uses the GPU
RUN uv pip install "jax[cuda12]"

# Copy full source and reinstall (picks up everything, reuses cached wheels)
COPY . .
RUN uv pip install -e ".[mujoco]" --no-deps

# Patch mujoco.cgl usage: CGLUnlockContext is macOS-only but called unconditionally.
# Guard _context_is_cgl with a platform check so Linux never tries to load the macOS CGL library.
RUN sed -i \
    -e 's/from mujoco import gl_context/import sys; from mujoco import gl_context/' \
    -e 's/self\._context_is_cgl = True/self._context_is_cgl = sys.platform == "darwin"/' \
    molmo_spaces/renderer/opengl_rendering.py

# Pre-download NLTK data so the container has no runtime network dependency
RUN .venv/bin/python -c "import nltk; nltk.download('wordnet'); nltk.download('wordnet2022')"

# Persist asset cache across runs
VOLUME /root/.cache/molmo-spaces-resources
VOLUME /root/.cache/molmospaces

ENV PYTHONPATH="/app:/app/scripts/datagen"

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD [".venv/bin/python", "scripts/datagen/run_pipeline.py", "--seed", "3"]
