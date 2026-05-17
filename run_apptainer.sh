#!/bin/bash
# Apptainer equivalent of docker-compose.yml.
# Translates both services to apptainer run/exec commands.
set -euo pipefail

SIF="${SIF:-molmospaces.sif}"
MOLMO_RESOURCES="${MOLMO_RESOURCES:-$HOME/.cache/molmo-spaces-resources}"
MOLMO_ASSETS="${MOLMO_ASSETS:-$HOME/.cache/molmospaces}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-$HOME/.cache/huggingface}"

BIND_ARGS=(
    --bind "${MOLMO_RESOURCES}:/root/.cache/molmo-spaces-resources"
    --bind "${MOLMO_ASSETS}:/root/.cache/molmospaces"
    --bind "${MODEL_WEIGHTS}:/root/.cache/huggingface"
)

ENV_ARGS=(
    --env MUJOCO_GL=egl
    --env NVIDIA_VISIBLE_DEVICES=all
    --env NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
)

usage() {
    cat <<EOF
Usage: $0 <command> [extra args...]

Commands:
  pull               Pull the SIF image from Docker Hub
  simulation         Run headless data generation (EGL GPU rendering, no display)
  simulation-viewer  Run with noVNC viewer at http://localhost:6080/vnc.html
  shell              Open an interactive shell in the container

Any extra args after the command are forwarded to the pipeline script.

Environment variables:
  SIF              SIF image path (default: molmospaces.sif)
  MOLMO_RESOURCES  Host path for molmo-spaces-resources cache (default: ~/.cache/molmo-spaces-resources)
  MOLMO_ASSETS     Host path for molmospaces assets cache     (default: ~/.cache/molmospaces)
  MODEL_WEIGHTS    Host path for HuggingFace model weights    (default: ~/.cache/huggingface)

Example:
  $0 pull
  $0 simulation --seed 42
  $0 simulation-viewer --seed 5
EOF
    exit 1
}

ensure_dirs() {
    mkdir -p "$MOLMO_RESOURCES" "$MOLMO_ASSETS" "$MODEL_WEIGHTS"
}

case "${1:-}" in
    pull)
        apptainer pull "$SIF" docker://yuvalbublil/molmospaces:1.0
        ;;

    simulation)
        ensure_dirs
        # Use exec (no entrypoint) — EGL rendering doesn't need Xvfb.
        # --writable-tmpfs allows ephemeral writes inside the container image.
        apptainer exec \
            --nv \
            --writable-tmpfs \
            "${BIND_ARGS[@]}" \
            "${ENV_ARGS[@]}" \
            "$SIF" \
            /app/.venv/bin/python /app/scripts/datagen/run_pipeline.py "${@:2}"
        ;;

    simulation-viewer)
        ensure_dirs

        # Kill any orphaned X/VNC processes and remove stale lock from a previous run.
        pkill -u "$USER" -f "Xvfb :99"      2>/dev/null || true
        pkill -u "$USER" -f "x11vnc"         2>/dev/null || true
        pkill -u "$USER" -f "websockify.*6080" 2>/dev/null || true
        rm -f /tmp/.X99-lock /tmp/.X11-unix/X99

        # On exit (normal finish, Ctrl-C, or error) clean up what the entrypoint left behind.
        cleanup() {
            pkill -u "$USER" -f "Xvfb :99"      2>/dev/null || true
            pkill -u "$USER" -f "x11vnc"         2>/dev/null || true
            pkill -u "$USER" -f "websockify.*6080" 2>/dev/null || true
            rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
        }
        trap cleanup EXIT

        echo "noVNC viewer will be available at http://localhost:6080/vnc.html"
        # Explicitly invoke the entrypoint (starts Xvfb + x11vnc + noVNC) instead
        # of using --compat, which triggers overlayfs writes that fail on NFS home dirs.
        # Apptainer uses host networking, so port 6080 is directly accessible.
        apptainer exec \
            --nv \
            --writable-tmpfs \
            "${BIND_ARGS[@]}" \
            "${ENV_ARGS[@]}" \
            "$SIF" \
            /usr/local/bin/docker-entrypoint.sh \
            vglrun -d egl /app/.venv/bin/python /app/scripts/datagen/run_pipeline.py --viewer "${@:2}"
        ;;

    shell)
        ensure_dirs
        apptainer shell \
            --nv \
            --writable-tmpfs \
            "${BIND_ARGS[@]}" \
            "${ENV_ARGS[@]}" \
            "$SIF"
        ;;

    *)
        usage
        ;;
esac
