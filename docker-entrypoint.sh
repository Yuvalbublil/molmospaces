#!/bin/bash
set -e

# Start virtual framebuffer
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99
until xdpyinfo -display :99 >/dev/null 2>&1; do sleep 0.1; done

# Start VNC server on the virtual display (port 5900)
x11vnc -display :99 -nopw -listen 0.0.0.0 -xkb -forever -shared &

# Start noVNC web proxy (port 6080 → VNC 5900)
websockify --web /usr/share/novnc 6080 localhost:5900 &

echo "noVNC viewer available at http://localhost:6080/vnc.html"

exec "$@"
