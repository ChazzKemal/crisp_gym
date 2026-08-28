#!/usr/bin/env bash

# =============================================================================
# ROS2 Robot Network Environment Setup for crisp_gym (jazzy-lerobot)
# =============================================================================
#
# Mirrors clearpath_remote_ws/tools/setup_robot_env.sh. Without these vars the
# FastDDS participants spawned by crisp_gym cannot reach the robot's Discovery
# Server, so subscribers never receive /current_pose or /joint_states and the
# Robot.wait_until_ready() call times out.
#
# Values come from clearpath_remote_ws/src/tum09_ridgeback/tum09_bringup/
# config/robot.yaml and the robot network layout:
#
#   Ethernet: 192.168.131.1:11811
#   WiFi:     192.168.50.100:11811
#
# FASTRTPS_DEFAULT_PROFILES_FILE + RMW_FASTRTPS_USE_QOS_FROM_XML are needed for
# stable Orbbec camera FPS during recording (large socket buffers, async
# publish).
# =============================================================================

_log() {
    echo "[crisp_gym setup_robot_env] $*" > /dev/tty 2>/dev/null \
        || echo "[crisp_gym setup_robot_env] $*" >&2
}

_log "Starting..."

export ROS_DOMAIN_ID="0"
export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
export ROS_AUTOMATIC_DISCOVERY_RANGE="SUBNET"
export ROS_SUPER_CLIENT="True"
unset ROS_STATIC_PEERS

# Orbbec FastDDS tuning (fixes low FPS for high-bandwidth images)
if [ -n "${PIXI_PROJECT_ROOT:-}" ] && [ -f "$PIXI_PROJECT_ROOT/scripts/fastdds_orbbec.xml" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$PIXI_PROJECT_ROOT/scripts/fastdds_orbbec.xml"
    export RMW_FASTRTPS_USE_QOS_FROM_XML=1
else
    _log "WARNING: fastdds_orbbec.xml not found under \$PIXI_PROJECT_ROOT/scripts/ — camera FPS tuning disabled."
fi

# Warn if socket buffers are too small for Orbbec image streams.
if [ "$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)" -lt 1048576 ]; then
    _log "WARNING: net.core.rmem_max is small — Orbbec FPS may suffer. Fix once with:"
    _log "  sudo sysctl -w net.core.rmem_max=2147483647"
    _log "  sudo sysctl -w net.core.wmem_max=2147483647"
fi

# Auto-detect which network we're on and point to the matching Discovery Server.
if ip -4 addr show 2>/dev/null | grep -q '192\.168\.131\.'; then
    export ROS_DISCOVERY_SERVER="192.168.131.1:11811;"
    _log "Ethernet detected — ROS_DISCOVERY_SERVER=$ROS_DISCOVERY_SERVER"
elif ip -4 addr show 2>/dev/null | grep -q '192\.168\.50\.'; then
    export ROS_DISCOVERY_SERVER="192.168.50.100:11811;"
    _log "WiFi detected — ROS_DISCOVERY_SERVER=$ROS_DISCOVERY_SERVER"
    if ! ip route show 2>/dev/null | grep -q '192\.168\.131\.0/24'; then
        _log "WARNING: On WiFi but no route to 192.168.131.0/24. Fix once with:"
        _log "  nmcli connection modify 'ASUS-ROUTER' ipv4.routes '192.168.131.0/24 192.168.50.100'"
        _log "  nmcli connection up 'ASUS-ROUTER'"
    fi
else
    _log "WARNING: Neither Ethernet (192.168.131.x) nor WiFi (192.168.50.x) detected."
    _log "ROS_DISCOVERY_SERVER not set — robot topics will not be visible."
fi

_log "Done."
