# Troubleshooting: "No controllers are currently loaded!"

## Symptom

```bash
ros2 control list_controllers
# → No controllers are currently loaded!
```

All topics still appear (e.g. `/r100_0207/...`), but no arm controllers
exist — the arm is unresponsive to ROS2 commands.

## Fix that worked (2026-04-16)

From an SSH session on the robot (`ssh robot@192.168.131.1`):

```bash
sudo systemctl restart clearpath-platform.service
```

Then **press the physical E-stop reset button** on the back of the
Ridgeback base (the robot should go from flashing red to solid
red/white).

Then from `clearpath_remote_ws` on the dev laptop:

```bash
pixi run python src/tum09_ridgeback/tum09_custom/scripts/startup_robot.py
```

After `startup_robot.py` completes, verify:

```bash
ros2 control list_controllers
# → should list joint_state_broadcaster, joint_trajectory_controller,
#   safety_observer_controller, etc.
```

## Why this sequence

1. **`restart clearpath-platform`** — restarts the base nodes (CAN bus,
   IMU, odometry). If the platform service was in a bad state (e.g.
   after a connectivity drop), the discovery server may not be routing
   properly to the arm service.

2. **E-stop reset button** — clears any latched hardware fault on the
   Ridgeback. The arm service (`tum09-arm`) won't fully activate
   controllers if the platform reports an active safety stop.

3. **`startup_robot.py`** — powers on the UR10e via the Dashboard
   Server, restarts `tum09-arm.service`, starts the External Control
   URCap, homes the arm, and sets the workspace box. This is the step
   that actually loads the controllers.
