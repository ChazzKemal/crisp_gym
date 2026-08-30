"""Two monkey-patches the deploy path cannot run without.

Both work around behaviour in crisp_py that is correct for teleop but wrong when a
producer owns the command stream:

* the env yamls set ``publish_target_pose: false`` so crisp_py's own 20 Hz timer does
  not fight the sender, which also removes the publisher the sender needs -- so it is
  re-created here and handed back;
* crisp_py's gripper subscribes to its own command topic, and the loopback callback
  overwrites ``_target`` with the normalized value. See
  ``docs/gripper_self_subscription_bug.md``.

Applied by every deploy and replay entry point immediately after ``make_env``.
"""

from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_system_default


# ---------------------------------------------------------------------------
# Monkey-patch: re-enable /target_pose publishing
# ---------------------------------------------------------------------------

def enable_target_pose_publishing(env) -> None:
    """Make the env's Robot client publish to /target_pose.

    ur10e_ridgeback_env.yaml sets publish_target_pose=false (for mocap
    recording). For replay, the env's Robot must own the topic. This
    destroys the external subscription, creates the publisher + 20 Hz
    timer that Robot.__init__ would have created with the flag set to
    true. See docs/ridgeback_target_pose_ownership.md.
    """
    robot = env.robot

    sub = getattr(robot, "_target_pose_subscriber", None)
    if sub is not None:
        robot.node.destroy_subscription(sub)
        robot._target_pose_subscriber = None

    if getattr(robot, "_target_pose_publisher", None) is None:
        robot._target_pose_publisher = robot.node.create_publisher(
            PoseStamped,
            robot.config.target_pose_topic,
            qos_profile_system_default,
        )

    robot.node.create_timer(
        1.0 / robot.config.publish_frequency,
        robot._callback_publish_target_pose,
        ReentrantCallbackGroup(),
    )

    robot.config.publish_target_pose = True


def fix_gripper_self_subscription(env) -> None:
    """Prevent the Gripper's self-subscription from corrupting _target.

    crisp_py bug: Gripper.__init__ subscribes to the same topic it
    publishes on (target_state_topic). set_target(target) stores
    _target = _unnormalize(target) (correct raw value), then publishes
    the NORMALIZED value on the topic. The self-subscription callback
    (_callback_target_state) stores msg.data directly into _target —
    overwriting the correct raw value with the normalized value. The
    50 Hz publish timer then reads the corrupted _target and sends the
    gripper in the OPPOSITE direction.

    Fix: no-op the publisher so set_target() never triggers the
    loopback. _target is still set correctly by set_target()'s direct
    assignment. The Gripper's 50 Hz _callback_publish_target (which
    sends GripperCommand goals) still works — it reads _target, not
    the topic.
    """
    gripper = getattr(env, "gripper", None)
    if gripper is None:
        return
    gripper._target_state_publisher.publish = lambda msg: None
