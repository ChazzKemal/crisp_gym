"""The target sender: the thread that actually publishes to the robot.

Moved verbatim out of ``examples/17_replay_dataset.py``. This is the actuation
end of the deploy path -- a producer fills a bounded queue with :class:`TargetItem`
and this thread publishes each one at its own ``deadline_mono``, so pose commands
leave on a schedule the producer computed rather than whenever the producer
happened to get around to it.

It runs on its own thread for a reason: rclpy publishes release the GIL inside the
C extension, so the sender keeps its cadence while the producer is busy running
inference. The C++ variant (:mod:`crisp_gym.deploy.cpp_sender`) exists for when
even that is not enough.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

logger = logging.getLogger(__name__)


@dataclass
class TargetItem:
    """One frame's worth of work for the sender thread.

    All fields are populated once by the producer; the sender does no
    arithmetic on them other than building the ROS message.
    """
    pose_xyz: np.ndarray            # (3,) float64
    pose_quat: np.ndarray           # (4,) float64 xyzw
    grip_raw: float | None          # raw value for /target_gripper_state; None disables
    action: np.ndarray              # (action_dim,) float32 — for the replay log
    deadline_mono: float            # time.monotonic() target
    frame_idx: int
    s_eff: float                    # effective speed factor for this segment
    cycles: int                     # integer cycle count; segment-boundary key


class TargetSenderThread(threading.Thread):
    """Dedicated daemon thread that publishes targets at item.deadline_mono.

    Pops TargetItem(s) from a bounded queue. When item.cycles changes vs the
    previous item, calls scaler.step_to(item.s_eff) before publishing so kp/kd
    jumps happen at integer-cycle transitions (one batched RPC per segment).
    Termination: producer pushes None as a sentinel.
    """

    def __init__(
        self,
        q: queue.Queue,
        *,
        target_pose_pub,
        gripper_raw_pub,
        gripper_action_client,
        gripper_max_effort: float,
        pose_msg: PoseStamped,
        clock,
        scaler: "ReplayScaler | None",
        replay_log: list,
        state_capture_fn,
        debug_publish: bool,
        log_interval_s: float = 5.0,
        dry_run: bool = False,
        gripper_edge_detect: bool = True,
        gripper_edge_eps: float = 1e-4,
        gripper_latch_frames: int = 0,
    ):
        super().__init__(name="TargetSender", daemon=True)
        self._q = q
        self._target_pose_pub = target_pose_pub
        self._gripper_raw_pub = gripper_raw_pub
        self._gripper_action_client = gripper_action_client
        self._gripper_max_effort = float(gripper_max_effort)
        # Edge-detect on the gripper publish. The Robotiq action server (and
        # the crisp_py /target_gripper_state ↔ goal bridge) PREEMPT the
        # active goal on every new send_goal_async. At 20 Hz, that means a
        # constant grip_raw value would trigger 20 preemptions per second,
        # each restarting the trajectory's acceleration ramp from rest —
        # the gripper effectively crawls. Edge-detect sends a new goal ONLY
        # when grip_raw changes by more than `gripper_edge_eps`, letting
        # the driver actually complete the trajectory between commands.
        # `gripper_edge_eps=1e-4` filters float-noise from the policy chunk
        # while still catching genuine ~0.01% transitions; bigger eps adds
        # hysteresis at the open↔close midpoint.
        self._gripper_edge_detect = bool(gripper_edge_detect)
        self._gripper_edge_eps = float(gripper_edge_eps)
        # Hysteresis latch: after the gripper target CHANGES (a new goal is
        # actually sent), block any further change for the next
        # `gripper_latch_frames` popped frames — the target is held at the
        # last sent value during the window. 0 disables (legacy behaviour).
        # Defends against a policy whose gripper channel oscillates at the
        # chunk seam (open↔close every chunk): without a latch each flip
        # preempts the in-flight Robotiq goal before the fingers finish
        # travelling, so the gripper chatters but never completes a grasp.
        # Note: latch only gates CHANGES; same-value re-sends still follow the
        # edge-detect rule. Only applies to the Python sender (not --cpp-sender).
        self._gripper_latch_frames = max(0, int(gripper_latch_frames))
        self._grip_latch_remaining: int = 0
        # Tracks the last value we ACTUALLY sent (not the last value seen),
        # so a skipped publish doesn't update this. None = no publish yet.
        self._last_published_grip_raw: float | None = None
        # Diagnostics: how many gripper publishes were elided by edge-detect.
        self.gripper_dedupe_count: int = 0
        # Diagnostics: how many gripper CHANGES were suppressed by the latch.
        self.gripper_latch_blocked_count: int = 0
        self._pose_msg = pose_msg
        self._clock = clock
        self._scaler = scaler
        self._replay_log = replay_log
        self._state_capture_fn = state_capture_fn
        self._debug_publish = debug_publish
        self._log_interval_s = log_interval_s
        # When True, the sender still pops + sleeps + logs + tracks queue
        # depth, but skips every ROS call: no scaler.step_to RPC, no
        # /target_pose publish, no gripper action/Float32. Used by
        # 19_deploy_policy.py's --dry-run to rehearse the full producer +
        # queue + sender pipeline (realistic cadence and underrun stats)
        # while leaving the robot completely untouched.
        self._dry_run = bool(dry_run)
        # Diagnostics (read by main after join()).
        self.pub_dt_samples: list[float] = []
        self.queue_depth_min: int = 2 ** 31
        self.queue_depth_max: int = 0
        self.n_published: int = 0
        self.underrun_count: int = 0
        # Per-frame deadline slack (ms). Positive = we slept this long
        # before publishing; negative = we were already past the deadline
        # when popping. `n_late_frames` is the count of slack < 0 entries —
        # a clearer name for the same thing `underrun_count` tracks.
        self.slack_samples_ms: list[float] = []
        self.n_late_frames: int = 0
        # Per-frame sender stage timings (ms). All entries are wall-clock
        # durations captured via time.perf_counter; summarised by the
        # deploy script at shutdown.
        #   pop_ms             : queue.get() block time (high = producer not feeding)
        #   scaler_rpc_ms      : scaler.step_to() RPC (only at segment boundaries)
        #   sleep_overshoot_ms : (actual_sleep - requested_sleep) (high = GIL contention)
        #   pub_pose_ms        : /target_pose publish call
        #   pub_grip_ms        : gripper publish call (action client OR raw Float32)
        #   loop_total_ms      : pop -> ready-for-next-pop wall time
        self.stage_samples: dict[str, list[float]] = {
            "pop_ms": [],
            "scaler_rpc_ms": [],
            "sleep_overshoot_ms": [],
            "pub_pose_ms": [],
            "pub_grip_ms": [],
            "loop_total_ms": [],
        }
        self.frame_rows: list[dict] = []

    def run(self) -> None:
        prev_cycles: int | None = None
        last_log = time.monotonic()
        while True:
            _t_loop_start = time.perf_counter()
            _t_pop = time.perf_counter()
            item = self._q.get()
            pop_ms = (time.perf_counter() - _t_pop) * 1000.0
            if item is None:
                break

            depth = self._q.qsize()
            if depth < self.queue_depth_min:
                self.queue_depth_min = depth
            if depth > self.queue_depth_max:
                self.queue_depth_max = depth

            # Segment-boundary gain jump. Skipped under dry_run so the
            # scaler never fires its fire-and-forget SetParameters RPC at
            # the live cartesian_controller — keeps the controller's kp/kd
            # untouched during dry-run rehearsals.
            scaler_rpc_ms = 0.0
            if (
                not self._dry_run
                and self._scaler is not None
                and item.cycles != prev_cycles
            ):
                _t_scaler = time.perf_counter()
                self._scaler.step_to(item.s_eff)
                scaler_rpc_ms = (time.perf_counter() - _t_scaler) * 1000.0
                prev_cycles = item.cycles

            # Sleep until the absolute deadline. Below-zero means we're
            # behind schedule; publish anyway (preserves pre-refactor
            # behaviour at 17_replay_dataset.py:1657–1661). Note we always
            # sleep — dry_run preserves the pacing so queue dynamics match
            # a real run. Capture per-frame slack so 19_deploy_policy.py
            # can report the full distribution (median / p10 / p1 / min)
            # at shutdown instead of just the underrun count.
            slack_s = item.deadline_mono - time.monotonic()
            self.slack_samples_ms.append(slack_s * 1000.0)
            sleep_overshoot_ms = 0.0
            if slack_s > 0.0:
                _t_sleep = time.perf_counter()
                time.sleep(slack_s)
                actual_sleep_s = time.perf_counter() - _t_sleep
                # Anything beyond the requested sleep is GIL / scheduler
                # delay before the thread resumed — the canonical
                # "thread starvation" indicator.
                sleep_overshoot_ms = max(
                    0.0, (actual_sleep_s - slack_s) * 1000.0,
                )
            else:
                self.underrun_count += 1
                self.n_late_frames += 1

            # Pose + gripper publishes. In dry_run we skip both — the
            # sender still pops, still sleeps, still logs, still tracks
            # underruns / queue depth, just doesn't touch ROS. The replay
            # log row below is still written (useful for diff against a
            # real run).
            pub_pose_ms = 0.0
            pub_grip_ms = 0.0
            # Whether the gripper publish actually fired this frame. False
            # either because (a) --dry-run skipped all ROS calls, (b) the
            # item carried no gripper command, or (c) edge-detect deduped
            # it against the previous publish. Logged per-frame in
            # frame_rows so analysis can distinguish "command flowed to
            # hardware" from "command was elided/skipped".
            grip_published = False
            # True iff a gripper CHANGE was suppressed this frame by the
            # hysteresis latch (held at the last sent value).
            grip_latched = False
            if not self._dry_run:
                # Fill pose msg in place (no allocation in hot path).
                self._pose_msg.header.stamp = self._clock.now().to_msg()
                self._pose_msg.pose.position.x = float(item.pose_xyz[0])
                self._pose_msg.pose.position.y = float(item.pose_xyz[1])
                self._pose_msg.pose.position.z = float(item.pose_xyz[2])
                self._pose_msg.pose.orientation.x = float(item.pose_quat[0])
                self._pose_msg.pose.orientation.y = float(item.pose_quat[1])
                self._pose_msg.pose.orientation.z = float(item.pose_quat[2])
                self._pose_msg.pose.orientation.w = float(item.pose_quat[3])
                _t_pub = time.perf_counter()
                self._target_pose_pub.publish(self._pose_msg)
                pub_pose_ms = (time.perf_counter() - _t_pub) * 1000.0
                if self._debug_publish:
                    self.pub_dt_samples.append(pub_pose_ms / 1000.0)

                # Gripper — action-client and raw-publish branches mirror
                # the pre-refactor main loop. With edge-detect enabled
                # (default), the raw-publish (Float32) branch only fires a NEW
                # publish when the requested grip_raw moves more than
                # `gripper_edge_eps` from the last value we actually sent. This
                # stops the action server's per-publish preemption from
                # continuously restarting the gripper's acceleration ramp; see
                # __init__ comment. The action-client (direct) branch is exempt
                # — it re-sends every frame (continuous re-drive); see the elif
                # condition below.
                if item.grip_raw is not None:
                    grip_raw_f = float(item.grip_raw)
                    # A "change" = grip_raw moved more than edge_eps from the
                    # last value we actually sent (or nothing sent yet).
                    is_change = (
                        self._last_published_grip_raw is None
                        or abs(grip_raw_f - self._last_published_grip_raw)
                            > self._gripper_edge_eps
                    )
                    # Hysteresis latch: a change is blocked while the latch
                    # armed by a previous change is still counting down. The
                    # latch is measured BEFORE consuming this frame's count.
                    blocked_by_latch = is_change and self._grip_latch_remaining > 0
                    if self._grip_latch_remaining > 0:
                        self._grip_latch_remaining -= 1
                    if blocked_by_latch:
                        # Hold last sent value; do not publish this change.
                        self.gripper_latch_blocked_count += 1
                        grip_latched = True
                    elif (
                        is_change
                        or not self._gripper_edge_detect
                        or self._gripper_action_client is not None
                    ):
                        # Publish when: a permitted change, a same-value re-send
                        # with edge-detect disabled (legacy per-tick), OR we're
                        # on the action-client (direct) path — which ALWAYS
                        # re-sends every frame so the gripper goal is driven
                        # continuously, mirroring crisp_py's 30 Hz
                        # /target_gripper_state relay. Without continuous
                        # re-drive a direct-action command is single-shot on
                        # edges; a chattering policy gripper channel then
                        # preempts the in-flight Robotiq goal before the fingers
                        # finish travelling and the grasp never completes.
                        _t_grip = time.perf_counter()
                        if self._gripper_action_client is not None:
                            goal = GripperCommand.Goal()
                            goal.command.position = grip_raw_f
                            goal.command.max_effort = self._gripper_max_effort
                            self._gripper_action_client.send_goal_async(goal)
                        elif self._gripper_raw_pub is not None:
                            self._gripper_raw_pub.publish(
                                Float32(data=grip_raw_f)
                            )
                        pub_grip_ms = (time.perf_counter() - _t_grip) * 1000.0
                        self._last_published_grip_raw = grip_raw_f
                        grip_published = True
                        # Arm the latch only on a genuine change.
                        if is_change:
                            self._grip_latch_remaining = self._gripper_latch_frames
                    else:
                        # edge-detect on + no change: driver still executing it.
                        self.gripper_dedupe_count += 1
            self.n_published += 1

            loop_total_ms = (time.perf_counter() - _t_loop_start) * 1000.0
            self.stage_samples["pop_ms"].append(pop_ms)
            self.stage_samples["scaler_rpc_ms"].append(scaler_rpc_ms)
            self.stage_samples["sleep_overshoot_ms"].append(sleep_overshoot_ms)
            self.stage_samples["pub_pose_ms"].append(pub_pose_ms)
            self.stage_samples["pub_grip_ms"].append(pub_grip_ms)
            self.stage_samples["loop_total_ms"].append(loop_total_ms)
            # Per-frame action columns for offline analysis (e.g. gripper
            # open/close chatter). `item.action` is the raw policy action
            # [x, y, z, roll, pitch, yaw, grip] BEFORE the 0.5-midpoint
            # binarization; `grip_cmd` is the value actually sent to the
            # gripper (post-binarize, unnormalized).
            act = item.action
            _act_labels = (
                "act_x", "act_y", "act_z",
                "act_roll", "act_pitch", "act_yaw", "act_grip",
            )
            act_cols = {
                lbl: (float(act[i]) if act is not None and i < len(act)
                      else float("nan"))
                for i, lbl in enumerate(_act_labels)
            }
            self.frame_rows.append({
                "frame_idx": item.frame_idx,
                "queue_depth": depth,
                "slack_ms": slack_s * 1000.0,
                "pop_ms": pop_ms,
                "scaler_rpc_ms": scaler_rpc_ms,
                "sleep_overshoot_ms": sleep_overshoot_ms,
                "pub_pose_ms": pub_pose_ms,
                "pub_grip_ms": pub_grip_ms,
                "loop_total_ms": loop_total_ms,
                **act_cols,
                "grip_cmd": (float(item.grip_raw)
                             if item.grip_raw is not None else float("nan")),
                # True iff the sender actually called send_goal_async /
                # Float32 publish for this frame. False = elided by
                # edge-detect, or no gripper, or dry-run.
                "grip_published": bool(grip_published),
                # True = gripper change suppressed by hysteresis latch.
                "grip_latched": bool(grip_latched),
            })

            # Replay log row.
            row: dict = {
                "frame_index": item.frame_idx,
                "timestamp": time.time(),
                "replay.s_eff": float(item.s_eff),
                "replay.cycles": int(item.cycles),
                "replay.action": item.action.copy(),
            }
            if self._state_capture_fn is not None:
                self._state_capture_fn(row)
            self._replay_log.append(row)

            now = time.monotonic()
            if now - last_log > self._log_interval_s:
                logger.info(
                    "  frame %d  q=%d  underruns=%d",
                    item.frame_idx, depth, self.underrun_count,
                )
                last_log = now
