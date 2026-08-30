"""Controller gains and gripper speed: turning a speed factor into hardware.

Moved out of ``examples/17_replay_dataset.py``. :class:`ReplayScaler` is the piece
that makes speedup physically realisable -- running a trajectory faster is not just a
matter of shorter deadlines, the arm must also track harder, so stiffness is scaled
with the square of the speed factor (and damping linearly). See
``docs/variable_impedance_design.md``.

Gripper speed is scaled on the same path, which is why the constants for the speed
controller live here too.
"""

import logging
import subprocess
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node as RclpyNode
from rclpy.qos import qos_profile_system_default
from std_msgs.msg import Float64MultiArray

logger = logging.getLogger(__name__)

# reset_crisp_kp.py (SetParameters) and gripper_speed_test.py (speed
# controller spawn on /gripper/controller_manager).
KP_TASK_KEYS = [
    "task.k_pos_x", "task.k_pos_y", "task.k_pos_z",
    "task.k_rot_x", "task.k_rot_y", "task.k_rot_z",
]
# Damping companion params. Auto-damping convention: a base value <= 0 means
# the C++ controller computes d = 2*sqrt(k) every cycle (cartesian_controller
# .cpp:441). When auto, we never push a scaled d — the formula scales it for
# us via the new k = k_base * s_eff**2. See docs/variable_impedance_design.md.
KD_TASK_KEYS = [
    "task.d_pos_x", "task.d_pos_y", "task.d_pos_z",
    "task.d_rot_x", "task.d_rot_y", "task.d_rot_z",
]
GRIPPER_MAX_SPEED_MPS = 0.150          # robotiq_driver kGripperMaxSpeed
SPEED_CTL_NAME = "gripper_speed_controller"
SPEED_CTL_TYPE = "forward_command_controller/ForwardCommandController"
SPEED_INTERFACE = "set_gripper_max_velocity"
SPEED_CMDS_TOPIC = f"/gripper/{SPEED_CTL_NAME}/commands"
KNUCKLE_JOINT = "arm_0_gripper_robotiq_85_left_knuckle_joint"

# --gripper-base-speed CLI default; defined here so the speed-queue arithmetic
# and the CLI share a single source of truth.
DEFAULT_GRIPPER_SPEED = 0.0375


# ---------------------------------------------------------------------------
# Kp + gripper-speed scaling (active when --scale-kp is set)
# ---------------------------------------------------------------------------

def _get_params_batch(
    helper: RclpyNode, target_node: str, names: list[str], timeout: float = 10.0,
) -> list[float | None] | None:
    """Call <target_node>/get_parameters via rclpy service (one round-trip).

    timeout is generous (10 s) because Phase 2b fires immediately after
    Phase 2 activates the controller, and FastDDS over Discovery Server +
    WiFi can take several seconds to publish the controller's parameter
    service endpoints. 10 s absorbs that without slowing the happy path
    (returns as soon as the service is reachable).
    """
    client = helper.create_client(GetParameters, f"{target_node}/get_parameters")
    try:
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(GetParameters.Request(names=list(names)))
        rclpy.spin_until_future_complete(helper, future, timeout_sec=timeout)
        res = future.result()
        if res is None:
            return None
        # PARAMETER_NOT_SET has type == 0.
        return [pv.double_value if pv.type != 0 else None for pv in res.values]
    finally:
        helper.destroy_client(client)


def _set_params_batch(
    helper: RclpyNode,
    target_node: str,
    named_values: list[tuple[str, float]],
    timeout: float = 10.0,
) -> list[tuple[str, str]]:
    """Call <target_node>/set_parameters via rclpy service (one round-trip).

    Returns list of (name, reason) failures; empty on success.

    timeout matches _get_params_batch (10 s) for the same discovery-slack
    reason. Only used for cold-path operations (apply/restore); the hot
    path goes through ReplayScaler's cached fire-and-forget client, not
    this function — so a slow timeout here can't stall the trajectory.
    """
    params = []
    for name, val in named_values:
        p = Parameter()
        p.name = name
        p.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE,
            double_value=float(val),
        )
        params.append(p)
    client = helper.create_client(SetParameters, f"{target_node}/set_parameters")
    try:
        if not client.wait_for_service(timeout_sec=timeout):
            return [(p.name, "set_parameters service unavailable") for p in params]
        future = client.call_async(SetParameters.Request(parameters=params))
        rclpy.spin_until_future_complete(helper, future, timeout_sec=timeout)
        res = future.result()
        if res is None:
            return [(p.name, "set_parameters timed out") for p in params]
        return [
            (p.name, r.reason)
            for p, r in zip(params, res.results)
            if not r.successful
        ]
    finally:
        helper.destroy_client(client)


def _list_controllers(cm: str) -> dict | None:
    # 15s is enough budget for first-call DDS discovery through
    # ROS_DISCOVERY_SERVER. Every `ros2` CLI invocation is a fresh
    # participant that has to introduce itself to the discovery server
    # — no state inherited from previous subprocesses (e.g. an earlier
    # init_gripper_speed.py warmup). On a quiet network 5 s was usually
    # fine; with many participants advertising (cameras, mocap, the
    # robot stack) cold discovery routinely takes 10-15 s.
    try:
        r = subprocess.run(
            ["ros2", "control", "list_controllers", "-c", cm],
            capture_output=True, text=True, timeout=15.0,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "list_controllers -c %s timed out (15s). Controller manager may "
            "not be discoverable — check `ros2 node list | grep %s`.",
            cm, cm.split("/")[1] if "/" in cm else cm,
        )
        return None
    if r.returncode != 0:
        return None
    out: dict = {}
    for line in r.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            out[parts[0]] = parts[-1]
    return out


def _spawn_gripper_speed_controller(cm: str) -> tuple[bool, str]:
    """Idempotently spawn gripper_speed_controller. Mirrors gripper_speed_test.py.

    All subprocess calls have timeouts so a missing/slow controller_manager
    fails this fast (returning (False, reason)) instead of hanging the
    replay. ReplayScaler treats a False return as "skip gripper speed
    scaling, keep kp/kd active" — the rest of replay continues normally.
    """
    ctrls = _list_controllers(cm) or {}
    state = ctrls.get(SPEED_CTL_NAME)
    if state == "active":
        return True, "already active"

    # Loaded + configured but never activated (e.g. previous replay hit the
    # 25 s activate timeout and bailed). All the type-declare / load /
    # param-set / configure steps below are redundant — and the configure
    # step would actually fail with
    #   "cannot put X in 'inactive' state from its current state inactive"
    # which the `"already"` substring check downstream doesn't catch. Skip
    # straight to activation.
    if state == "inactive":
        try:
            r = subprocess.run(
                ["ros2", "control", "set_controller_state", SPEED_CTL_NAME,
                 "active", "-c", cm],
                capture_output=True, text=True, timeout=25.0,
            )
        except subprocess.TimeoutExpired:
            return False, "set_controller_state active timed out (25s)"
        if r.returncode != 0:
            return False, f"activate (from inactive): {(r.stderr or r.stdout).strip()}"
        return True, "activated from inactive"

    # Each subprocess pays its own cold-discovery cost; 15 s matches the
    # bumped _list_controllers timeout above. If we got this far, either
    # _list_controllers saw nothing for SPEED_CTL_NAME (controller not
    # loaded, normal first-run path) or _list_controllers itself timed out
    # (slow discovery). Either way, give the param set/get calls room to
    # settle DDS before assuming the controller_manager is unreachable.
    try:
        subprocess.run(
            ["ros2", "param", "set", cm, f"{SPEED_CTL_NAME}.type", SPEED_CTL_TYPE],
            capture_output=True, text=True, timeout=15.0,
        )
        chk = subprocess.run(
            ["ros2", "param", "get", cm, f"{SPEED_CTL_NAME}.type"],
            capture_output=True, text=True, timeout=15.0,
        )
    except subprocess.TimeoutExpired:
        return False, f"ros2 param set/get on {cm} timed out (15s)"
    if SPEED_CTL_TYPE not in chk.stdout:
        return False, f"could not declare {SPEED_CTL_NAME}.type on {cm}"

    if state is None:
        try:
            r = subprocess.run(
                ["ros2", "control", "load_controller", SPEED_CTL_NAME, "-c", cm],
                capture_output=True, text=True, timeout=10.0,
            )
        except subprocess.TimeoutExpired:
            return False, f"load_controller on {cm} timed out (10s)"
        if r.returncode != 0:
            return False, f"load_controller: {(r.stderr or r.stdout).strip()}"

    ns = cm[:-len("/controller_manager")] if cm.endswith("/controller_manager") else ""
    ctl_node = f"{ns}/{SPEED_CTL_NAME}"
    time.sleep(0.5)
    for pname, pvalue in [
        ("joints", f"[{KNUCKLE_JOINT}]"),
        ("interface_name", SPEED_INTERFACE),
    ]:
        try:
            r = subprocess.run(
                ["ros2", "param", "set", ctl_node, pname, pvalue],
                capture_output=True, text=True, timeout=15.0,
            )
        except subprocess.TimeoutExpired:
            return False, f"set {ctl_node} {pname} timed out (15s)"
        if r.returncode != 0:
            return False, f"set {ctl_node} {pname}: {(r.stderr or r.stdout).strip()}"

    # Cold-start `inactive` (= on_configure) can take 10+ s through the ROS
    # discovery server when many topics are advertising. Subsequent activate
    # is fast. Generous 25 s budget covers cold first spawn; on warm runs
    # the early "already active" check above short-circuits this loop.
    for target_state in ("inactive", "active"):
        try:
            r = subprocess.run(
                ["ros2", "control", "set_controller_state", SPEED_CTL_NAME,
                 target_state, "-c", cm],
                capture_output=True, text=True, timeout=25.0,
            )
        except subprocess.TimeoutExpired:
            return False, f"set_controller_state {target_state} timed out (25s)"
        if r.returncode != 0 and "already" not in (r.stderr + r.stdout).lower():
            return False, f"{target_state}: {(r.stderr or r.stdout).strip()}"

    return True, "spawned"


class ReplayScaler:
    """Scale CRISP cartesian kp + kd and gripper speed from a per-frame s_eff.

    Implements the xVLA-aligned formula documented in
    docs/variable_impedance_design.md:

        k_pos/k_rot  =  k_base * s_eff**kp_exp     (default kp_exp=2.0)
        d_pos/d_rot  =  d_base * s_eff**kd_exp     (default kd_exp=1.0)
        gripper.spd  =  base_gripper_speed * s_eff (linear, clamped)

    Auto-damping sentinel: a base d_* <= 0 means the C++ controller computes
    d = 2*sqrt(k) every cycle (cartesian_controller.cpp:441). For axes whose
    base d is auto, we never push a scaled d — the new k = k_base * s_eff**2
    feeds the controller's own sqrt and yields d_eff = 2*sqrt(k_base)*|s_eff|
    for free, exactly the linear-in-s_eff damping we want.

    apply():       cache current kp + kd, mark auto-damping axes, spawn the
                   gripper speed controller, prime the gripper speed, push
                   the initial s_eff (so frame 0 doesn't pay RPC latency
                   inside the rate-limited loop).
    step_to(s):    if s differs from the currently-applied factor, push new
                   kp + (non-auto) kd via one batched SetParameters call and
                   republish the scaled gripper speed.
    restore():     put kp + kd back to cached originals; publish base gripper
                   speed. Safe to call even if apply() partially failed —
                   restores only what was cached.
    """

    def __init__(
        self,
        env,
        s_eff: np.ndarray,
        base_gripper_speed: float,
        controller_node: str,
        gripper_cm: str,
        kp_warn_threshold: float,
        *,
        kp_exp: float = 2.0,
        kd_exp: float = 1.0,
        gripper_stride: int = 1,
    ):
        self.env = env
        self.s_eff = np.asarray(s_eff, dtype=np.float64)
        if self.s_eff.ndim != 1 or self.s_eff.size == 0:
            raise ValueError(
                f"s_eff must be a non-empty 1-D array; got shape {self.s_eff.shape}"
            )
        self.base_gripper_speed = base_gripper_speed
        self.controller_node = controller_node
        self.gripper_cm = gripper_cm
        self.kp_warn_threshold = kp_warn_threshold
        self.kp_exp = float(kp_exp)
        self.kd_exp = float(kd_exp)
        # xVLA's lerobot_eval.py:440 — `gripper.speed *= effective_speed *
        # action_stride`. We bake the stride factor in at init; step_to then
        # publishes base_gripper_speed * s_eff * gripper_stride per segment.
        self.gripper_stride = max(1, int(gripper_stride))
        self._original_kp: dict[str, float] = {}
        self._original_kd: dict[str, float] = {}
        # _kd_is_auto[name] == True  -> base d <= 0 -> let the C++ controller
        # auto-track k via 2*sqrt(k); never push a scaled d for this axis.
        self._kd_is_auto: dict[str, bool] = {}
        self._speed_pub = None
        self._applied = False
        self._current_factor: float = 1.0  # baseline (unscaled)
        self._gripper_clamp_warned: bool = False
        self._segment_count: int = 0
        # Dedicated helper node for Get/SetParameters. Not attached to any
        # executor, so `rclpy.spin_until_future_complete(helper, future)` is
        # safe even while env.robot.node is being spun by crisp_py's own
        # background executor.
        self._helper: RclpyNode | None = None
        # Long-lived SetParameters client. Created once in apply() (after
        # the helper exists + service discovery is done) and reused for
        # every step_to. This is the key to making the in-loop call
        # fire-and-forget: we never re-discover, never re-create, never
        # destroy mid-run — just call_async and return. Destroyed in
        # _destroy_helper alongside the helper node.
        self._set_params_client = None

    def _ensure_helper(self) -> RclpyNode:
        if self._helper is None:
            self._helper = RclpyNode("replay_scaler_helper")
        return self._helper

    def _destroy_helper(self) -> None:
        if self._set_params_client is not None and self._helper is not None:
            try:
                self._helper.destroy_client(self._set_params_client)
            except Exception:
                logger.exception("failed to destroy cached SetParameters client")
            self._set_params_client = None
        if self._helper is not None:
            try:
                self._helper.destroy_node()
            except Exception:
                logger.exception("failed to destroy scaler helper node")
            self._helper = None

    def apply(self) -> None:
        if self._applied:
            return
        self._applied = True

        peak_s = float(self.s_eff.max())
        floor_s = float(self.s_eff.min())
        peak_kp_factor = peak_s ** self.kp_exp
        if peak_kp_factor > self.kp_warn_threshold:
            logger.warning(
                "Kp peak factor %.2f (s_eff_peak=%.2f ** %.1f) exceeds "
                "--kp-scale-warn=%.2f. Controller torques will be %.1fx larger "
                "on the same error — risk of joint-limit repulsion kicking in "
                "or commanded torque clamping. See variable_impedance_design.md "
                "(YAML caps stiffness at 5000; with k_base ~500, s_eff ~3 is "
                "the safe ceiling). Consider a lower --max-speed.",
                peak_kp_factor, peak_s, self.kp_exp,
                self.kp_warn_threshold, peak_kp_factor,
            )

        helper = self._ensure_helper()

        # 1. Cache original kp + kd values. One batched GetParameters covers
        # both lists (12 params, one round-trip).
        all_keys = KP_TASK_KEYS + KD_TASK_KEYS
        originals = _get_params_batch(helper, self.controller_node, all_keys)
        if originals is None:
            logger.warning(
                "scale-kp: %s/get_parameters unavailable; kp/kd will NOT be "
                "scaled this run.", self.controller_node,
            )
        else:
            kp_originals = originals[:len(KP_TASK_KEYS)]
            kd_originals = originals[len(KP_TASK_KEYS):]
            for name, orig in zip(KP_TASK_KEYS, kp_originals):
                if orig is None:
                    logger.warning(
                        "scale-kp: %s not set on %s; skipping",
                        name, self.controller_node,
                    )
                    continue
                self._original_kp[name] = orig
            auto_axes = []
            for name, orig in zip(KD_TASK_KEYS, kd_originals):
                if orig is None:
                    # Parameter not declared at all — treat as auto so we
                    # never try to push a value for it.
                    self._kd_is_auto[name] = True
                    continue
                self._original_kd[name] = orig
                is_auto = orig <= 0.0
                self._kd_is_auto[name] = is_auto
                if is_auto:
                    auto_axes.append(name.rsplit(".", 1)[-1])
            if self._original_kp:
                logger.info(
                    "scale-kp: cached %d kp + %d kd originals on %s "
                    "(s_eff peak=%.3f → kp peak factor %.3f, floor=%.3f; "
                    "kp_exp=%.2f, kd_exp=%.2f).",
                    len(self._original_kp), len(self._original_kd),
                    self.controller_node, peak_s, peak_kp_factor, floor_s,
                    self.kp_exp, self.kd_exp,
                )
                if auto_axes:
                    logger.info(
                        "scale-kp: kd auto-damping on %d axis(es): %s — "
                        "C++ controller will auto-track 2*sqrt(k) every "
                        "cycle; no explicit kd RPC for these.",
                        len(auto_axes), ", ".join(auto_axes),
                    )

        # Pre-create + warm the long-lived SetParameters client. wait_for_service
        # pays the DDS discovery cost ONCE, here, so step_to's call_async in the
        # hot path is genuinely instant. Without this, the first in-loop call
        # would block the sender thread for ~hundreds of ms to several seconds
        # while discovery + first-call routing settles — exactly the lag we
        # observed in the prior run (248 underruns / 3 s set_parameters timeout
        # at the first segment boundary).
        if self._original_kp or self._original_kd:
            self._set_params_client = helper.create_client(
                SetParameters, f"{self.controller_node}/set_parameters",
            )
            if not self._set_params_client.wait_for_service(timeout_sec=5.0):
                logger.warning(
                    "scale-kp: %s/set_parameters not discovered in 5 s; "
                    "fire-and-forget calls may be dropped until the service "
                    "comes up. (kp scaling still attempted.)",
                    self.controller_node,
                )

        # 2. Spawn gripper_speed_controller and create the speed publisher.
        # TEMP_DISABLE_GRIPPER_SPEED: gripper_speed_controller adjustment is
        # currently disabled. _speed_pub stays None (initialized at line 783),
        # so the downstream `if self._speed_pub is not None` guards in
        # step_to() and restore() automatically skip all gripper-speed
        # publishes. To re-enable, change `if False:` back to `if True:`
        # (or remove the gate).
        if False:
            ok, msg = _spawn_gripper_speed_controller(self.gripper_cm)
            if not ok:
                logger.warning(
                    "gripper speed controller spawn failed: %s. Gripper speed "
                    "will NOT be scaled (kp scaling still active).", msg,
                )
            else:
                logger.info("gripper_speed_controller: %s", msg)
                self._speed_pub = self.env.robot.node.create_publisher(
                    Float64MultiArray, SPEED_CMDS_TOPIC, qos_profile_system_default,
                )
                # Prime with the baseline so the controller is hot before the
                # replay loop starts adjusting it. POST_SET_DELAY mirrors
                # gripper_speed_test.py.
                prime = Float64MultiArray()
                prime.data = [float(self.base_gripper_speed)]
                self._speed_pub.publish(prime)
                time.sleep(0.3)

        # 3. Apply the first-frame factor before replay starts so frame 0
        # doesn't pay the RPC latency inside the rate-limited loop.
        self.step_to(float(self.s_eff[0]))

    def step_to(self, s: float) -> None:
        """Push kp + (non-auto) kd + gripper speed for an effective speed `s`.

        Idempotent: returns immediately if `s` already matches the currently
        applied factor. The sender thread calls this only at integer-cycle
        segment boundaries, so one batched SetParameters round-trip per
        segment is the natural rate.
        """
        if not self._applied:
            return
        s = float(s)
        if abs(s - self._current_factor) < 1e-9:
            return

        # Push scaled kp + (non-auto) kd via a SINGLE fire-and-forget
        # SetParameters call. No spin_until_future_complete — we never wait
        # for the controller's response. The cached client (set up in apply())
        # has already done service discovery, so call_async returns
        # immediately and the sender thread is back to publishing within
        # microseconds. If the controller's set_parameters callback is
        # slow / loaded, the change just applies a bit later; if a single
        # request gets dropped, the next segment transition supersedes it
        # with new values anyway. Replay correctness does not depend on
        # individual parameter writes succeeding — only on the AVERAGE
        # tracking being close to the requested schedule.
        if self._set_params_client is not None and (
            self._original_kp or any(
                (not auto) for auto in self._kd_is_auto.values()
            )
        ):
            params: list[Parameter] = []
            kp_scale = s ** self.kp_exp
            kd_scale = s ** self.kd_exp
            for name, orig in self._original_kp.items():
                p = Parameter()
                p.name = name
                p.value = ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(orig * kp_scale),
                )
                params.append(p)
            for name, orig in self._original_kd.items():
                if self._kd_is_auto.get(name, True):
                    continue
                p = Parameter()
                p.name = name
                p.value = ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(orig * kd_scale),
                )
                params.append(p)
            if params:
                self._set_params_client.call_async(
                    SetParameters.Request(parameters=params)
                )

        # Push scaled gripper speed (no POST_SET_DELAY needed — controller
        # was primed in apply()). Formula matches xVLA lerobot_eval.py:440 —
        # gripper.speed = base * s_eff * action_stride.
        if self._speed_pub is not None:
            scaled = self.base_gripper_speed * s * self.gripper_stride
            clamped = min(scaled, GRIPPER_MAX_SPEED_MPS)
            if scaled > GRIPPER_MAX_SPEED_MPS and not self._gripper_clamp_warned:
                logger.warning(
                    "gripper speed %.4f m/s (base %.4f * s_eff %.2f * stride "
                    "%d) exceeds driver max %.3f m/s; clamped at s_eff=%.3f "
                    "(further increases will not affect the gripper).",
                    scaled, self.base_gripper_speed, s, self.gripper_stride,
                    GRIPPER_MAX_SPEED_MPS, s,
                )
                self._gripper_clamp_warned = True
            msg = Float64MultiArray()
            msg.data = [float(clamped)]
            self._speed_pub.publish(msg)

        self._segment_count += 1
        logger.debug(
            "scale-kp: s_eff=%.3f → kp×%.3f kd×%.3f (segment #%d)",
            s, s ** self.kp_exp, s ** self.kd_exp, self._segment_count,
        )
        self._current_factor = s

    def restore(self) -> None:
        # Restore gripper baseline speed (if we ever published anything).
        if self._speed_pub is not None:
            try:
                msg = Float64MultiArray()
                msg.data = [float(self.base_gripper_speed)]
                self._speed_pub.publish(msg)
                logger.info(
                    "gripper speed restored to base %.4f m/s",
                    self.base_gripper_speed,
                )
            except Exception:
                logger.exception("failed to restore gripper speed")

        # Restore kp + kd in a single batched SetParameters call. For kd
        # axes that were originally auto (<=0) we push the original value
        # back (-1.0 by convention) so the controller resumes auto-tracking.
        #
        # CRITICAL: if rclpy is already shut down (e.g. main got a SIGINT
        # and rclpy's signal handler invalidated the context before this
        # finally block ran), creating a new client errors out with
        # "rcl node's context is invalid" — and the cached kp/kd values
        # NEVER get pushed back. The controller is left at the last
        # step_to() values. After the next launch you'd be tuning against
        # those inflated gains. Skip the RPC + log a loud warning so the
        # operator knows to run `ros2 run tum09_custom reset_crisp_kp.py`
        # manually.
        if (self._original_kp or self._original_kd) and not rclpy.ok():
            logger.error(
                "scale-kp: rclpy context already shut down (likely from "
                "SIGINT) — CANNOT restore kp/kd. Controller is still at the "
                "last applied gains (kp ≈ baseline × %.2f). Run "
                "`ros2 run tum09_custom reset_crisp_kp.py` in another "
                "terminal to restore yaml defaults before the next replay.",
                self._current_factor ** self.kp_exp,
            )
            self._original_kp.clear()
            self._original_kd.clear()
            self._kd_is_auto.clear()
        elif self._original_kp or self._original_kd:
            helper = self._ensure_helper()
            named_values: list[tuple[str, float]] = []
            named_values.extend(self._original_kp.items())
            named_values.extend(self._original_kd.items())
            try:
                failures = _set_params_batch(
                    helper, self.controller_node, named_values,
                )
                if failures:
                    for name, reason in failures:
                        logger.warning(
                            "scale-kp: failed to restore %s: %s", name, reason,
                        )
                else:
                    logger.info(
                        "scale-kp: restored %d kp + %d kd params to originals",
                        len(self._original_kp), len(self._original_kd),
                    )
            except Exception:
                logger.exception("scale-kp: exception during restore")
            self._original_kp.clear()
            self._original_kd.clear()
            self._kd_is_auto.clear()

        if self._segment_count > 0:
            logger.info(
                "scale-kp: %d gain transitions applied across replay.",
                self._segment_count,
            )

        self._destroy_helper()
