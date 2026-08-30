"""Asynchronous SmolVLA policy with Real-Time Chunking (RTC).

Lives next to ``async_lerobot_policy.py`` and follows the same multiprocessing
shape: a spawn-context subprocess owns the GPU policy; the parent talks to it
over an mp.Pipe. The two differ in:

* the policy is always ``SmolVLAPolicy`` from the lerobot fork that ships
  ``lerobot.policies.smolvla`` and ``lerobot.policies.rtc``,
* the request carries a language ``task`` plus RTC state
  (``prev_chunk_left_over``, ``inference_delay`` hint, queue index snapshot),
* the response carries the raw chunk AND the post-processed chunk plus the
  measured inference latency, so the parent can call ``ActionQueue.merge``
  with the real delay.

The deploy script (``examples/30_deploy_smolvla.py``) owns the
``ActionQueue`` and the background dispatcher thread; this module only
exposes the subprocess + IPC primitives.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
from collections import deque
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import torch
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import (
    concatenate_state_features,
    numpy_obs_to_torch,
)


logger = logging.getLogger(__name__)


@register_policy("smolvla_rtc_policy")
class AsyncSmolVLAPolicy(Policy):
    """SmolVLA + RTC inference subprocess wrapper.

    The constructor signature mirrors what ``crisp_gym.policy.policy.make_policy``
    will pass after loading ``2_smolvla_rtc_config.yaml``:

      AsyncSmolVLAPolicy(pretrained_path, env, rtc=<dict>,
                         queue_refill_threshold=..., num_inference_steps=...)

    The ``env`` arg is accepted (and stored) only so the constructor signature
    matches ``AsyncLerobotPolicy`` — the inference subprocess never sees the
    live ROS env (env is not picklable for spawn).
    """

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        *,
        rtc: dict | None = None,
        queue_refill_threshold: int = 12,
        num_inference_steps: int | None = None,
        allow_empty_cameras: int | None = None,
    ):
        # Spawn (not fork) so the child can init CUDA cleanly.
        ctx = mp.get_context("spawn")
        self.parent_conn, self.child_conn = ctx.Pipe()
        self.env = env
        self.rtc_cfg = dict(rtc or {})
        self.queue_refill_threshold = int(queue_refill_threshold)
        self.num_inference_steps = num_inference_steps

        # SmolVLA always uses n_obs_steps=1, but read from the checkpoint for
        # forward-compatibility. Reading via PreTrainedConfig is cheap
        # (config.json only — no torch weights, no GPU).
        from lerobot.configs.policies import PreTrainedConfig

        peeked_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
        if peeked_cfg is None:
            raise ValueError(
                f"Policy configuration is missing in the pretrained path: "
                f"{pretrained_path}."
            )
        self.n_obs = int(getattr(peeked_cfg, "n_obs_steps", 1))
        self.n_act = int(getattr(peeked_cfg, "n_action_steps", 50))
        # Action dim is published as input_features["action"].shape[0] in lerobot
        # 0.4.3; fall back to 7 (xyz + rotvec + gripper) if absent so legacy
        # checkpoints don't crash the load.
        action_feat = peeked_cfg.output_features.get("action") if peeked_cfg.output_features else None
        self.action_dim = int(action_feat.shape[0]) if action_feat is not None else 7

        self.inf_proc = ctx.Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "rtc_cfg": self.rtc_cfg,
                "num_inference_steps": num_inference_steps,
                "allow_empty_cameras": allow_empty_cameras,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self):  # noqa: ANN201
        """Not used by the deploy script. Kept abstract-satisfying."""
        raise NotImplementedError(
            "AsyncSmolVLAPolicy does not provide make_data_fn — the deploy "
            "script drives the RTC queue directly."
        )

    @override
    def reset(self) -> None:
        self.parent_conn.send({"type": "RESET"})

    @override
    def shutdown(self) -> None:
        try:
            self.parent_conn.send({"type": "SHUTDOWN"})
        except (BrokenPipeError, EOFError):
            pass
        _drain_conn(self.parent_conn)
        try:
            self.inf_proc.join(timeout=5.0)
        finally:
            if self.inf_proc.is_alive():
                self.inf_proc.terminate()

    # --- Request/response convenience ----------------------------------

    def request(
        self,
        obs_seq: list[Observation],
        task: str,
        prev_chunk_left_over: np.ndarray | None,
        action_index_before_inference: int,
        inference_delay_hint: int,
    ) -> dict:
        """Send one inference request and block on the reply.

        Returns a dict with keys: ``original`` (np.ndarray, raw policy chunk
        pre-postprocess), ``processed`` (np.ndarray, post-processed), and
        ``latency_s`` (float, measured inference wall time in seconds).
        """
        self.parent_conn.send(
            {
                "type": "OBS_SEQ_RTC",
                "obs_seq": obs_seq,
                "task": task,
                "prev_chunk_left_over": prev_chunk_left_over,
                "action_index_before_inference": int(action_index_before_inference),
                "inference_delay_hint": int(inference_delay_hint),
            }
        )
        resp = self.parent_conn.recv()
        if isinstance(resp, dict) and resp.get("type") == "ERROR":
            raise RuntimeError(f"inference worker error: {resp.get('error')!r}")
        return resp


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    rtc_cfg: dict,
    num_inference_steps: int | None,
    allow_empty_cameras: int | None = None,
) -> None:
    """Subprocess entry-point.

    Loads SmolVLA + RTCConfig on GPU, then services request/response over the
    pipe until the parent sends ``{"type": "SHUTDOWN"}`` (or closes).
    """
    # All heavy imports happen in the child process so the parent stays light.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.utils.constants import ACTION

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
    if policy_config is None:
        raise ValueError(
            f"Policy configuration is missing in the pretrained path: "
            f"{pretrained_path}."
        )

    # Apply optional flow-matching denoise-steps override. SmolVLA exposes
    # this as `num_inference_steps` on the config (configuration_smolvla.py).
    if num_inference_steps is not None and hasattr(policy_config, "num_inference_steps"):
        policy_config.num_inference_steps = int(num_inference_steps)
        logger.info("[Inference] override: num_inference_steps=%s", num_inference_steps)

    # Override how many camera slots the model is allowed to zero-pad when
    # the batch doesn't supply them. Use this when the checkpoint declares
    # more `image_features` than the deploy env produces (e.g. trained with
    # 3 cameras, deployed with 2 — set allow_empty_cameras=1 so the missing
    # camera is filled with zeros instead of raising).
    if allow_empty_cameras is not None and hasattr(policy_config, "empty_cameras"):
        policy_config.empty_cameras = int(allow_empty_cameras)
        logger.info("[Inference] override: empty_cameras=%s", allow_empty_cameras)

    # Build the RTCConfig and attach it to the policy config BEFORE
    # from_pretrained() — SmolVLAPolicy.__init__ calls init_rtc_processor()
    # using whatever it finds on self.config.rtc_config at construction time.
    if "prefix_attention_schedule" in rtc_cfg:
        sched = rtc_cfg["prefix_attention_schedule"]
        if isinstance(sched, str):
            rtc_cfg = {**rtc_cfg, "prefix_attention_schedule": RTCAttentionSchedule[sched.upper()]}
    if not hasattr(policy_config, "rtc_config"):
        raise RuntimeError(
            "Loaded policy config has no `rtc_config` attribute — is "
            "`lerobot` resolved to the speedup fork (lerobot 0.4.3+ with "
            "lerobot.policies.rtc)? Check the pixi env `jazzy-lerobot-rtc`."
        )
    policy_config.rtc_config = RTCConfig(**rtc_cfg)

    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(pretrained_path, config=policy_config)
    policy.reset()
    policy.to(device).eval()
    # SmolVLAPolicy.__init__ already calls init_rtc_processor(); calling it
    # again is harmless and guards against fork paths where __init__ skipped
    # it (e.g. some `from_pretrained` overrides reconstruct without __init__).
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=pretrained_path
    )

    # ------------------------------------------------------------------
    # Trim postprocessor action stats to match the model's output dim.
    #
    # Why: this checkpoint was trained on a 7-dim recorded dataset
    # (xyz + rotvec + gripper) but configured for a 6-dim policy output
    # (the "nogrip" variant drops the gripper action channel). The
    # saved unnormalizer.safetensors still carries 7-element mean/std/min/...
    # for the `action` feature, so multiplying the model's 6-dim output
    # by 7-element stats broadcasts and explodes (RuntimeError: tensor
    # size 6 must match 7 at non-singleton dim 2).
    #
    # Walk the loaded postprocessor steps, locate the one carrying
    # `_tensor_stats["action"]`, and slice every action stat tensor down
    # to the model's expected dim. No-op when the dims already match.
    # ------------------------------------------------------------------
    model_action_dim = None
    action_feat = policy.config.output_features.get("action") if policy.config.output_features else None
    if action_feat is not None and getattr(action_feat, "shape", None):
        model_action_dim = int(action_feat.shape[0])
    if model_action_dim is not None:
        import numpy as _np

        def _trim_one(value, n):
            # Trim the trailing dim of a numpy array / python list / torch tensor
            # down to n entries. Scalars (count) and short vectors pass through.
            try:
                import torch as _torch
                if isinstance(value, _torch.Tensor):
                    if value.ndim == 0 or value.shape[-1] <= n:
                        return value, False
                    return value[..., :n].clone(), True
            except ImportError:
                pass
            if isinstance(value, _np.ndarray):
                if value.ndim == 0 or value.shape[-1] <= n:
                    return value, False
                return value[..., :n].copy(), True
            if isinstance(value, list):
                if len(value) <= n:
                    return value, False
                return value[:n], True
            return value, False

        for step in getattr(postprocessor, "steps", []) or []:
            # Trim BOTH `self.stats` (the source dict — used to re-derive
            # `_tensor_stats` on every .to() call) AND `_tensor_stats`
            # (the current tensors). Trimming only one of them gets undone
            # on the next device/dtype migration.
            for attr_name in ("stats", "_tensor_stats"):
                stats_dict = getattr(step, attr_name, None)
                if not isinstance(stats_dict, dict) or "action" not in stats_dict:
                    continue
                action_stats = stats_dict["action"]
                if not isinstance(action_stats, dict):
                    continue
                trimmed = 0
                for stat_name, value in list(action_stats.items()):
                    new_value, did_trim = _trim_one(value, model_action_dim)
                    if did_trim:
                        action_stats[stat_name] = new_value
                        trimmed += 1
                if trimmed:
                    logger.info(
                        "[Inference] Trimmed %d action stat entries in "
                        "postprocessor.%s down to %d dims.",
                        trimmed, attr_name, model_action_dim,
                    )

    state_subkeys = [
        k for k in policy.config.input_features if k.startswith("observation.state.")
    ]
    logger.info(
        "[Inference] SmolVLA loaded from %s on %s. RTC enabled=%s "
        "(exec=%d, expected_delay=%d). state_subkeys=%s",
        pretrained_path,
        device,
        policy_config.rtc_config.enabled,
        policy_config.rtc_config.execution_horizon,
        policy_config.rtc_config.inference_delay,
        state_subkeys,
    )

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if not isinstance(msg, dict):
            logger.warning("[Inference] non-dict msg: %r", msg)
            continue

        mtype = msg.get("type")
        if mtype == "SHUTDOWN":
            break
        if mtype == "RESET":
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            continue
        if mtype != "OBS_SEQ_RTC":
            logger.warning("[Inference] unknown msg type: %r", mtype)
            continue

        try:
            resp = _serve_one_request(
                msg=msg,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                state_subkeys=state_subkeys,
                device=device,
                action_key=ACTION,
            )
            conn.send(resp)
        except Exception as exc:  # noqa: BLE001 — surface any error to the parent
            logger.exception("[Inference] request failed")
            try:
                conn.send({"type": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            except (BrokenPipeError, EOFError):
                break

    try:
        conn.close()
    except OSError:
        pass
    logger.info("[Inference] worker shutting down")


def _serve_one_request(
    msg: dict[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    state_subkeys: list[str],
    device: torch.device,
    action_key: str,
) -> dict[str, Any]:
    obs_seq = msg["obs_seq"]
    task: str | None = msg.get("task")
    prev_arr: np.ndarray | None = msg.get("prev_chunk_left_over")
    action_index_before = int(msg.get("action_index_before_inference", 0))
    inference_delay_hint = int(msg.get("inference_delay_hint", 0))

    obs_raw = dict(obs_seq[-1])  # most-recent observation
    if state_subkeys:
        obs_raw["observation.state"] = np.concatenate(
            [
                np.asarray(obs_raw[k], dtype=np.float32).reshape(-1)
                for k in state_subkeys
            ]
        )
    else:
        obs_raw["observation.state"] = concatenate_state_features(obs_raw)

    state_feat = policy.config.input_features.get("observation.state")
    if state_feat is not None:
        expected_dim = int(state_feat.shape[0])
        actual_dim = int(np.asarray(obs_raw["observation.state"]).reshape(-1).shape[0])
        if actual_dim != expected_dim:
            raise ValueError(
                f"observation.state dim mismatch: env produced {actual_dim}, "
                f"policy expects {expected_dim}."
            )

    batch = numpy_obs_to_torch(obs_raw)
    # SmolVLA's preprocessor reads the language instruction from `task`.
    # eval_with_real_robot.py passes it as a 1-element list so the batched
    # tokenizer step (TokenizerProcessorStep) handles it uniformly with
    # batched inputs.
    if task is not None:
        batch["task"] = [task]

    prev_tensor: torch.Tensor | None = None
    if prev_arr is not None and len(prev_arr) > 0:
        prev_tensor = torch.as_tensor(prev_arr, dtype=torch.float32, device=device).unsqueeze(0)

    batch = preprocessor(batch)

    t0 = time.perf_counter()
    # NOTE: we deliberately do NOT wrap in torch.inference_mode() here.
    # The RTC processor (modeling_rtc.RTCProcessor.denoise_step) computes a
    # guidance correction via torch.autograd.grad on the model output, which
    # requires gradient tracking on `x_t`. inference_mode unconditionally
    # disables grad for any tensor created inside its scope, raising
    # "RuntimeError: element 0 of tensors does not require grad" on the
    # second inference (when prev_chunk_left_over kicks RTC's guidance path
    # in). predict_action_chunk has its own @torch.no_grad() decorator
    # which is the right granularity — RTC re-enables grad locally.
    chunk = policy.predict_action_chunk(
        batch,
        inference_delay=inference_delay_hint,
        prev_chunk_left_over=prev_tensor,
    )
    original = chunk.squeeze(0).detach().clone()
    processed = postprocessor(chunk).squeeze(0)
    latency_s = time.perf_counter() - t0

    return {
        "type": "CHUNK",
        "original": original.cpu().numpy(),
        "processed": processed.detach().cpu().numpy(),
        "latency_s": float(latency_s),
        "action_index_before_inference": action_index_before,
    }


def _drain_conn(conn: Connection) -> None:
    try:
        while conn.poll(0):
            _ = conn.recv()
    except (EOFError, OSError):
        pass
