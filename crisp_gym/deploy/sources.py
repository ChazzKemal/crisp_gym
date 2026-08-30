"""Chunk sources: where the producer loop gets its next (K, >=7) action array.

Moved out of ``examples/19_deploy_policy.py``. The loop is deliberately
policy-agnostic -- it asks for a chunk, schedules it, and pushes TargetItems -- so
anything satisfying :class:`ChunkSource` can drive the robot: a real LeRobot policy
in a subprocess, the same policy in-process for debugging, or a synthetic source that
needs no checkpoint and no GPU at all.

The heavy imports (torch, lerobot) stay deferred inside the constructors on purpose:
importing this module must not drag in a GPU stack, and ``cv2`` has to be imported
before ``torch`` to avoid a libjpeg symbol clash.
"""

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from crisp_gym.envs.manipulator_env_config import OrientationRepresentation
from crisp_gym.policy.async_lerobot_policy import AsyncLerobotPolicy

logger = logging.getLogger(__name__)


@runtime_checkable
class ChunkSource(Protocol):
    """What the producer loop requires of whatever is generating actions.

    This was an informal duck type -- three classes that happened to share four
    members, with the agreement recorded only in a docstring ("Mirrors the
    AsyncLerobotPolicy interface"). Writing it down makes a fourth implementation a
    matter of satisfying a declared contract rather than reading the other three.
    """

    #: Rolling observation-buffer length the source expects, from the checkpoint.
    n_obs: int
    #: Actions per returned chunk, from the checkpoint's ``n_action_steps``.
    n_act: int

    def request(self, obs_buf) -> np.ndarray:
        """Block until the next chunk is ready; return it as ``(K, >=7)``."""
        ...

    def shutdown(self) -> None:
        """Release the subprocess / files / IPC the source owns."""
        ...


# ---------------------------------------------------------------------------
# Chunk source abstraction
#
# The main producer loop is policy-agnostic: it requests a chunk, computes
# the speed schedule + cycle-snap on it, and pushes K TargetItems to the
# queue. The chunk source is whatever produces the (K, 7) action array.
#
# Two implementations:
#   _LeRobotChunkSource: wraps AsyncLerobotPolicy → torch in a subprocess.
#   _FakeChunkSource:    synthesises chunks without any model. For smoke
#                        tests of the deploy pipeline (sender thread, queue,
#                        scaler, restore, etc.) without needing a checkpoint.
# ---------------------------------------------------------------------------


class _LeRobotChunkSource:
    """Adapter around AsyncLerobotPolicy that exposes a request/shutdown API.

    Skips AsyncLerobotPolicy.make_data_fn() — that closure calls env.step(),
    which would bypass our cycle-snapped sender thread. We talk to
    parent_conn directly so the chunk lands here, not in the env.
    """

    def __init__(
        self,
        pretrained_path: str,
        env,
        *,
        num_inference_steps: int | None = None,
        noise_scheduler_type: str | None = None,
        n_action_steps: int | None = None,
    ):
        self._policy = AsyncLerobotPolicy(
            pretrained_path=pretrained_path,
            env=env,
            num_inference_steps=num_inference_steps,
            noise_scheduler_type=noise_scheduler_type,
            n_action_steps=n_action_steps,
        )
        self.n_obs = self._policy.n_obs
        self.n_act = self._policy.n_act

    def request(self, obs_buf) -> np.ndarray:
        self._policy.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
        # recv_chunk (not raw recv) so a dead worker raises instead of
        # hanging the producer forever. See AsyncLerobotPolicy.recv_chunk.
        chunk = self._policy.recv_chunk()
        return chunk

    def shutdown(self) -> None:
        self._policy.shutdown()


class _SyncLeRobotChunkSource:
    """In-process counterpart to ``_LeRobotChunkSource`` — no subprocess, no IPC.

    Loads the LeRobot policy + pre/post-processors in the deploy process and
    runs ``predict_action_chunk`` directly in ``request()``. Mirrors the
    state-assembly + dim-check logic from
    ``crisp_gym.policy.async_lerobot_policy.inference_worker`` so behaviour
    matches the async path exactly — only the IPC layer is removed.

    Supports both ACT (``n_obs_steps=1``) and diffusion-family policies
    (``n_obs_steps>=2``); ``n_obs`` and ``n_act`` are read from the checkpoint
    config in ``__init__`` rather than hardcoded, matching ``AsyncLerobotPolicy``.

    Trade-off vs the async source:
      + simpler to debug (drop ``pdb.set_trace()`` inside ``request``;
        no spawn, no pickling of obs dicts through a Pipe).
      + skips the ~5-10 ms parent->child->parent IPC roundtrip.
      - torch inference now holds the GIL inside the deploy process for the
        full denoising loop (~25-35 ms for ACT, ~50-300 ms for diffusion
        depending on ``num_inference_steps``). That keeps the sender thread
        off the CPU while a chunk is being computed -> expect
        ``sleep_overshoot_ms`` p99 and the late-frame count to rise
        relative to the async path, more so for diffusion.

    Intended for debugging / development. Use the async (default) path for
    production deploys where sender timing matters.
    """

    def __init__(
        self,
        pretrained_path: str,
        env,
        *,
        num_inference_steps: int | None = None,
        noise_scheduler_type: str | None = None,
        n_action_steps: int | None = None,
    ):
        # Local imports keep the heavy torch / lerobot import cost off the
        # deploy script's startup path when --sync is not requested.
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class
        try:
            from lerobot.policies.factory import make_pre_post_processors
            use_processors = True
        except ImportError:
            make_pre_post_processors = None  # noqa: N806
            use_processors = False

        from crisp_gym.util.lerobot_features import (
            concatenate_state_features,
            numpy_obs_to_torch,
        )
        # Same diffusion-queue helpers as the async worker. Imported here
        # (inside the sync source's __init__) so the deploy script's
        # heavy-import startup path is unaffected when --sync is not used.
        # ACTION constant is needed to drop the None action key the
        # preprocessor leaves in the batch (see async-worker import comment).
        from lerobot.policies.utils import populate_queues
        from lerobot.utils.constants import ACTION, OBS_IMAGES

        self._torch = torch
        self._concatenate_state_features = concatenate_state_features
        self._numpy_obs_to_torch = numpy_obs_to_torch
        self._populate_queues = populate_queues
        self._OBS_IMAGES = OBS_IMAGES
        self._ACTION = ACTION

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
        if policy_config is None:
            raise ValueError(
                f"Policy configuration is missing in the pretrained path: "
                f"{pretrained_path}."
            )
        # Apply diffusion-only overrides BEFORE policy load. The diffusion
        # model reads `num_inference_steps` and constructs its noise scheduler
        # in __init__ (modeling_diffusion.py:184-198), so post-load mutation
        # is silently ignored. getattr-guarded so non-diffusion configs (ACT)
        # don't acquire spurious attributes.
        if num_inference_steps is not None and hasattr(policy_config, "num_inference_steps"):
            policy_config.num_inference_steps = int(num_inference_steps)
        if noise_scheduler_type is not None and hasattr(policy_config, "noise_scheduler_type"):
            policy_config.noise_scheduler_type = str(noise_scheduler_type)
        # n_action_steps override (e.g. trained with chunk_size=100, deploy
        # at 50 for tighter replan cadence). Validate against the policy
        # family's invariants — `chunk_size` (ACT) or
        # `horizon - n_obs_steps + 1` (diffusion) — before mutating.
        if n_action_steps is not None and hasattr(policy_config, "n_action_steps"):
            steps_req = int(n_action_steps)
            horizon = getattr(policy_config, "chunk_size", None)
            if horizon is None:
                horizon = getattr(policy_config, "horizon", None)
            if horizon is not None and steps_req >= int(horizon):
                raise ValueError(
                    f"--n-act={steps_req} must be < horizon/chunk_size={horizon}."
                )
            n_obs_steps_cfg = int(getattr(policy_config, "n_obs_steps", 1))
            if horizon is not None and steps_req > int(horizon) - n_obs_steps_cfg + 1:
                raise ValueError(
                    f"--n-act={steps_req} violates `n_action_steps <= horizon - "
                    f"n_obs_steps + 1` (horizon={horizon}, n_obs_steps={n_obs_steps_cfg})."
                )
            policy_config.n_action_steps = steps_req
        policy_cls = get_policy_class(policy_config.type)
        policy = policy_cls.from_pretrained(pretrained_path, config=policy_config)
        policy.reset()
        policy.to(device).eval()
        self._policy = policy

        # ACT has n_obs_steps=1, diffusion defaults to 2. Mirror
        # AsyncLerobotPolicy: read both from the checkpoint config so the
        # producer's rolling obs buffer matches what the policy expects.
        self.n_obs = int(getattr(policy.config, "n_obs_steps", 1))
        self.n_act = int(getattr(policy.config, "n_action_steps", 1))

        logger.info(
            "[sync] loaded %s policy (type=%s) from %s on device %s "
            "(n_obs=%d, n_act=%d)",
            policy.name, policy.config.type, pretrained_path, device,
            self.n_obs, self.n_act,
        )

        self._preprocessor = self._postprocessor = None
        if use_processors:
            self._preprocessor, self._postprocessor = make_pre_post_processors(
                policy_cfg=policy.config, pretrained_path=pretrained_path,
            )

        # Same fallback logic as inference_worker: prefer the policy's declared
        # observation.state.* sub-keys (their order is authoritative). If the
        # policy declares only the flat observation.state, concatenate every
        # observation.state.* the env emits (insertion order). The downstream
        # dim-check in request() catches mismatches before they reach the
        # normalizer as opaque broadcast errors.
        self._state_subkeys = [
            k for k in policy.config.input_features
            if k.startswith("observation.state.")
        ]
        logger.info(
            "[sync] observation.state built from: %s",
            self._state_subkeys or "<all observation.state.* via fallback>",
        )

    def request(self, obs_buf) -> np.ndarray:
        # Always feed the most-recent observation as a single-step batch.
        # Diffusion-family policies maintain their own n_obs_steps queue
        # internally (`policy._queues`); we populate it below before calling
        # predict_action_chunk. ACT consumes the batch directly. See the
        # async-worker comment for the deploy-time train-time gap (one obs
        # per replan, not per control step).
        obs_raw = obs_buf[-1]

        if self._state_subkeys:
            obs_raw["observation.state"] = np.concatenate(
                [np.asarray(obs_raw[k], dtype=np.float32).reshape(-1)
                 for k in self._state_subkeys]
            )
        else:
            obs_raw["observation.state"] = self._concatenate_state_features(obs_raw)

        # Fail loudly on state-dim mismatch (mirrors inference_worker).
        state_feat = self._policy.config.input_features.get("observation.state")
        if state_feat is not None:
            expected_dim = int(state_feat.shape[0])
            actual_dim = int(
                np.asarray(obs_raw["observation.state"]).reshape(-1).shape[0]
            )
            if actual_dim != expected_dim:
                src = (
                    f"input_features sub-keys {self._state_subkeys}"
                    if self._state_subkeys
                    else "concatenate_state_features over all observation.state.* keys"
                )
                raise ValueError(
                    f"observation.state dim mismatch: env produced {actual_dim}, "
                    f"policy expects {expected_dim} (built from {src}). Check the "
                    f"env config's `observations_to_include_to_state` matches the "
                    f"state the policy was trained on."
                )

        torch = self._torch
        with torch.inference_mode():
            batch = self._numpy_obs_to_torch(obs_raw)
            if self._preprocessor is not None:
                batch = self._preprocessor(batch)

            # Diffusion path: populate the policy's internal obs queue so
            # predict_action_chunk has a non-empty TensorList to stack.
            if hasattr(self._policy, "_queues") and self._policy._queues is not None:
                queue_batch = dict(batch)
                # Drop ACTION=None left by the preprocessor (see async worker
                # for why — populate_queues would otherwise fill the action
                # deque with Nones and torch.stack later trips).
                queue_batch.pop(self._ACTION, None)
                if self._policy.config.image_features:
                    queue_batch[self._OBS_IMAGES] = torch.stack(
                        [queue_batch[key] for key in self._policy.config.image_features],
                        dim=-4,
                    )
                self._populate_queues(self._policy._queues, queue_batch)
                chunk = self._policy.predict_action_chunk(queue_batch)
            else:
                # ACT path unchanged.
                chunk = self._policy.predict_action_chunk(batch)

            if self._postprocessor is not None:
                chunk = self._postprocessor(chunk)
        return chunk.squeeze(0).to(device="cpu").numpy()

    def shutdown(self) -> None:
        # Free GPU memory eagerly so a subsequent run in the same process
        # (or a tight bash loop) starts from a clean slate.
        try:
            del self._policy, self._preprocessor, self._postprocessor
        except Exception:
            pass
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass


# Camera observation keys we require the fake source to receive. Matches the
# ManipulatorEnv obs schema for ur10e_ridgeback_dual_cam_env (one entry per
# camera_configs[*].camera_name in the env yaml).
_REQUIRED_FAKE_CAM_KEYS = ("observation.images.camera", "observation.images.d405")


class DatasetExhausted(Exception):
    """Raised by _FakeChunkSource (dataset mode, loop=False) when the
    recorded episode has been fully traversed. The producer loop catches
    this, sets stopped_by='dataset_exhausted', breaks cleanly, and the
    finally block runs the summary write + teardown.
    """


class _FakeChunkSource:
    """No-model chunk generator for smoke-testing the deploy pipeline.

    Mirrors the AsyncLerobotPolicy interface (n_obs, n_act, request, shutdown)
    so main() doesn't care which source is feeding the queue.

    The fake source DOES inspect `obs_buf` — it validates that the latest
    observation contains both expected camera image keys and logs a per-chunk
    summary (shape / dtype / mean / first 16-byte fingerprint) so we can
    confirm the live camera frames are actually flowing into the policy
    interface end-to-end. The chunk OUTPUT is still synthetic (anchor pose
    for 'hold', dataset slice for 'dataset') — the policy is fake, the
    obs path is real.

    Modes:
      hold:     every frame of every chunk is the env's EE pose captured on
                the first request. Zero motion; just exercises queue +
                sender + scaler.step_to (no-op since s_eff stays at the
                same value) + shutdown. Never terminates on its own.
      dataset:  slice a recorded LeRobot episode into consecutive chunks of
                size n_act and return them. When loop=False (default), the
                last chunk before wrap-around is returned as a partial
                chunk of whatever remains, and the NEXT request raises
                DatasetExhausted so the producer can exit cleanly. When
                loop=True, wraps around indefinitely (useful for sustained
                stress tests).
    """

    def __init__(
        self,
        env,
        *,
        mode: str,
        n_act: int,
        n_obs: int = 2,
        dataset_actions: np.ndarray | None = None,
        loop: bool = False,
    ):
        if mode not in ("hold", "dataset"):
            raise ValueError(f"Unknown fake mode: {mode!r}")
        if mode == "dataset" and (
            dataset_actions is None or dataset_actions.shape[0] == 0
        ):
            raise ValueError(
                "fake dataset mode requires a non-empty dataset_actions array"
            )
        self.env = env
        self.mode = mode
        self.n_act = int(n_act)
        self.n_obs = int(n_obs)
        self._dataset_actions = dataset_actions
        self._dataset_idx = 0
        self._loop = bool(loop)
        self._exhausted = False  # only meaningful in dataset mode with loop=False
        self._anchor: np.ndarray | None = None  # captured on first request
        self._chunk_count = 0

    def _capture_anchor(self) -> np.ndarray:
        """Read current EE pose from crisp_py's buffer, freeze it as anchor."""
        if self._anchor is None:
            ee = self.env.robot.end_effector_pose
            ee_arr = ee.to_array(representation=OrientationRepresentation.EULER)
            grip = 1.0  # open
            self._anchor = np.concatenate([ee_arr.astype(np.float64), [grip]])
            logger.info(
                "fake source anchor: pos=(%.3f, %.3f, %.3f) rpy=(%.3f, %.3f, %.3f)",
                *self._anchor[:6],
            )
        return self._anchor

    # Only emit the per-camera fingerprint log every N-th chunk. Cuts ~95%
    # of producer-side logger overhead vs every-chunk logging while still
    # catching a "camera frozen / cv_bridge stale" regression quickly.
    _CAMERA_LOG_EVERY_N_CHUNKS = 30

    @classmethod
    def _validate_and_log_cameras(cls, obs_buf, chunk_idx: int) -> None:
        """Confirm both expected camera keys are in the latest obs + log them.

        Validation always runs (cheap dict lookup); fingerprint logging is
        throttled to every _CAMERA_LOG_EVERY_N_CHUNKS-th chunk. Raises
        KeyError if a required key is missing — the caller likely has the
        wrong --env-config (single-cam) or the camera launch is down.
        """
        if not obs_buf:
            raise RuntimeError(
                "obs_buf is empty; producer loop must call env._get_obs() "
                "before chunk_source.request()"
            )
        latest = obs_buf[-1]
        missing = [k for k in _REQUIRED_FAKE_CAM_KEYS if k not in latest]
        if missing:
            raise KeyError(
                f"Fake policy expects both cameras in observation; "
                f"missing key(s): {missing}. Check --env-config is "
                f"ur10e_ridgeback_dual_cam_env and both Orbbec + D405 "
                f"launches are running (`pixi run orbbec` + `pixi run "
                f"realsense` in clearpath_remote_ws)."
            )
        if chunk_idx % cls._CAMERA_LOG_EVERY_N_CHUNKS != 0:
            return
        for key in _REQUIRED_FAKE_CAM_KEYS:
            img = latest[key]
            arr = np.asarray(img)
            fp = int(arr.tobytes()[:16].__hash__()) if arr.size else 0
            logger.info(
                "  fake.cam[%s]: shape=%s dtype=%s mean=%.1f fp=%016x",
                key.rsplit(".", 1)[-1], tuple(arr.shape), arr.dtype,
                float(arr.mean()) if arr.size else float("nan"),
                fp & 0xFFFFFFFFFFFFFFFF,
            )

    def request(self, obs_buf) -> np.ndarray:
        # If we've already returned the dataset's final partial chunk, the
        # episode is done — bail. Producer catches DatasetExhausted.
        if self.mode == "dataset" and self._exhausted:
            raise DatasetExhausted(
                f"dataset exhausted after {self._chunk_count} chunks "
                f"(loop=False)"
            )

        # 1. Inspect the cameras we'd feed a real policy with. The OUTPUT
        # below is still synthetic, but we exercise the full obs path here
        # so deployment of a real policy with the same env config doesn't
        # surprise us with a missing-key error after we swap chunk sources.
        self._validate_and_log_cameras(obs_buf, self._chunk_count)

        K = self.n_act
        if self.mode == "hold":
            anchor = self._capture_anchor()
            chunk = np.tile(anchor, (K, 1))
        else:  # dataset
            assert self._dataset_actions is not None
            T = self._dataset_actions.shape[0]
            start = self._dataset_idx
            end = start + K
            if end <= T:
                # Normal path — full chunk inside the episode.
                chunk = self._dataset_actions[start:end].astype(np.float64, copy=True)
                self._dataset_idx = end
            else:
                # End-of-episode boundary. Two regimes:
                if self._loop:
                    # Wrap around — take the tail then loop to the head.
                    tail = self._dataset_actions[start:].astype(np.float64, copy=True)
                    head_len = K - tail.shape[0]
                    head = self._dataset_actions[:head_len].astype(np.float64, copy=True)
                    chunk = np.concatenate([tail, head], axis=0)
                    self._dataset_idx = head_len
                else:
                    # Single-pass: return whatever's left as a partial
                    # chunk (size T - start, between 1 and K-1). Mark
                    # exhausted so the NEXT request raises and the
                    # producer exits cleanly.
                    chunk = self._dataset_actions[start:].astype(np.float64, copy=True)
                    self._exhausted = True
                    self._dataset_idx = T
                    logger.info(
                        "fake dataset source: end of episode — returning "
                        "final partial chunk of %d frames (next request will "
                        "raise DatasetExhausted)",
                        chunk.shape[0],
                    )

        self._chunk_count += 1
        return chunk

    def shutdown(self) -> None:
        # Nothing to clean up — no subprocess.
        pass
