"""Asynchronous Lerobot Policy Module."""

import logging
from collections import deque
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Callable, Tuple

# cv2 MUST be imported before torch/lerobot. cv2's bundled libtiff needs the
# system libjpeg's `jpeg12_write_raw_data`; if torch loads a different libjpeg
# into the process first, the later `crisp_py.camera` -> cv2 import dies with
# "undefined symbol: jpeg12_write_raw_data". This module is re-imported from
# scratch by the spawned inference worker, so without this the worker dies on
# startup and the parent blocks forever in recv(). Same trap documented in
# examples/29_clean_deploy.py.
import cv2  # noqa: F401  (import for side effect: pins the correct libjpeg)

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import (
    concatenate_state_features,
    numpy_obs_to_torch,
)

# Diffusion-family policies maintain their own n_obs_steps deques inside the
# policy (`policy._queues`). `predict_action_chunk` reads from those queues
# (ignoring the batch values), so the caller must populate them via
# `populate_queues` before each chunk call. ACT does not use these queues.
#
# ACTION must be popped from the batch before populate_queues runs: the
# LeRobot preprocessor's `transition_to_batch` always re-inserts `ACTION`
# (set to None at inference time), and `populate_queues` would otherwise
# fill `policy._queues[ACTION]` with Nones. `predict_action_chunk` then
# trips `torch.stack([None, None, ...])` → TypeError. select_action handles
# this with `batch.pop(ACTION)` (modeling_diffusion.py:124); we mirror that.
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES

# LeRobot >= 0.4 moved input/output normalization out of the policy and into
# separate pre/post-processor pipelines (loaded from the checkpoint). Older
# versions normalized inside the policy. Mirror lerobot_policy.py and detect.
try:
    from lerobot.policies.factory import make_pre_post_processors

    USE_LEROBOT_PROCESSORS = True
except ImportError:
    USE_LEROBOT_PROCESSORS = False


@register_policy("async_lerobot_policy")
class AsyncLerobotPolicy(Policy):
    """Asynchronous Lerobot Policy."""

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        *,
        num_inference_steps: int | None = None,
        noise_scheduler_type: str | None = None,
        n_action_steps: int | None = None,
    ):
        """Initialize the policy.

        ``n_obs`` and ``n_act`` are now read from the checkpoint's
        ``PreTrainedConfig`` so the same code path supports both ACT
        (``n_obs_steps=1``, ``n_action_steps`` from training) and diffusion
        (``n_obs_steps=2`` by default, smaller ``n_action_steps`` per the
        ``n_action_steps <= horizon - n_obs_steps + 1`` constraint). This
        replaces the previous hardcoded ``n_obs=2, n_act=5`` defaults.

        ``n_action_steps`` overrides the checkpoint's per-inference action
        horizon at load time. Use this when the model was trained with a
        large chunk (e.g. 100) but you want to replan more often (e.g. 50).
        Must satisfy ``n_action_steps <= chunk_size`` (ACT) or
        ``n_action_steps <= horizon - n_obs_steps + 1`` (diffusion); the
        inference worker validates this and raises if violated. ``None``
        (default) = use the value from the checkpoint config.

        ``num_inference_steps`` / ``noise_scheduler_type`` override the
        diffusion policy's denoising-loop config at load time. None = use
        checkpoint setting. Silently ignored by non-diffusion policies.

        ``replan_time`` is still hardcoded — the deploy producer in
        ``examples/19_deploy_policy.py`` ignores ``make_data_fn`` entirely and
        replans on every chunk boundary, so this only matters for callers
        that still use the legacy ``make_data_fn`` path.
        """
        # Spawn (not fork) so the worker can initialize CUDA cleanly: a forked
        # child cannot re-init a CUDA context the parent already touched
        # ("Cannot re-initialize CUDA in forked subprocess"). Spawn re-imports
        # this module and re-pickles the kwargs below — all of which are now
        # picklable (no live `env`; see inference_worker).
        ctx = mp.get_context("spawn")
        self.parent_conn, self.child_conn = ctx.Pipe()
        self.env = env

        # Peek at the policy config once in the parent so the deploy main loop
        # can size its rolling obs buffer to match what the inference worker
        # will demand. PreTrainedConfig.from_pretrained reads only config.json
        # — no torch weights, no GPU — so this is cheap and side-effect-free.
        peeked_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
        if peeked_cfg is None:
            raise ValueError(
                f"Policy configuration is missing in the pretrained path: "
                f"{pretrained_path}."
            )
        # ACT has n_obs_steps=1, diffusion defaults to 2. Both expose
        # n_action_steps directly. getattr fallbacks tolerate older configs.
        self.n_obs = int(getattr(peeked_cfg, "n_obs_steps", 1))
        ckpt_n_act = int(getattr(peeked_cfg, "n_action_steps", 1))
        # CLI override (e.g. --n-act 50 on a chunk_size=100 model) wins.
        # The inference worker re-validates against `chunk_size` / `horizon`
        # before applying, so a value that violates the policy's invariant
        # raises there instead of corrupting the producer's rolling buffer.
        self.n_act = int(n_action_steps) if n_action_steps is not None else ckpt_n_act
        self.replan_time = 3
        self.inpainting = False

        self.inf_proc = ctx.Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "steps": self.n_act,
                "inpainting": self.inpainting,
                "replan_time": self.replan_time,
                "num_inference_steps": num_inference_steps,
                "noise_scheduler_type": noise_scheduler_type,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Return a function that returns (obs, action) each frame by talking to the worker.

        Behaviour:
         - On first call: collect n_obs observations into a rolling buffer.
         - Request chunks according to the replan_time parameter.
         - Each call executes one action from the current chunk and returns (obs, action) for sorting/recording.
        """
        # Before starting, fill the observation buffer
        obs_buf: deque = deque(maxlen=self.n_obs)
        for _ in range(self.n_obs):
            obs_buf.append(self.env._get_obs())

        # Prepare first chunk in the case that n_act != replan_time
        if self.n_act != self.replan_time:
            self.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
            print("Starting new inference")

        i = 0
        next_chunk = None
        current_chunk = None

        def _fn() -> tuple:
            nonlocal i, next_chunk, current_chunk  # Required to mutate across calls
            if i == 0:
                if (
                    self.n_act == self.replan_time
                ):  # Edge case when we want to make a new prediction after all action chunks have been used up
                    obs_buf.append(self.env._get_obs())
                    self.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
                    print("Starting new inference")
                next_chunk = self.recv_chunk()
                current_chunk = next_chunk[self.n_act - self.replan_time :]
                print("Length ot the new current chunk:", len(current_chunk))

            # execute action
            action = current_chunk[i]
            print("Process element:", i)
            obs, *_ = self.env.step(action, block=False)
            obs_buf.append(obs)

            # Start prediction
            if i == (2 * self.replan_time - self.n_act):
                self.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
                print("Starting new inference")

            # step done
            i += 1

            # when done with one episode reset the counter
            if i >= (len(current_chunk)):
                i = 0

            return obs, action

        return _fn

    # Default ceiling for a single inference round-trip. Generous: a cold
    # laptop GPU at idle clocks can take well over a second per chunk.
    RECV_TIMEOUT_S = 30.0

    def recv_chunk(self, timeout: float | None = None):
        """Receive one chunk, raising instead of blocking forever.

        A bare ``parent_conn.recv()`` hangs indefinitely if the inference
        worker died (e.g. an ImportError at spawn time): the pipe stays open
        from the parent's side and no reply ever arrives. In a live deploy
        that means the producer silently stops feeding the queue and the arm
        holds its last pose with no error. Poll instead, and check the
        worker is alive before declaring a timeout.
        """
        deadline = timeout if timeout is not None else self.RECV_TIMEOUT_S
        if self.parent_conn.poll(deadline):
            return self.parent_conn.recv()
        if not self.inf_proc.is_alive():
            raise RuntimeError(
                f"inference worker (pid={self.inf_proc.pid}) died with exit "
                f"code {self.inf_proc.exitcode} — see its traceback above. "
                f"No chunk will ever arrive."
            )
        raise TimeoutError(
            f"no chunk from inference worker after {deadline:.1f}s "
            f"(worker still alive — inference is hung or extremely slow)."
        )

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        _drain_conn(self.parent_conn)
        self.inf_proc.join()


def inference_worker(  # noqa: D417
    conn: Connection,
    pretrained_path: str,
    steps: int | None,
    inpainting: bool,
    replan_time: int,
    num_inference_steps: int | None = None,
    noise_scheduler_type: str | None = None,
):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        steps (int): How many actions are executed from the prediction
        inpainting (bool): Whether to use inpainting in the prediction of a new chunk or not
        replan_time (int): After how many steps to start predicting a new action chunk
        num_inference_steps: Override diffusion's denoising-loop length. None = use checkpoint setting.
        noise_scheduler_type: Override diffusion's noise scheduler ("DDPM" / "DDIM"). None = use checkpoint setting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the policy (architecture) config from config.json. We deliberately do
    # not load TrainPipelineConfig from train_config.json here: that file can
    # carry training-only fields from forked LeRobot variants (e.g. rabc_* /
    # speedup_*) that stock lerobot's strict draccus parser rejects. Deployment
    # only needs the policy type, which config.json already provides.
    policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
    if policy_config is None:
        raise ValueError(
            f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
            "Please ensure the policy is correctly configured."
        )
    # Apply diffusion-only overrides BEFORE policy load. The diffusion
    # model constructs its noise scheduler in __init__ and reads
    # `num_inference_steps` there too (modeling_diffusion.py:184-198),
    # so any post-load mutation is silently ignored. Guarded by hasattr
    # so ACT configs don't gain spurious attributes.
    if num_inference_steps is not None and hasattr(policy_config, "num_inference_steps"):
        policy_config.num_inference_steps = int(num_inference_steps)
        logging.info(
            f"[Inference] override: num_inference_steps={num_inference_steps}"
        )
    if noise_scheduler_type is not None and hasattr(policy_config, "noise_scheduler_type"):
        policy_config.noise_scheduler_type = str(noise_scheduler_type)
        logging.info(
            f"[Inference] override: noise_scheduler_type={noise_scheduler_type}"
        )
    policy_cls = get_policy_class(policy_config.type)

    if steps is not None:
        # Prediction-horizon attribute differs by policy family:
        #   ACT       → `chunk_size` (ACTConfig)
        #   Diffusion → `horizon`    (DiffusionConfig)
        # Some forked configs may carry both; prefer the explicit `chunk_size`
        # when present (ACT semantics) and fall back to `horizon`.
        horizon = getattr(policy_config, "chunk_size", None)
        if horizon is None:
            horizon = getattr(policy_config, "horizon", None)
        if horizon is None:
            raise ValueError(
                f"Policy config {type(policy_config).__name__} exposes neither "
                f"`chunk_size` nor `horizon`; cannot validate steps={steps}."
            )
        if steps >= horizon:
            raise ValueError(
                f"The policy steps={steps} must be smaller than the horizon={horizon}."
                "Please modify your cli."
            )
        # Diffusion adds the stricter `n_action_steps <= horizon - n_obs_steps + 1`
        # constraint. n_obs_steps defaults to 1 for ACT (which has no such
        # constraint) so this check degrades to `steps <= horizon` there.
        n_obs_steps_cfg = int(getattr(policy_config, "n_obs_steps", 1))
        if steps > horizon - n_obs_steps_cfg + 1:
            raise ValueError(
                f"steps={steps} violates `n_action_steps <= horizon - "
                f"n_obs_steps + 1` (horizon={horizon}, "
                f"n_obs_steps={n_obs_steps_cfg}). Lower --n-act or retrain "
                f"with a larger horizon."
            )
        policy_config.n_action_steps = int(steps)

    # `inpainting_lengh` (sic) is a speedup-fork ACT-only attribute; diffusion
    # configs don't carry it. Guard with hasattr so wiring inpainting=True for
    # a diffusion checkpoint silently no-ops instead of corrupting the config
    # with a stray attribute the policy never reads.
    if inpainting is True and hasattr(policy_config, "inpainting_lengh"):
        policy_config.inpainting_lengh = max(
            0, int(policy_config.n_action_steps) - int(replan_time)
        )

    policy = policy_cls.from_pretrained(pretrained_path, config=policy_config)

    logging.info(
        f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
    )

    policy.reset()
    policy.to(device).eval()

    # LeRobot 0.4+ does normalization in external pre/post-processor pipelines
    # loaded from the checkpoint, not inside the policy.
    preprocessor = postprocessor = None
    if USE_LEROBOT_PROCESSORS:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=pretrained_path
        )

    # The flat `observation.state` the policy consumes is the concatenation of
    # these observation.state.* sub-features, in this order. Derive them from
    # the policy's declared inputs so the vector matches training even when the
    # env exposes extra state sub-keys (gripper, gripper_target, target, ...).
    state_subkeys = [
        k for k in policy.config.input_features if k.startswith("observation.state.")
    ]
    logging.info(f"[Inference] observation.state built from: {state_subkeys}")

    # n_obs_steps determines whether the worker stacks the rolling obs_seq
    # (diffusion: 2+) or just consumes the most recent observation (ACT: 1).
    # ACTConfig hardcodes n_obs_steps=1 and asserts it stays 1; reading it via
    # getattr keeps us forward-compatible with future configs that omit the
    # field entirely.
    n_obs_steps = int(getattr(policy.config, "n_obs_steps", 1))
    logging.info(
        f"[Inference] policy.type={policy.config.type!r} n_obs_steps={n_obs_steps}"
    )

    print("Ready to recive information")

    while True:
        # Check if messages are recieved correctly
        msg = conn.recv()
        if msg is None:
            break
        if msg == "reset":
            logging.info("[Inference] Resetting policy")
            policy.reset()
            if USE_LEROBOT_PROCESSORS:
                preprocessor.reset()
                postprocessor.reset()
            continue
        if not (isinstance(msg, dict) and msg.get("type") == "OBS_SEQ"):
            logging.warning(f"[Inference] Unknown message: {type(msg)}")
            continue

        # obs_seq is a list of observation dicts, oldest first. We always
        # feed the MOST RECENT one as a single-step batch. The temporal
        # history (for diffusion) is maintained by the policy's own internal
        # `_queues` — see the populate_queues call below. The producer's
        # rolling buffer is incidental for diffusion (one obs per replan
        # already populates the queue's tail correctly).
        obs_seq = msg["obs_seq"]
        obs_raw = obs_seq[-1]

        # Build the flat observation.state from exactly the sub-features the
        # policy declares (state_subkeys), in its order. The env exposes extra
        # observation.state.* keys (gripper, target, ...) this policy never saw
        # in training; concatenate_state_features would blindly include them.
        if state_subkeys:
            obs_raw["observation.state"] = np.concatenate(
                [np.asarray(obs_raw[k], dtype=np.float32).reshape(-1) for k in state_subkeys]
            )
        else:
            obs_raw["observation.state"] = concatenate_state_features(obs_raw)

        # Fail loudly on a state-dim mismatch (silent length errors otherwise
        # surface as opaque broadcast failures inside the normalizer).
        state_feat = policy.config.input_features.get("observation.state")
        if state_feat is not None:
            expected_dim = int(state_feat.shape[0])
            actual_dim = int(np.asarray(obs_raw["observation.state"]).reshape(-1).shape[0])
            if actual_dim != expected_dim:
                src = (
                    f"input_features sub-keys {state_subkeys}"
                    if state_subkeys
                    else "concatenate_state_features over all observation.state.* keys"
                )
                raise ValueError(
                    f"observation.state dim mismatch: env produced {actual_dim}, "
                    f"policy expects {expected_dim} (built from {src}). Check the "
                    f"env config's `observations_to_include_to_state` matches the "
                    f"state the policy was trained on."
                )

        with torch.inference_mode():
            batch = numpy_obs_to_torch(obs_raw)
            if USE_LEROBOT_PROCESSORS:
                batch = preprocessor(batch)

            # Diffusion path: predict_action_chunk reads from `policy._queues`,
            # not from `batch` (see modeling_diffusion.py:96). Mirror what
            # `select_action` does up to the queue check: stack the per-camera
            # image keys into the single OBS_IMAGES key (dim=-4 = new camera
            # axis), then populate the queues with this latest single-step
            # batch. populate_queues auto-fills the deque on its first call
            # by repeating the first observation `n_obs_steps` times.
            #
            # Trade-off: feeding one obs per chunk replan means the policy's
            # `n_obs_steps` queue holds observations spaced by `n_action_steps`
            # control steps, not by 1 control step as in training. With FPS=20
            # and n_act=8 that's 400 ms between queued obs vs ~50 ms during
            # training — functionally works, but is a known deploy-time
            # train-time gap for diffusion (no fix yet; revisit if behaviour
            # is poor).
            if hasattr(policy, "_queues") and policy._queues is not None:
                queue_batch = dict(batch)
                # Drop ACTION (preprocessor's transition_to_batch leaves it as
                # None at inference; otherwise populate_queues stuffs the
                # action deque with Nones and torch.stack later trips).
                queue_batch.pop(ACTION, None)
                if policy.config.image_features:
                    queue_batch[OBS_IMAGES] = torch.stack(
                        [queue_batch[key] for key in policy.config.image_features],
                        dim=-4,
                    )
                populate_queues(policy._queues, queue_batch)
                chunk = policy.predict_action_chunk(queue_batch)
            else:
                # ACT path: predict_action_chunk consumes `batch` directly
                # and gathers config.image_features internally — no stacking.
                chunk = policy.predict_action_chunk(batch)

            if USE_LEROBOT_PROCESSORS:
                chunk = postprocessor(chunk)

        chunk = chunk.squeeze(0).to(device="cpu").numpy()
        logging.debug(f"[Inference] Computed chunk with shape {tuple(chunk.shape)}")
        conn.send(chunk)

    conn.close()
    logging.info("[Inference] Worker shutting down")


# To be implemented later to avoid stale messages in the pipe
def _drain_conn(conn):  # noqa: ANN001
    """Non-blocking: remove any pending messages so we don't reuse stale chunks."""
    try:
        while conn.poll(0):
            _ = conn.recv()
    except (EOFError, OSError):
        pass
