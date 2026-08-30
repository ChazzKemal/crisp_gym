"""A real policy run alongside a fake chunk source, purely to cost the inference.

Moved out of ``examples/19_deploy_policy.py``. Lets a deploy run measure realistic
inference latency and GIL contention while the actions actually sent to the robot
still come from a synthetic source -- so the timing is real but the motion is not.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class _ShadowACTPolicy:
    """ACT-shaped shadow run alongside the dataset fake source for timing.

    Tries to instantiate a real `lerobot.policies.act.ACTPolicy` with random
    weights and input/output features auto-derived from a sample observation.
    Falls back to a small torchvision ResNet18-based stub if the LeRobot
    instantiation path errors out (e.g. version mismatch, optional dep
    missing, normalization stats lookup that needs a real dataset).

    Output is RANDOM. The cycle-snap queue always uses the dataset slice;
    the shadow exists only to measure how long real ACT inference takes
    under live obs, on the actual hardware/GPU.
    """

    def __init__(
        self,
        obs_sample: dict,
        n_act: int,
        action_dim: int = 7,
        device: str | None = None,
        temporal_ensemble_coeff: float | None = None,
    ):
        import torch
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Canonicalize: a bare "cuda" string resolves to cuda:current_device
        # at tensor-creation time (usually cuda:0). Our subsequent
        # `p.device != target_device` checks compare to a `torch.device`
        # which DOES include an index — so without canonicalizing, every
        # cuda:0 parameter would falsely look "off" against a cuda(no-idx)
        # target. Pin the index now.
        if device == "cuda" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        self.device = device
        # cuDNN + matmul perf flags. cudnn.benchmark auto-tunes conv kernels
        # the first time each unique input shape is seen — critical for
        # ResNet18 throughput (2-5x). TF32 on Ampere+ GPUs is essentially
        # free precision drop for matmuls. Skipped on CPU.
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.n_act = int(n_act)
        self.action_dim = int(action_dim)
        self.flavor = "uninitialized"
        self.policy = None

        # Image keys in the obs — order matters for stub fallback (so we can
        # pair tensors with backbones).
        self._img_keys: list[str] = sorted(
            k for k in obs_sample
            if k.startswith("observation.images.")
            and np.asarray(obs_sample[k]).ndim == 3
        )
        # State key handling. crisp_gym's ManipulatorEnv._get_obs() returns
        # the proprioceptive components as sub-keys (observation.state.cartesian,
        # .joints, .gripper, .gripper_target, .target) rather than a flat
        # `observation.state`. ACT expects a single `observation.state` tensor
        # though, so we auto-build it by concatenating whichever sub-keys exist,
        # in a stable sorted order. `_obs_to_batch` does the runtime concat.
        # If a flat `observation.state` IS present in obs, we just use it as-is.
        self._state_key: str | None = None
        self._state_sub_keys: list[str] = []
        self._state_total_dim: int = 0
        if "observation.state" in obs_sample:
            self._state_key = "observation.state"
            self._state_total_dim = int(np.asarray(obs_sample["observation.state"]).size)
        else:
            self._state_sub_keys = sorted(
                k for k in obs_sample
                if k.startswith("observation.state.")
                and np.ndim(obs_sample[k]) <= 1
            )
            if self._state_sub_keys:
                self._state_key = "observation.state"  # synthetic; built in _obs_to_batch
                self._state_total_dim = sum(
                    int(np.asarray(obs_sample[k]).size) for k in self._state_sub_keys
                )

        # Construct the policy INSIDE a default-device context so any tensor
        # created without an explicit device argument (e.g. lazy
        # positional embeddings, BatchNorm running stats, torchvision
        # ResNet18 conv weights) lands on cuda from the start. Without
        # this, `.to(device)` only moves *registered* parameters/buffers
        # and a follow-up rescue can miss tensors stored as plain Python
        # attributes that submodules consult during forward — the
        # mechanism that was keeping SM% at 0 even after rescue.
        target_device = torch.device(self.device)
        # The `with target_device:` context (PyTorch 2.0+) sets the default
        # device for tensor constructors. Don't also call
        # torch.cuda.set_device() — it requires an indexed device and
        # raises ValueError on a bare "cuda" instance (which is why the
        # previous run silently fell through to the act-stub fallback).
        try:
            with target_device:
                self.policy = self._build_real_act(
                    obs_sample, n_act, action_dim, temporal_ensemble_coeff,
                )
            self.flavor = "act-real"
        except Exception:
            logger.exception(
                "real ACT instantiation failed; falling back to torchvision stub",
            )
            with target_device:
                self.policy = self._build_stub(obs_sample, n_act, action_dim)
            self.flavor = "act-stub"

        self.policy.eval()
        # Belt-and-braces: explicit .to(device) for any tensors the context
        # manager didn't catch, then audit + rescue stragglers.
        self.policy.to(target_device)
        params_off = [
            (name, str(p.device)) for name, p in self.policy.named_parameters()
            if p.device != target_device
        ]
        buffers_off = [
            (name, str(b.device)) for name, b in self.policy.named_buffers()
            if b.device != target_device
        ]
        for name, p in self.policy.named_parameters():
            if p.device != target_device:
                p.data = p.data.to(target_device)
        for name, b in self.policy.named_buffers():
            if b.device != target_device:
                b.data = b.data.to(target_device)
        # Post-rescue audit: anything that's STILL off after both passes
        # is a tensor held outside the standard parameters/buffers tree
        # (e.g. a Python attribute on a Processor or a closure-captured
        # constant). We'd need a deeper walker for those.
        still_off_params = [
            (name, str(p.device)) for name, p in self.policy.named_parameters()
            if p.device != target_device
        ]
        still_off_buffers = [
            (name, str(b.device)) for name, b in self.policy.named_buffers()
            if b.device != target_device
        ]
        if params_off or buffers_off:
            logger.warning(
                "shadow policy: %d params + %d buffers were OFF target device "
                "after default-device-context + .to(%s); rescued in-place. "
                "Offending names (first 5): params=%s buffers=%s",
                len(params_off), len(buffers_off), self.device,
                params_off[:5], buffers_off[:5],
            )
        else:
            logger.info(
                "shadow policy: all params + buffers constructed on %s",
                self.device,
            )
        if still_off_params or still_off_buffers:
            logger.error(
                "shadow policy: rescue FAILED for %d params + %d buffers — "
                "these are non-standard tensors and won't move via .to(). "
                "Forward pass will likely fall back to CPU for these. "
                "Still-off (first 5): params=%s buffers=%s",
                len(still_off_params), len(still_off_buffers),
                still_off_params[:5], still_off_buffers[:5],
            )

        state_desc = (
            f"{self._state_key} (dim={self._state_total_dim}"
            + (
                f", concatenated from {self._state_sub_keys}"
                if self._state_sub_keys else ""
            )
            + ")"
            if self._state_key else "None"
        )
        logger.info(
            "shadow policy ready: flavor=%s device=%s n_act=%d action_dim=%d "
            "img_keys=%s state=%s temporal_ensemble=%s",
            self.flavor, self.device, self.n_act, self.action_dim,
            self._img_keys, state_desc, temporal_ensemble_coeff,
        )

        # Warmup: run a few forwards on the sample obs to pay the cudnn
        # autotune + kernel compilation cost UP FRONT so the first measured
        # chunk doesn't include it. Without this, N=5 chunk runs see the
        # warmup amortized over only 5 samples and median ≈ steady-state
        # is masked.
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                t0 = time.monotonic()
                for _ in range(3):
                    _ = self.predict(obs_sample)
                torch.cuda.synchronize()
                logger.info(
                    "shadow warmup: 3 forwards in %.0f ms (next .predict() "
                    "should be steady-state)",
                    (time.monotonic() - t0) * 1000.0,
                )
            except Exception:
                logger.exception("shadow warmup failed; continuing without it")

    def _build_real_act(
        self,
        obs_sample: dict,
        n_act: int,
        action_dim: int,
        temporal_ensemble_coeff: float | None,
    ):
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.configs.types import FeatureType, PolicyFeature

        # CHW shapes from HWC image obs. ACT expects a (C, H, W) PolicyFeature
        # and pulls feature_map from a ResNet backbone, so it'll resize
        # internally during the forward pass.
        input_features: dict = {}
        for key in self._img_keys:
            arr = np.asarray(obs_sample[key])
            chw = (arr.shape[2], arr.shape[0], arr.shape[1])
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=chw)
        if self._state_key is not None and self._state_total_dim > 0:
            # Either a real observation.state lives in obs (flat use), or
            # we'll synthesize one by concatenating sub-keys in _obs_to_batch.
            # Either way, ACT just needs the dimension up front.
            input_features[self._state_key] = PolicyFeature(
                type=FeatureType.STATE, shape=(int(self._state_total_dim),),
            )

        if not input_features:
            raise RuntimeError(
                "No usable visual/state features in obs_sample — ACT needs at "
                "least one observation.images.* or observation.state key."
            )

        output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        }

        # n_action_steps must be 1 when temporal_ensemble_coeff is set
        # (ACTConfig validates this in __post_init__).
        n_action_steps = 1 if temporal_ensemble_coeff is not None else n_act

        config = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=n_act,
            n_action_steps=n_action_steps,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            # Skip torchvision pretrained weights download. Random init is
            # fine — we never use the output.
            pretrained_backbone_weights=None,
            # Faster forward: skip the VAE branch entirely. ACT trained with
            # VAE; for our random-weight shadow it's just extra compute.
            use_vae=False,
            # Pin the config's device up front instead of leaving it None
            # and watching PreTrainedConfig.__post_init__ warn + auto-switch.
            # Some LeRobot policy paths read config.device to decide where
            # to allocate auxiliary buffers; setting it here guarantees
            # they're created on cuda from the start, not on cpu then
            # moved (which is what was causing SM%≈0 during the shadow
            # forward — the model was bouncing between cpu and gpu).
            device=self.device,
        )
        return ACTPolicy(config)

    def _build_stub(self, obs_sample: dict, n_act: int, action_dim: int):
        """Last-resort: torchvision ResNet18 per camera + MLP head.

        Forward-pass cost roughly comparable to ACT (the dominant term is the
        backbone, not the transformer). Returns a torch.nn.Module exposing
        `predict_action_chunk(batch)` mimicking ACTPolicy's signature.
        """
        import torch
        import torch.nn as nn
        import torchvision.models

        img_keys = list(self._img_keys)
        n_cams = len(img_keys)
        state_dim = int(self._state_total_dim)

        class _Stub(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.backbones = nn.ModuleList([
                    torchvision.models.resnet18(weights=None) for _ in range(n_cams)
                ])
                feat_dim = 1000 * n_cams + state_dim  # ResNet18 final FC = 1000
                self_inner.head = nn.Sequential(
                    nn.Linear(feat_dim, 512), nn.ReLU(),
                    nn.Linear(512, n_act * action_dim),
                )
                self_inner.n_act = n_act
                self_inner.action_dim = action_dim
                self_inner.img_keys = img_keys
                self_inner.state_key = "observation.state" if state_dim > 0 else None

            def predict_action_chunk(self_inner, batch):
                feats = [bb(batch[k]) for bb, k in zip(self_inner.backbones, self_inner.img_keys)]
                if self_inner.state_key is not None and self_inner.state_key in batch:
                    feats.append(batch[self_inner.state_key])
                cat = torch.cat(feats, dim=-1)
                actions = self_inner.head(cat)
                return actions.view(-1, self_inner.n_act, self_inner.action_dim)

        return _Stub()

    def _obs_to_batch(self, obs: dict) -> dict:
        torch = self._torch
        batch: dict = {}
        for key in self._img_keys:
            arr = np.asarray(obs[key])
            # H2D copy as uint8 (1 B/pixel) BEFORE float conversion (4 B/pixel).
            # Cuts the host→device bandwidth by 4× and offloads the divide-by-255
            # cast to the GPU where it's much cheaper.
            t = torch.from_numpy(arr).to(self.device, non_blocking=True)
            t = t.float() / 255.0
            if t.ndim == 3:
                t = t.permute(2, 0, 1)
            batch[key] = t.unsqueeze(0)
        # State: flat key first, else concatenate sub-keys in stable order.
        if self._state_key is not None:
            if self._state_sub_keys:
                parts = [
                    np.asarray(obs[k], dtype=np.float32).flatten()
                    for k in self._state_sub_keys
                    if k in obs
                ]
                state_arr = np.concatenate(parts) if parts else np.zeros(
                    self._state_total_dim, dtype=np.float32,
                )
            elif self._state_key in obs:
                state_arr = np.asarray(obs[self._state_key]).astype(np.float32).flatten()
            else:
                state_arr = np.zeros(self._state_total_dim, dtype=np.float32)
            batch[self._state_key] = (
                self._torch.from_numpy(state_arr).unsqueeze(0).to(self.device, non_blocking=True)
            )
        return batch

    def predict(self, obs: dict) -> np.ndarray:
        """Forward pass on the latest obs. Returns (n_act, action_dim) numpy."""
        torch = self._torch
        on_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        # Per-stage timing on the first few calls to localise the bottleneck.
        # cuda.synchronize() before each timestamp so we measure GPU work
        # completion, not enqueue. Logged only for the first 3 predict()
        # calls (covers warmup + first chunk) to avoid spamming.
        if not hasattr(self, "_predict_debug_count"):
            self._predict_debug_count = 0
        debug_this_call = self._predict_debug_count < 3

        if debug_this_call and on_cuda:
            torch.cuda.synchronize()
        t0 = time.monotonic()
        batch = self._obs_to_batch(obs)
        if debug_this_call and on_cuda:
            torch.cuda.synchronize()
        t1 = time.monotonic()

        with torch.device(self.device), torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        if on_cuda:
            torch.cuda.synchronize()
        t2 = time.monotonic()

        out = chunk.squeeze(0).detach().cpu().numpy()
        t3 = time.monotonic()

        if debug_this_call:
            # Sample one tensor from the batch to confirm device.
            sample_key = next(iter(batch))
            sample_tensor = batch[sample_key]
            # Sample one param to confirm policy device.
            sample_param = next(self.policy.parameters(), None)
            logger.info(
                "shadow predict[#%d]: obs→batch=%.1fms (out %s on %s) | "
                "model forward=%.1fms (param0 %s on %s) | "
                ".cpu().numpy()=%.1fms | total=%.1fms",
                self._predict_debug_count,
                (t1 - t0) * 1000.0,
                tuple(sample_tensor.shape), sample_tensor.device,
                (t2 - t1) * 1000.0,
                tuple(sample_param.shape) if sample_param is not None else "?",
                sample_param.device if sample_param is not None else "?",
                (t3 - t2) * 1000.0,
                (t3 - t0) * 1000.0,
            )
            self._predict_debug_count += 1
        return out

    def shutdown(self) -> None:
        try:
            del self.policy
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            logger.exception("shadow policy shutdown raised")
