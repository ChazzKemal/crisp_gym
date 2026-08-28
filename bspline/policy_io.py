"""Load a LeRobot checkpoint the way LeRobot itself does at inference time.

LeRobot 0.4.x keeps normalisation *outside* the policy, in separate pre/post
processor pipelines saved next to the weights::

    policy_preprocessor_step_3_normalizer_processor.safetensors
    policy_postprocessor_step_0_unnormalizer_processor.safetensors

``ACTPolicy.from_pretrained()`` restores only the network. Call it alone and
``select_action`` hands back an action in *normalised* space, and observations
go in unnormalised -- both silently wrong, with no error raised. Everything
downstream (decoded waypoints, error in millimetres) is then meaningless.

The pipeline order, matching ``lerobot.utils.control_utils.predict_action``::

    observation -> preprocessor -> policy -> postprocessor -> action
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_policy(ckpt: str | Path, device: str = "cuda"):
    """Return ``(policy, preprocessor, postprocessor)`` ready for inference."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    ckpt = str(ckpt)
    policy = ACTPolicy.from_pretrained(ckpt).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def predict_action(policy, preprocessor, postprocessor, sample: dict, device: str = "cuda"):
    """Run one dataset sample through the full inference pipeline.

    Returns the unnormalised action as a 1-D numpy array.
    """
    obs = {
        k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
        for k, v in sample.items()
        if k.startswith("observation") or k == "task"
    }
    with torch.inference_mode():
        policy.reset()
        action = policy.select_action(preprocessor(obs))
        action = postprocessor(action)
    return action[0].float().cpu().numpy().reshape(-1)


__all__ = ["load_policy", "predict_action"]
