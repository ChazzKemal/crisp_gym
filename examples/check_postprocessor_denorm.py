#!/usr/bin/env python3
"""Diagnostic: does the checkpoint's postprocessor de-normalize the policy output?

Loads the policy exactly the way crisp_gym/policy/async_lerobot_policy.py's
inference_worker does, runs one inference on a dummy observation, and prints
the action chunk BEFORE and AFTER the postprocessor — so we can see whether
predict_action_chunk emits normalized values that the postprocessor then
un-normalizes into real Cartesian/gripper units.

Run:
    pixi run -e jazzy-lerobot python examples/check_postprocessor_denorm.py \
        --pretrained-path outputs/train/act_speedup_smoke/checkpoints/100000/pretrained_model
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

try:
    from lerobot.policies.factory import make_pre_post_processors
    HAVE_PROCESSORS = True
except ImportError:
    HAVE_PROCESSORS = False


def _summ(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    print(f"  {name:<28} shape={arr.shape}")
    print(f"  {'':<28} min={np.array2string(arr.min(0), precision=4)}")
    print(f"  {'':<28} max={np.array2string(arr.max(0), precision=4)}")
    print(f"  {'':<28} mean={np.array2string(arr.mean(0), precision=4)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-path", required=True, type=Path)
    args = ap.parse_args()

    pp = str(args.pretrained_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== checkpoint: {pp}")
    print(f"=== device: {device}   HAVE_PROCESSORS(make_pre_post_processors)={HAVE_PROCESSORS}")

    # --- raw un-normalizer stats straight from the safetensors -------------
    unfile = next(args.pretrained_path.glob("*unnormalizer*safetensors"), None)
    print(f"\n--- postprocessor un-normalizer file: {unfile}")
    if unfile is not None:
        from safetensors.torch import load_file
        stats = load_file(str(unfile))
        for k, v in stats.items():
            if "action" in k.lower():
                print(f"  {k}: {np.array2string(v.cpu().numpy(), precision=4)}")

    # --- load policy exactly like inference_worker -------------------------
    policy_config = PreTrainedConfig.from_pretrained(pp)
    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(pp, config=policy_config)
    policy.reset()
    policy.to(device).eval()
    print(f"\n--- loaded {policy_config.type} policy")
    print(f"  input_features:  {list(policy.config.input_features)}")
    print(f"  output_features: {list(policy.config.output_features)}")

    pre = post = None
    if HAVE_PROCESSORS:
        pre, post = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=pp
        )
        print("  built pre/post processors from checkpoint")

    # --- build a dummy observation matching the policy input_features ------
    obs = {}
    for key, feat in policy.config.input_features.items():
        shape = tuple(feat.shape)
        if key.startswith("observation.images"):
            # lerobot stores visual features CHW; numpy_obs_to_torch wants HWC uint8
            c, h, w = shape
            obs[key] = np.random.randint(0, 255, size=(h, w, c), dtype=np.uint8)
        elif key.startswith("observation.state"):
            obs[key] = np.zeros(shape, dtype=np.float32)

    # flat observation.state = concat of the state.* subkeys (worker line 230)
    state_subkeys = [
        k for k in policy.config.input_features if k.startswith("observation.state.")
    ]
    if state_subkeys:
        obs["observation.state"] = np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in state_subkeys]
        )

    # --- to torch batch (mirror crisp_gym.util.lerobot_features) -----------
    batch = {}
    for key, value in obs.items():
        if key.startswith("observation.state"):
            batch[key] = torch.from_numpy(value).unsqueeze(0).to(device).float()
        elif key.startswith("observation.images"):
            batch[key] = (
                torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).to(device).float()
                / 255.0
            )

    # --- inference ---------------------------------------------------------
    with torch.inference_mode():
        batch_in = pre(batch) if pre is not None else batch
        chunk_raw = policy.predict_action_chunk(batch_in)
        chunk_post = post(chunk_raw) if post is not None else chunk_raw

    raw = chunk_raw.squeeze(0).detach().cpu().numpy()
    pst = chunk_post.squeeze(0).detach().cpu().numpy()

    print("\n=== ACTION CHUNK — raw (out of predict_action_chunk) ===")
    _summ("raw[:, :7]", raw)
    print("\n=== ACTION CHUNK — after postprocessor ===")
    _summ("post[:, :7]", pst)

    same = np.allclose(raw, pst, atol=1e-6)
    print(f"\n=== raw == post ?  {same}")
    if same:
        print("  -> postprocessor did NOT change the values "
              "(either no-op, or un-normalization happens inside the policy).")
    else:
        print("  -> postprocessor CHANGED the values: it is de-normalizing the output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
