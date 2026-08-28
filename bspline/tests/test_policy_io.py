"""Guard against the silent-normalisation trap.

LeRobot 0.4.x stores normalisation in pre/post-processor pipelines saved beside
the weights, not as buffers inside the policy. ``ACTPolicy.from_pretrained()``
restores only the network, so calling ``select_action`` on raw observations
returns an action in *normalised* space -- with no error raised. Decoded
waypoints then look catastrophically wrong for a model that is training fine.

These tests skip unless a checkpoint is on disk.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from policy_io import load_policy, predict_action

DATASET = Path("/home/batur/Coding/data/merged_bspline_20260528")
CKPT_ROOT = Path(
    "/home/batur/Coding/ur10_clearpath/Yunfei/crisp_gym/outputs/train/"
    "bspline_act_merged_20260528/checkpoints"
)


def _latest_ckpt():
    if not CKPT_ROOT.exists():
        return None
    steps = sorted(p for p in CKPT_ROOT.iterdir() if p.name.isdigit())
    return (steps[-1] / "pretrained_model") if steps else None


@pytest.fixture(scope="module")
def loaded():
    ckpt = _latest_ckpt()
    if ckpt is None or not (DATASET / "meta" / "bspline.json").exists():
        pytest.skip("no trained checkpoint / converted dataset on disk")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DATASET.name, root=DATASET, video_backend="pyav")
    policy, pre, post = load_policy(ckpt, "cuda")
    meta = json.loads((DATASET / "meta" / "bspline.json").read_text())
    return ds, policy, pre, post, meta


def test_checkpoint_ships_processor_files():
    """If these disappear, the pipeline is silently unnormalised."""
    ckpt = _latest_ckpt()
    if ckpt is None:
        pytest.skip("no checkpoint on disk")
    names = {p.name for p in ckpt.iterdir()}
    assert any("preprocessor" in n and n.endswith(".safetensors") for n in names)
    assert any("postprocessor" in n and n.endswith(".safetensors") for n in names)


def test_prediction_lands_in_the_dataset_range(loaded):
    """The whole point: a postprocessed action is on the same scale as the data.

    Skipping the postprocessor leaves the action normalised, i.e. roughly zero
    mean and unit scale -- which is what this catches.
    """
    ds, policy, pre, post, meta = loaded
    stats = json.loads((DATASET / "meta" / "stats.json").read_text())["action"]
    lo = np.asarray(stats["min"])
    hi = np.asarray(stats["max"])
    span = hi - lo

    preds = np.stack([predict_action(policy, pre, post, ds[i]) for i in (10, 4000, 12000)])
    for p in preds:
        inside = ((p >= lo - span) & (p <= hi + span)).mean()
        assert inside > 0.9, f"only {inside:.0%} of channels within the data range"


def test_knot_column_is_on_a_frame_scale(loaded):
    """Knots are measured in frames; normalised output would be near zero."""
    ds, policy, pre, post, meta = loaded
    NS, NC, K = meta["n_action_steps"], meta["n_action_channels"], meta["degree"]
    grid = predict_action(policy, pre, post, ds[4000]).reshape(NS, NC)
    span = grid[NS - K - 1, 0] - grid[K, 0]
    assert span > 5, f"chunk span {span:.1f} frames -- suspiciously small, check normalisation"
    assert span < 400, f"chunk span {span:.1f} frames exceeds any episode"


def test_postprocessor_actually_changes_the_action(loaded):
    """A direct assertion that the missing step is not a no-op."""
    import torch

    ds, policy, pre, post, meta = loaded
    sample = ds[4000]
    obs = {k: (v.unsqueeze(0).to("cuda") if torch.is_tensor(v) else v)
           for k, v in sample.items() if k.startswith("observation") or k == "task"}
    with torch.inference_mode():
        policy.reset()
        raw = policy.select_action(pre(obs))
        done = post(raw)
    # the postprocessor also moves the action back to cpu
    raw, done = raw.cpu(), done.cpu()
    assert not torch.allclose(raw, done, atol=1e-3), "postprocessor is a no-op -- stats missing?"
