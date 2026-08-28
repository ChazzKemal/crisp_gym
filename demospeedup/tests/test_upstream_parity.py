"""Numerical parity with the genuine upstream implementation.

Every other test file checks that this port behaves sensibly. This one checks
that it behaves *identically to DemoSpeedup* where it claims to, by importing
the real functions out of a clone of

    https://github.com/lingxiao-guo/DemoSpeedup

and running them side by side. Point ``DEMOSPEEDUP_UPSTREAM`` at the clone; the
whole module skips when it is absent, so the suite still runs without it.

    git clone --depth 1 https://github.com/lingxiao-guo/DemoSpeedup /tmp/DemoSpeedup
    DEMOSPEEDUP_UPSTREAM=/tmp/DemoSpeedup conda run -n lerobot-041 \
        python -m pytest tests/test_upstream_parity.py -q

Upstream's modules pull in mujoco/hydra/robomimic at import time, so the two
functions under test are loaded from source text rather than imported.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from demospeedup_core.entropy import kde_entropy
from demospeedup_core.retiming import process_action_label_upstream, select_keep_indices
from demospeedup_core.sampling import TemporalSampleBuffer

UPSTREAM = os.environ.get("DEMOSPEEDUP_UPSTREAM", "")
pytestmark = pytest.mark.skipif(
    not (UPSTREAM and (Path(UPSTREAM) / "aloha" / "act").is_dir()),
    reason="set DEMOSPEEDUP_UPSTREAM to a DemoSpeedup clone to run parity tests",
)


def _load(relpath: str, *names):
    """Exec just the named top-level defs/classes out of an upstream file.

    Importing the module would drag in mujoco, hydra and a diffusion_policy
    fork; the functions themselves only need numpy + torch.
    """
    source = (Path(UPSTREAM) / relpath).read_text()
    tree = ast.parse(source)
    wanted = set(names)
    kept = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    missing = wanted - {node.name for node in kept}
    if missing:
        raise AssertionError(f"{relpath}: no top-level {sorted(missing)}")
    module = ast.Module(body=kept, type_ignores=[])
    ns: dict = {"np": np, "torch": torch}
    exec(compile(module, f"<upstream:{relpath}>", "exec"), ns)  # noqa: S102
    return [ns[n] for n in names]


# --------------------------------------------------------------------------
# 1. the KDE entropy DemoSpeedup labels with
# --------------------------------------------------------------------------
def _upstream_kde():
    gaussian_kernel, KDE = _load(
        "aloha/act/detr/models/entropy_utils.py", "gaussian_kernel", "KDE"
    )
    # KDE.kde_entropy calls the module-level gaussian_kernel; exec put both in
    # the same namespace, so the bound method already resolves it.
    return KDE().kde_entropy


@pytest.mark.parametrize("seed", range(5))
def test_kde_entropy_matches_upstream(seed):
    """Our estimator must be upstream's, to floating point."""
    upstream_kde = _upstream_kde()
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 120, 7, generator=g) * (0.05 + seed)
    ours = kde_entropy(x, bandwidth=1.0)
    theirs = upstream_kde(x)
    assert ours.shape == theirs.shape
    assert torch.allclose(ours, theirs, atol=1e-6), f"{ours.item()} vs {theirs.item()}"


def test_kde_entropy_matches_upstream_on_a_degenerate_cloud():
    """The 1e-8 log floor has to fire in the same place in both."""
    upstream_kde = _upstream_kde()
    x = torch.zeros(1, 64, 7)
    assert torch.allclose(kde_entropy(x), upstream_kde(x), atol=1e-6)


# --------------------------------------------------------------------------
# 2. the stride walk that decides which frames survive
# --------------------------------------------------------------------------
def _upstream_walk(labels: np.ndarray, horizon: int | None = None) -> list[int]:
    """Run upstream's real ``process_action_label`` and recover its indices.

    It returns re-packed tensors rather than the index list, so the indices are
    recovered by feeding it an action column that is simply the frame number:
    ``new_actions[i] == indices[i]`` by construction.
    """
    (process_action_label,) = _load("aloha/act/act_utils.py", "process_action_label")
    n = len(labels)
    action = torch.arange(n, dtype=torch.float32).unsqueeze(1)  # (T, 1) == frame id
    label = torch.from_numpy(np.asarray(labels)).float()
    is_pad = torch.zeros(n, dtype=torch.bool)
    new_actions, _ = process_action_label(action, label, is_pad)
    # trailing rows are zero-filled; the walk never emits index 0, so a 0 after
    # the first entry marks the end of the real output.
    flat = new_actions[:, 0].tolist()
    out = []
    for v in flat:
        if v == 0 and out:
            break
        out.append(int(round(v)))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_stride_walk_matches_upstream(seed):
    """Our vendored walk must reproduce upstream's index sequence exactly."""
    labels = np.random.default_rng(seed).integers(0, 2, size=200)
    assert process_action_label_upstream(labels, start=-1) == _upstream_walk(labels)


@pytest.mark.parametrize(
    "labels",
    [
        np.zeros(60, dtype=np.int64),                                  # all precision
        np.ones(60, dtype=np.int64),                                   # all free motion
        np.concatenate([np.ones(20), np.zeros(20), np.ones(20)]).astype(np.int64),
        np.concatenate([np.zeros(30), np.ones(30)]).astype(np.int64),
        np.array(([1] * 3 + [0]) * 15, dtype=np.int64),                # short free runs
    ],
)
def test_stride_walk_matches_upstream_on_structured_labels(labels):
    assert process_action_label_upstream(labels, start=-1) == _upstream_walk(labels)


# --------------------------------------------------------------------------
# 3. the adaptation: retiming the episode instead of the chunk
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_stride_walk_is_memoryless(seed):
    """The claim the whole port rests on.

    Upstream subsamples the action chunk starting at *every* frame. We drop
    frames instead, so consecutive frames of the retimed episode have to *be*
    that subsampled chunk. That holds because the stride rule is memoryless --
    the next step depends only on the labels at the current position, and its
    ``i + low_v < horizon`` boundary is start-invariant (``i_local + k < n -
    start`` iff ``i_global + k < n``). So a walk launched from any frame already
    on the global walk reproduces the global walk's own continuation.

    Checked here against the *rule*, launched cleanly at the current frame.
    Upstream's literal loop starts at ``i = -1`` instead -- see the next test.
    """
    labels = np.random.default_rng(seed).integers(0, 2, size=240)
    keep = select_keep_indices(labels, keep_last=False)
    for i, start in enumerate(keep[:-1]):
        theirs = [start + k for k in process_action_label_upstream(labels[start:], start=0)]
        ours = keep[i + 1:].tolist()
        overlap = min(len(theirs), len(ours))
        assert overlap > 0
        assert theirs[:overlap] == ours[:overlap], f"diverged from frame {start}"


@pytest.mark.parametrize("last_label,expected_first", [(0, 1), (1, 3)])
def test_upstream_first_waypoint_is_decided_by_the_last_label(last_label, expected_first):
    """Characterises the ``i = -1`` quirk, against the real upstream function.

    ``process_action_label`` begins its loop at ``i = -1``, so the first label
    it consults is ``current_label[-1]`` -- the *last* frame of the chunk, an
    arbitrary distance in the future. The first emitted waypoint is therefore
    1 (when that label is 0) or 3 (when it is 1, because ``label[-1:3]`` slices
    to empty and ``torch.all`` of an empty tensor is True), regardless of what
    the labels near the start of the chunk say. Every subsequent step follows
    the normal rule.

    Consequence for this port: our retimed episodes match upstream's *intended*
    stride walk, and differ from its literal first step. We start the walk at
    the frame we are actually standing on.
    """
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=120)
    labels[0] = 0          # a start whose own label says "step 2"
    labels[-1] = last_label
    assert _upstream_walk(labels)[0] == expected_first
    assert process_action_label_upstream(labels, start=0)[0] == 2  # the honest step


# --------------------------------------------------------------------------
# 4. temporal aggregation: ring buffer == upstream's dense table
# --------------------------------------------------------------------------
@pytest.mark.parametrize("chunk_size,n_steps,n_samples", [(4, 12, 3), (8, 9, 2), (5, 5, 4)])
def test_temporal_buffer_pools_the_same_samples_as_upstream(chunk_size, n_steps, n_samples):
    """Upstream keeps a (T, T+chunk, S, D) tensor and slices column t.

    ``all_time_samples[[t], t : t + num_queries] = action_samples`` then
    ``samples_for_curr_step = all_time_samples[:, t]`` filtered to the rows that
    were written. Our ring buffer must pool exactly that set, in that order.
    """
    dim = 2
    g = torch.Generator().manual_seed(0)
    chunks = [torch.randn(n_samples, chunk_size, dim, generator=g) for _ in range(n_steps)]

    dense = torch.zeros(n_steps, n_steps + chunk_size, n_samples, dim)
    written = torch.zeros(n_steps, n_steps + chunk_size, dtype=torch.bool)
    buffer = TemporalSampleBuffer(chunk_size)

    for t, chunk in enumerate(chunks):
        # upstream stores (chunk_len, num_samples, dim) at [t, t : t+chunk]
        dense[t, t : t + chunk_size] = chunk.permute(1, 0, 2)
        written[t, t : t + chunk_size] = True
        buffer.add(chunk)

        rows = written[:, t]
        theirs = dense[:, t][rows].flatten(0, 1)   # (n_pred * S, D)
        ours = buffer.current()
        assert ours.shape == theirs.shape, f"t={t}: {tuple(ours.shape)} vs {tuple(theirs.shape)}"
        # upstream pools oldest-prediction-first; ours is newest-first. The KDE
        # estimator is permutation invariant, so compare as sets of rows.
        assert torch.allclose(
            ours.sort(dim=0).values, theirs.sort(dim=0).values, atol=1e-6
        ), f"t={t}: different sample set"
        assert torch.allclose(kde_entropy(ours.unsqueeze(0)), kde_entropy(theirs.unsqueeze(0)))


# --------------------------------------------------------------------------
# 5. the latent the samples are drawn from
# --------------------------------------------------------------------------
def test_prior_latent_matches_upstream_reparametrisation():
    """Upstream's ``get_samples`` draws z ~ N(0, I); so must our patch.

    ``DETRVAE.get_samples`` sets ``mu = logvar = 0`` and calls
    ``reparametrize_n(mu, logvar.div(2).exp(), n)``. With logvar 0 the scale is
    ``exp(0) == 1`` exactly, so the draw is the standard normal -- which is what
    ``_PriorLatentProj`` substitutes. Checked here on upstream's own function.
    """
    from torch.autograd import Variable

    source = (Path(UPSTREAM) / "aloha/act/detr/models/detr_vae.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "reparametrize_n")
    ns = {"torch": torch, "Variable": Variable}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<u>", "exec"), ns)  # noqa: S102

    latent_dim, n = 32, 20000
    mu = logvar = torch.zeros(1, latent_dim)
    std = logvar.div(2).exp()
    assert torch.equal(std, torch.ones_like(std)), "logvar=0 must give a unit scale"

    torch.manual_seed(0)
    theirs = ns["reparametrize_n"](mu, std, n).reshape(n, latent_dim)
    torch.manual_seed(0)
    ours = torch.randn(n, latent_dim)

    for sample in (theirs, ours):
        assert abs(float(sample.mean())) < 0.02
        assert abs(float(sample.std()) - 1.0) < 0.02
    # identical construction from the same generator state, not merely the same law
    assert torch.allclose(theirs, ours, atol=1e-6)


# --------------------------------------------------------------------------
# 6. the precision / non-precision merge rule
# --------------------------------------------------------------------------
def _upstream_merge(entropy_z: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    """Upstream's ``hdbscan_with_custom_merge`` with the clustering held fixed.

    The clusterer is stubbed to return ``cluster_labels`` so the test compares
    the *merge rule* -- the part this port re-implements -- rather than two
    different HDBSCAN builds.
    """
    class _Stub:
        def __init__(self, *a, **k):
            self.labels_ = np.asarray(cluster_labels).copy()

        def fit(self, X):
            return self

    source = (Path(UPSTREAM) / "aloha/act/imitate_episodes.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "hdbscan_with_custom_merge")
    ns = {"np": np, "hdbscan": type("m", (), {"HDBSCAN": _Stub})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<u>", "exec"), ns)  # noqa: S102
    X = np.stack([np.arange(len(entropy_z), dtype=float), entropy_z], axis=-1)
    return ns["hdbscan_with_custom_merge"](X, "", 0, plot=False)


def _our_merge(entropy_z: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    """The same assembly step this port performs in ``segment_entropy``."""
    from demospeedup_core.segmentation import NON_PRECISION, _cluster_vote

    clusters = np.asarray(cluster_labels).copy()
    clusters[:50] = -1  # upstream's fixed prefix, reproduced by fast_prefix=50
    labels = np.full(len(entropy_z), NON_PRECISION, dtype=np.int64)
    for cid in np.unique(clusters[clusters >= 0]):
        mask = clusters == cid
        labels[mask] = _cluster_vote(entropy_z, mask)
    return labels


@pytest.mark.parametrize("seed", range(6))
def test_merge_rule_matches_upstream(seed):
    """Cluster -> {precision, non-precision} must reproduce upstream exactly."""
    rng = np.random.default_rng(seed)
    n = 200
    entropy_z = rng.normal(size=n)
    clusters = rng.integers(-1, 4, size=n)  # -1 is HDBSCAN's noise label
    assert np.array_equal(_our_merge(entropy_z, clusters),
                          _upstream_merge(entropy_z, clusters))


def test_merge_rule_marks_a_straddling_cluster_as_precision():
    """Upstream's quirk: ``np.mean(points < 0)`` is truthy if *any* point is.

    One below-mean frame is enough to make an otherwise high-entropy cluster
    'precision'. Verified against upstream rather than merely asserted.
    """
    n = 200
    entropy_z = np.full(n, 1.0)
    clusters = np.zeros(n, dtype=np.int64)
    entropy_z[120] = -0.01  # a single below-mean frame in the one cluster
    ours, theirs = _our_merge(entropy_z, clusters), _upstream_merge(entropy_z, clusters)
    assert np.array_equal(ours, theirs)
    assert ours[60] == 0  # precision, decided by that lone frame


# --------------------------------------------------------------------------
# 7. the k-NN estimator's sign, against analytic ground truth
# --------------------------------------------------------------------------
def test_knn_entropy_obeys_the_scaling_law():
    """Differential entropy must satisfy H(cX) = H(X) + d log c.

    This is the check that settles the sign. Upstream writes
    ``digamma_n - digamma_k - dim * log(...)``, which gives H(cX) = H(X) - d log c:
    a *wider* cloud would score as *lower* entropy. Our corrected ``+`` recovers
    the law. Nothing in the pipeline uses this estimator -- DemoSpeedup labels
    with the KDE one, which is reproduced verbatim -- but an inverted
    cross-check is worse than none.
    """
    from demospeedup_core.entropy import k_nn_distance, kozachenko_leonenko_entropy

    torch.manual_seed(0)
    x = torch.randn(1, 400, 5)
    c, d, k = 3.0, 5, 5
    ours = float(kozachenko_leonenko_entropy(x, k))
    ours_scaled = float(kozachenko_leonenko_entropy(x * c, k))
    assert ours_scaled - ours == pytest.approx(d * np.log(c), abs=1e-3)

    # upstream's expression, inlined, fails the same law by exactly -2 d log c
    from scipy.special import digamma

    def upstream(sample):
        _, num_samples, dim = sample.size()
        avg = k_nn_distance(sample, k).mean(dim=2)
        return (torch.tensor(digamma(num_samples)) - torch.tensor(digamma(k))
                - dim * torch.log(avg).mean(dim=1, keepdim=True))

    theirs = float(upstream(x))
    theirs_scaled = float(upstream(x * c))
    assert theirs_scaled - theirs == pytest.approx(-d * np.log(c), abs=1e-3)
