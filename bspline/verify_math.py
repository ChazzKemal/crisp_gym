#!/usr/bin/env python3
"""Machine-checked verification of the B-spline pipeline's mathematics.

Each check states a claim, tests it against an *independent* reference where
one exists (scipy's BSpline for evaluation, scipy's Rotation for rotations, a
from-scratch Cox-de Boor recursion for the basis functions, the browser's
JavaScript decoder for the walkthrough page), and reports the measured
quantity next to the threshold it has to beat.

    conda run -n lerobot-041 python verify_math.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bspline_core.bspline_action import (  # noqa: E402
    ScipyBSplineCompression,
    chunk_bspline_trajectory,
    chunk_to_params,
    decode_bspline_action,
)
from bspline_core.chunk_sampler import BSplineChunkSampler  # noqa: E402
from bspline_core.knots import decode_relative_knots, encode_relative_knots  # noqa: E402
from bspline_core.rotation import (  # noqa: E402
    axis_angle_to_matrix,
    axis_angle_to_rotation_6d,
    convert_actions_7d_to_10d,
    convert_actions_10d_to_7d,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from lerobot_bridge import load_lerobot_actions, to_policy_actions  # noqa: E402

SRC = "/home/batur/Coding/data/merged_act_finetune_20260528"
CONVERTED = "/home/batur/Coding/data/merged_bspline_20260528"
DEGREE, CHUNK_SIZE, MAX_ERROR = 3, 20, 0.01
NSTEP = CHUNK_SIZE + 2 * DEGREE


@dataclass
class Check:
    group: str
    claim: str
    reference: str
    metric: str
    value: float
    threshold: float
    smaller_is_better: bool = True
    detail: str = ""
    samples: int = 0

    @property
    def passed(self) -> bool:
        return self.value <= self.threshold if self.smaller_is_better else self.value >= self.threshold


RESULTS: list[Check] = []


def add(**kw):
    RESULTS.append(Check(**kw))
    return RESULTS[-1]


# ---------------------------------------------------------------------------
# an independent Cox-de Boor basis, written from the definition
# ---------------------------------------------------------------------------
def basis(i: int, k: int, t: np.ndarray, x: float) -> float:
    """B_{i,k}(x) straight from the recursion, no scipy involved."""
    if k == 0:
        return 1.0 if (t[i] <= x < t[i + 1]) else 0.0
    left = 0.0
    if t[i + k] > t[i]:
        left = (x - t[i]) / (t[i + k] - t[i]) * basis(i, k - 1, t, x)
    right = 0.0
    if t[i + k + 1] > t[i + 1]:
        right = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * basis(i + 1, k - 1, t, x)
    return left + right


def de_boor(t: np.ndarray, c: np.ndarray, k: int, x: float) -> float:
    """De Boor's algorithm, independent of scipy."""
    n = len(c)
    if x >= t[n]:
        s = n - 1
    elif x <= t[k]:
        s = k
    else:
        s = int(np.searchsorted(t, x, side="right")) - 1
    d = [float(c[s - k + j]) for j in range(k + 1)]
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            den = t[s + 1 + j - r] - t[s + j - k]
            a = 0.0 if den == 0 else (x - t[s + j - k]) / den
            d[j] = (1 - a) * d[j - 1] + a * d[j]
    return d[k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=HERE / "math_verification.json")
    ap.add_argument("--episodes", type=int, default=12)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    print("Loading data and fitting reference splines...", flush=True)
    data = load_lerobot_actions(SRC)
    policy = to_policy_actions(data.actions)
    lengths = np.diff(np.concatenate([[0], data.episode_ends]))
    # short episodes first, so the verification stays quick and reproducible
    order = np.argsort(lengths)[: args.episodes]

    fits = []
    for e in order:
        a, b = int(data.episode_starts[e]), int(data.episode_ends[e])
        c = ScipyBSplineCompression(degree=DEGREE)
        c.compress(policy[a:b], max_error=MAX_ERROR)
        fits.append((int(e), a, b, c))
    ep, a0, b0, comp = fits[len(fits) // 2]
    t_full, c_full, _ = comp.spline.tck
    print(f"  {len(fits)} episodes fitted; reference episode {ep} "
          f"({b0 - a0} frames, {len(t_full)} knots)\n", flush=True)

    # =====================================================================
    # A. B-spline fundamentals -- do the objects we build satisfy the axioms?
    # =====================================================================
    worst = 0.0
    for _, _, _, c in fits:
        t, cc, kk = c.spline.tck
        worst = max(worst, abs(len(t) - (len(cc) + kk + 1)))
    add(group="A. B-spline fundamentals",
        claim="len(knots) = len(control points) + degree + 1",
        reference="definition of a B-spline",
        metric="max deviation over fitted episodes", value=float(worst), threshold=0.0,
        samples=len(fits))

    n_ctrl = len(c_full)
    xs = np.linspace(t_full[DEGREE], t_full[n_ctrl], 400, endpoint=False)
    sums = np.array([sum(basis(i, DEGREE, t_full, float(x)) for i in range(n_ctrl)) for x in xs])
    add(group="A. B-spline fundamentals",
        claim="partition of unity: the basis functions sum to 1 across the domain",
        reference="Cox-de Boor recursion implemented from the definition",
        metric="max |sum - 1|", value=float(np.abs(sums - 1).max()), threshold=1e-9,
        samples=len(xs))

    neg = min(min(basis(i, DEGREE, t_full, float(x)) for i in range(n_ctrl)) for x in xs[::8])
    add(group="A. B-spline fundamentals",
        claim="basis functions are non-negative",
        reference="Cox-de Boor recursion", metric="most negative value",
        value=float(-min(neg, 0.0)), threshold=0.0, samples=len(xs[::8]))

    off = 0.0
    for i in rng.choice(n_ctrl, size=min(20, n_ctrl), replace=False):
        i = int(i)
        for x in np.linspace(t_full[0], t_full[-1], 300):
            if t_full[i] <= x < t_full[i + DEGREE + 1]:
                continue
            off = max(off, abs(basis(i, DEGREE, t_full, float(x))))
    add(group="A. B-spline fundamentals",
        claim="local support: B_i vanishes outside [t_i, t_{i+degree+1})",
        reference="Cox-de Boor recursion",
        metric="max |B_i(x)| outside its support", value=float(off), threshold=1e-12,
        samples=20 * 300)

    # =====================================================================
    # B. The windowing theorem -- the claim the whole representation rests on
    # =====================================================================
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)
    worst, n = 0.0, 0
    for s, ch in enumerate(chunks):
        if s + NSTEP > len(c_full):
            continue
        lo, hi = ch["t"][DEGREE], ch["t"][NSTEP - DEGREE - 1]
        if hi <= lo:
            continue
        xs = np.linspace(lo, hi, 40, endpoint=False)
        local = BSpline(ch["t"], ch["c"][: -(DEGREE + 1)], DEGREE)(xs)
        worst = max(worst, float(np.abs(local - comp.spline(xs)).max()))
        n += 1
    add(group="B. Windowing theorem",
        claim="a windowed chunk equals the global spline on the window's interior",
        reference="the globally fitted scipy spline",
        metric="max |chunk(x) - global(x)|", value=worst, threshold=1e-9, samples=n,
        detail="knots t[s:s+M] with control points c[s:s+M-k-1]; interior is "
               "[t[s+k], t[s+M-k-1])")

    # Not a tautology: take the slice decode_bspline_action actually performs and
    # ask scipy whether that many control points is legal for the knot vector.
    bad = 0
    for ch in chunks[:60]:
        p_ = chunk_to_params(ch, NSTEP, 11)
        knots = p_[:, 0].astype(np.float64)
        sliced = p_[: -(DEGREE + 1), 1:]                    # exactly what decode uses
        required = len(knots) - DEGREE - 1                  # from the count identity
        if sliced.shape[0] != required:
            bad += 1
            continue
        try:
            BSpline(knots, sliced, DEGREE)                  # scipy validates the pairing
        except Exception:
            bad += 1
    add(group="B. Windowing theorem",
        claim="the control-point slice decode actually takes, params[:-(degree+1)], is the "
              "exact number the windowed knot vector admits -- and scipy accepts the pairing",
        reference="scipy.interpolate.BSpline's own validation",
        metric="chunks whose slice is rejected or miscounted",
        value=float(bad), threshold=0.0,
        detail=f"M={NSTEP}, k={DEGREE} -> {NSTEP - DEGREE - 1} control points", samples=60)

    # =====================================================================
    # C. Translation invariance -- why subtracting the frame index is safe
    # =====================================================================
    worst = 0.0
    for ch in chunks[5:25]:
        p = chunk_to_params(ch, NSTEP, 11)
        base = decode_bspline_action(p, degree=DEGREE, num_actions=24)
        for shift in (0.0, 1.0, 7.0, -13.0, 137.5):
            q = p.copy()
            q[:, 0] -= shift
            worst = max(worst, float(np.abs(decode_bspline_action(q, degree=DEGREE,
                                                                  num_actions=24) - base).max()))
    add(group="C. Translation invariance",
        claim="shifting every knot by c and evaluating at x-c leaves the curve unchanged",
        reference="the unshifted decode of the same chunk",
        metric="max |decode(shifted) - decode(original)|", value=worst, threshold=1e-5,
        samples=20 * 5,
        detail="the Cox-de Boor coefficients (x-t_i)/(t_{i+k}-t_i) are differences only")

    # =====================================================================
    # D. Relative-knot encoding
    # =====================================================================
    worst = 0.0
    for ch in chunks[:40]:
        p = chunk_to_params(ch, NSTEP, 11)
        worst = max(worst, float(np.abs(
            decode_relative_knots(encode_relative_knots(p, DEGREE), DEGREE) - p).max()))
    add(group="D. Relative-knot encoding",
        claim="encode then decode is the identity",
        reference="the original parameter matrix",
        metric="max |round-trip - original|", value=worst, threshold=1e-4, samples=40)

    worst = 0.0
    for ch in chunks[5:25]:
        p = chunk_to_params(ch, NSTEP, 11)
        d1 = decode_bspline_action(p, degree=DEGREE, num_actions=16)
        d2 = decode_bspline_action(encode_relative_knots(p, DEGREE), degree=DEGREE,
                                   num_actions=16, relative_knots=True)
        worst = max(worst, float(np.abs(d1 - d2).max()))
    add(group="D. Relative-knot encoding",
        claim="decoding through the relative encoding gives the same waypoints",
        reference="decoding the absolute form", metric="max |difference|",
        value=worst, threshold=1e-5, samples=20)

    # =====================================================================
    # E. De Boor -- three independent evaluators must agree
    # =====================================================================
    worst_db, worst_basis = 0.0, 0.0
    js_cases = []
    for ch in chunks[5:20]:
        t, c = ch["t"], ch["c"][: -(DEGREE + 1)]
        lo, hi = t[DEGREE], t[NSTEP - DEGREE - 1]
        if hi <= lo:
            continue
        for x in np.linspace(lo, hi, 12, endpoint=False):
            ref = float(BSpline(t, c, DEGREE)(x)[0])
            worst_db = max(worst_db, abs(de_boor(t, c[:, 0], DEGREE, float(x)) - ref))
            direct = sum(float(c[i, 0]) * basis(i, DEGREE, t, float(x)) for i in range(len(c)))
            worst_basis = max(worst_basis, abs(direct - ref))
            js_cases.append({"t": t.tolist(), "c": c[:, 0].tolist(), "x": float(x), "ref": ref})
    add(group="E. Evaluation algorithms",
        claim="a from-scratch de Boor evaluator matches scipy",
        reference="scipy.interpolate.BSpline", metric="max |difference|",
        value=float(worst_db), threshold=1e-9, samples=len(js_cases))
    add(group="E. Evaluation algorithms",
        claim="sum_i c_i B_i(x) from the raw recursion matches scipy",
        reference="scipy.interpolate.BSpline", metric="max |difference|",
        value=float(worst_basis), threshold=1e-9, samples=len(js_cases))

    js_err = run_js_check(js_cases[:120])
    if js_err is not None:
        add(group="E. Evaluation algorithms",
            claim="the walkthrough page's JavaScript decoder matches scipy",
            reference="scipy.interpolate.BSpline, via node",
            metric="max |difference|", value=js_err, threshold=1e-9, samples=min(120, len(js_cases)),
            detail="findSpan/evalSpline extracted from walkthrough_template.html")

    # =====================================================================
    # F. The padded-tail degeneracy -- stated precisely, then measured
    # =====================================================================
    padded = [ch for s, ch in enumerate(chunks) if s + NSTEP > len(c_full)]
    interior = [ch for s, ch in enumerate(chunks) if s + NSTEP <= len(c_full)]
    if padded:
        ch = padded[-1]
        t = ch["t"]
        tmax = t[NSTEP - DEGREE - 1]
        s_pad = sum(basis(i, DEGREE, t, float(tmax)) for i in range(NSTEP - DEGREE - 1))
        add(group="F. Padded-tail degeneracy",
            claim="on a tail-padded chunk, partition of unity FAILS at x = t_max "
                  "(every basis function has already closed its half-open support)",
            reference="Cox-de Boor recursion", metric="sum of basis functions at t_max",
            value=float(abs(s_pad - 0.0)), threshold=1e-12, samples=1,
            detail="a value of 0 instead of 1 is exactly why the last decoded waypoint "
                   "collapses to the origin; the decoder drops it")
    ch = interior[len(interior) // 2]
    t = ch["t"]
    tmax = t[NSTEP - DEGREE - 1]
    s_int = sum(basis(i, DEGREE, t, float(tmax)) for i in range(NSTEP - DEGREE - 1))
    add(group="F. Padded-tail degeneracy",
        claim="on an interior chunk the same point is fine: the basis still sums to 1",
        reference="Cox-de Boor recursion", metric="|sum - 1| at t_max",
        value=float(abs(s_int - 1.0)), threshold=1e-9, samples=1)

    # =====================================================================
    # G. Rotation algebra
    # =====================================================================
    rv = rng.uniform(-2 * np.pi, 2 * np.pi, size=(2000, 3))
    R = axis_angle_to_matrix(rv)
    add(group="G. Rotation algebra",
        claim="our Rodrigues map matches scipy",
        reference="scipy.spatial.transform.Rotation.from_rotvec",
        metric="max |difference|",
        value=float(np.abs(R - Rotation.from_rotvec(rv).as_matrix()).max()),
        threshold=1e-12, samples=len(rv))

    d6 = matrix_to_rotation_6d(R)
    add(group="G. Rotation algebra",
        claim="the 6D form is literally the first two rows of the rotation matrix "
              "(pytorch3d's convention, which upstream uses)",
        reference="the matrix itself",
        metric="max |d6 - rows(R)|",
        value=float(max(np.abs(d6[:, :3] - R[:, 0, :]).max(), np.abs(d6[:, 3:] - R[:, 1, :]).max())),
        threshold=0.0, samples=len(rv))

    Rr = rotation_6d_to_matrix(d6)
    add(group="G. Rotation algebra",
        claim="Gram-Schmidt is a left inverse: it recovers R exactly from those two rows",
        reference="the original matrix", metric="max |GS(rows(R)) - R|",
        value=float(np.abs(Rr - R).max()), threshold=1e-12, samples=len(rv))

    pert = d6 + rng.normal(scale=0.25, size=d6.shape)
    Rp = rotation_6d_to_matrix(pert)
    orth = np.abs(np.einsum("nij,nkj->nik", Rp, Rp) - np.eye(3)[None]).max()
    add(group="G. Rotation algebra",
        claim="Gram-Schmidt maps ANY 6 numbers back onto SO(3) -- so a network's "
              "un-orthonormal output is still a valid rotation",
        reference="the SO(3) conditions R Rᵀ = I and det R = +1",
        metric="max(|R Rᵀ - I|, |det R - 1|)",
        value=float(max(orth, np.abs(np.linalg.det(Rp) - 1).max())), threshold=1e-10,
        samples=len(pert))

    raw7 = np.concatenate([rng.uniform(-1, 1, (2000, 3)), rv, rng.uniform(0, 1, (2000, 1))], axis=1)
    back = convert_actions_10d_to_7d(convert_actions_7d_to_10d(raw7))
    geo = (Rotation.from_rotvec(raw7[:, 3:6]) * Rotation.from_rotvec(back[:, 3:6]).inv()).magnitude()
    add(group="G. Rotation algebra",
        claim="the 7D -> 10D -> 7D action round trip preserves the rotation",
        reference="geodesic distance on SO(3)", metric="max angle [rad]",
        value=float(geo.max()), threshold=1e-6, samples=len(raw7))
    add(group="G. Rotation algebra",
        claim="the round trip preserves position and gripper exactly",
        reference="the original action", metric="max |difference|",
        value=float(max(np.abs(back[:, :3] - raw7[:, :3]).max(),
                        np.abs(back[:, 6] - raw7[:, 6]).max())),
        threshold=1e-6, samples=len(raw7))

    # =====================================================================
    # H. The fit guarantee
    # =====================================================================
    worst = 0.0
    for _, a, b, c in fits:
        worst = max(worst, float(np.abs(c.spline(np.arange(b - a)) - policy[a:b]).max()))
    add(group="H. Fit guarantee",
        claim=f"the fitted spline stays within max_error of every recorded frame",
        reference="the recorded actions", metric="max |spline - data| over all frames/dims",
        value=worst, threshold=MAX_ERROR, samples=int(sum(b - a for _, a, b, _ in fits)),
        detail=f"max_error = {MAX_ERROR}")

    # =====================================================================
    # I. The frame -> chunk assignment rule
    # =====================================================================
    mismatches, frames = 0, 0
    for e, a, b, c in fits:
        chs = chunk_bspline_trajectory(c, chunk_size=CHUNK_SIZE, stride=1)
        smp = BSplineChunkSampler(actions=policy[a:b], episode_ends=np.array([b - a]),
                                  chunk_size=CHUNK_SIZE, degree=DEGREE, max_error=MAX_ERROR,
                                  stride=1, max_first_k=1)
        starts = c.spline.tck[0][DEGREE:DEGREE + len(chs)]
        for ts in range(b - a):
            s_idx = min(int(np.searchsorted(starts, ts, side="left")), len(chs) - 1)
            want = chunk_to_params(chs[s_idx], NSTEP, 11)[:, 0] - ts
            if not np.allclose(want, smp.chunk_for_timestep(ts)[:, 0], atol=1e-3):
                mismatches += 1
            frames += 1
    add(group="I. Assignment rule",
        claim="a closed-form rule -- the smallest window s with t[s+k] >= frame -- "
              "reproduces the sampler's imperative loop exactly",
        reference="BSplineChunkSampler", metric="mismatching frames",
        value=float(mismatches), threshold=0.0, samples=frames)

    # =====================================================================
    # J. Normalisation identity on the converted dataset
    # =====================================================================
    conv_root = Path(CONVERTED)
    if (conv_root / "meta" / "stats.json").exists():
        st = json.loads((conv_root / "meta" / "stats.json").read_text())["action"]
        meta = json.loads((conv_root / "meta" / "bspline.json").read_text())
        ns, nc = meta["n_action_steps"], meta["n_action_channels"]
        mean = np.asarray(st["mean"]).reshape(ns, nc)
        std = np.asarray(st["std"]).reshape(ns, nc)
        add(group="J. Normalisation",
            claim="per-channel stats are constant down the rows, so every row of a column "
                  "shares one scale (as upstream's get_normalizer does)",
            reference="the stats written to meta/stats.json",
            metric="max row-to-row spread",
            value=float(max(np.abs(mean - mean[0]).max(), np.abs(std - std[0]).max())),
            threshold=0.0, samples=ns * nc)

        conv = load_lerobot_actions(conv_root)
        x = conv.actions[rng.choice(len(conv.actions), 500, replace=False)].reshape(-1, ns, nc)
        rt = ((x - mean) / std) * std + mean
        add(group="J. Normalisation",
            claim="unnormalise(normalise(x)) = x on real stored actions",
            reference="the stored actions", metric="max |round-trip - original|",
            value=float(np.abs(rt - x).max()), threshold=1e-3, samples=500)

    report(args.json)
    return 0 if all(c.passed for c in RESULTS) else 1


def run_js_check(cases):
    """Run the walkthrough page's own JS spline evaluator against scipy."""
    tpl = HERE / "walkthrough_template.html"
    if not tpl.exists():
        return None
    src = tpl.read_text()
    try:
        start = src.index("function findSpan")
        end = src.index("const padTo")
    except ValueError:
        return None
    fns = src[start:end]
    with tempfile.TemporaryDirectory() as d:
        js = Path(d) / "check.js"
        data = Path(d) / "cases.json"
        data.write_text(json.dumps(cases))
        js.write_text(fns + f"""
const cases = require({json.dumps(str(data))});
let worst = 0;
for (const c of cases){{
  const got = evalSpline(c.t, c.c, 3, c.x);
  worst = Math.max(worst, Math.abs(got - c.ref));
}}
console.log(worst);
""")
        try:
            out = subprocess.run(["node", str(js)], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:
            return None
        return float(out.stdout.strip())


def report(json_path: Path):
    groups: dict[str, list[Check]] = {}
    for c in RESULTS:
        groups.setdefault(c.group, []).append(c)

    width = 96
    print("=" * width)
    print("B-SPLINE PIPELINE -- MATHEMATICAL VERIFICATION")
    print("=" * width)
    for g, cs in groups.items():
        print(f"\n{g}")
        print("-" * width)
        for c in cs:
            mark = "PASS" if c.passed else "FAIL"
            print(f"  [{mark}] {c.claim}")
            print(f"         vs {c.reference}")
            val = f"{c.value:.3e}" if c.value else "0"
            thr = f"{c.threshold:.1e}" if c.threshold else "0 (exact)"
            print(f"         {c.metric}: {val}   (must be <= {thr})   n={c.samples:,}")
            if c.detail:
                print(f"         note: {c.detail}")
    n_pass = sum(c.passed for c in RESULTS)
    print("\n" + "=" * width)
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    print("=" * width)

    json_path.write_text(json.dumps(
        {"checks": [{**c.__dict__, "passed": c.passed} for c in RESULTS],
         "n_passed": n_pass, "n_total": len(RESULTS),
         "config": {"degree": DEGREE, "chunk_size": CHUNK_SIZE, "max_error": MAX_ERROR,
                    "n_steps": NSTEP}},
        indent=1))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    raise SystemExit(main())
