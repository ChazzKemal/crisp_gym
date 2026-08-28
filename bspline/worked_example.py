#!/usr/bin/env python3
"""A fully worked numerical example of the B-spline action pipeline.

Carries one real recorded episode through every stage, printing actual numbers
and shapes at each step, plus a hand-checkable de Boor evaluation small enough
to verify with a pencil.

    conda run -n lerobot-041 python worked_example.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bspline_core.bspline_action import (  # noqa: E402
    ScipyBSplineCompression, chunk_bspline_trajectory, chunk_to_params, decode_bspline_action)
from bspline_core.rotation import (  # noqa: E402
    axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix,
    convert_actions_7d_to_10d, convert_actions_10d_to_7d)
from lerobot_bridge import load_lerobot_actions, to_policy_actions  # noqa: E402

SRC = "/home/batur/Coding/data/merged_act_finetune_20260528"
EPISODE, DEGREE, CHUNK_SIZE, MAX_ERROR = 58, 3, 20, 0.01
NSTEP = CHUNK_SIZE + 2 * DEGREE
WINDOW = 60          # the chunk we follow all the way through
FRAME = 100          # the frame we localise it to

E = {}               # exhibits, dumped to JSON for the report page


def r(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=HERE / "worked_example.json")
    args = ap.parse_args()
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    # =====================================================================
    rule("STEP 0 — what one recorded frame is")
    data = load_lerobot_actions(SRC)
    a, b = int(data.episode_starts[EPISODE]), int(data.episode_ends[EPISODE])
    raw = data.actions[a:b]
    print(f"episode {EPISODE}:  raw actions shape = {raw.shape}   (frames, channels)")
    print(f"                  {b - a} frames at {data.fps} Hz = {(b - a) / data.fps:.1f} s")
    print("\ncolumns:  [ x  y  z | rx ry rz | gripper ]   metres, axis-angle radians, 0/1")
    print("\nfirst three frames:")
    for i in range(3):
        print(f"  frame {i}: {raw[i]}")
    E["step0"] = {"shape": list(raw.shape), "fps": data.fps,
                  "rows": [r(raw[i], 5) for i in range(3)],
                  "seconds": (b - a) / data.fps}

    # =====================================================================
    rule("STEP 1 — one rotation, converted (7 numbers -> 10)")
    v = raw[FRAME, 3:6].astype(np.float64)
    R = axis_angle_to_matrix(v)
    d6 = matrix_to_rotation_6d(R)
    print(f"axis-angle at frame {FRAME}:  {v}     (shape {v.shape})")
    print(f"  angle = |v| = {np.linalg.norm(v):.4f} rad = {np.degrees(np.linalg.norm(v)):.1f}°")
    print(f"  axis  = v/|v| = {v / np.linalg.norm(v)}")
    print(f"\nrotation matrix R = exp([v]x)   (shape {R.shape}):")
    for row in R:
        print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
    print(f"\nrot6d = first two ROWS of R, flattened   (shape {d6.shape}):")
    print(f"    {d6}")
    print(f"    R[0,:] = {R[0]}   -> d6[0:3]")
    print(f"    R[1,:] = {R[1]}   -> d6[3:6]")
    Rb = rotation_6d_to_matrix(d6)
    vb = Rotation.from_matrix(Rb).as_rotvec()
    ang = (Rotation.from_rotvec(v) * Rotation.from_rotvec(vb).inv()).magnitude()
    print(f"\nback:  Gram-Schmidt(d6) -> R,  max|R_back - R| = {np.abs(Rb - R).max():.2e}")
    print(f"       R -> axis-angle = {vb}")
    print(f"       geodesic error  = {ang:.2e} rad")
    print(f"\nthe third row is NOT stored — it is recovered as b1 x b2:")
    print(f"    R[2,:]      = {R[2]}")
    print(f"    b1 x b2     = {np.cross(Rb[0], Rb[1])}")
    print(f"\nso the action row grows:  7 -> 10")
    print(f"    raw   (7,)  = {raw[FRAME]}")
    print(f"    policy(10,) = {convert_actions_7d_to_10d(raw[FRAME:FRAME + 1])[0]}")
    E["step1"] = {
        "axis_angle": r(v, 5), "angle_rad": float(np.linalg.norm(v)),
        "angle_deg": float(np.degrees(np.linalg.norm(v))), "axis": r(v / np.linalg.norm(v), 5),
        "R": [r(row, 4) for row in R], "d6": r(d6, 4),
        "third_row": r(R[2], 4), "cross": r(np.cross(Rb[0], Rb[1]), 4),
        "roundtrip_matrix_err": float(np.abs(Rb - R).max()), "geodesic_err": float(ang),
        "raw7": r(raw[FRAME], 5),
        "policy10": r(convert_actions_7d_to_10d(raw[FRAME:FRAME + 1])[0], 5)}

    policy = to_policy_actions(raw)
    print(f"\nwhole episode: {raw.shape} -> {policy.shape}")

    # =====================================================================
    rule("STEP 2 — fit ONE spline to the whole episode")
    comp = ScipyBSplineCompression(degree=DEGREE)
    comp.compress(policy, max_error=MAX_ERROR)
    t_full, c_full, k = comp.spline.tck
    print(f"input   : {policy.shape} sampled at t = 0,1,...,{len(policy) - 1}")
    print(f"degree  : {k}")
    print(f"knots   : shape {t_full.shape}")
    print(f"coeffs  : shape {c_full.shape}   (one control point per row, 10 dims each)")
    print(f"\nthe counting identity:  len(t) = len(c) + k + 1")
    print(f"                        {len(t_full)} = {len(c_full)} + {k} + 1"
          f"  -> {len(c_full) + k + 1}   {'OK' if len(t_full) == len(c_full) + k + 1 else 'MISMATCH'}")
    print(f"\nfirst 12 knots: {t_full[:12]}")
    print(f"   note t[0..3] are all 0 — the boundary knot is repeated k+1 = {k + 1} times")
    print(f"last  12 knots: {t_full[-12:]}")
    d = np.diff(t_full[DEGREE:-DEGREE])
    d = d[d > 0]
    print(f"\nknot spacing (interior, frames):  min {d.min():.2f}   "
          f"median {np.median(d):.2f}   mean {d.mean():.2f}   max {d.max():.2f}")
    print("   -> spacing is NOT uniform: that is the adaptivity the method trades on")
    fit = comp.spline(np.arange(len(policy)))
    err = np.abs(fit - policy)
    print(f"\ncompression: {len(t_full)} knots for {len(policy)} frames "
          f"= {len(t_full) / len(policy):.3f} knots/frame")
    print(f"fit error  : max over all {policy.shape[0]}x{policy.shape[1]} values = {err.max():.5f}"
          f"   (tolerance {MAX_ERROR})")
    print(f"             position only: {err[:, :3].max() * 1000:.2f} mm")
    E["step2"] = {"knots_shape": list(t_full.shape), "coef_shape": list(c_full.shape),
                  "degree": int(k), "n_frames": int(len(policy)),
                  "identity": [int(len(t_full)), int(len(c_full)), int(k)],
                  "first_knots": r(t_full[:12], 3), "last_knots": r(t_full[-12:], 3),
                  "spacing": {"min": float(d.min()), "median": float(np.median(d)),
                              "mean": float(d.mean()), "max": float(d.max())},
                  "knots_per_frame": float(len(t_full) / len(policy)),
                  "fit_err": float(err.max()), "pos_err_mm": float(err[:, :3].max() * 1000)}

    # =====================================================================
    rule(f"STEP 3 — slice out ONE chunk (window s = {WINDOW})")
    M = NSTEP
    kt = t_full[WINDOW:WINDOW + M]
    kc = c_full[WINDOW:WINDOW + M]
    print(f"take M = chunk_size + 2*degree = {CHUNK_SIZE} + 2*{DEGREE} = {M} consecutive knots,")
    print(f"and the control points with the SAME indices:")
    print(f"    knots  = t_full[{WINDOW}:{WINDOW + M}]  -> shape {kt.shape}")
    print(f"    coeffs = c_full[{WINDOW}:{WINDOW + M}]  -> shape {kc.shape}")
    print(f"\nthe 26 knots (in frames):\n    {kt}")
    t_min, t_max = kt[DEGREE], kt[M - DEGREE - 1]
    print(f"\nvalid domain = [ knots[{DEGREE}], knots[{M - DEGREE - 1}] ) "
          f"= [{t_min:.2f}, {t_max:.2f})")
    print(f"    span = {t_max - t_min:.2f} frames = {(t_max - t_min) / data.fps:.2f} s")
    print(f"    the {DEGREE} knots at each end are the overhang the cubic basis needs")
    print(f"\ndecode uses params[:-(degree+1)] control points = first {M - DEGREE - 1} rows")
    print(f"    and a knot vector of length {M} at degree {DEGREE} admits exactly")
    print(f"    {M} - {DEGREE} - 1 = {M - DEGREE - 1} control points.  Same number.")
    xs = np.linspace(t_min, t_max, 5, endpoint=False)
    loc = BSpline(kt, kc[: -(DEGREE + 1)], DEGREE)(xs)
    glo = comp.spline(xs)
    print(f"\nlocal-support check — windowed spline vs the global one, z channel (index 2):")
    print(f"    {'x (frame)':>10} {'windowed':>12} {'global':>12} {'difference':>12}")
    for x, l, g in zip(xs, loc[:, 2], glo[:, 2]):
        print(f"    {x:10.3f} {l:12.6f} {g:12.6f} {abs(l - g):12.2e}")
    E["step3"] = {"M": M, "window": WINDOW, "knots": r(kt, 3),
                  "knots_shape": list(kt.shape), "coef_shape": list(kc.shape),
                  "t_min": float(t_min), "t_max": float(t_max),
                  "span_frames": float(t_max - t_min), "span_s": float((t_max - t_min) / data.fps),
                  "n_ctrl": M - DEGREE - 1,
                  "support_check": [{"x": float(x), "local": float(l), "global": float(g),
                                     "diff": float(abs(l - g))}
                                    for x, l, g in zip(xs, loc[:, 2], glo[:, 2])]}

    # =====================================================================
    rule("STEP 4 — pack it into the parameter matrix the policy predicts")
    chunks = chunk_bspline_trajectory(comp, chunk_size=CHUNK_SIZE, stride=1)
    P = chunk_to_params(chunks[WINDOW], M, 11)
    print(f"parameter matrix shape = {P.shape}   (rows = knots, cols = 1 + action_dim)")
    print(f"    column 0      = the knot value")
    print(f"    columns 1..10 = the control point [x y z r0..r5 grip]")
    print(f"\nrows 0-5 and 22-25 (of {M}):")
    hdr = f"    {'row':>4} {'knot':>8} " + " ".join(f"{n:>8}" for n in
                                                    ["x", "y", "z", "r0", "r1", "r2", "r3", "r4", "r5", "grip"])
    print(hdr)
    for i in list(range(6)) + [None] + list(range(22, 26)):
        if i is None:
            print(f"    {'...':>4}")
            continue
        print(f"    {i:>4} {P[i, 0]:8.2f} " + " ".join(f"{v:8.4f}" for v in P[i, 1:]))
    print(f"\nflattened, that is {M} x 11 = {M * 11} numbers — one vector per observation")
    E["step4"] = {"shape": list(P.shape), "flat": M * 11,
                  "rows": [{"i": i, "knot": float(P[i, 0]), "cp": r(P[i, 1:], 4)}
                           for i in list(range(6)) + list(range(22, 26))]}

    # =====================================================================
    rule(f"STEP 5 — make the knots relative to a frame (frame {FRAME})")
    Pr = P.copy()
    Pr[:, 0] -= FRAME
    print(f"stored form subtracts the current frame index from column 0 only:")
    print(f"    {'row':>4} {'absolute':>10} {'- frame':>10} {'relative':>10}")
    for i in [0, 3, 10, 22, 25]:
        print(f"    {i:>4} {P[i, 0]:10.2f} {FRAME:10d} {Pr[i, 0]:10.2f}")
    print(f"\nvalid domain becomes [{Pr[DEGREE, 0]:.2f}, {Pr[M - DEGREE - 1, 0]:.2f}) "
          f"— i.e. 'starts {Pr[DEGREE, 0]:.2f} frames from now, lasts "
          f"{Pr[M - DEGREE - 1, 0] - Pr[DEGREE, 0]:.2f} frames'")
    d_abs = decode_bspline_action(P, degree=DEGREE, num_actions=6)
    d_rel = decode_bspline_action(Pr, degree=DEGREE, num_actions=6)
    print(f"\ndecoding both gives identical waypoints — max difference "
          f"{np.abs(d_abs - d_rel).max():.2e}")
    print("    (the Cox-de Boor coefficients are ratios of differences, so a shift cancels)")
    E["step5"] = {"frame": FRAME,
                  "rows": [{"i": i, "abs": float(P[i, 0]), "rel": float(Pr[i, 0])}
                           for i in [0, 3, 10, 22, 25]],
                  "rel_domain": [float(Pr[DEGREE, 0]), float(Pr[M - DEGREE - 1, 0])],
                  "shift_diff": float(np.abs(d_abs - d_rel).max())}

    # =====================================================================
    rule("STEP 6 — decode back to waypoints the arm can run")
    n_wp = 6
    dec10 = decode_bspline_action(P, degree=DEGREE, num_actions=n_wp)
    dec7 = convert_actions_10d_to_7d(dec10)
    ts = np.linspace(t_min, t_max, n_wp)
    print(f"evaluate the spline at {n_wp} points spaced uniformly over [{t_min:.2f}, {t_max:.2f}]")
    print(f"    decoded (10 dims) shape = {dec10.shape}")
    print(f"    converted to cart7      = {dec7.shape}")
    print(f"\n{'frame':>8} {'x':>9} {'y':>9} {'z':>9} {'gripper':>9}   "
          f"{'recorded x':>11} {'y':>9} {'z':>9}   {'err mm':>7}")
    frames = np.arange(len(raw))
    for x, w in zip(ts, dec7):
        truth = np.array([np.interp(x, frames, raw[:, j]) for j in range(3)])
        e = np.linalg.norm(w[:3] - truth) * 1000
        print(f"{x:8.2f} {w[0]:9.4f} {w[1]:9.4f} {w[2]:9.4f} {w[6]:9.3f}   "
              f"{truth[0]:11.4f} {truth[1]:9.4f} {truth[2]:9.4f}   {e:7.2f}")
    print(f"\nnumber of waypoints is a DECODE-TIME choice — the same 286 numbers give:")
    for n in (4, 8, 32):
        dd = decode_bspline_action(P, degree=DEGREE, num_actions=n)
        print(f"    num_actions={n:3d} -> shape {dd.shape},  first row x = {dd[0, 0]:.5f}, "
              f"last row x = {dd[-1, 0]:.5f}")
    E["step6"] = {"n_wp": n_wp, "shape10": list(dec10.shape), "shape7": list(dec7.shape),
                  "rows": [{"frame": float(x), "wp": r(w, 4),
                            "truth": r([np.interp(x, frames, raw[:, j]) for j in range(3)], 4),
                            "err_mm": float(np.linalg.norm(
                                w[:3] - np.array([np.interp(x, frames, raw[:, j])
                                                  for j in range(3)])) * 1000)}
                           for x, w in zip(ts, dec7)]}

    # =====================================================================
    rule("STEP 7 — de Boor by hand (small enough to check with a pencil)")
    t = np.array([0., 0., 0., 1., 2., 3., 3., 3.])
    c = np.array([0., 2., 1., 3., 1.])
    kk, x = 2, 1.5
    print(f"knots t = {t}        (shape {t.shape})")
    print(f"coeffs c = {c}                    (shape {c.shape})")
    print(f"degree k = {kk};  identity: len(t) = len(c)+k+1 -> {len(t)} = {len(c)}+{kk}+1  OK")
    s = int(np.searchsorted(t, x, side="right")) - 1
    print(f"\nevaluate at x = {x}")
    print(f"  find the span: t[{s}] = {t[s]} <= {x} < t[{s + 1}] = {t[s + 1]}  ->  s = {s}")
    d = [float(c[s - kk + j]) for j in range(kk + 1)]
    print(f"  initialise d_j = c[s-k+j] for j=0..{kk}:  "
          + ", ".join(f"d{j}=c[{s - kk + j}]={d[j]:g}" for j in range(kk + 1)))
    triangle = [list(d)]
    steps = []
    for rr in range(1, kk + 1):
        print(f"\n  round r = {rr}:")
        for j in range(kk, rr - 1, -1):
            lo, hi = t[s + j - kk], t[s + 1 + j - rr]
            alpha = 0.0 if hi == lo else (x - lo) / (hi - lo)
            new = (1 - alpha) * d[j - 1] + alpha * d[j]
            print(f"    j={j}:  alpha = (x - t[{s + j - kk}]) / (t[{s + 1 + j - rr}] - t[{s + j - kk}])"
                  f" = ({x} - {lo:g}) / ({hi:g} - {lo:g}) = {alpha:.4f}")
            print(f"           d{j} <- (1-{alpha:.4f})*{d[j - 1]:.4f} + {alpha:.4f}*{d[j]:.4f}"
                  f" = {new:.4f}")
            steps.append({"r": rr, "j": j, "lo": float(lo), "hi": float(hi),
                          "alpha": float(alpha), "d_prev": float(d[j - 1]),
                          "d_cur": float(d[j]), "out": float(new)})
            d[j] = new
        triangle.append(list(d))
    ref = float(BSpline(t, c, kk)(x))
    print(f"\n  result d{kk} = {d[kk]:.6f}")
    print(f"  scipy BSpline(t, c, {kk})({x}) = {ref:.6f}")
    print(f"  difference = {abs(d[kk] - ref):.2e}")
    E["step7"] = {"t": r(t, 2), "c": r(c, 2), "k": kk, "x": x, "span": s,
                  "init": r(triangle[0], 4), "steps": steps,
                  "result": float(d[kk]), "scipy": ref, "diff": float(abs(d[kk] - ref))}

    # =====================================================================
    rule("STEP 8 — the padded tail, with numbers")
    s_pad = len(chunks) - 1
    Pp = chunk_to_params(chunks[s_pad], M, 11)
    kp = Pp[:, 0]
    print(f"last chunk (window s = {s_pad}) runs past the end of the knot vector.")
    print(f"its knots: {kp}")
    print(f"    knots[{M - DEGREE - 1}] = t_max = {kp[M - DEGREE - 1]:.2f}")
    print(f"    knots[{M - 1}]            = {kp[M - 1]:.2f}   <- padded, identical")
    from verify_math import basis
    ssum = sum(basis(i, DEGREE, kp.astype(np.float64), float(kp[M - DEGREE - 1]))
               for i in range(M - DEGREE - 1))
    Pi = chunk_to_params(chunks[WINDOW], M, 11)
    ki = Pi[:, 0].astype(np.float64)
    isum = sum(basis(i, DEGREE, ki, float(ki[M - DEGREE - 1])) for i in range(M - DEGREE - 1))
    print(f"\nsum of basis functions at x = t_max:")
    print(f"    padded chunk   : {ssum:.6f}   <- partition of unity FAILS")
    print(f"    interior chunk : {isum:.6f}   <- fine")
    dp = decode_bspline_action(Pp, degree=DEGREE, num_actions=5)
    print(f"\nso the last decoded waypoint collapses:")
    for i, row in enumerate(dp):
        tag = "  <- dropped by the decoder" if i == len(dp) - 1 else ""
        print(f"    waypoint {i}: x={row[0]:9.4f} y={row[1]:9.4f} z={row[2]:9.4f}{tag}")
    E["step8"] = {"window": s_pad, "knots": r(kp, 2), "t_max": float(kp[M - DEGREE - 1]),
                  "padded_sum": float(ssum), "interior_sum": float(isum),
                  "waypoints": [r(row[:3], 4) for row in dp]}

    # =====================================================================
    rule("STEP 9 — shapes, end to end")
    conv = Path("/home/batur/Coding/data/merged_bspline_20260528")
    n_total = 21560
    tbl = [
        ("recorded episode", f"({b - a}, 7)", "x y z, axis-angle, gripper"),
        ("after rotation conversion", f"({b - a}, 10)", "axis-angle -> rot6d"),
        ("fitted knot vector", f"({len(t_full)},)", f"{len(t_full) / (b - a):.3f} knots/frame"),
        ("fitted control points", f"({len(c_full)}, 10)", f"= {len(t_full)} - {DEGREE} - 1"),
        ("chunks produced", f"{len(chunks)}", "one per knot window, stride 1"),
        ("one chunk", f"({M}, 11)", "knot column + 10 control dims"),
        ("flattened for the policy", f"({M * 11},)", "what ACT regresses"),
        ("whole dataset", f"({n_total}, {M * 11})", "70 episodes, all frames"),
    ]
    w = max(len(x[0]) for x in tbl)
    for name, shape, note in tbl:
        print(f"    {name:<{w}}  {shape:>16}   {note}")
    E["step9"] = [{"name": n, "shape": s, "note": t} for n, s, t in tbl]
    E["meta"] = {"episode": EPISODE, "degree": DEGREE, "chunk_size": CHUNK_SIZE,
                 "max_error": MAX_ERROR, "n_steps": M, "fps": data.fps,
                 "window": WINDOW, "frame": FRAME}

    args.json.write_text(json.dumps(E, indent=1))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
