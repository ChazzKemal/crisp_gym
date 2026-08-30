#!/usr/bin/env python3
"""Pick a window of consecutive deploy chunks; compare smoothing options.

Loads a ``deploy_runs/<ts>/trace.npz``, picks chunks ``[first:first+N]``,
and renders four trajectories on the same axes:

  raw chunks (faint)         - one polyline per chunk, colour-coded by t
  naive executed             - concatenate chunks[t][0..K-1] for all t
  Hermite cubic blend        - boundary fix: match position AND velocity
                               over `--blend-window` control frames
  cubic smoothing spline     - scipy UnivariateSpline with smoothing
                               parameter `s` auto-picked from the noise
                               std implied by overlapping chunk predictions

Outputs a 2x2 figure:
  top-left   xy projection (top-down view of the workspace)
  top-right  xz projection (side view; the most-active boundary axis is z)
  bottom     |Delta v| at chunk boundary for each method, per-axis bars
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def hermite_blend(prev_tail: np.ndarray, next_head: np.ndarray,
                  window: int, dt: float) -> np.ndarray:
    """Per-axis cubic Hermite over `window` frames matching pos+vel at both ends.

    prev_tail: shape (>=2, D)  -- last few frames of old chunk
    next_head: shape (>=2, D)  -- first few frames of new chunk
    window:    int             -- number of blended frames returned
    dt:        float           -- inter-frame interval (seconds), for vel scale

    Returns (window, D). The blend replaces the would-be transition between
    prev_tail[-1] and next_head[0]; sample 0 of the returned array equals
    prev_tail[-1], sample window-1 equals next_head[0] (with continuity in
    velocity at both ends).
    """
    p0 = prev_tail[-1]                                  # (D,)
    p1 = next_head[0]                                   # (D,)
    v0 = (prev_tail[-1] - prev_tail[-2]) / dt           # (D,) vel at end of old
    v1 = (next_head[1] - next_head[0]) / dt             # (D,) vel at start of new
    T = window * dt                                     # blend duration
    # Cubic Hermite on [0, T]:
    #   p(s) = h00 p0 + h10 T v0 + h01 p1 + h11 T v1, with s in [0, 1]
    s = np.linspace(0.0, 1.0, window)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return (
        h00[:, None] * p0
        + (h10[:, None] * T) * v0
        + h01[:, None] * p1
        + (h11[:, None] * T) * v1
    )


def make_naive_executed(chunks: np.ndarray) -> np.ndarray:
    """Concatenate chunks[t][0..K-1] in order -- what the sender would emit
    without any blending. shape (N*K, 7)."""
    return chunks.reshape(-1, chunks.shape[-1])


def make_hermite_executed(chunks: np.ndarray, window: int, dt: float) -> np.ndarray:
    """Replace `window` frames at every chunk boundary with a Hermite blend.

    Layout: for chunks 0..N-1 each with K frames,
      [chunk_0[0..K-1-w], blend_01(w), chunk_1[1..K-1-w], blend_12(w), ...,
       chunk_{N-1}[1:]]
    The first w samples of every chunk except chunk_0 are absorbed into the
    blend (we replace them, not append). That keeps the absolute timeline
    aligned: each new chunk still occupies K control frames.
    """
    N, K, D = chunks.shape
    half = window // 2
    out = []
    out.append(chunks[0][: K - half])                            # head of first chunk
    for t in range(N - 1):
        prev_tail = chunks[t][K - 2 - half: K - half]            # 2 frames; last is the boundary anchor
        next_head = chunks[t + 1][half: half + 2]                # 2 frames; first is the boundary anchor
        blend = hermite_blend(prev_tail, next_head, window=window, dt=dt)
        out.append(blend[1:-1])                                  # exclude anchor frames (already in adjacent slices)
        if t < N - 2:
            out.append(chunks[t + 1][half: K - half])
        else:
            out.append(chunks[t + 1][half:])
    return np.concatenate(out, axis=0)


def make_rolling2_warm_executed(chunks: np.ndarray, dt: float,
                                  s_scale: float = 1.0) -> tuple[np.ndarray, float]:
    """Rolling 2-chunk smoothing spline, warm-started from emitted history.

    Same as ``make_rolling2_executed`` but instead of using the RAW previous
    chunk as one of the two buffer entries, we use whatever was actually
    emitted for the previous chunk's slot (the output of the previous
    fit). That makes successive splines see a consistent "past" point at
    the chunk boundary, eliminating the inter-spline discontinuity that
    cold rolling-2 suffers from.
    """
    from scipy.interpolate import UnivariateSpline
    N, K, D = chunks.shape
    times = np.arange(K) * dt
    out = np.empty((N * K, D), dtype=np.float32)
    out[:K] = chunks[0]
    s_used = 0.0
    n_fits = 0
    for t in range(1, N):
        prev_emitted = out[(t - 1) * K: t * K]   # what we ACTUALLY sent for chunk t-1
        t_obs = np.concatenate([
            (t - 1) * K * dt + times,
            t       * K * dt + times,
        ])
        a_obs = np.concatenate([prev_emitted, chunks[t]], axis=0)
        order = np.argsort(t_obs, kind='stable')
        t_sorted = t_obs[order]
        a_sorted = a_obs[order]
        s_per_axis = []
        for ax in range(D):
            med = np.median(a_sorted[:, ax])
            mad = np.median(np.abs(a_sorted[:, ax] - med)) + 1e-9
            sigma = 1.4826 * mad
            s_per_axis.append(s_scale * (sigma ** 2) * len(t_sorted) * 0.05)
        t_query = t * K * dt + times
        chunk_out = np.empty((K, D), dtype=np.float32)
        for ax in range(D):
            spl = UnivariateSpline(t_sorted, a_sorted[:, ax], k=3,
                                    s=s_per_axis[ax])
            chunk_out[:, ax] = spl(t_query)
        out[t * K:(t + 1) * K] = chunk_out
        s_used += float(np.mean(s_per_axis))
        n_fits += 1
    return out, (s_used / max(n_fits, 1))


def make_rolling2_executed(chunks: np.ndarray, dt: float,
                            s_scale: float = 1.0) -> tuple[np.ndarray, float]:
    """Rolling 2-chunk smoothing spline — fits over (prev, current) only.

    At each chunk arrival, the buffer contains exactly two chunks: the
    previous one (already partially executed) and the brand-new one. We fit
    a cubic smoothing spline through their combined 2*K points (per axis),
    then sample the spline at the K absolute timestamps that fall within
    the new chunk's slot. Those K samples replace the chunk's raw values.

    The first chunk in the window has no predecessor; we use it as-is.

    This mimics the smallest possible rolling-window deploy (M=2). Hot-path
    cost per chunk = 1 spline refit over 64 points per axis (~1 ms total).
    """
    from scipy.interpolate import UnivariateSpline
    N, K, D = chunks.shape
    times = np.arange(K) * dt

    out = np.empty((N * K, D), dtype=np.float32)
    out[:K] = chunks[0]
    s_used = 0.0
    n_fits = 0
    for t in range(1, N):
        # Buffer = chunk[t-1] (absolute time (t-1)*K*dt + i*dt) and
        #          chunk[t]   (absolute time  t   *K*dt + i*dt).
        t_obs = np.concatenate([
            (t - 1) * K * dt + times,
            t       * K * dt + times,
        ])
        a_obs = np.concatenate([chunks[t - 1], chunks[t]], axis=0)
        order = np.argsort(t_obs, kind='stable')
        t_sorted = t_obs[order]
        a_sorted = a_obs[order]

        # Per-axis s scaled to the local MAD of the 2*K window. This adapts
        # smoothing strength to the local action range so axes that aren't
        # moving don't get oversmoothed.
        s_per_axis = []
        for ax in range(D):
            med = np.median(a_sorted[:, ax])
            mad = np.median(np.abs(a_sorted[:, ax] - med)) + 1e-9
            sigma = 1.4826 * mad
            s_per_axis.append(s_scale * (sigma ** 2) * len(t_sorted) * 0.05)

        # Sample at the K timestamps inside chunk t's slot.
        t_query = t * K * dt + times
        chunk_out = np.empty((K, D), dtype=np.float32)
        for ax in range(D):
            spl = UnivariateSpline(t_sorted, a_sorted[:, ax], k=3,
                                    s=s_per_axis[ax])
            chunk_out[:, ax] = spl(t_query)
        out[t * K:(t + 1) * K] = chunk_out
        s_used += float(np.mean(s_per_axis))
        n_fits += 1
    return out, (s_used / max(n_fits, 1))


def make_lookahead_executed(chunks: np.ndarray, dt: float,
                              s_scale: float = 1.0,
                              past_chunks: int = 1,
                              future_chunks: int = 1) -> tuple[np.ndarray, float]:
    """Rolling spline with ONE-chunk lookahead.

    Models the realistic async-deploy situation: by the time we need to
    emit chunk t's frames, chunk t+1 has ALREADY been computed (because
    inference latency ~30 ms << one chunk's duration ~1.6 s). So the
    spline can fit through chunks [t-past_chunks .. t .. t+future_chunks].

    Window for chunk t's emission: chunks [t-past_chunks, t+future_chunks],
    clamped to [0, N-1]. Default (1, 1) = previous + current + next = 3
    chunks, 96 points per axis per fit. Cheap enough for real time
    (~1 ms per refit).

    The "lag" cost of this in deploy is roughly the inference latency
    of one chunk (~30-50 ms), not the chunk duration. Confirmed by:
    when chunk t is about to be EMITTED, chunk t+1 has been REQUESTED
    immediately after chunk t pushed -- so chunk t+1's data is in hand
    after one inference cycle, long before chunk t's last frame is sent.
    """
    from scipy.interpolate import UnivariateSpline
    N, K, D = chunks.shape
    times = np.arange(K) * dt
    out = np.empty((N * K, D), dtype=np.float32)
    out[:K] = chunks[0]
    s_used = 0.0
    n_fits = 0
    for t in range(1, N):
        first = max(0, t - past_chunks)
        last = min(N - 1, t + future_chunks)
        bufs = []
        ts = []
        for k in range(first, last + 1):
            bufs.append(chunks[k])
            ts.append(k * K * dt + times)
        t_obs = np.concatenate(ts)
        a_obs = np.concatenate(bufs, axis=0)
        order = np.argsort(t_obs, kind='stable')
        t_sorted = t_obs[order]
        a_sorted = a_obs[order]
        s_per_axis = []
        for ax in range(D):
            med = np.median(a_sorted[:, ax])
            mad = np.median(np.abs(a_sorted[:, ax] - med)) + 1e-9
            sigma = 1.4826 * mad
            s_per_axis.append(s_scale * (sigma ** 2) * len(t_sorted) * 0.05)
        t_query = t * K * dt + times
        chunk_out = np.empty((K, D), dtype=np.float32)
        for ax in range(D):
            spl = UnivariateSpline(t_sorted, a_sorted[:, ax], k=3,
                                    s=s_per_axis[ax])
            chunk_out[:, ax] = spl(t_query)
        out[t * K:(t + 1) * K] = chunk_out
        s_used += float(np.mean(s_per_axis))
        n_fits += 1
    return out, (s_used / max(n_fits, 1))


def make_causal_global_executed(chunks: np.ndarray, dt: float,
                                  s_scale: float = 1.0) -> tuple[np.ndarray, float]:
    """Causal growing-window spline — at each chunk arrival, fit through
    chunks 0..t (no future data ever used).

    This is what you could actually run online: a global smoothing spline,
    but with the spline refit each chunk using only the predictions known
    so far. The window grows monotonically; by the last chunk it has the
    same data the offline-global spline had.

    For chunk t, we fit a spline through ALL chunk points 0..t and sample
    it at chunk t's K absolute timestamps -- the spline's value at those
    timestamps becomes what we emit for chunk t's slot.

    This is the online analogue of `make_spline_executed`; the difference
    between the two tells us how much "future knowledge" the offline
    version was exploiting.
    """
    from scipy.interpolate import UnivariateSpline
    N, K, D = chunks.shape
    times = np.arange(K) * dt
    out = np.empty((N * K, D), dtype=np.float32)
    out[:K] = chunks[0]
    s_used = 0.0
    n_fits = 0
    for t in range(1, N):
        # Buffer = ALL chunks emitted so far + the brand-new one.
        # Use raw chunks (cold) to mirror the offline global; using emitted
        # history would be the "warm" variant.
        bufs = []
        ts = []
        for k in range(0, t + 1):
            bufs.append(chunks[k])
            ts.append(k * K * dt + times)
        t_obs = np.concatenate(ts)
        a_obs = np.concatenate(bufs, axis=0)
        order = np.argsort(t_obs, kind='stable')
        t_sorted = t_obs[order]
        a_sorted = a_obs[order]
        s_per_axis = []
        for ax in range(D):
            med = np.median(a_sorted[:, ax])
            mad = np.median(np.abs(a_sorted[:, ax] - med)) + 1e-9
            sigma = 1.4826 * mad
            s_per_axis.append(s_scale * (sigma ** 2) * len(t_sorted) * 0.05)
        t_query = t * K * dt + times
        chunk_out = np.empty((K, D), dtype=np.float32)
        for ax in range(D):
            spl = UnivariateSpline(t_sorted, a_sorted[:, ax], k=3,
                                    s=s_per_axis[ax])
            chunk_out[:, ax] = spl(t_query)
        out[t * K:(t + 1) * K] = chunk_out
        s_used += float(np.mean(s_per_axis))
        n_fits += 1
    return out, (s_used / max(n_fits, 1))


def make_spline_executed(chunks: np.ndarray, dt: float,
                          s_scale: float = 1.0) -> tuple[np.ndarray, float]:
    """Cubic smoothing spline through ALL chunk points (per axis).

    Each frame of every chunk is an observation at absolute time
    t_chunk * K * dt + i * dt. Multiple chunks may predict the same t; the
    spline naturally averages them (weighted equally).

    Smoothing parameter `s` is auto-picked from the std of disagreement
    between overlapping predictions and the per-axis count, then multiplied
    by `s_scale` (1.0 = the natural choice).

    Returns (executed, s_used). executed shape: (N*K, 7).
    """
    from scipy.interpolate import UnivariateSpline
    N, K, D = chunks.shape
    times = np.arange(K) * dt
    # Build observation arrays: per-chunk time stamps shifted by chunk_idx*K*dt.
    # In the trace, chunks are issued every K control frames, so chunk t is
    # anchored at absolute time t * K * dt. (Async/overlap details don't matter
    # for the demo -- we're showing the policy's *intent*.)
    t_obs = []
    a_obs = []  # (M, D)
    for t in range(N):
        t_obs.append(t * K * dt + times)
        a_obs.append(chunks[t])
    t_obs = np.concatenate(t_obs)         # (N*K,)
    a_obs = np.concatenate(a_obs, axis=0) # (N*K, D)

    # Auto s: cumulative variance of residuals from a per-axis straight line
    # gives a baseline; we want s ~ a few * (per-axis noise variance) * n_obs
    # so the spline absorbs the inconsistency without over-smoothing.
    s_per_axis = []
    for ax in range(D):
        # Use median absolute deviation around a 1-second moving median as a
        # robust noise estimate; multiply by N_obs to match scipy's s scaling
        # (scipy expects sum of squared residuals tolerance).
        med = np.median(a_obs[:, ax])
        mad = np.median(np.abs(a_obs[:, ax] - med)) + 1e-9
        sigma = 1.4826 * mad
        s_per_axis.append(s_scale * (sigma ** 2) * len(t_obs) * 0.05)

    # Sort by time for scipy.
    order = np.argsort(t_obs, kind='stable')
    t_sorted = t_obs[order]
    a_sorted = a_obs[order]

    # Sample at the same absolute timestamps as the naive executed: one
    # frame per control step from t=0 to t=N*K*dt.
    t_query = np.arange(N * K) * dt

    executed = np.empty((len(t_query), D), dtype=np.float64)
    s_used_avg = 0.0
    for ax in range(D):
        spl = UnivariateSpline(t_sorted, a_sorted[:, ax], k=3, s=s_per_axis[ax])
        executed[:, ax] = spl(t_query)
        s_used_avg += s_per_axis[ax]
    return executed.astype(np.float32), s_used_avg / D


def plot_comparison(chunks, naive, hermite, rolling2, rolling2_warm, lookahead, causal_global, spline, dt, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    N, K, D = chunks.shape
    fig = plt.figure(figsize=(14, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    axes = np.empty((3, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[1, 0] = fig.add_subplot(gs[1, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1])
    axes[2, 0] = fig.add_subplot(gs[2, 0])
    axes[2, 1] = fig.add_subplot(gs[2, 1])

    # --- top-left: xy projection -----------------------------------------
    ax = axes[0, 0]
    # Faint raw chunks, coloured by chunk index
    cmap = plt.get_cmap('viridis')
    for t in range(N):
        ax.plot(chunks[t, :, 0] * 1000, chunks[t, :, 1] * 1000,
                color=cmap(t / max(N - 1, 1)), alpha=0.35, lw=0.9)
    # Naive executed (the kinky polyline)
    ax.plot(naive[:, 0] * 1000, naive[:, 1] * 1000, color='black',
            lw=1.6, label='naive executed (concat chunks)')
    # Hermite blend
    ax.plot(hermite[:, 0] * 1000, hermite[:, 1] * 1000, color='tab:blue',
            lw=1.6, alpha=0.9, label='Hermite cubic blend (w=blend_window)')
    # Rolling 2-chunk spline (cold)
    ax.plot(rolling2[:, 0] * 1000, rolling2[:, 1] * 1000, color='tab:green',
            lw=1.4, alpha=0.7, ls=':',
            label='rolling-2 cold (raw prev + cur)')
    # Rolling 2-chunk spline, warm
    ax.plot(rolling2_warm[:, 0] * 1000, rolling2_warm[:, 1] * 1000, color='tab:purple',
            lw=1.6, alpha=0.9,
            label='rolling-2 warm (emitted prev + cur)')
    # Lookahead spline (1 past + 1 future chunk)
    ax.plot(lookahead[:, 0] * 1000, lookahead[:, 1] * 1000, color='tab:cyan',
            lw=1.8, alpha=0.95,
            label='lookahead(1,1): 1 past + 1 future chunk (~30 ms lag)')
    # Causal global (online: chunks 0..t at each step, no future)
    ax.plot(causal_global[:, 0] * 1000, causal_global[:, 1] * 1000, color='tab:orange',
            lw=1.4, alpha=0.8, ls=':',
            label='causal global (chunks 0..t, no future)')
    # Smoothing spline (offline global)
    ax.plot(spline[:, 0] * 1000, spline[:, 1] * 1000, color='tab:red',
            lw=1.8, alpha=0.9, label='offline global (all chunks, oracle)')
    # Boundary markers on the naive line
    for t in range(1, N):
        idx = t * K
        ax.plot(naive[idx, 0] * 1000, naive[idx, 1] * 1000,
                'o', color='black', mfc='white', ms=5, zorder=5)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'Top-down (xy)  --  chunks {N} x K={K}  (faint = raw chunks)')
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)

    # --- top-right: xz projection ----------------------------------------
    ax = axes[0, 1]
    for t in range(N):
        ax.plot(chunks[t, :, 0] * 1000, chunks[t, :, 2] * 1000,
                color=cmap(t / max(N - 1, 1)), alpha=0.35, lw=0.9)
    ax.plot(naive[:, 0] * 1000, naive[:, 2] * 1000, color='black', lw=1.6)
    ax.plot(hermite[:, 0] * 1000, hermite[:, 2] * 1000, color='tab:blue', lw=1.6, alpha=0.9)
    ax.plot(rolling2[:, 0] * 1000, rolling2[:, 2] * 1000, color='tab:green', lw=1.4, alpha=0.7, ls=':')
    ax.plot(rolling2_warm[:, 0] * 1000, rolling2_warm[:, 2] * 1000, color='tab:purple', lw=1.6, alpha=0.9)
    ax.plot(lookahead[:, 0] * 1000, lookahead[:, 2] * 1000, color='tab:cyan', lw=1.8, alpha=0.95)
    ax.plot(causal_global[:, 0] * 1000, causal_global[:, 2] * 1000, color='tab:orange', lw=1.4, alpha=0.8, ls=':')
    ax.plot(spline[:, 0] * 1000, spline[:, 2] * 1000, color='tab:red', lw=1.8, alpha=0.9)
    for t in range(1, N):
        idx = t * K
        ax.plot(naive[idx, 0] * 1000, naive[idx, 2] * 1000,
                'o', color='black', mfc='white', ms=5, zorder=5)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_title('Side (xz)  --  o = chunk boundary')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)

    # --- middle row + bottom-left: position vs time, one panel per axis ----
    t_axis = np.arange(naive.shape[0]) * dt
    method_styles = [
        ('naive',          naive,          'black',      '-',  1.0, 0.55),
        ('Hermite',        hermite,        'tab:blue',   '--', 1.4, 0.95),
        ('rolling-2 cold', rolling2,       'tab:green',  ':',  1.1, 0.6),
        ('rolling-2 warm', rolling2_warm,  'tab:purple', '-.', 1.1, 0.7),
        ('lookahead(1,1)', lookahead,      'tab:cyan',   '-',  1.6, 0.95),
        ('causal global',  causal_global,  'tab:orange', ':',  1.0, 0.7),
        ('offline global', spline,         'tab:red',    '-',  1.6, 0.95),
    ]
    for panel_pos, axis_i, axis_name in [((1, 0), 0, 'x'),
                                          ((1, 1), 1, 'y'),
                                          ((2, 0), 2, 'z')]:
        ax = axes[panel_pos]
        for label, traj, color, ls, lw, alpha in method_styles:
            ax.plot(t_axis, traj[:, axis_i] * 1000,
                    color=color, ls=ls, lw=lw, alpha=alpha, label=label)
        for t in range(1, N):
            ax.axvline(t * K * dt, color='gray', alpha=0.3, lw=0.5)
        ax.set_xlabel('time [s]')
        ax.set_ylabel(f'{axis_name} position [mm]')
        ax.set_title(f'{axis_name}(t)')
        ax.grid(alpha=0.3)
        if panel_pos == (1, 0):
            ax.legend(fontsize=8, loc='best')

    # --- bottom-right: |dv| histogram at boundaries -----------------------
    ax = axes[2, 1]
    def boundary_dv(traj):
        # dv at every chunk boundary index = t*K (t=1..N-1)
        out = []
        for t in range(1, N):
            i = t * K
            if i + 1 < len(traj) and i - 2 >= 0:
                v_prev = traj[i - 1, :6] - traj[i - 2, :6]
                v_next = traj[i + 1, :6] - traj[i, :6]
                out.append(np.linalg.norm(v_next - v_prev))
        return np.asarray(out) * 1000  # convert to mm-equivalent

    dv_naive          = boundary_dv(naive)
    dv_hermite        = boundary_dv(hermite)
    dv_rolling2       = boundary_dv(rolling2)
    dv_rolling2_warm  = boundary_dv(rolling2_warm)
    dv_lookahead      = boundary_dv(lookahead)
    dv_causal_global  = boundary_dv(causal_global)
    dv_spline         = boundary_dv(spline)

    bins = np.linspace(0, max(dv_naive.max(), 1e-6), 14)
    ax.hist(dv_naive,         bins=bins, alpha=0.35, color='black',      label=f'naive (p50={np.median(dv_naive):.2f})')
    ax.hist(dv_hermite,       bins=bins, alpha=0.35, color='tab:blue',   label=f'Hermite (p50={np.median(dv_hermite):.2f})')
    ax.hist(dv_rolling2,      bins=bins, alpha=0.3,  color='tab:green',  label=f'rolling-2 cold (p50={np.median(dv_rolling2):.2f})')
    ax.hist(dv_rolling2_warm, bins=bins, alpha=0.3,  color='tab:purple', label=f'rolling-2 warm (p50={np.median(dv_rolling2_warm):.2f})')
    ax.hist(dv_lookahead,     bins=bins, alpha=0.45, color='tab:cyan',   label=f'lookahead(1,1) (p50={np.median(dv_lookahead):.2f})')
    ax.hist(dv_causal_global, bins=bins, alpha=0.3,  color='tab:orange', label=f'causal global (p50={np.median(dv_causal_global):.2f})')
    ax.hist(dv_spline,        bins=bins, alpha=0.4,  color='tab:red',    label=f'offline global (p50={np.median(dv_spline):.2f})')
    ax.set_xlabel('|delta v| at chunk boundary (cart 6-vec, mm-equiv)')
    ax.set_ylabel('count')
    ax.set_title('Velocity discontinuity at boundaries (smaller = smoother)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f'Chunk smoothing demo  --  chunks [{args.first}:{args.first + args.n}]'
                 f'  from {Path(args.trace).parent.name}',
                 fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote figure: {out_path}')


def report_stats(label, methods, K):
    """Print boundary-discontinuity numbers for each method."""
    first_traj = next(iter(methods.values()))
    N_bnd = (first_traj.shape[0] // K) - 1

    def measure(traj):
        pos_jump = []
        ang = []
        for t in range(1, N_bnd + 1):
            i = t * K
            if i + 1 >= len(traj) or i - 2 < 0:
                continue
            pos_jump.append(np.linalg.norm(traj[i, :3] - traj[i - 1, :3]))
            v_prev = traj[i - 1, :6] - traj[i - 2, :6]
            v_next = traj[i + 1, :6] - traj[i, :6]
            n_prev = np.linalg.norm(v_prev) + 1e-9
            n_next = np.linalg.norm(v_next) + 1e-9
            cos = float(np.dot(v_prev, v_next) / (n_prev * n_next))
            ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        return np.asarray(pos_jump) * 1000, np.asarray(ang)

    print(f'\n[{label}] boundary stats (n_boundaries={N_bnd}):')
    print(f'  {"method":<14}  pos_jump p50 / p90 (mm)   angle_change p50 (deg)   reversals')
    for name, traj in methods.items():
        pj, ang = measure(traj)
        rev = int((ang > 90).sum())
        print(f'  {name:<14}  {np.percentile(pj, 50):6.2f} / {np.percentile(pj, 90):6.2f}'
              f'           {np.percentile(ang, 50):6.1f}'
              f'              {rev}/{len(ang)}')


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--trace', type=str,
        default=str(Path.home() / '.cache/huggingface/lerobot/deploy_runs/20260522T225850/trace.npz'),
        help='Path to a deploy_runs/<ts>/trace.npz.',
    )
    ap.add_argument('--first', type=int, default=80, help='First chunk index in the window.')
    ap.add_argument('--n', type=int, default=10, help='Number of consecutive chunks to include.')
    ap.add_argument('--dt', type=float, default=0.05, help='Inter-frame dt (seconds); 20Hz default.')
    ap.add_argument('--blend-window', type=int, default=4, help='Hermite blend length (frames).')
    ap.add_argument('--spline-s-scale', type=float, default=1.0,
                    help='Multiplier on the auto smoothing strength.')
    ap.add_argument('--out', type=str, default='/tmp/chunk_smoothing.png')
    args = ap.parse_args()

    d = np.load(args.trace, allow_pickle=True)
    chunks_full = d['chunk']
    N_full = chunks_full.shape[0]
    if args.first + args.n > N_full:
        print(f'Run has {N_full} chunks; clipping window to last {min(args.n, N_full - args.first)}.')
        args.n = N_full - args.first

    chunks = chunks_full[args.first: args.first + args.n].astype(np.float32)
    print(f'Loaded {chunks_full.shape}; using chunks [{args.first}:{args.first + args.n}]'
          f' shape {chunks.shape}')

    naive = make_naive_executed(chunks)
    hermite = make_hermite_executed(chunks, window=args.blend_window, dt=args.dt)
    # Ensure same length for plotting; trim hermite if it ended up off by a frame.
    rolling2, s_used_r = make_rolling2_executed(chunks, dt=args.dt,
                                                  s_scale=args.spline_s_scale)
    print(f'Rolling-2 cold spline avg s per refit = {s_used_r:.4g}')
    rolling2_warm, s_used_w = make_rolling2_warm_executed(chunks, dt=args.dt,
                                                            s_scale=args.spline_s_scale)
    print(f'Rolling-2 warm spline avg s per refit = {s_used_w:.4g}')
    lookahead, s_used_la = make_lookahead_executed(chunks, dt=args.dt,
                                                     s_scale=args.spline_s_scale,
                                                     past_chunks=1, future_chunks=1)
    print(f'Lookahead (past=1,fut=1) spline avg s per refit = {s_used_la:.4g}')
    causal_global, s_used_c = make_causal_global_executed(chunks, dt=args.dt,
                                                            s_scale=args.spline_s_scale)
    print(f'Causal-global spline avg s per refit = {s_used_c:.4g}')
    spline, s_used = make_spline_executed(chunks, dt=args.dt,
                                            s_scale=args.spline_s_scale)
    print(f'Offline-global spline avg s = {s_used:.4g}')
    # All trajectories must share length for plotting.
    L = min(len(naive), len(hermite), len(rolling2), len(rolling2_warm),
            len(lookahead), len(causal_global), len(spline))
    naive_for_plot = naive[:L]
    hermite = hermite[:L]
    rolling2 = rolling2[:L]
    rolling2_warm = rolling2_warm[:L]
    lookahead = lookahead[:L]
    causal_global = causal_global[:L]
    spline = spline[:L]

    methods = {
        'naive':           naive_for_plot,
        'Hermite':         hermite,
        'rolling-2 cold':  rolling2,
        'rolling-2 warm':  rolling2_warm,
        'lookahead(1,1)':  lookahead,
        'causal global':   causal_global,
        'offline global':  spline,
    }
    report_stats('boundary metrics', methods, K=chunks.shape[1])

    plot_comparison(chunks, naive_for_plot, hermite, rolling2, rolling2_warm,
                    lookahead, causal_global, spline, dt=args.dt, out_path=args.out)

    return 0


if __name__ == '__main__':
    sys.exit(main())
