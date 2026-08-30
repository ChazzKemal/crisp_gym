"""Gripper event detection, shared by every method's response to it.

The gripper's stroke rate is fixed hardware -- roughly 0.57 s for the 85 mm stroke
even at the driver's 0.150 m/s maximum, and 2.27 s at the 0.0375 m/s the deploy path
actually uses. Measured over five recorded datasets, the commanded gripper channel is
*binary*: all 768 transitions in those episodes are exactly one frame wide. A policy
smears that into a 2-4 frame ramp, which is still far shorter than the stroke.

So every method that compresses a demonstration has to give the gripper time back,
and the arithmetic differs per method -- but the *question* does not. "Where does the
gripper act, including a window that straddles a chunk boundary?" is asked identically
by all of them, so it is answered once here; the responses live with the methods.

Both detectors are stateful across chunks on purpose. A window opened near the end of
one chunk must continue into the next, and an edge exactly on a seam is only visible
if the previous chunk's final level is remembered.
"""

from __future__ import annotations

import numpy as np

#: Gripper channel index in a ``(K, >=7)`` action row.
GRIP_COL = 6


class GripperCloseWindow:
    """Frames within ``n_frames`` of an open->close edge.

    Edge-triggered on the *close*, not level-triggered: staying closed through a
    carry fires nothing, so transport keeps whatever speedup the method chose and
    only the grasp itself is protected.
    """

    def __init__(self, n_frames: int, *, invert: bool = False):
        self.n_frames = max(0, int(n_frames))
        self.invert = bool(invert)
        self.prev_closed: bool | None = None
        self.remaining = 0

    def mask(self, actions: np.ndarray) -> np.ndarray:
        """``(K,)`` bool: frames inside a close window, carry included."""
        k = actions.shape[0]
        out = np.zeros(k, dtype=bool)
        if self.n_frames <= 0 or k == 0:
            return out

        g = np.clip(actions[:, GRIP_COL], 0.0, 1.0)
        if self.invert:
            g = 1.0 - g
        closed = g < 0.5

        if self.remaining > 0:                      # window opened in a prior chunk
            c = min(self.remaining, k)
            out[:c] = True
            self.remaining -= c

        was = bool(self.prev_closed) if self.prev_closed is not None else False
        for i in range(k):
            if closed[i] and not was:               # open->close edge = a grab
                end = i + self.n_frames
                out[i:min(end, k)] = True
                if end > k:
                    self.remaining = max(self.remaining, end - k)
            was = bool(closed[i])
        self.prev_closed = bool(closed[-1])
        return out


class GripperMotionRun:
    """Frames where the gripper command is *moving*, not merely settled.

    Deliberately not "rows that agree with the new value": after a close the command
    agrees with itself for the whole carry, so that definition would cover the entire
    transport and destroy exactly the speedup the method exists to produce. The run is
    where ``|dgrip|`` exceeds ``eps``, which is bounded by construction.
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = float(eps)
        self.prev_value: float | None = None

    def mask(self, actions: np.ndarray) -> np.ndarray:
        """``(K,)`` bool: frames where the command changed from the previous frame."""
        k = actions.shape[0]
        if k == 0:
            return np.zeros(0, dtype=bool)
        g = np.clip(actions[:, GRIP_COL], 0.0, 1.0).astype(np.float64)
        prev = self.prev_value if self.prev_value is not None else g[0]
        deltas = np.abs(np.diff(np.concatenate(([prev], g))))
        self.prev_value = float(g[-1])
        return deltas > self.eps
