"""axis-angle <-> rotation_6d conversions, numpy-only.

The upstream repo reaches this through ``diffusion_policy``'s
``RotationTransformer``, which wraps ``pytorch3d.transforms``. pytorch3d is a
heavy dependency we do not want in the crisp_gym environment, so the two
functions actually used are reimplemented here against the *same* conventions:

``matrix_to_rotation_6d``
    ``matrix[..., :2, :].reshape(..., 6)`` -- the first two **rows** of the
    rotation matrix, flattened row-major.
``rotation_6d_to_matrix``
    Gram-Schmidt (Zhou et al. 2019) on those two rows, stacked back as rows.

``tests/test_rotation.py`` checks both against ``scipy.spatial.transform``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vector -> (..., 3, 3) rotation matrix."""
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    flat = axis_angle.reshape(-1, 3)
    mats = Rotation.from_rotvec(flat).as_matrix()
    return mats.reshape(axis_angle.shape[:-1] + (3, 3))


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 3) rotation vector."""
    matrix = np.asarray(matrix, dtype=np.float64)
    flat = matrix.reshape(-1, 3, 3)
    vecs = Rotation.from_matrix(flat).as_rotvec()
    return vecs.reshape(matrix.shape[:-2] + (3,))


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two rows, flattened."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,)).copy()


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt; tolerates non-orthonormal input."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def axis_angle_to_rotation_6d(axis_angle: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vector -> (..., 6) continuous rotation representation."""
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def rotation_6d_to_axis_angle(d6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3) rotation vector."""
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def convert_actions_7d_to_10d(raw_actions: np.ndarray) -> np.ndarray:
    """``[xyz(3), axis_angle(3), gripper(1)]`` -> ``[xyz(3), rot6d(6), gripper(1)]``.

    Mirrors ``diffusion_policy``'s ``_convert_actions`` for ``raw_dim == 7``
    with ``target_action_dim == 10`` and ``abs_action=True`` -- the exact path
    the upstream YAM single-arm config takes.
    """
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if raw_actions.shape[-1] != 7:
        raise ValueError(f"expected last dim 7, got {raw_actions.shape[-1]}")
    pos = raw_actions[..., :3]
    rot = axis_angle_to_rotation_6d(raw_actions[..., 3:6])
    gripper = raw_actions[..., 6:7]
    return np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)


def convert_actions_10d_to_7d(actions: np.ndarray) -> np.ndarray:
    """Inverse of :func:`convert_actions_7d_to_10d` (rotation is re-orthonormalised)."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape[-1] != 10:
        raise ValueError(f"expected last dim 10, got {actions.shape[-1]}")
    pos = actions[..., :3]
    rot = rotation_6d_to_axis_angle(actions[..., 3:9])
    gripper = actions[..., 9:10]
    return np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)


__all__ = [
    "axis_angle_to_matrix",
    "axis_angle_to_rotation_6d",
    "convert_actions_10d_to_7d",
    "convert_actions_7d_to_10d",
    "matrix_to_axis_angle",
    "matrix_to_rotation_6d",
    "rotation_6d_to_axis_angle",
    "rotation_6d_to_matrix",
]
