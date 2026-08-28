"""axis-angle <-> rotation_6d must match the pytorch3d conventions upstream uses."""

import numpy as np
from scipy.spatial.transform import Rotation

from bspline_core.rotation import (
    axis_angle_to_matrix,
    axis_angle_to_rotation_6d,
    convert_actions_7d_to_10d,
    convert_actions_10d_to_7d,
    matrix_to_rotation_6d,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
)


def _rotvecs(n=500, seed=0):
    return np.random.default_rng(seed).uniform(-2 * np.pi, 2 * np.pi, size=(n, 3))


def test_axis_angle_to_matrix_matches_scipy():
    rv = _rotvecs()
    assert np.allclose(axis_angle_to_matrix(rv), Rotation.from_rotvec(rv).as_matrix())


def test_rotation_6d_is_first_two_rows():
    """pytorch3d's matrix_to_rotation_6d is matrix[..., :2, :].reshape(6)."""
    mats = Rotation.from_rotvec(_rotvecs(50)).as_matrix()
    d6 = matrix_to_rotation_6d(mats)
    assert d6.shape == (50, 6)
    assert np.allclose(d6[:, :3], mats[:, 0, :])
    assert np.allclose(d6[:, 3:], mats[:, 1, :])


def test_rotation_round_trip_is_exact():
    rv = _rotvecs()
    back = rotation_6d_to_axis_angle(axis_angle_to_rotation_6d(rv))
    # Compare as rotations: rotvecs are only unique modulo 2*pi wrapping.
    diff = (Rotation.from_rotvec(rv) * Rotation.from_rotvec(back).inv()).magnitude()
    assert diff.max() < 1e-8


def test_gram_schmidt_repairs_perturbed_6d():
    """A network's raw 6d output is not orthonormal; decoding must still give SO(3)."""
    rv = _rotvecs(200)
    d6 = axis_angle_to_rotation_6d(rv) + np.random.default_rng(1).normal(scale=0.1, size=(200, 6))
    mats = rotation_6d_to_matrix(d6)
    assert np.allclose(np.linalg.det(mats), 1.0)
    eye = np.einsum("nij,nkj->nik", mats, mats)
    assert np.allclose(eye, np.eye(3)[None], atol=1e-10)


def test_action_7d_10d_round_trip():
    rng = np.random.default_rng(2)
    raw = np.concatenate(
        [
            rng.uniform(-1, 1, size=(300, 3)),
            _rotvecs(300, seed=3),
            rng.uniform(0, 1, size=(300, 1)),
        ],
        axis=1,
    ).astype(np.float32)
    ten = convert_actions_7d_to_10d(raw)
    assert ten.shape == (300, 10)
    back = convert_actions_10d_to_7d(ten)
    assert np.allclose(back[:, :3], raw[:, :3], atol=1e-6)
    assert np.allclose(back[:, 6], raw[:, 6], atol=1e-6)
    diff = (Rotation.from_rotvec(raw[:, 3:6]) * Rotation.from_rotvec(back[:, 3:6]).inv()).magnitude()
    assert diff.max() < 1e-6


def test_real_actions_convert(real_actions_7d):
    ten = convert_actions_7d_to_10d(real_actions_7d)
    assert ten.shape == (len(real_actions_7d), 10)
    assert np.isfinite(ten).all()
    # rot6d entries live in [-1, 1] because they are rotation-matrix rows.
    assert np.abs(ten[:, 3:9]).max() <= 1.0 + 1e-6
