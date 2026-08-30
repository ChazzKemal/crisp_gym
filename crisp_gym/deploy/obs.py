"""Observation gathering that survives a silent sensor.

Moved out of ``examples/19_deploy_policy.py``. crisp_py raises ``RuntimeError`` out
of ``current_image`` / ``joint_values`` / ``Gripper.value`` when a topic has never
produced a message, which would abort a deploy run over one camera that did not come
up. Instead the schema is captured once while everything is healthy, and later gaps
are zero-filled to the right shape so the chunk source still sees a well-formed obs
dict -- with a count per distinct error surfaced in ``summary.json``, so a run that
was quietly missing a sensor is visible afterwards rather than merely survivable.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

_ZEROFILL_WARNED: set[str] = set()
_ZEROFILL_COUNTS: dict[str, int] = {}


def _build_obs_schema(env) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """(shape, dtype) for every obs key env._get_obs() is expected to produce.

    Derived from env config — does NOT require sensors to be alive.
    Cameras: from cam.config.resolution (a (H, W) tuple — crisp_py
    unpacks it as target_h, target_w in `_resize_with_aspect_ratio`).
    State sub-keys: fixed dims (cartesian/target=6, joints=6, gripper=1).
    """
    schema: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for cam in getattr(env, "cameras", []) or []:
        res = getattr(cam.config, "resolution", None)
        if res is None or len(res) != 2:
            # Falls back to a reasonable default; the actual frame shape
            # would be determined at runtime by the CameraInfo callback.
            # If the user truly has no camera info, zero-fill at this
            # shape is still better than RuntimeError.
            h, w = 480, 640
        else:
            h, w = int(res[0]), int(res[1])
        key = f"observation.images.{cam.config.camera_name}"
        schema[key] = ((h, w, 3), np.dtype(np.uint8))
    state_dims = {
        "observation.state.cartesian": (6,),
        "observation.state.joints": (6,),
        "observation.state.gripper": (1,),
        "observation.state.gripper_target": (1,),
        "observation.state.target": (6,),
    }
    state_keys = getattr(env.config, "observations_to_include_to_state", []) or []
    for k in state_keys:
        if k in state_dims:
            schema[k] = (state_dims[k], np.dtype(np.float32))
    return schema


def _get_obs_zerofill(env, schema, last_obs_holder):
    """Call env._get_obs(); on RuntimeError, build an obs dict from `schema`
    with zeros for every key. `last_obs_holder` is a [obs] list so we can
    preserve previously-good sub-keys (e.g. `task`) when only some sensors
    fail. First occurrence per unique error message logs a WARNING;
    subsequent occurrences are counted silently in `_ZEROFILL_COUNTS`.
    """
    try:
        obs = env._get_obs()
        last_obs_holder[0] = obs
        return obs
    except RuntimeError as e:
        msg = str(e)
        if msg not in _ZEROFILL_WARNED:
            _ZEROFILL_WARNED.add(msg)
            logger.warning(
                "env._get_obs() raised (%s) — zero-filling missing sensor data; "
                "subsequent occurrences will be counted silently and surfaced "
                "in summary.json.", msg,
            )
        _ZEROFILL_COUNTS[msg] = _ZEROFILL_COUNTS.get(msg, 0) + 1
        obs = dict(last_obs_holder[0] or {})
        for key, (shape, dtype) in schema.items():
            obs.setdefault(key, np.zeros(shape, dtype=dtype))
        obs.setdefault("task", "")
        return obs
