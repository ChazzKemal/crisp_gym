"""Policy module for crisp_gym."""

from crisp_gym.policy.policy import Policy, make_policy, register_policy

__all__ = [
    "LerobotPolicy",
    "AsyncLerobotPolicy",
    "Policy",
    "register_policy",
    "make_policy",
]


def __getattr__(name: str):
    # Lazy-import heavy policy classes to avoid pulling in ManipulatorEnv
    # (and its camera/cv2 dependencies) at package import time.
    if name == "AsyncLerobotPolicy":
        from crisp_gym.policy.async_lerobot_policy import AsyncLerobotPolicy
        return AsyncLerobotPolicy
    if name == "LerobotPolicy":
        from crisp_gym.policy.lerobot_policy import LerobotPolicy
        return LerobotPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
