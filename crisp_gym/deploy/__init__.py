"""Deploy-time actuation: the components a policy runner drives the robot with.

Extracted from ``examples/17_replay_dataset.py`` and ``examples/19_deploy_policy.py``.
Those two scripts had grown into libraries-in-disguise: ``19`` loaded ``17`` by file
path through importlib (a leading digit is not a valid identifier), executed all of
its ~2900 lines, and lifted 16 names back out by hand. Nothing under ``examples/``
ships in the wheel either, so none of it was importable by anything installed.

The rule for what belongs here: **a step transforms the chunk's contents; the loop
owns time, queueing, and when to call the policy.** Components live here; the
program that drives them lives in its runner.
"""
