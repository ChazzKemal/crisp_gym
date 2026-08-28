# `/target_pose` ownership on the Ridgeback UR10e

This doc covers a recurring footgun in the Ridgeback `crisp_gym` flow: who is
allowed to publish `/target_pose` to the `cartesian_controller`, and what
breaks when the wrong process owns the topic. It also records the workarounds
currently used by the recording (`13_ridgeback_mocap_record.py`) and replay
(`14_ridgeback_replay.py`) example scripts so we can replace them with a
proper fix later.

## TL;DR

- `ur10e_ridgeback_env.yaml` sets `publish_target_pose: false`. That is
  correct for **mocap recording** (so `track_mocap.py` can own the topic) but
  silently wrong for **replay** (the replay script needs the env's `Robot`
  client to publish).
- Both example scripts now monkey-patch the `Robot` client at runtime to
  flip the topic ownership in opposite directions:
  - script 13 *silences* `Robot._target_pose_publisher`,
  - script 14 *re-enables* it (creates the publisher + timer and drops the
    external subscription).
- A real fix should replace the monkey-patches with explicit env
  configuration (an override at `make_env` time, or two separate YAMLs).

## What `publish_target_pose` actually does

`crisp_py.Robot.__init__` (`crisp_py/robot/robot.py:108-186`) branches on the
flag at construction time:

```python
if self.config.publish_target_pose:
    # Robot client OWNS /target_pose:
    self._target_pose_publisher = self.node.create_publisher(
        PoseStamped, self.config.target_pose_topic, qos_profile_system_default
    )
    self._target_pose_subscriber = None
    # ... and a 20 Hz timer re-publishes self._target_pose:
    self.node.create_timer(
        1.0 / self.config.publish_frequency,
        self._callback_publish_target_pose,
        ReentrantCallbackGroup(),
    )
else:
    # Some EXTERNAL node owns /target_pose; mirror it locally so
    # `robot.target_pose` reflects the live commanded value:
    self._target_pose_publisher = None
    self._target_pose_subscriber = self.node.create_subscription(
        PoseStamped,
        self.config.target_pose_topic,
        self._callback_external_target_pose,   # writes self._target_pose
        qos_profile_system_default,
        callback_group=ReentrantCallbackGroup(),
    )
    # NO publish timer is created.
```

Important consequences of `publish_target_pose=False`:

1. There is no publisher *at all* on `target_pose_topic` from the env's
   `Robot` client. The 20 Hz republish timer is also skipped.
2. `Robot.set_target(pose=...)` (`robot.py:425-437`) only assigns the
   in-memory `self._target_pose`; with no publisher and no timer, **nothing
   ever reaches the controller**.
3. The external subscription continually overwrites `self._target_pose` with
   whatever the most recent message on `/target_pose` was. So even the
   in-memory buffer cannot be relied on as your own command.

## Why script 13 (mocap recording) needs `publish_target_pose=false`

`13_ridgeback_mocap_record.py` is paired with the external mocap tracker
(`tools/master_launch.sh up --track --controller crisp` →
`clearpath_remote_ws/.../track_mocap.py`). Mocap is the authoritative source
of `/target_pose` for the entire episode.

If `publish_target_pose=true`:

- The env's `Robot` client *also* publishes to `/target_pose` at 20 Hz from
  its internal `_target_pose` buffer.
- That buffer was initialised from `current_pose` at startup and never
  refreshed (the recorder never calls `set_target`), so the env streams a
  stale "hold this start pose" command.
- The `cartesian_controller` ends up averaging two competing publishers
  (mocap's live commands + the env's stale hold), and the arm barely moves.

Setting `publish_target_pose=false` solves the dual-publisher problem at
construction time but introduces a second issue: the recorder needs to read
"what was just commanded" for action logging, and the env's `_target_pose`
buffer would still be the stale start pose. The external-subscription branch
above fixes that by mirroring the live mocap value into `_target_pose`.

### Workaround currently in script 13

Even with `publish_target_pose=false`, **script 13 still applies an extra
runtime patch** (`13_ridgeback_mocap_record.py:110-131`,
`silence_env_target_publishers`). That helper replaces
`env.robot._target_pose_publisher.publish` and
`env.robot._target_joint_publisher.publish` with a no-op.

Why it exists, given the YAML already disables target_pose publishing:

- `_target_joint_publisher` is **always** created regardless of
  `publish_target_pose`, and its 20 Hz timer (`robot.py:193-199`) keeps
  re-sending `self._target_joint`. After the home move that buffer holds the
  arm's joint positions at episode start; that is a competing command path
  for the controller stack and worth silencing during mocap teleop.
- The `_target_pose_publisher` no-op is defensive — it is `None` when
  `publish_target_pose=false`, so `publish = _noop` would throw
  `AttributeError`. **This is a latent bug in the silencer**: today it works
  only because the YAML flag is consistent. If the YAML were ever flipped
  back to `true` without remembering to update the silencer, the recorder
  would crash on startup. Worth tightening when we do the real fix.

(In practice the silencer's docstring is mostly historical — it predates
the YAML change. The joint-publisher half is still load-bearing.)

## Why script 14 (replay) needs `publish_target_pose=true`

`14_ridgeback_replay.py` reads `observation.state.target` from a recorded
LeRobot v3 episode and calls `env.robot.set_target(pose=pose)` once per
frame. That call only updates `self._target_pose`; **the script relies on
crisp_py's internal 20 Hz timer to forward the buffer to the controller**.

With `publish_target_pose=false` (the YAML default), neither the publisher
nor the timer exists, so:

- The arm sees no new `/target_pose` messages and stays where it is.
- The gripper still moves because `env.gripper` is a separate client that
  publishes on its own topic (`/gripper/...`) and is unaffected.

This matches the observed symptom: "the replay seems to be not moving the
robot, the gripper is doing something."

### Workaround currently in script 14

After `make_env(...)` and **before** `wait_until_ready()`, the script
monkey-patches the env's `Robot` to look as if it had been built with
`publish_target_pose=true`:

1. Destroy `env.robot._target_pose_subscriber` (so incoming `/target_pose`
   messages — e.g. stray mocap traffic — cannot overwrite `_target_pose`).
2. Create `env.robot._target_pose_publisher` on the same topic with
   `qos_profile_system_default`.
3. Create a `1 / publish_frequency` timer that calls
   `env.robot._callback_publish_target_pose`, mirroring the timer that
   `Robot.__init__` would have created at line 178-186.
4. Flip `env.robot.config.publish_target_pose = True` for any code that
   inspects the flag later.

This is the inverse of script 13's silencer and lives next to the
replay-loop code so the dependency is visible.

## Why this is fragile

- Both scripts mutate `crisp_py.Robot` private fields after construction.
  Any rename inside `crisp_py` (e.g. `_target_pose_publisher` →
  `target_pose_publisher`) will silently break both scripts.
- The "right" topic ownership depends on the *script*, not the *env*. The
  YAML can only be right for one of them.
- Script 13's silencer has a latent `AttributeError` if the YAML flag is
  ever flipped back to `true` (see above).
- The same `Robot` object would behave very differently in `set_target` /
  `target_pose` semantics depending on which monkey-patch (if any) ran
  first. That is invisible to a casual reader of either script.

## Suggested proper fixes (for later)

In rough order of effort:

1. **Plumb `publish_target_pose` through `make_env` overrides.** Let the
   caller pass `robot_config={"publish_target_pose": True}` (or a
   dataclass/dict override) so each script can declare the topic ownership
   it needs without touching the YAML or the Robot internals. Then both
   monkey-patches go away. Likely the smallest correct change.

2. **Add a `Robot.set_target_pose_ownership(owner: "client" | "external")`
   method in `crisp_py`.** Encapsulates the destroy-subscription /
   create-publisher / create-timer dance (and the inverse) so we are not
   reaching into `_target_pose_*` from outside. The example scripts then
   call this method explicitly.

3. **Two YAMLs:** keep `ur10e_ridgeback_env.yaml` for mocap recording
   (`publish_target_pose: false`) and add
   `ur10e_ridgeback_replay_env.yaml` (`publish_target_pose: true`). Most
   explicit, zero coupling, costs an extra config file. Picks itself if
   options 1 and 2 turn out to be too invasive in `crisp_py`.

4. **Independently of the above, fix the latent
   `silence_env_target_publishers` bug in script 13** so it tolerates
   `_target_pose_publisher is None` (e.g. `if self._target_pose_publisher
   is not None: self._target_pose_publisher.publish = _noop`). One-liner.

5. **Decide what `_target_joint_publisher` should do under
   `publish_target_pose=False`.** Today it is always on, even when the
   intent of the flag is "external owns the target topics". If we go with
   option 2 above, the new method should also handle the joint publisher
   so the recorder no longer needs its own silencer.

---

# Gripper convention mismatch in the recorder dataset

While debugging the replay we found a second, independent issue: the
recorder writes the gripper into the parquet under **two opposite
conventions**, one in the action column and one in the observation
columns. The replay script now defaults to the action column to sidestep
this, but the underlying inconsistency should be fixed in the recorder.

## What the recorder writes

`13_ridgeback_mocap_record.py:287-296` (`data_fn`) writes the action:

```python
if env.gripper._target is not None:
    grip_action = float(env.gripper.target)   # crisp_py normalised
else:
    grip_action = float(env.gripper.value)    # crisp_py normalised
action = np.concatenate([target_pose, np.array([grip_action], dtype=np.float32)])
```

`crisp_py.Gripper` normalises with `(raw - min) / (max - min)`. The
`ur10e_ridgeback_env.yaml` gripper config has `min_value: 0.8,
max_value: 0.0`, so:

- `gripper.value`  → 1 = open, 0 = closed
- `gripper.target` → 1 = open, 0 = closed
- `gripper.set_target(x)` consumes the same convention.

So `action[6]` is in **crisp_py convention: 1 = open, 0 = closed**.

`ManipulatorEnv._get_obs` (`manipulator_env.py:352-378`) writes the
observation columns:

```python
gripper_value = 1 - np.array([self.gripper.value])           # GRIPPER_OBS
...
target_val = (self.gripper.target if self.gripper._target is not None
              else self.gripper.value)
obs[GRIPPER_TARGET_OBS] = np.array([1 - target_val], dtype=np.float32)
```

Both observation columns are flipped (`1 - ...`) to match the LeRobot
convention used by Franka recordings (where the raw gripper has
`min < max`, i.e. `gripper.value` already means 1 = closed). For the
ridgeback YAML the flip means:

- `observation.state.gripper`         → **0 = open, 1 = closed**
- `observation.state.gripper_target`  → **0 = open, 1 = closed**

For one frame where the gripper is fully open:

| column                                 | value | convention                |
|----------------------------------------|-------|---------------------------|
| `observation.state.gripper`            | `0.0` | LeRobot (0 = open)        |
| `observation.state.gripper_target`     | `0.0` | LeRobot (0 = open)        |
| `action[6]`                            | `1.0` | crisp_py (1 = open)       |

That is genuinely a bug: an obs/action pair for the same physical state
encodes "gripper open" with two opposite numbers. A naive policy trained
on this dataset will learn the inverse mapping; a notebook user
inspecting the columns side by side will be confused; and the original
header docstring of `14_ridgeback_replay.py` even asserted the wrong
convention.

There is also a secondary issue with `observation.state.gripper_target`:
the recorder falls back to `self.gripper.value` for any frame *before*
the first mocap `/target_gripper_state` message lands. So the prefix of
the column is not actually the operator's target — it is the measured
state at recording time. The action column has the same fallback path,
so it is contaminated identically, but the user explicitly asked us to
prefer it because it is at least internally consistent with the
recorder's own convention.

## Workaround currently in script 14

`14_ridgeback_replay.py` now:

1. Adds `action` to `--gripper-source` and makes it the default.
2. Reads `row["action"][6]` straight through (no flip) when the source
   is `action`.
3. Still supports `target` and `obs` for completeness, but flips them
   internally to crisp_py convention before passing to `set_target`.
4. Updates the header docstring and `print_summary` to describe the
   crisp_py convention rather than the (wrong) LeRobot one.

This matches the previous gripper-convention behaviour the *user* was
expecting — just sourced from the column that actually means what it
says.

## Suggested proper fixes (gripper)

1. **Flip the action column in the recorder.** Make
   `13_ridgeback_mocap_record.py` write `1 - gripper.target` into
   `action[6]` so the action and the observation columns agree on
   "0 = open, 1 = closed". Then the replay just needs to do
   `set_target(1 - action[6])`. Smallest semantic change but it
   silently breaks any policy already trained on the current dataset
   convention.

2. **Stop flipping the observation columns in the env.** Have
   `ManipulatorEnv._get_obs` write `gripper.value` and `gripper.target`
   straight into the parquet, matching `set_target`'s convention. This
   is consistent with `crisp_py` semantics but breaks compatibility
   with existing Franka recordings that rely on the flip — would need
   to be conditional on the gripper config (e.g. only flip when
   `min < max`).

3. **Fix the `_target is None` fallback in
   `ManipulatorEnv._get_obs`.** Either drop the frame, raise, or write
   a sentinel (e.g. `np.nan`), so `observation.state.gripper_target`
   stops silently changing meaning for the prefix of every episode.
   Apply the same fix to the recorder's action path.

4. **Add a `gripper.is_target_initialized` property to `crisp_py`** so
   downstream code can check this without poking at `_target`.

5. **Document the gripper convention explicitly in
   `ur10e_ridgeback_env.yaml`** so the next person editing it does not
   silently flip min/max and break every dataset that came before.
