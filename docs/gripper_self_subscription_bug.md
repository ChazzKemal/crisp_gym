# Gripper self-subscription bug in crisp_py

## Symptom

During replay (or any code path that calls `env.step()` →
`gripper.set_target()`), the gripper does the **opposite** of what was
commanded — open when it should close, close when it should open — and
exhibits stuttering "stop-start" motions as it rapidly alternates between
the correct and inverted command.

## Root cause

`crisp_py.Gripper.__init__` (`crisp_py/gripper/gripper.py:105-114`)
creates both a **publisher** and a **subscriber** on the same topic
(`/target_gripper_state`):

```python
# Publisher (line 105-106)
self._target_state_publisher = self.node.create_publisher(
    Float32, self.config.target_state_topic, qos_profile_system_default
)

# Subscriber on the SAME topic (line 108-114)
self.node.create_subscription(
    Float32,
    self.config.target_state_topic,
    self._callback_target_state,
    qos_profile_system_default,
    callback_group=ReentrantCallbackGroup(),
)
```

When `set_target(target)` is called (`gripper.py:337-350`):

```python
def set_target(self, target: float, *, epsilon: float = 0.1):
    # Step A: store the UNNORMALIZED (raw) value — CORRECT
    self._target = self._unnormalize(target)

    # Step B: publish the NORMALIZED value to the topic
    msg = Float32()
    msg.data = float(target)       # ← normalized (0–1)
    self._target_state_publisher.publish(msg)
```

Step A correctly stores the raw joint angle in `_target`. But step B
publishes the *normalized* value on the topic. Because FastDDS delivers
messages to subscribers on the same node (intra-process loopback), the
Gripper's own subscriber fires:

```python
def _callback_target_state(self, msg: Float32):
    # Step C: OVERWRITES _target with the normalized value — BUG
    self._target = float(msg.data)
```

Now `_target` holds a normalized value (e.g. `0.0`) where the rest of
the code expects a raw joint angle (e.g. `0.8`).

## How the inversion happens

For the Ridgeback UR10e Robotiq 2F-85 gripper config:

```yaml
min_value: 0.8   # raw closed position (radians)
max_value: 0.0   # raw open position (radians)
```

Normalization: `normalize(raw) = (raw - 0.8) / (0.0 - 0.8) = (0.8 - raw) / 0.8`

### Example: commanding "close" with `set_target(0.0)`

| Step | `_target` value | Meaning |
|------|-----------------|---------|
| A: `_unnormalize(0.0)` | `0.8` | raw closed — **correct** |
| C: callback stores `0.0` | `0.0` | raw 0.0 = open position — **wrong** |

The 50 Hz publish timer (`_callback_publish_target`) then reads the
corrupted `_target = 0.0`:

```
_normalize(0.0) = (0.8 - 0.0) / 0.8 = 1.0  →  "fully open"
```

It sends a GripperCommand goal to **open** the gripper — the exact
opposite of the intended "close."

### The stuttering effect

Steps A and C race against each other. On every `set_target()` call:
1. `_target` is briefly correct (step A)
2. A few milliseconds later, the loopback callback fires and inverts it
   (step C)
3. The publish timer may fire between A and C (sending correct command)
   or after C (sending inverted command)

This causes the gripper to rapidly alternate between correct and
inverted commands — visible as stuttering "open-close-stop" motions.

## Why it doesn't affect mocap recording

During mocap teleop (`13_ridgeback_mocap_record.py`), the env's
Gripper instance never calls `set_target()`. The mocap tracker
(`track_mocap.py`) drives the gripper directly via its own
`GripperCommand` action client and publishes raw joint angles to
`/target_gripper_state`. The env's Gripper only reads via the
subscription — no self-publishing, so no loopback race.

## Fix (workaround in replay script)

In `17_replay_dataset.py`, after creating the env and before any
`env.step()` calls:

```python
def fix_gripper_self_subscription(env):
    gripper = getattr(env, "gripper", None)
    if gripper is None:
        return
    gripper._target_state_publisher.publish = lambda msg: None
```

This no-ops the publisher so `set_target()` never triggers the
loopback. `_target` is still set correctly by `set_target()`'s direct
assignment (step A). The Gripper's 50 Hz `_callback_publish_target`
(which sends GripperCommand goals to the hardware) still works — it
reads `_target`, not the topic.

### Why replacing the callback doesn't work

The first attempt was to replace the callback method:

```python
gripper._callback_target_state = lambda msg: None  # DOESN'T WORK
```

This replaces the attribute on the Python object, but rclpy's
subscription internally holds a reference to the **original bound
method** object captured at `create_subscription()` time. Replacing the
attribute after the fact doesn't affect the subscription's stored
reference. The original callback continues to fire.

## Proper fix (for crisp_py)

The subscription callback should unnormalize the received value before
storing it, matching what `set_target()` does:

```python
def _callback_target_state(self, msg: Float32):
    # msg.data is normalized (0–1), _target must be raw
    self._target = self._unnormalize(float(msg.data))
```

Or, if the intent is that the topic carries raw values (as
`track_mocap.py` uses it), then `set_target()` should publish the
raw value:

```python
msg.data = float(self._unnormalize(target))  # publish raw, not normalized
```

Either way, the publisher and subscriber must agree on the convention.
Currently they don't — `set_target()` publishes normalized,
`_callback_target_state` stores as raw, and the mismatch causes the
inversion.

## Additional note: track_mocap.py convention mismatch

`track_mocap.py` publishes **raw joint angles** to
`/target_gripper_state` (`0.0` = open, `0.8` = closed), while
`Gripper.set_target()` publishes **normalized** values (`0.0` = closed,
`1.0` = open) to the same topic. Any subscriber must know which source
is active to interpret the values correctly. This is documented in
`docs/ridgeback_target_pose_ownership.md` under "Gripper convention
mismatch."
