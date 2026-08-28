# `15_ridgeback_mocap_record.py` — FPS collapse and the image-writer fix

Problem-and-solution writeup for the FPS drop observed while recording mocap
teleop episodes with `examples/15_ridgeback_mocap_record.py`.

## TL;DR

`crisp_gym/record/recording_manager.py:_create_dataset` calls
`LeRobotDataset.create(..., use_videos=True)` but never calls
`dataset.start_image_writer(...)`. With `use_videos=True` and no image writer,
`dataset.add_frame` encodes each camera frame to the episode's PNG staging pool
**synchronously on the writer subprocess**. For the Ridgeback rig
(multiple Orbbec RGB streams at 20 Hz) that encode cannot keep up, the 16-slot
`mp.JoinableQueue` between the recording loop and the writer subprocess fills,
the parent-side `queue.put(...)` blocks, and the recording loop drops below
20 Hz — visible in the console as

```
Frame processing took too long: … seconds too long i.e. XX.XX FPS.
Consider decreasing the FPS or optimizing the data function.
```

The fix is to call `dataset.start_image_writer(num_processes=8, num_threads=1)`
immediately after `LeRobotDataset.create(...)`, *inside the writer subprocess*
(i.e., inside `_create_dataset`, which is invoked from `_writer_proc`). 8
workers do the PNG encode off the subprocess's critical path, `add_frame`
becomes a fast handoff, the queue drains, the parent loop hits its 1/fps
budget.

This fix is orthogonal to the `root=` dataset-location argument and to
`batch_encoding_size=...`, both of which appeared in the discussion but do
not affect FPS.

## Symptom

- Expected: `15_ridgeback_mocap_record.py --fps 20` records at 20 Hz.
- Observed: main loop warns `Frame processing took too long … XX.XX FPS` from
  `crisp_gym/record/recording_manager.py:319`, effective FPS significantly
  below 20.
- 12/13 (non-mocap or pre-fix mocap recorders) share the same writer code
  path and are vulnerable to the same symptom, but the mocap topology in 15
  puts the most pressure on it because the env's ROS executor is already
  contended with the extra `/target_pose` subscription
  (`examples/15_ridgeback_mocap_record.py:234`).

## Architecture of the recording pipeline

Relevant files and lines in `Yunfei/crisp_gym/crisp_gym/record/recording_manager.py`:

1. `RecordingManager.__init__` spawns the writer subprocess via
   `mp.Process(target=self._writer_proc, ...)` at line 77 and starts it at
   line 83. On Linux this is a `fork`, so the parent process state is
   inherited wholesale into the child.
2. `_writer_proc` (line 167) runs **in the child**. It calls
   `self._create_dataset()` at line 170, signals `self.dataset_ready`, then
   enters a message loop over `self.queue`.
3. `_create_dataset` (line 130) is where the dataset object is constructed.
   Current code:
   ```python
   dataset = LeRobotDataset.create(
       repo_id=self.config.repo_id,
       fps=self.config.fps,
       robot_type=self.config.robot_type,
       features=self.config.features,
       use_videos=True,
   )
   ```
   No `root=`, no `start_image_writer(...)`.
4. The parent-side recording loop is in `record_episode` (line 274). Its
   inner loop (lines 301-323):
   ```python
   while self.state == "recording":
       frame_start = time.time()
       obs, action = data_fn()                           # parent
       ...
       self.queue.put({"type": "FRAME", "data": (obs, action, task)})
       sleep_time = 1 / self.config.fps - (time.time() - frame_start)
       if sleep_time > 0:
           time.sleep(sleep_time)
       else:
           logger.warning("Frame processing took too long: ...")
   ```
5. `self.queue` is `mp.JoinableQueue(self.config.queue_size)` with default
   `queue_size = 16` from `recording_manager_config.py:32`. `put()` is
   blocking when the queue is full.
6. The child-side frame consumer is in `_writer_proc` (lines 180-211):
   ```python
   if mtype == "FRAME":
       obs, action, task = msg["data"]
       frame = {"action": action.astype(np.float32)}
       for feature_name in self.config.features: ...
       frame["observation.state"] = concatenate_state_features(obs, ...)
       if _ADD_FRAME_HAS_TASK:
           dataset.add_frame(frame, task=task)
       else:
           frame["task"] = task
           dataset.add_frame(frame)
   ```

## Root cause: synchronous image encoding in the writer subprocess

LeRobot `LeRobotDataset.create(..., use_videos=True)` produces a dataset that
stores images as **per-episode PNGs** during recording and concatenates them
into an MP4 inside `save_episode()`. The *encode* step (numpy RGB array →
PNG bytes on disk) has two code paths:

- **With an image writer started**: `add_frame` hands image arrays to a
  pool of worker processes (started by `dataset.start_image_writer(...)`) and
  returns immediately. The PNG encode runs in parallel on the pool, off the
  caller's critical path.
- **Without an image writer started**: `add_frame` performs the PNG encode
  inline on the caller — i.e., on the writer subprocess, blocking the rest
  of the `_writer_proc` message loop.

`_create_dataset` is in the second state today. The encode cost per frame
scales with:

- Number of camera streams declared in `features` (here: the Orbbec streams
  the env exposes for Ridgeback).
- Image resolution (Orbbec at 1080p is dramatically worse than 720p).
- PNG compression level (LeRobot default, single-threaded).

At 20 Hz × 3 × 1080p RGB the synchronous encode cannot fit into the 50 ms
budget per frame on a single worker subprocess. As soon as the subprocess
falls behind, the backpressure chain kicks in:

1. Child stops calling `queue.get()` because it is stuck inside
   `dataset.add_frame(...)`.
2. Parent-side `self.queue.put(FRAME)` blocks on the full 16-slot queue.
3. Parent loop's per-iteration wall clock exceeds `1 / fps`, `sleep_time`
   goes negative, the warning at line 319 fires, and measured FPS
   collapses to whatever the synchronous encode rate happens to be.

This is exactly the failure mode the `"Frame processing took too long"`
warning was designed to surface. It is not a bug in 15 specifically — it is
a latent limitation of `_create_dataset` that only becomes visible when the
total image-encode cost per frame approaches `1 / fps`, which is the case
for the Ridgeback camera rig but not for lower-resolution or
single-camera rigs.

## Ancillary, orthogonal edits in the pasted snippet

The snippet that prompted this writeup also contained:

```python
root="/mnt/DataExtern/LSY-lab/real_world_5",
# image_writer_threads=1,
# image_writer_processes=16,
# batch_encoding_size=8,
```

None of these affect FPS:

- `root=` controls *where on disk* the dataset is written. Default is
  `~/.cache/huggingface/lerobot/<repo_id>` via `HF_LEROBOT_HOME`. Only
  relevant if the home partition is too small or too slow (an HDD might in
  principle introduce I/O stalls, but not at the rates recording produces
  before encode, and the symptom would be different). If the recording
  target is an external SSD, `root=` is the right knob, but it has to be
  threaded through both branches of `_create_dataset` (the `create` branch
  and the `resume` branch at line 135) **and** through the collision check
  on line 150, which currently only looks at `HF_LEROBOT_HOME / repo_id`.
  That is a separate change and should not be bundled with the FPS fix.
- `image_writer_threads=1, image_writer_processes=16` as ctor kwargs to
  `LeRobotDataset.create(...)` is an alternate shape of the same fix as
  `start_image_writer(...)`, with a different worker count. Semantically
  equivalent, not additive. Pick one.
- `batch_encoding_size=8` batches the MP4 encode at `save_episode()` time.
  It reduces save-episode latency, not `add_frame` throughput. Irrelevant to
  the live FPS loop.

## Alternative FPS causes the image-writer fix does NOT address

Before committing to this as the full fix, two other causes should be ruled
out with a short dry run:

1. **`data_fn()` itself being slow.** In
   `examples/15_ridgeback_mocap_record.py:290` `data_fn` calls
   `env.get_obs()`, which reads from ROS camera subscriptions. If the
   Orbbec topic rate is below 20 Hz or the env's ROS executor is contended
   with the extra `/target_pose` subscription at
   `examples/15_ridgeback_mocap_record.py:234`, `get_obs` itself stalls
   and the starting line of each iteration is already late. Quick diagnosis:
   `ros2 topic hz <orbbec topic>` and `ros2 topic hz /target_pose`
   alongside a dry run. The image writer fix does nothing for this case.
2. **Parent-side pickle cost on `queue.put`.** `mp.JoinableQueue.put(msg)`
   pickles the frame (including camera numpy arrays) in the parent before
   the child can receive it. At 3 × 1080p RGB × 20 Hz this is ~360 MB/s of
   pickle throughput on the parent side, which can alone exhaust the
   per-frame budget. Mitigations are downsampling images in `data_fn`
   before they hit the queue, or switching to shared memory — not the image
   writer. Diagnosis: add a `logger.debug(self.queue.qsize())` or wrap
   `queue.put` with a timer.

Confirm which of the three causes is dominant before editing, because the
image-writer fix is the correct response only if cause (3) — child-side
synchronous encode backpressure — is the one actually hurting.

## Proposed fix (not yet applied)

Recommended shape is **Option A**: extend `RecordingManagerConfig` and
`_create_dataset`, so the fix is available to every recorder (12, 13, 15,
spacemouse) without forking per-script code.

### 1. `crisp_gym/record/recording_manager_config.py`

Add two fields to `RecordingManagerConfig` (around line 32, alongside the
existing system-configuration block):

```python
image_writer_processes: int = 0   # 0 = feature disabled, preserves current behavior
image_writer_threads: int = 1
```

Default `image_writer_processes = 0` is deliberate: it keeps 12/13 and every
other recorder bit-for-bit identical until they opt in. The fix is fully
backwards compatible.

### 2. `crisp_gym/record/recording_manager.py:_create_dataset`

Immediately after the existing `LeRobotDataset.create(...)` call at line 157,
before the `logger.debug(f"Dataset created with meta: {dataset.meta}")` at
line 164:

```python
if self.config.image_writer_processes > 0:
    logger.info(
        f"Starting image writer: processes={self.config.image_writer_processes}, "
        f"threads={self.config.image_writer_threads}"
    )
    dataset.start_image_writer(
        num_processes=self.config.image_writer_processes,
        num_threads=self.config.image_writer_threads,
    )
    logger.info("Image writer started inside subprocess.")
```

Placement detail that matters: `_create_dataset` is called from
`_writer_proc` (line 170) which already runs inside the writer subprocess.
Calling `start_image_writer` here means the writer workers are spawned as
children of the writer subprocess, which is exactly what LeRobot's internal
bookkeeping expects. Do not move this call up into `RecordingManager.__init__`
or into the parent process — the workers would be in the wrong process tree
and frames would not reach them.

### 3. `examples/15_ridgeback_mocap_record.py`

Pass the new fields through when constructing `RecordingManagerConfig` at
line 270:

```python
rec_config = RecordingManagerConfig(
    features=features,
    repo_id=args.repo_id,
    robot_type="ur10e",
    fps=args.fps,
    num_episodes=args.num_episodes,
    resume=args.resume,
    push_to_hub=args.push_to_hub,
    image_writer_processes=args.image_writer_processes,
    image_writer_threads=args.image_writer_threads,
)
```

And add the matching CLI flags near the other `parser.add_argument(...)`
calls (around lines 163-216):

```python
parser.add_argument(
    "--image-writer-processes",
    type=int,
    default=8,
    help="LeRobot image writer worker process count. 0 disables the writer "
         "(synchronous encode — causes FPS drops on multi-camera rigs).",
)
parser.add_argument(
    "--image-writer-threads",
    type=int,
    default=1,
    help="LeRobot image writer threads per process.",
)
```

Defaults of `8, 1` match the pasted snippet's `start_image_writer` form. 8
workers is enough for 3 Orbbec 1080p streams at 20 Hz with headroom.

### 4. Do NOT touch 12/13 in the same pass

Unless explicitly scoped. They share the same latent limitation but their
current defaults (`image_writer_processes = 0`) keep them bit-for-bit
identical. Opting them in is a trivial follow-up of adding the two CLI flags
to each recorder script; do it only when those recorders are actively being
used and the FPS drop is observed there too.

## Worker count choice

Two sizings were discussed:

| form | workers |
|---|---|
| `dataset.start_image_writer(num_processes=8, num_threads=1)` | 8 |
| `LeRobotDataset.create(..., image_writer_processes=16, image_writer_threads=1)` | 16 |

For 3 Orbbec RGB streams at 1080p/20 Hz, 8 workers fully cover the encode
cost with headroom. 16 is overkill unless the rig grows to 4+ streams or the
resolution steps up to 4K. Default to 8; bump to 16 if a 60-second dry run
still trips `"Frame processing took too long"` at 8.

`num_threads=1` is correct for PNG encoding in LeRobot: the underlying
encode (PIL) is single-threaded, so threads-per-process above 1 yields no
benefit. Keep it at 1.

## Rejected alternatives

- **Option B: subclass `RecordingManager` inside `15_…` and override
  `_create_dataset`.** Keeps the change contained to script 15. Rejected
  because it duplicates the resume branch, the collision check, and the
  `_ADD_FRAME_HAS_TASK` shim — all three of which already have drift-prone
  history in this file. One source of truth is cheaper to maintain.
- **Option C: class-level monkey-patch of `_create_dataset` in `15_…`
  before `make_recording_manager(...)`.** Smallest diff, but fragile and
  spooky. Not suitable for a change that ships.
- **Dropping `fps` from 20 to 15 or 10.** Sidesteps the root cause rather
  than fixing it, and the downstream imitation-learning work expects 20 Hz
  data.
- **Disabling `use_videos=True`.** Would avoid PNG encoding but produce
  per-frame images on disk instead of an MP4 per episode, which bloats the
  dataset and breaks every downstream consumer that expects the video
  layout.

## Verification plan after the fix lands

1. Dry run: `python examples/15_ridgeback_mocap_record.py --require-mocap
   --num-episodes 1 --task "fps test"` with the mocap pipeline live. Record
   a single 10-second episode.
2. Expect: no `"Frame processing took too long"` warnings in the console.
3. Expect: `dataset.meta.info` reports 200 frames (10 s × 20 Hz) for the
   episode, not fewer.
4. Optional instrumentation: temporarily add
   `logger.debug(f"queue size: {self.queue.qsize()}")` inside
   `record_episode`'s inner loop; with the fix in place, qsize should
   stabilize below ~4, not oscillate up to 16.
5. If (4) shows the queue still filling, the dominant cause is not the
   image writer — fall back to the two alternative-cause diagnostics in
   the "Alternative FPS causes" section above.

## References

- `Yunfei/crisp_gym/crisp_gym/record/recording_manager.py:130` —
  `_create_dataset`, current definition without `start_image_writer`.
- `Yunfei/crisp_gym/crisp_gym/record/recording_manager.py:167` —
  `_writer_proc`, the subprocess entry point.
- `Yunfei/crisp_gym/crisp_gym/record/recording_manager.py:274` —
  `record_episode`, the parent-side recording loop and the
  `"Frame processing took too long"` warning at line 319.
- `Yunfei/crisp_gym/crisp_gym/record/recording_manager_config.py:13` —
  `RecordingManagerConfig`, where the two new fields belong.
- `Yunfei/crisp_gym/examples/15_ridgeback_mocap_record.py:270` —
  `RecordingManagerConfig(...)` construction site that needs to pass the
  new fields.
- `Yunfei/crisp_gym/examples/15_ridgeback_mocap_record.py:234` — extra
  `/target_pose` subscription on the env node, a possible contributor to
  `get_obs` latency independent of the image writer.
- `Yunfei/crisp_gym/docs/ridgeback_target_pose_ownership.md` — topic
  ownership background for the mocap flow, referenced from script 15's
  docstring.
