"""The deploy CLI, as a parser anyone can build.

Moved out of ``main()`` in ``examples/19_deploy_policy.py``, where roughly 500 of its
~2150 lines were argparse construction -- so most of the "main function" was really a
CLI definition with a program stuck to the end of it.

Separating it means the flag set can be read, diffed and reused without executing a
deploy, and a different front end (a draccus config in Pace, say) can consult the
same defaults instead of restating them.
"""

import argparse

from crisp_gym.deploy.gains import DEFAULT_GRIPPER_SPEED, GRIPPER_MAX_SPEED_MPS


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Every flag the deploy entry point accepts.

    Args:
        description: Shown in ``--help``; the caller passes its own ``__doc__`` so
            the help text still belongs to the script the user actually ran.
    """
    parser = argparse.ArgumentParser(description=description)
    src_group = parser.add_argument_group("chunk source (real policy OR fake)")
    src_group.add_argument(
        "--pretrained-path", default=None,
        help="Path to the LeRobot pretrained model directory. Mutually "
             "exclusive with --fake-mode.",
    )
    src_group.add_argument(
        "--fake-mode", default=None, choices=["hold", "dataset"],
        help="Run with a fake chunk source instead of a real model. "
             "Useful for smoke-testing the deploy pipeline without a "
             "trained checkpoint. 'hold' keeps the arm at the EE pose "
             "captured on the first request. 'dataset' slices a recorded "
             "episode into chunks of --fake-n-act.",
    )
    src_group.add_argument(
        "--fake-n-act", type=int, default=16,
        help="Chunk size for fake mode (default: 16). Must be > "
             "--lookahead. Ignored when --pretrained-path is set "
             "(real policy's n_action_steps wins).",
    )
    src_group.add_argument(
        "--fake-n-obs", type=int, default=2,
        help="Rolling obs buffer size for fake mode (default: 2). The fake "
             "chunk source ignores observations but the producer loop still "
             "fills env._get_obs() each iteration to exercise that path.",
    )
    src_group.add_argument(
        "--fake-repo-id", default=None,
        help="LeRobot dataset repo_id for --fake-mode dataset.",
    )
    src_group.add_argument(
        "--fake-episode-idx", type=int, default=0,
        help="Episode index in --fake-repo-id (default: 0).",
    )
    src_group.add_argument(
        "--fake-loop", action="store_true",
        help="Keep slicing the dataset in a loop forever (the old behaviour). "
             "By default the dataset fake source returns the final partial "
             "chunk on the last pass and then raises DatasetExhausted, so the "
             "deploy script exits cleanly with stopped_by='dataset_exhausted' "
             "and summary.json is written. Pass this flag if you want a "
             "sustained inference-stress test that runs until Ctrl-C.",
    )
    src_group.add_argument(
        "--fake-drop-holds", action="store_true",
        help="In --fake-mode dataset, strip held frames from the recorded "
             "action array before chunking. A frame is held if every "
             "channel (xyz + rpy + gripper) differs from the previous "
             "frame by less than --hold-eps; the first frame of each "
             "held run is kept as an anchor. Shortens the trajectory; "
             "the producer + sender + speed schedule downstream are "
             "unchanged. Distinct from --drop-holds (which modifies the "
             "per-chunk speed schedule, not the trajectory length).",
    )
    parser.add_argument(
        "--env-config", default="ur10e_ridgeback_dual_cam_env",
        help="crisp_gym env config name (default: ur10e_ridgeback_dual_cam_env). "
             "The dual-cam env is the canonical deploy target — both Orbbec "
             "(third-person) and D405 (wrist) feed into env._get_obs() so the "
             "policy sees the same image streams used during recording. Pass "
             "ur10e_ridgeback_env for single-cam smoke tests.",
    )
    parser.add_argument(
        "--fps", type=float, default=20.0,
        help="Baseline target rate (default: 20). Sets dt_base = 1/fps for "
             "the cycle-snap pipeline. Cycle-snap rounds dt_eff up to the "
             "next multiple of CONTROL_DT = 2 ms (500 Hz), NOT of dt_base, "
             "so speedup (s_eff > 1) routinely publishes faster than fps — "
             "e.g. fps=20, s_raw=2 → dt_eff ≈ 26 ms (~37 Hz). At s_eff=1.0 "
             "the publish rate matches fps. Should approximately match "
             "the policy's training data fps so the implicit velocity demand "
             "at s_eff=1.0 matches what was demonstrated.",
    )
    parser.add_argument(
        "--scale-kp", action="store_true",
        help="Enable xVLA-aligned scaling (kp²/kd/gripper/time) per "
             "docs/variable_impedance_design.md.",
    )
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--min-speed", type=float, default=1.0)
    parser.add_argument("--clamp-deg", type=float, default=5.0)
    parser.add_argument(
        "--lookahead", type=int, default=2,
        help="Forward window within each chunk for compute_speed_schedule. "
             "Must be < policy.n_action_steps; bigger values give better "
             "slow-before-curve at the cost of latency within the chunk.",
    )
    parser.add_argument(
        "--lookbehind", type=int, default=0,
        help="Backward window for compute_speed_schedule, symmetric "
             "counterpart to --lookahead. The producer holds a deque of "
             "the last N action rows it pushed to the sender and prepends "
             "them to the current chunk before computing the schedule, so "
             "the arm stays slow on the EXIT of a curve, not just the "
             "entry. 0 (default) = forward-only (legacy behaviour). For "
             "the very first chunk the buffer is empty and the centered "
             "window edge-pads on the left.",
    )
    parser.add_argument(
        "--cum-lookahead", type=int, default=0,
        help="Cumulative-angle forward window. When > 0, the chunk schedule "
             "uses compute_speed_schedule_cumangle "
             "(factor = clip(90 - cum, 0) / 90) instead of "
             "compute_speed_schedule's averaging variant "
             "(factor = clip(90*(N+1) - cum, 0) / (90*(N+1))). The cumangle "
             "formula slows the arm much more aggressively as a long window "
             "adds up small bends — useful when the policy emits many "
             "sub-threshold direction changes that the averaging path "
             "under-weights. Takes precedence over --lookahead when both "
             "are > 0. Mirrors the cum_lookahead slider in "
             "27_speedup_slider_viewer.py.",
    )
    parser.add_argument(
        "--hold-eps", type=float, default=1e-6,
        help="Minimum per-channel delta (xyz/rpy/grip) below which a frame "
             "is considered 'held' for --fake-drop-holds (default: 1e-6). "
             "Recording noise floor is usually above 1e-6 — try 1e-4 if "
             "no frames are stripped.",
    )
    parser.add_argument("--kp-exp", type=float, default=2.0)
    parser.add_argument("--kd-exp", type=float, default=1.0)
    parser.add_argument(
        "--kp-scale-warn", type=float, default=3.0,
        help="Warn if peak kp factor (s_eff_max ** kp_exp) exceeds this.",
    )
    parser.add_argument(
        "--gripper-base-speed", type=float, default=DEFAULT_GRIPPER_SPEED,
    )
    parser.add_argument("--controller-node", default="/cartesian_controller")
    parser.add_argument("--gripper-cm", default="/gripper/controller_manager")
    parser.add_argument(
        "--gripper-direct-action", action="store_true",
        help="Send gripper commands via env.gripper._command_action_client "
             "(action goal) instead of /target_gripper_state Float32 pub.",
    )
    parser.add_argument(
        "--gripper-no-edge-detect", action="store_true",
        help="DISABLE the sender's gripper edge-detection. By default the "
             "sender only re-sends the gripper goal/Float32 when the "
             "requested value changes by more than 1e-4 from the last "
             "publish — this prevents the Robotiq action server from "
             "PREEMPTING the active trajectory every 50 ms (which caps "
             "effective close speed at `accel × dt` instead of `speed_limit`). "
             "Pass this flag to restore the legacy per-frame publish "
             "behavior (every-tick preemption). Useful only if you need to "
             "reproduce older behaviour or if a downstream subscriber "
             "expects a constant publish rate.",
    )
    parser.add_argument(
        "--gripper-latch-frames", type=int, default=0, metavar="N",
        help="Hysteresis/latch on the gripper command: once the target "
             "CHANGES, lock it for the next N published frames (it cannot "
             "change again until N frames have elapsed). 0 (default) disables. "
             "Use when the policy's gripper channel oscillates open↔close at "
             "the chunk seam (every chunk) — without a latch each flip preempts "
             "the in-flight Robotiq goal before the fingers finish travelling, "
             "so the gripper chatters but never completes a grasp. Latch only "
             "gates CHANGES (same-value re-sends still follow edge-detect). "
             "Python sender only — ignored with --cpp-sender. At 20 fps, N=5 "
             "≈ 250 ms minimum dwell between open/close transitions.",
    )
    parser.add_argument(
        "--gripper-slowdown-frames", type=int, default=0, metavar="N",
        help="Trajectory-speed brake during a grasp: on each open→close "
             "gripper transition, force the arm speed factor s_eff back to "
             "1.0 (real-time) for that frame and the next N-1, then resume the "
             "speedup schedule. 0 (default) disables. Edge-triggered on the "
             "CLOSE transition only — the arm runs real-time *while it grabs*, "
             "but staying closed during the carry/lift is unaffected (no new "
             "transition), so speedup resumes for transport. Forces s_eff=1.0 "
             "(not a clamp). No effect on baseline runs (s_eff already 1.0). "
             "Applied in the producer before cycle-snap, so it's baked into the "
             "per-frame deadlines and works with --cpp-sender. At 20 fps, N=10 "
             "≈ 0.5 s (about one 2F-85 full close at --gripper-max-speed).",
    )
    parser.add_argument(
        "--gripper-max-speed", action="store_true",
        help="Force the Robotiq driver's runtime speed_limit to "
             f"GRIPPER_MAX_SPEED_MPS ({GRIPPER_MAX_SPEED_MPS} m/s = 150 mm/s, "
             "the driver's hardware cap). At startup, spawns "
             "gripper_speed_controller (if not active) and publishes "
             "GRIPPER_MAX_SPEED_MPS to /gripper/gripper_speed_controller/"
             "commands ONCE. Independent of --scale-kp: works whether or "
             "not the scaler is active. Use this when the policy commands "
             "a fast open/close window (~1-2 s) and you observe the "
             "gripper only partially traveling — the default driver "
             "speed_limit on first boot is usually conservative. NOTE: "
             "this only raises the driver's max; the action-server "
             "preemption pattern (one goal every 50 ms from either the "
             "sender or crisp_py's timer) still bottlenecks effective "
             "rate to roughly accel * dt — but a higher speed_limit "
             "means the per-cycle progress is larger.",
    )
    parser.add_argument(
        "--invert-gripper", action="store_true",
        help="Flip the gripper action (1-grip). For policies trained on "
             "older datasets recorded with 0=open convention.",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Action-chunk stride: take every Nth action from the policy's "
             "chunk before pushing to the sender. Achieves trajectory "
             "speedup by decimation rather than time-compression — sender "
             "still runs at dt_eff (no extra OS-timing pressure), but the "
             "robot traverses N× more of the recorded path per published "
             "target. Combines multiplicatively with --max-speed: stride=2 "
             "+ max-speed=2 → 4× effective speedup with dt_eff=17ms "
             "(manageable for sender) and 2× larger per-step deltas. "
             "Stride=1 (default) = no decimation. Note: speed schedule + "
             "cycle-snap run on the strided chunk, so PACE's curvature "
             "estimate uses the post-stride trajectory.",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Destroy camera subscriptions + timers during deploy. ONLY safe "
             "if the policy doesn't use images — otherwise env._get_obs() "
             "returns stale frames.",
    )
    parser.add_argument(
        "--no-gripper-state", action="store_true",
        help="Destroy the 500 Hz /gripper/joint_states subscription. Cheap "
             "GIL-hygiene win; gripper still moves. WARNING: this also "
             "freezes observation.state.gripper (and observation.state, "
             "which concatenates it) at whatever value the last callback "
             "wrote — every subsequent policy input will see a stale "
             "gripper reading. Safe for replay (17_replay_dataset.py: dataset "
             "actions don't condition on obs) but NOT recommended when a "
             "policy / shadow ACT consumes obs.state. Omit this flag for "
             "deploy with --shadow-act or --pretrained-path.",
    )
    parser.add_argument(
        "--no-safety-clip", action="store_true",
        help="Disable env's safety box position clipping. Use with caution.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip the bits that require a controller-manager: the "
             "controller-manager wait inside env.wait_until_ready, "
             "env.home(), env.switch_controller('cartesian'). Topic-based "
             "readiness waits (robot/gripper/cameras) still run — pair "
             "with fake_sensors.py for those. The ReplayScaler still runs "
             "if --scale-kp; for realistic scaler RPC latency the "
             "fake_sensors.py default also publishes a /cartesian_controller "
             "node with the right parameter services. The sender's "
             "/target_pose publish also still runs unless --dry-run is set. "
             "Drop --dry-run + add --scale-kp + --max-speed 1.5 for the "
             "most realistic publish + scaler cost measurement.",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=-1,
        help="Stop after this many inference chunks (-1 = run until Ctrl-C).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Rehearse the full deploy pipeline (obs reads, fake source, "
             "shadow ACT, speed schedule, cycle-snap, queue push, sender "
             "pacing) without touching the robot. The producer still fills "
             "the queue and waits on drain to overlap_threshold like a real "
             "run; the sender thread still pops, sleeps to deadlines, logs "
             "underruns and queue depth. The ONLY thing dry_run gates is the "
             "ROS-facing side of the sender: no /target_pose publish, no "
             "gripper command, no scaler.step_to RPC to /cartesian_controller. "
             "Queue cadence, underrun stats, percentile measurements all "
             "match a real deploy.",
    )
    parser.add_argument(
        "--shadow-act", action="store_true",
        help="Run a real LeRobot ACT policy in parallel with the fake dataset "
             "source, on every chunk request. Forward pass uses random weights "
             "(no pretrained checkpoint); output is discarded; we measure "
             "inference latency. Lets you smoke-test the full ACT inference "
             "path (image preprocess + transformer forward + action head) "
             "before swapping in a real model. Falls back to a torchvision "
             "ResNet18 stub if ACT instantiation fails.",
    )
    parser.add_argument(
        "--shadow-temporal-ensemble", type=float, default=None,
        help="If set, configure ACTConfig with this temporal_ensemble_coeff. "
             "Forces n_action_steps=1 inside ACT (its requirement). Note: our "
             "producer is per-chunk, so the ensembler is constructed and ACT "
             "runs under RTC architecture, but the per-step .update() blend "
             "is dormant. Set to e.g. 0.01 to exercise the RTC code path "
             "without changing producer cadence.",
    )
    parser.add_argument(
        "--shadow-device", default=None,
        help="Torch device for the shadow (e.g. 'cuda', 'cuda:1', 'cpu'). "
             "Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--shadow-inpaint-tail", type=int, default=2,
        help="When >0, blend the shadow ACT's new chunk into a separate "
             "shadow-history deque, replacing its last N items with a 50/50 "
             "weighted average against the new chunk's first N items "
             "(xVLA-style 'inpaint'). This is a SMOKE TEST of the blending "
             "code path — the shadow history is never consumed by the "
             "robot. With --shadow-inpaint-tail 0, the shadow's chunks are "
             "still tracked but no blending happens. Set to match "
             "--overlap-threshold for the most realistic test (default 2).",
    )
    parser.add_argument(
        "--overlap-threshold", type=int, default=2,
        help="Trigger the next chunk inference when the sender's queue drops "
             "to <= this many items. Default 2 means inference fires while "
             "the last 2 actions of the current chunk are still in flight, "
             "so the sender thread never sees an empty queue at chunk "
             "boundaries (assuming inference latency < threshold * dt_eff). "
             "Tune to match your model's measured latency: at 30 Hz / 33 ms "
             "per frame, threshold=2 hides ~66 ms of inference; threshold=3 "
             "hides ~100 ms. Set to 0 to disable overlap (wait for full "
             "drain — the old behaviour).",
    )
    parser.add_argument(
        "--blend-overlap", type=int, default=0, metavar="N",
        help="Temporal-ensemble the chunk seam to remove the per-chunk "
             "'push'. Hold back the last N frames of each chunk and average "
             "them with the next chunk's first N frames, ramping the weight "
             "old->new (frame i weight w=(i+1)/(N+1) toward the NEW chunk) so "
             "the seam stays continuous with what's executing but converges "
             "to the fresher prediction. Averages the pose channels "
             "(xyz + rotvec) only; the gripper channel is taken from the new "
             "chunk (NEVER averaged). Producer-side, so it applies to both "
             "the Python and C++ senders. N is clamped to K//2 (half the "
             "per-chunk frame count) so the blended head and held-back tail "
             "never overlap. 0 (default) disables — chunks are stitched "
             "head-to-tail.",
    )
    parser.add_argument(
        "--blend-mode", type=str, default="linear",
        choices=("linear", "hermite"),
        help="How to interpolate the chunk seam when --blend-overlap > 0. "
             "'linear' (default, existing behaviour): per-frame weighted "
             "average  out=(1-w)*prev_pred + w*new_pred  with w ramping "
             "0->1 over the overlap. 'hermite': replace the overlap slots "
             "with a cubic Hermite curve from the last actually-emitted "
             "frame (p_start, v_start) to the first verbatim new-chunk "
             "frame (p_end, v_end) — matches both position AND velocity at "
             "both ends, so the executed trajectory has C1 continuity at "
             "the boundary (no direction reversal). Gripper channel [6] "
             "is NEVER interpolated in either mode — always takes the new "
             "chunk's value. Requires --blend-overlap > 0 to have any "
             "effect; --blend-skip is honoured in linear mode but ignored "
             "in hermite (Hermite always starts the bridge at slot 0).",
    )
    parser.add_argument(
        "--blend-skip", type=int, default=0, metavar="S",
        help="Commit horizon for --blend-overlap: execute the first S frames "
             "of the overlap VERBATIM from the previous chunk (pose AND "
             "gripper) before blending begins. These timesteps are treated as "
             "already-committed/in-flight, so the fresh chunk's (possibly "
             "noisy) first predictions don't perturb them. Blending then runs "
             "over the remaining N-S overlap frames, with the old->new ramp "
             "restarted across that shorter region (so frame S stays close to "
             "the committed old plan and converges to new by frame N). "
             "Requires 0 <= S < --blend-overlap; clamped to the overlap "
             "length. 0 (default) blends from the very first overlap frame.",
    )
    parser.add_argument(
        "--startup-delay", type=float, default=0.0,
        help="Sleep this many seconds after the sender starts but before "
             "the producer loop pushes the first chunk. Gives the "
             "cartesian_controller's /target_pose subscriber time to "
             "discover the sender's publisher (especially with "
             "--cpp-sender, which spawns a fresh publisher in the C++ "
             "subprocess that has to be matched by the controller's "
             "subscriber via FastDDS). Symptom this fixes: the first "
             "chunk's targets land before subscriber-match completes and "
             "get dropped on the wire, so the arm doesn't move until "
             "chunk 2. 0 (default) preserves prior behaviour; 0.5-1.0 is "
             "usually enough to absorb the discovery race.",
    )
    parser.add_argument(
        "--debug-publish", action="store_true",
        help="Record per-publish timing on the sender thread; log p50/p90/p99 "
             "at shutdown.",
    )
    parser.add_argument(
        "--record-trace", action="store_true",
        help="Save per-chunk obs→action trace for post-hoc debugging. Writes "
             "`trace.npz` alongside summary.json containing, for every chunk: "
             "chunk_idx, wall_ns, mono_ns, the predicted (K, action_dim) "
             "action chunk, and every `observation.state.*` sub-key the env "
             "emitted at that moment. Pairs with chunks.csv (producer "
             "timings) and frames.csv (sender-side per-publish state) via "
             "chunk_idx. Also dumps the per-camera RGB image at chunk-"
             "request time as JPEGs under `trace_images/` (one file per "
             "chunk per camera) — disable with --record-trace-no-images. "
             "Designed for 'policy looks OK then slips at object contact' "
             "investigations: open trace_images/chunk_{N}_*.jpg at the "
             "slip frame and inspect what the policy saw + planned.",
    )
    parser.add_argument(
        "--record-trace-every", type=int, default=1,
        help="With --record-trace, only capture every Nth chunk. Default 1 "
             "(every chunk). Set higher (e.g. 5) to keep file size + write "
             "overhead down on long deploys.",
    )
    parser.add_argument(
        "--record-trace-no-images", action="store_true",
        help="With --record-trace, skip JPEG image capture (numerical state "
             "and chunk only). Cuts ~30-50 KB per camera per chunk.",
    )
    parser.add_argument(
        "--cpp-sender", action="store_true",
        help="Use the C++ crisp_sender subprocess for the timing-critical "
             "publish loop (pose + gripper Float32 + scaler RPC). Requires "
             "`pixi run colcon build --packages-select tum09_custom` first. "
             "Eliminates Python's time.sleep + GIL jitter (~5 ms p99 + 70+ ms "
             "outliers) seen at 3-4x speedup. Mutually exclusive with "
             "--gripper-direct-action (action-client gripper stays Python-only).",
    )
    parser.add_argument(
        "--rt-priority", type=int, default=0,
        help="SCHED_FIFO priority (1-99) for the C++ sender thread. Only "
             "applied with --cpp-sender. 0 (default) = no RT. Needs "
             "CAP_SYS_NICE — falls back to no-RT with a warning if the kernel "
             "rejects the syscall. Try 80 first; expect ~10x p99 improvement.",
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="Run policy inference IN-PROCESS instead of spawning the "
             "AsyncLerobotPolicy subprocess. The chunk source becomes "
             "_SyncLeRobotChunkSource (same interface, no IPC). "
             "Intended for debugging — you can drop pdb/pudb inside "
             "request() and step through inference without spawning. "
             "WARNING: torch inference now holds the GIL inside the "
             "deploy process for ~25-35 ms/chunk, stalling the sender "
             "thread; expect higher sleep_overshoot_ms p99 and more late "
             "frames than the async (default) path. Not for production "
             "deploys. Ignored with --fake-mode (no real policy to load).",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=None,
        help="Override the diffusion policy's `num_inference_steps` at load "
             "time (denoising-loop length). Defaults to whatever's in the "
             "checkpoint config (often None → falls back to "
             "num_train_timesteps = 100). Linear cost: each step is ~17 ms "
             "on an RTX A3000 for this size UNet, so 100 → 1.75s, 20 → "
             "365ms, 10 → 190ms, 5 → 100ms. Combine with `--noise-scheduler"
             "-type DDIM` for low step counts. No effect for non-diffusion "
             "policies (ACT, etc.) — silently ignored.",
    )
    parser.add_argument(
        "--noise-scheduler-type", default=None, choices=["DDPM", "DDIM"],
        help="Override the diffusion policy's noise scheduler at load time. "
             "DDPM is the LeRobot default and trained-with; DDIM allows "
             "aggressive --num-inference-steps reductions (10 or 5) with "
             "minimal quality loss. DDPM quality degrades fast below ~50 "
             "steps because the noise schedule wasn't trained for it. No "
             "effect for non-diffusion policies — silently ignored.",
    )
    parser.add_argument(
        "--n-act", type=int, default=None,
        help="Override the policy's n_action_steps (per-inference action "
             "horizon) at load time. Use when the model was trained with a "
             "large chunk (e.g. chunk_size=100) but you want to replan more "
             "often — e.g. `--n-act 50` makes the producer consume 50 actions "
             "per chunk and request a new one. Must satisfy n_act < "
             "chunk_size (ACT) or n_act <= horizon - n_obs_steps + 1 "
             "(diffusion); load fails loudly otherwise. None (default) = use "
             "the checkpoint's n_action_steps unchanged.",
    )
    parser.add_argument(
        "--run-tag", default=None,
        help="Optional human-readable tag appended to the deploy_runs folder "
             "name. Result: deploy_runs/<YYYYMMDDTHHMMSS>_<tag>/. Useful for "
             "tagging which demo / task a run corresponds to. Non "
             "filesystem-safe chars are sanitised to '-'.",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Record a video of the run into deploy_runs/<...>/video_<cam>.mp4 "
             "via a writer subprocess (mp4v codec, plays in any standard "
             "player). Captures from the camera named by --video-camera at "
             "--video-fps Hz. Bounded queue with drop-on-overflow so video "
             "I/O can never stall the deploy loop. Dropped-frame count is "
             "logged at shutdown.",
    )
    parser.add_argument(
        "--video-camera", default="all",
        help="Camera(s) to record when --save-video is set. Accepts: "
             "'all' (default — every camera in the env, so both Orbbec + "
             "D405 in ur10e_ridgeback_dual_cam_env), a single camera_name "
             "(e.g. 'camera'), or a comma-separated list (e.g. "
             "'camera,d405'). One subprocess per camera, each writes "
             "video_<name>.mp4 + video_recorder_<name>.log into the run "
             "folder. Names not found in the env are skipped with a warning.",
    )
    parser.add_argument(
        "--video-fps", type=float, default=20.0,
        help="Capture cadence for --save-video (Hz). Default 20 matches "
             "the typical --fps; lower values cut CPU cost.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the [Y/n] prompt.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser
