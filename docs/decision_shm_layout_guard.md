# Decision: guarding the crisp_sender shared-memory layout

**Status:** implemented, 2026-08-30 · **Layout version:** 1

## What the protocol is

`--cpp-sender` splits the deploy loop across two processes:

```
PYTHON (producer)                             C++ (crisp_sender)
read obs -> run inference                 ->  sleep until each deadline
-> compute chunk, speeds, deadlines       ->  publish PoseStamped / gripper
-> write bytes into /dev/shm              ->
```

Only the *waiting and publishing* moved to C++, because Python's timing was
unreliable under GIL contention. Everything upstream is still Python;
`crisp_gym/deploy/cpp_sender.py` is a drop-in for `TargetSenderThread` with the same
`.put()` / `.qsize()` / `.join()` surface, so `19_deploy_policy.py` cannot tell which
one it is driving.

The handoff has **no serialization format**. Four `/dev/shm` segments are mapped by
both processes, and the two sides agree only on byte *positions*:

```
one 64-byte action ring slot
byte: 0        8         16          28           44     48     52      56      60
      seqnum   deadline  xyz (3xf)   quat (4xf)   grip   flags  frame   s_eff   cycles
```

Python writes `struct.pack_into("<fff", mm, slot + 16, x, y, z)`; C++ reads
`memcpy(&pose, base + slot + 16, 12)`. Both hardcode `16`, in two files, in two
languages — roughly 30 offsets duplicated by hand.

## Why it worked before, and why that changed

It worked because the two lists were written together and lived in **one tree**:
`crisp_gym/deploy/cpp_sender.py` and
`clearpath_remote_ws/.../src/crisp_sender.cpp` were pulled and rebuilt together, so
they physically could not disagree. Nothing verified the agreement; nothing needed to.

Two handshakes already existed, but both concern **timing, not layout**:

- `kSetupCompleteOff` — Python writes it last; C++ waits for nonzero, so it never
  reads a half-written setup block.
- `kSetupReadyOff` — C++ sets it after rclcpp init and publisher discovery; Python
  waits. This is the subscriber-match race that `--startup-delay 1.0` also covers.

The architecture change is what introduces the risk. Once Pace pins `crisp_gym` by
SHA, the Python offsets are frozen at that commit while `clearpath_remote_ws` moves
independently — a pinned producer can meet a binary built from newer C++.

**The failure is silent.** Insert one field mid-struct and every later offset shifts;
Python writes the deadline where C++ reads the pose, and the arm is commanded to a
nonsense **absolute** target (`use_relative_actions: false`) with no error raised. A
*large* mismatch would scramble the ring's head/tail counters and most likely hang —
safer. It is the small, plausible change that is dangerous.

## Options considered

**A. Overload the existing setup-complete flag.** Python already writes `1` to
`kSetupCompleteOff` (a `uint64`) as its final act; C++ waits for nonzero. Write
`MAGIC << 32 | VERSION` instead and require that exact value.
*+* No new offsets — and a new offset is itself one more number to keep in sync,
which is the failure mode being guarded. Validation happens at exactly the moment
C++ first trusts Python's bytes.
*-* A stale C++ binary still testing `!= 0` accepts it silently. Unavoidable for any
scheme's first version: a guard only binds once both sides have it.

**B. Dedicated fields in the ring header.** `kActionHeaderSize` is 64 bytes but only
40 are used (`HEAD`/`TAIL`/`CAPACITY`/`STOP`/`DT_HINT`), leaving 40–63 free.
*+* A real field rather than an overloaded flag.
*-* Two more offsets to maintain, and it is checked later in the sequence, after
Python has already written the setup block.

**C. Defer until the pin goes live.** It protects nothing while both sides still ship
together.
*-* The moment it starts mattering is exactly the moment nobody is thinking about it.

Rejected outright: putting magic at offset 0 of each segment (the plan's original
wording). Offset 0 is already occupied in all three — `kSetupCompleteOff`,
`kAH_HeadOff`, `kSC_ReqOff` — so it would shift every subsequent offset, i.e. perform
the exact breaking change the guard exists to detect.

## Decision: A

```python
_SHM_LAYOUT_MAGIC     = 0x43535044        # "CSPD", crisp sender protocol
_SHM_LAYOUT_VERSION   = 1
_SETUP_COMPLETE_VALUE = (MAGIC << 32) | VERSION   # 0x4353504400000001
```

Python stamps this into `kSetupCompleteOff` as the last step of setup, which both
releases C++ from its wait *and* declares the layout. C++ validates before reading any
other offset and refuses to start on mismatch, distinguishing the two cases so the
message tells you what actually happened.

("Magic" is only a recognizable constant, the way a ZIP starts with `PK` — the value
means nothing beyond being distinctive enough not to arise by accident.)

## Maintaining it

**Bump `_SHM_LAYOUT_VERSION` and `kShmLayoutVersion` together whenever any offset or
slot size changes** in `crisp_gym/deploy/cpp_sender.py` or `crisp_sender.cpp`. Both
files carry a comment saying so, next to the constants.

## Verification

Built with `colcon build --packages-select tum09_custom`, then run against
hand-stamped `/dev/shm` segments, no robot involved:

| producer stamp | outcome |
|---|---|
| `1` (stale, pre-guard) | `layout magic mismatch: got 0x00000000, expected 0x43535044` |
| `MAGIC \| 2` (newer than binary) | `layout version mismatch: producer speaks v2, this crisp_sender was built for v1` |
| `MAGIC \| 1` (correct) | passes, runs normally |
