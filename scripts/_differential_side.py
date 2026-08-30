import os, sys, importlib.util, numpy as np
from scipy.spatial.transform import Rotation
side, out = sys.argv[1], sys.argv[2]

class Args:
    max_speed=2.0; min_speed=1.0; clamp_deg=5.0
    lookahead=2; lookbehind=0; cum_lookahead=0
    invert_gripper=False
args = Args()

if side == "orig":                      # the untouched reference
    import os
    p = os.environ["ORIG_TREE"] + "/examples/19_deploy_policy.py"
    spec = importlib.util.spec_from_file_location("d19", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    pre_compute, sched = m._pre_compute_chunk_arrays, m._build_chunk_speed_schedule
    bsq = m.build_speed_queue_arrays
else:                                   # the refactored library
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crisp_gym.deploy.timing import _pre_compute_chunk_arrays as pre_compute, build_speed_queue_arrays as bsq
    from crisp_gym.deploy.pipeline import _build_chunk_speed_schedule as sched

chunks = np.load(os.environ["CHUNKS"])
res = {}
for i, c in enumerate(chunks):
    c = c.astype(np.float64)
    xyz, quat, grip, af32 = pre_compute(
        c, args=args, gripper_enabled=True,
        gripper_unnormalize_fn=lambda g: g * 0.085,      # Robotiq stroke, deterministic
        rotation_from_action=lambda v: Rotation.from_rotvec(v),
    )
    s = sched(c, args, past_buffer=None)
    cyc, dt_eff, s_eff = bsq(s, 0.05, len(c), retime=True)
    res[f"xyz{i}"]=xyz; res[f"quat{i}"]=quat; res[f"grip{i}"]=grip
    res[f"s{i}"]=s; res[f"cyc{i}"]=cyc; res[f"dt{i}"]=dt_eff; res[f"seff{i}"]=s_eff
np.savez(out, **res)
print(f"{side}: wrote {len(res)} arrays")
