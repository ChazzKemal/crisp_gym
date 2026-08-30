#!/usr/bin/env bash
# Differential check: this library vs the untouched original, on real policy output.
#
# Runs the chunk-shaping math from BOTH trees over action chunks produced by a real
# trained checkpoint, and asserts they agree bit-for-bit. Two processes, because the
# two trees each provide a `crisp_gym` and cannot share one interpreter.
#
# This is the check that covers what --fake-mode cannot: chunks with the curvature and
# gripper structure a policy actually emits, rather than synthesised ones.
#
#   scripts/differential_vs_original.sh [checkpoint_dir]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ORIG="${ORIG_TREE:-/home/ali/Coding/Robot_Control/Yunfei/crisp_gym}"
CKPT="${1:-$ORIG/outputs/train/act_cart7_v2_angleaxis_nogrip_chunk100_ft_20260528/checkpoints/030000/pretrained_model}"
PYTHON="${SMOKE_PYTHON:-$ORIG/.pixi/envs/jazzy-lerobot/bin/python}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }
[ -f "$ORIG/examples/19_deploy_policy.py" ] || { echo "no reference tree at $ORIG" >&2; exit 1; }

echo "reference : $ORIG"
echo "checkpoint: $CKPT"

echo "[1] generating chunks from the real policy"
"$PYTHON" - "$CKPT" "$TMP/chunks.npy" <<'PY'
import sys, cv2, numpy as np, torch          # cv2 before torch: libjpeg symbol clash
ck, out = sys.argv[1], sys.argv[2]
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
cfg = PreTrainedConfig.from_pretrained(ck); cfg.pretrained_path = ck
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
pol = get_policy_class(cfg.type).from_pretrained(ck, config=cfg)
pol.reset(); pol.to(cfg.device).eval()
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=ck)
chunks = []
for i in range(4):
    g = torch.Generator(device=cfg.device).manual_seed(i)
    obs = {k: (torch.rand(1, *f.shape, device=cfg.device, generator=g) if "image" in k
               else torch.full((1, *f.shape), 0.1 * i, device=cfg.device))
           for k, f in cfg.input_features.items()}
    obs["task"] = "pick up the object"
    with torch.no_grad():
        chunks.append(post(pol.predict_action_chunk(pre(obs))).detach().cpu().numpy()[0])
np.save(out, np.stack(chunks))
PY

echo "[2] running both implementations"
CHUNKS="$TMP/chunks.npy" ORIG_TREE="$ORIG" "$PYTHON" scripts/_differential_side.py orig "$TMP/orig.npz"
CHUNKS="$TMP/chunks.npy" ORIG_TREE="$ORIG" "$PYTHON" scripts/_differential_side.py fork "$TMP/fork.npz"

echo "[3] comparing"
"$PYTHON" - "$TMP/orig.npz" "$TMP/fork.npz" <<'PY'
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
groups = {"target_xyz":"xyz","target_quat":"quat","grip_raw":"grip",
          "speed_schedule":"s","cycles":"cyc","dt_eff":"dt","s_eff":"seff"}
bad = 0
for name, pre in groups.items():
    keys = [k for k in a.files if k.startswith(pre) and k[len(pre):].isdigit()]
    n = sum(a[k].size for k in keys)
    d = max(float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))) for k in keys)
    print(f"  {name:16s} {n:>6} elements  max|diff| {d:.1e}  {'OK' if d == 0 else 'DIFFERS'}")
    bad += d != 0
sys.exit(1 if bad else 0)
PY
echo "PASS — bit-identical on real policy output"
