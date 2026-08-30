#!/usr/bin/env bash
# Offline regression suite for the deploy path. No robot, no cameras.
#
# Written for the crisp_gym/deploy/ extraction: these checks must stay green
# through it, because the extraction moves ~6400 lines out of two example
# scripts and nothing else verifies that the tree still imports and resolves.
#
# Usage:   scripts/smoke_test.sh [--wheel]
#   --wheel   also build a wheel and verify a clean-venv install resolves
#             bundled YAML through importlib.resources (slow, ~60s)
#
# PYTHON must be an interpreter with torch/lerobot/rclpy/cv2 available.
# Override with SMOKE_PYTHON=/path/to/python.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"

PYTHON="${SMOKE_PYTHON:-/home/ali/Coding/Robot_Control/Yunfei/crisp_gym/.pixi/envs/jazzy-lerobot/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "FAIL: no usable interpreter at $PYTHON" >&2
    echo "      set SMOKE_PYTHON to a python with torch/lerobot/rclpy/cv2" >&2
    exit 1
fi
# Import this checkout, not whatever crisp_gym the environment has installed.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

pass=0; fail=0
ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail+1)); }

echo "=== crisp_gym smoke test ==="
echo "repo:   $REPO"
echo "python: $PYTHON"
echo

echo "[1] syntax"
if "$PYTHON" -m compileall -q crisp_gym examples >/dev/null 2>&1; then
    ok "crisp_gym/ and examples/ compile"
else
    bad "compileall reported syntax errors"
fi

echo "[2] the checkout is the one under test"
actual=$("$PYTHON" -c "import crisp_gym; print(crisp_gym.__file__)" 2>/dev/null)
case "$actual" in
    "$REPO"/*) ok "crisp_gym resolves to this checkout" ;;
    *)         bad "crisp_gym resolved to $actual (PYTHONPATH override lost)" ;;
esac

echo "[3] deploy entry points import and build their CLI"
for s in 19_deploy_policy.py 17_replay_dataset.py; do
    if (cd examples && timeout 300 "$PYTHON" "$s" --help >/dev/null 2>&1); then
        ok "$s --help"
    else
        bad "$s --help"
    fi
done

echo "[4] env configs resolve (incl. nested robot/gripper/camera from_yaml)"
if "$PYTHON" - <<'PY' 2>/dev/null
import sys
from crisp_gym.envs.manipulator_env_config import make_env_config, list_env_configs
bad = 0
for n in sorted(str(x) for x in list_env_configs()):
    if "ur10e" not in n:
        continue
    try:
        make_env_config(n)
        print(f"  PASS  env {n}")
    except Exception as e:
        print(f"  FAIL  env {n}: {type(e).__name__}: {e}")
        bad += 1
sys.exit(1 if bad else 0)
PY
then ok "all env configs"; else bad "an env config failed to load"; fi

echo "[5] every registered policy name resolves to a class"
if "$PYTHON" - <<'PY' 2>/dev/null
import importlib, sys
from crisp_gym.policy.policy import policy_registry, _LAZY_POLICY_MODULES
bad = 0
for name, mod in sorted(_LAZY_POLICY_MODULES.items()):
    try:
        importlib.import_module(mod)
        cls = policy_registry.get(name)
        assert cls is not None, "module imported but did not register"
        print(f"  PASS  policy {name} -> {cls.__name__}")
    except Exception as e:
        print(f"  FAIL  policy {name}: {type(e).__name__}: {e}")
        bad += 1
sys.exit(1 if bad else 0)
PY
then ok "all policy names"; else bad "a policy name did not resolve"; fi

echo "[6] unit tests (pure-numpy deploy math, no robot)"
if "$PYTHON" -m pytest tests/ -q >/dev/null 2>&1; then
    ok "pytest tests/"
else
    bad "pytest tests/ -- run '\''$PYTHON -m pytest tests/ -q'\'' to see why"
fi

if [ "${1:-}" = "--wheel" ]; then
    echo "[7] wheel ships the bundled YAML and the deploy-path modules"
    tmp=$(mktemp -d)
    if "$PYTHON" -m pip wheel --no-deps --no-build-isolation -w "$tmp" . >/dev/null 2>&1 \
       && "$PYTHON" -c "
import glob, sys, zipfile
w = glob.glob('$tmp/*.whl')[0]
names = zipfile.ZipFile(w).namelist()
need = ['crisp_gym/config/path.py', 'crisp_gym/util/lerobot_features.py',
        'crisp_gym/config/envs/ur10e_ridgeback_dual_cam_deploy_env_rotvec.yaml']
missing = [n for n in need if n not in names]
yamls = [n for n in names if n.endswith('.yaml')]
print(f'  wheel has {len(yamls)} yaml, {len(names)} files')
sys.exit(1 if missing else 0)
"; then ok "wheel contents"; else bad "wheel missing deploy-path files or YAML"; fi
    rm -rf "$tmp"
fi

echo
echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
