#!/usr/bin/env bash
# Train ACT, then Diffusion, back-to-back on the merged pickplace dataset.
#
# Diffusion only starts if ACT finishes cleanly (`set -e`). Run this inside the
# `lerobot-041` conda env. Intended to be launched in a detached tmux session
# so it survives closing the terminal:
#
#   cd ur10_clearpath/Yunfei/crisp_gym
#   tmux new-session -d -s act_diff_train \
#     "conda run -n lerobot-041 bash examples/26_train_act_then_diffusion.sh \
#      2>&1 | tee outputs/train/act_diff_train.log"
#
#   tmux attach -t act_diff_train     # check on it later
#
# Any arguments passed to this script are forwarded to BOTH wrappers, e.g.
#   bash examples/26_train_act_then_diffusion.sh --steps 50000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRISP_GYM_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CRISP_GYM_DIR"

echo "=============================================================="
echo "[1/2] Training ACT  -> outputs/train/act_merged"
echo "=============================================================="
python examples/24_train_act.py "$@"

echo
echo "=============================================================="
echo "[2/2] Training Diffusion Policy -> outputs/train/diffusion_merged"
echo "=============================================================="
python examples/25_train_diffusion.py "$@"

echo
echo "Done. ACT + Diffusion training complete."
