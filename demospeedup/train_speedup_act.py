#!/usr/bin/env python3
"""Step 4 of DemoSpeedup: train ACT on the accelerated dataset.

The only differences from the proxy run (step 1) are the dataset and the
chunk size. Upstream halves it -- 50 -> 25 for ACT, 48 -> 24 for DP -- "to
maintain geometrical consistency": one accelerated frame covers 2-4 source
frames, so a chunk of half the length still spans about as much *motion* as
the proxy's did. Our ACT baselines use ``chunk_size=100``, hence 50 here.

Refuses to run unless ``--dataset-root`` carries the
``meta/demospeedup_source.json`` sidecar that ``convert_lerobot_to_speedup.py``
writes, so a baseline dataset cannot be trained under this name by accident.

    conda run -n lerobot-041 python train_speedup_act.py --wandb

Anything after ``--`` is forwarded verbatim to ``lerobot-train``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_proxy_act import (  # noqa: E402
    PHOTOMETRIC_TFS,
    add_common_arguments,
    build_command,
    run,
)

DEFAULT_ROOT = Path("/home/batur/Coding/data/merged_speedup_20260528")
DEFAULT_REPO_ID = "merged_speedup_20260528"
DEFAULT_OUTPUT_DIR = Path("outputs/train/demospeedup_act")
DEFAULT_CHUNK_SIZE = 50  # half the proxy's 100, as upstream halves 50 -> 25
DEFAULT_WANDB_PROJECT = "demospeedup_act"

assert PHOTOMETRIC_TFS  # re-exported for symmetry with the proxy script


def build_speedup_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    sidecar = args.dataset_root / "meta" / "demospeedup_source.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"{sidecar} not found -- {args.dataset_root} is not an accelerated "
            f"dataset; run convert_lerobot_to_speedup.py first"
        )
    meta = json.loads(sidecar.read_text())
    print(
        f"accelerated dataset: {meta['source_frames']} -> {meta['kept_frames']} frames "
        f"({meta['speedup']:.2f}x), stride {meta['low_v']}x/{meta['high_v']}x, "
        f"source {meta['source_dataset']}"
    )
    return build_command(args, extra)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(ap)
    ap.set_defaults(
        repo_id=DEFAULT_REPO_ID,
        dataset_root=DEFAULT_ROOT,
        output_dir=DEFAULT_OUTPUT_DIR,
        chunk_size=DEFAULT_CHUNK_SIZE,
        wandb_project=DEFAULT_WANDB_PROJECT,
    )
    return run(build_speedup_command, ap)


if __name__ == "__main__":
    raise SystemExit(main())
