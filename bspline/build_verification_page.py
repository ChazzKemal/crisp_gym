#!/usr/bin/env python3
"""Render verify_math.py's results into a self-contained report page.

    conda run -n lerobot-041 python verify_math.py
    conda run -n lerobot-041 python build_verification_page.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "math_verification.json")
    ap.add_argument("--template", type=Path, default=HERE / "verification_template.html")
    ap.add_argument("--out", type=Path, default=HERE / "math_verification.html")
    args = ap.parse_args()

    results = json.loads(args.results.read_text())
    blob = json.dumps(results, separators=(",", ":"))
    tpl = args.template.read_text()
    marker = "/*__RESULTS__*/"
    if marker not in tpl:
        raise RuntimeError(f"{marker} not found in {args.template}")
    args.out.write_text(tpl.replace(marker, blob))
    print(f"wrote {args.out}  ({results['n_passed']}/{results['n_total']} checks, "
          f"{len(blob) / 1024:.0f} KB of results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
