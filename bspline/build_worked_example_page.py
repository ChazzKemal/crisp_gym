#!/usr/bin/env python3
"""Render worked_example.py's exhibits into a self-contained page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=HERE / "worked_example.json")
    ap.add_argument("--template", type=Path, default=HERE / "worked_example_template.html")
    ap.add_argument("--out", type=Path, default=HERE / "worked_example.html")
    args = ap.parse_args()
    blob = json.dumps(json.loads(args.data.read_text()), separators=(",", ":"))
    tpl = args.template.read_text()
    if "/*__DATA__*/" not in tpl:
        raise RuntimeError("marker /*__DATA__*/ not found in template")
    args.out.write_text(tpl.replace("/*__DATA__*/", blob))
    print(f"wrote {args.out} ({len(blob) / 1024:.0f} KB of numbers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
