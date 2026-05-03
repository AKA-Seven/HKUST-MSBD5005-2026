#!/usr/bin/env python3
"""Q2 visuals entrypoint: runs each present script under visualization/q2/ (3D bundle graph, sankey,
reliability bubble, bundle dashboard). Optional scripts omitted from the repo are skipped."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _run(mod_name: str, file_name: str) -> None:
    path = Path(__file__).resolve().parent / file_name
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()


def main() -> None:
    scripts = (
        "build_q2_dandelion_3d.py",
        "build_q2_trade_sankey_q1_q2.py",
        "build_q2_bundle_reliability_bubble.py",
        "build_q2_bundle_dashboard.py",
    )
    here = Path(__file__).resolve().parent
    for fname in scripts:
        path = here / fname
        if path.is_file():
            _run(path.stem, fname)
        else:
            print(f"[build_q2_figures] skip missing script: {fname}")


if __name__ == "__main__":
    main()
