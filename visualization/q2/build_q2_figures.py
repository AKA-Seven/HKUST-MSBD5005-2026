#!/usr/bin/env python3
"""Q2 可视化入口：3D 蒲公英网络、Q1+Q2 桑基、bundle→评级桑基、Bundle 可靠性气泡图、连通分量柱状对比。"""

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
    _run("build_q2_dandelion_3d", "build_q2_dandelion_3d.py")
    _run("build_q2_trade_sankey_q1_q2", "build_q2_trade_sankey_q1_q2.py")
    _run("build_q2_sankey", "build_q2_sankey.py")
    _run("build_q2_bundle_reliability_bubble", "build_q2_bundle_reliability_bubble.py")
    _run("build_q2_connected_components_bar", "build_q2_connected_components_bar.py")
    _run("build_q2_bundle_dashboard", "build_q2_bundle_dashboard.py")


if __name__ == "__main__":
    main()
