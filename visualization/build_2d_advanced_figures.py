#!/usr/bin/env python3
"""统一入口：依次生成 Q1-Q4 图表。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent


def _load(subdir: str, module_file: str):
    """动态加载 visualization/qX/ 下的构建脚本。"""
    spec = importlib.util.spec_from_file_location(module_file, _here / subdir / module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_q1 = _load("q1", "build_q1_figure.py").main
build_q2 = _load("q2", "build_q2_figures.py").main
build_q3 = _load("q3", "build_q3_figures.py").main
build_q4 = _load("q4", "build_q4_figures.py").main


def main() -> None:
    build_q1()
    build_q2()
    build_q3()
    build_q4()
    print("Done. Q1-Q4 figures generated.")


if __name__ == "__main__":
    main()
