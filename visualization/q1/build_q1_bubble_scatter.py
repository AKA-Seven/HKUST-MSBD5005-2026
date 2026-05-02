#!/usr/bin/env python3
"""Q1：公司气泡分布图（活跃度 × 伙伴广度）。

数据来源：`outputs/q1/q1_temporal_patterns.json`（全量公司）。
- X：活跃月数 `active_months`
- Y：伙伴数量 `partner_count`
- 气泡面积：`total_links`（开方后映射到显示尺寸，符合面积感知）
- 颜色：`temporal_pattern`，固定为同一套蓝色系色相
- 坐标轴：X、Y 均为以 10 为底的对数刻度，拉开小活跃月数、小伙伴数处的拥挤（原点附近过密）。
- HTML 气泡：与 PNG 使用相同的 `scatter` 面积参数 `s`（`_scatter_sizes`），按半径比例映射到像素后再略缩小。

输出：`visualization/figures_2d/q1/q1_bubble_activity_partners.png`、`.html`、`_data.json`（汇总统计）。

运行（conda 环境 `dv`）：
`conda run -n dv python visualization/q1/build_q1_bubble_scatter.py`
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_vis_root = Path(__file__).resolve().parents[1]
if str(_vis_root) not in sys.path:
    sys.path.insert(0, str(_vis_root))

from figures2d_common import FIG_DIR, ensure_dirs, load_q1_patterns

# 时序模式展示顺序与蓝色系配色（由深到浅、可区分）
PATTERN_ORDER: tuple[str, ...] = (
    "stable",
    "bursty",
    "periodic",
    "short_term",
    "general",
)

# 和谐蓝色系（hex）；未知类别用蓝灰
PATTERN_HEX: dict[str, str] = {
    "stable": "#0a4da8",
    "bursty": "#1565c0",
    "periodic": "#1e88d8",
    "short_term": "#42a5f5",
    "general": "#78909c",
}
PATTERN_HEX_DEFAULT = "#607d8b"

PATTERN_LABEL_ZH: dict[str, str] = {
    "stable": "稳定",
    "bursty": "突发",
    "periodic": "周期",
    "short_term": "短期",
    "general": "一般",
}

PATTERN_LABEL_EN: dict[str, str] = {
    "stable": "Stable",
    "bursty": "Bursty",
    "periodic": "Periodic",
    "short_term": "Short-term",
    "general": "General",
}

# 气泡：s ∝ total_links；缩放使得最大点约占合理像素
SIZE_BASE = 2.2
SIZE_CAP_PT2 = 420.0

# 对数轴：计数为 0 时抬到该正数，避免 log(0)；常见数据为 ≥1，几乎无影响。
LOG_AXIS_FLOOR = 0.8

# Plotly 气泡：与 PNG 共用 _scatter_sizes 的 s（面积 ∝ 链数）；半径 ∝ √s，再整体略小于 PNG 观感
PLOTLY_DIAM_MIN = 2.6
PLOTLY_DIAM_MAX = 22.0
PLOTLY_SMALLER_THAN_PNG = 0.88


def _plotly_marker_sizes_from_mpl(s_mpl: np.ndarray) -> np.ndarray:
    """与 Matplotlib scatter 的 s 一致；换算为 Plotly 的 marker.size（px），略小于 PNG 同比例大小。"""
    r = np.sqrt(np.maximum(s_mpl, 0.0))
    rmax = float(np.max(r)) if r.size else 1.0
    if rmax <= 0:
        rmax = 1.0
    u = r / rmax
    d = PLOTLY_DIAM_MIN + u * (PLOTLY_DIAM_MAX - PLOTLY_DIAM_MIN)
    d = d * PLOTLY_SMALLER_THAN_PNG
    return np.clip(d, 2.0, 28.0)


def _values_for_log_axis(a: np.ndarray) -> np.ndarray:
    """保证元素为正，供 log10 坐标使用。"""
    return np.where(a > 0, a, LOG_AXIS_FLOOR)


def _load_arrays(patterns: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    xs: list[int] = []
    ys: list[int] = []
    links: list[int] = []
    companies: list[str] = []
    pats: list[str] = []

    for row in patterns:
        c = row.get("company")
        if c is None or c == "":
            continue
        try:
            am = int(row.get("active_months") or 0)
            pc = int(row.get("partner_count") or 0)
            tl = int(row.get("total_links") or 0)
        except (TypeError, ValueError):
            continue
        pat = str(row.get("temporal_pattern") or "general").strip() or "general"
        xs.append(am)
        ys.append(pc)
        links.append(max(tl, 0))
        companies.append(str(c))
        pats.append(pat)

    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(links, dtype=np.float64),
        companies,
        pats,
    )


def _scatter_sizes(links: np.ndarray) -> np.ndarray:
    """matplotlib scatter 的 s 为点数平方；面积感 ∝ total_links → s ∝ links."""
    if links.size == 0:
        return links
    raw = SIZE_BASE * np.sqrt(np.maximum(links, 1.0))
    cap = np.percentile(raw, 99.5) if raw.size > 10 else float(np.max(raw))
    if cap <= 0:
        cap = 1.0
    raw = np.clip(raw, None, cap * 1.15)
    max_s = SIZE_CAP_PT2
    raw = max_s * (raw / float(np.max(raw)))
    return raw


def _matplotlib_cjk_sans() -> str | None:
    """尽量选用系统已安装的无衬线中文字体，避免中文缺字。"""
    from matplotlib import font_manager

    candidates = (
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
    )
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in avail:
            return name
    return None


def plot_matplotlib(
    x: np.ndarray,
    y: np.ndarray,
    sizes: np.ndarray,
    patterns: list[str],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dirs()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cjk = _matplotlib_cjk_sans()
    rc: dict = {"axes.unicode_minus": False}
    if cjk:
        rc["font.sans-serif"] = [cjk, "DejaVu Sans", "Arial"]

    with plt.rc_context(rc):
        _plot_matplotlib_body(x, y, sizes, patterns, out_png, plt)


def _plot_matplotlib_body(
    x: np.ndarray,
    y: np.ndarray,
    sizes: np.ndarray,
    patterns: list[str],
    out_png: Path,
    plt,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=120)
    fig.patch.set_facecolor("#f7fafc")
    ax.set_facecolor("#f7fafc")

    x_log = _values_for_log_axis(x)
    y_log = _values_for_log_axis(y)

    for pat in PATTERN_ORDER:
        mask = np.array([p == pat for p in patterns], dtype=bool)
        if not np.any(mask):
            continue
        color = PATTERN_HEX.get(pat, PATTERN_HEX_DEFAULT)
        label = PATTERN_LABEL_ZH.get(pat, pat)
        ax.scatter(
            x_log[mask],
            y_log[mask],
            s=sizes[mask],
            c=color,
            alpha=0.38,
            edgecolors="white",
            linewidths=0.35,
            label=f"{label} ({int(np.sum(mask))})",
            rasterized=True,
            zorder=2,
        )

    other_mask = np.array([p not in PATTERN_ORDER for p in patterns], dtype=bool)
    if np.any(other_mask):
        ax.scatter(
            x_log[other_mask],
            y_log[other_mask],
            s=sizes[other_mask],
            c=PATTERN_HEX_DEFAULT,
            alpha=0.35,
            edgecolors="white",
            linewidths=0.3,
            label=f"其他 ({int(np.sum(other_mask))})",
            rasterized=True,
            zorder=1,
        )

    ax.set_xlabel("活跃月数（对数轴，底数 10）", fontsize=12)
    ax.set_ylabel("伙伴数量（对数轴，底数 10）", fontsize=12)
    ax.set_title(
        "公司时序画像 · 气泡大小 ∝ 总链接数（开方缩放）\n"
        f"N = {len(patterns):,} · 双对数坐标 · q1_temporal_patterns.json",
        fontsize=11,
        pad=12,
    )
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.grid(True, which="major", linestyle="--", alpha=0.4, color="#90a4ae")
    ax.grid(True, which="minor", linestyle=":", alpha=0.22, color="#b0bec5")
    ax.set_axisbelow(True)

    leg = ax.legend(
        title="时序模式",
        loc="upper left",
        framealpha=0.92,
        edgecolor="#cfd8dc",
        fontsize=9,
        title_fontsize=10,
    )
    leg.get_frame().set_facecolor("#ffffff")

    # 参考：典型链接规模
    ax.text(
        0.98,
        0.02,
        "颜色：时序模式（蓝色系）\n面积感对应总链接数\n对数轴缓解低×低区域拥挤",
        transform=ax.transAxes,
        fontsize=8,
        color="#455a64",
        ha="right",
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=160, facecolor=fig.patch.get_facecolor())
    plt.close(fig)


def export_plotly_html(
    x: np.ndarray,
    y: np.ndarray,
    links: np.ndarray,
    companies: list[str],
    patterns: list[str],
    sizes_mpl: np.ndarray,
    out_html: Path,
) -> None:
    import plotly.graph_objects as go

    def pat_label_en(p: str) -> str:
        return PATTERN_LABEL_EN.get(p, p)

    x_log = _values_for_log_axis(x)
    y_log = _values_for_log_axis(y)
    diam = _plotly_marker_sizes_from_mpl(sizes_mpl)

    fig = go.Figure()
    for pat in PATTERN_ORDER:
        mask = np.array([p == pat for p in patterns], dtype=bool)
        if not np.any(mask):
            continue
        color = PATTERN_HEX.get(pat, PATTERN_HEX_DEFAULT)
        label = pat_label_en(pat)
        xi, yi, li = x_log[mask], y_log[mask], links[mask]
        di = diam[mask]
        idxs = np.flatnonzero(mask)
        fig.add_trace(
            go.Scatter(
                x=xi,
                y=yi,
                mode="markers",
                name=f"{label} ({int(np.sum(mask))})",
                marker=dict(
                    size=di,
                    color=color,
                    opacity=0.55,
                    line=dict(width=0.5, color="white"),
                ),
                text=[
                    f"<b>{companies[i]}</b><br>Total links: {int(links[i])}<br>Pattern: {pat_label_en(patterns[i])}"
                    for i in idxs
                ],
                hoverinfo="text",
            )
        )

    other_mask = np.array([p not in PATTERN_ORDER for p in patterns], dtype=bool)
    if np.any(other_mask):
        xi, yi, li = x_log[other_mask], y_log[other_mask], links[other_mask]
        di = diam[other_mask]
        idxs = np.flatnonzero(other_mask)
        n_other = int(np.sum(other_mask))
        fig.add_trace(
            go.Scatter(
                x=xi,
                y=yi,
                mode="markers",
                name=f"Other ({n_other})",
                marker=dict(
                    size=di,
                    color=PATTERN_HEX_DEFAULT,
                    opacity=0.5,
                    line=dict(width=0.5, color="white"),
                ),
                text=[
                    f"<b>{companies[i]}</b><br>Total links: {int(links[i])}<br>Pattern: {pat_label_en(patterns[i])}"
                    for i in idxs
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Companies: active months × partners (log–log; bubble size ∝ total links)",
        xaxis_title="Active months (log10)",
        yaxis_title="Partner count (log10)",
        paper_bgcolor="#f7fafc",
        plot_bgcolor="#f7fafc",
        font=dict(size=12, color="#37474f"),
        legend_title_text="Temporal pattern",
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#cfd8dc", borderwidth=1),
        margin=dict(l=56, r=24, t=64, b=48),
        width=980,
        height=700,
    )
    fig.update_xaxes(
        type="log",
        showgrid=True,
        gridcolor="rgba(144,164,174,0.35)",
        gridwidth=1,
    )
    fig.update_yaxes(
        type="log",
        showgrid=True,
        gridcolor="rgba(144,164,174,0.35)",
        gridwidth=1,
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    ensure_dirs()
    out_dir = FIG_DIR / "q1"
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = load_q1_patterns()
    x, y, links, companies, pats = _load_arrays(patterns)
    if x.size == 0:
        print("无可用记录。", file=sys.stderr)
        sys.exit(1)

    sizes = _scatter_sizes(links)
    pat_counts = Counter(pats)

    stem = "q1_bubble_activity_partners"
    data_path = out_dir / f"{stem}_data.json"
    summary = {
        "n_companies": int(x.size),
        "x_field": "active_months",
        "y_field": "partner_count",
        "size_field": "total_links",
        "color_field": "temporal_pattern",
        "pattern_order": list(PATTERN_ORDER),
        "pattern_colors_hex": {p: PATTERN_HEX.get(p, PATTERN_HEX_DEFAULT) for p in PATTERN_ORDER},
        "pattern_counts": {p: pat_counts.get(p, 0) for p in PATTERN_ORDER},
        "other_pattern_count": sum(pat_counts[p] for p in pat_counts if p not in PATTERN_ORDER),
        "x_max": int(np.max(x)),
        "y_max": int(np.max(y)),
        "links_max": int(np.max(links)),
        "x_scale": "log10",
        "y_scale": "log10",
        "log_axis_floor": LOG_AXIS_FLOOR,
    }
    data_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入：{data_path}")

    png_path = out_dir / f"{stem}.png"
    plot_matplotlib(x, y, sizes, pats, png_path)
    print(f"已写入：{png_path}")

    html_path = out_dir / f"{stem}.html"
    export_plotly_html(x, y, links, companies, pats, sizes, html_path)
    print(f"已写入：{html_path}")


if __name__ == "__main__":
    main()
