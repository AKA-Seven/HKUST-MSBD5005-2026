#!/usr/bin/env python3
"""Q1：Top 50 公司 × 月份 的月度贸易链数热力图。

数据来源：仅 `outputs/q1/q1_temporal_patterns.json` 中的 `monthly_counts`（原始图谱链数）。
公司排序：与 `build_q1_ridge_river.py` 一致，按 `total_links` 降序取前 50。
时间轴：所选 50 家公司所有活跃月份的全局最小月～最大月（含链数为 0 的空白月）。

输出：`visualization/figures_2d/q1/q1_monthly_heatmap_top50.png`、`.html`、`_data.json`。

运行（conda 环境名 `dv`）：
`conda run -n dv python visualization/q1/build_q1_monthly_heatmap.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_vis_root = Path(__file__).resolve().parents[1]
if str(_vis_root) not in sys.path:
    sys.path.insert(0, str(_vis_root))

from figures2d_common import FIG_DIR, ensure_dirs, load_q1_patterns

TOP_N = 50
MAX_LABEL_LEN = 42


def top_companies_from_q1(patterns: list[dict], n: int) -> list[str]:
    ranked = sorted(patterns, key=lambda r: int(r.get("total_links") or 0), reverse=True)
    out = []
    seen = set()
    for row in ranked:
        c = row.get("company")
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(str(c))
        if len(out) >= n:
            break
    return out


def _month_tuple(m: str) -> tuple[int, int]:
    y, mo = m.split("-", 1)
    return int(y), int(mo)


def month_range_inclusive(start: str, end: str) -> list[str]:
    y1, m1 = _month_tuple(start)
    y2, m2 = _month_tuple(end)
    out: list[str] = []
    y, m = y1, m1
    while (y, m) < (y2, m2) or (y, m) == (y2, m2):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def build_heatmap_matrix(
    patterns: list[dict],
    companies: list[str],
) -> tuple[np.ndarray, list[str], list[dict]]:
    by_company = {str(r.get("company")): r for r in patterns if r.get("company")}

    all_months: set[str] = set()
    for c in companies:
        row = by_company.get(c)
        if not row:
            continue
        mc = row.get("monthly_counts") or {}
        if isinstance(mc, dict):
            all_months.update(mc.keys())

    if not all_months:
        raise RuntimeError("无可用月份（检查 q1_temporal_patterns.json）。")

    months_sorted = sorted(all_months)
    month_axis = month_range_inclusive(months_sorted[0], months_sorted[-1])

    n_c = len(companies)
    n_t = len(month_axis)
    month_index = {m: i for i, m in enumerate(month_axis)}
    mat = np.zeros((n_c, n_t), dtype=np.float64)

    meta_rows: list[dict] = []
    for i, c in enumerate(companies):
        row = by_company.get(c, {})
        mc = row.get("monthly_counts") or {}
        if isinstance(mc, dict):
            for mk, v in mc.items():
                j = month_index.get(mk)
                if j is not None:
                    try:
                        mat[i, j] = float(v)
                    except (TypeError, ValueError):
                        pass
        meta_rows.append(
            {
                "company": c,
                "temporal_pattern": row.get("temporal_pattern", ""),
                "total_links": int(row.get("total_links") or 0),
            }
        )

    return mat, month_axis, meta_rows


def _short_label(name: str, max_len: int = MAX_LABEL_LEN) -> str:
    s = str(name).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def plot_png(
    mat: np.ndarray,
    companies: list[str],
    months: list[str],
    meta_rows: list[dict],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    ensure_dirs()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 视觉：链数跨度大，用 log1p 上色；colorbar 仍标「链数」量级
    z = np.log1p(mat)
    zmax = float(np.nanmax(z)) if z.size else 0.0
    if zmax <= 0:
        zmax = 1.0

    fig_h = max(8.0, 0.22 * len(companies))
    fig_w = max(10.0, 0.12 * len(months))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    phase1_cmap = LinearSegmentedColormap.from_list(
        "phase1_teal",
        ["#e8dcc8", "#d9e5e2", "#b7d4cf", "#5ea8a1", "#0f766e", "#0a5c56"],
    )

    im = ax.imshow(
        z,
        aspect="auto",
        interpolation="nearest",
        cmap=phase1_cmap,
        norm=Normalize(vmin=0.0, vmax=zmax),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("log(1 + monthly link count)")

    tick_step = max(1, len(months) // 24)
    ax.set_xticks(np.arange(0, len(months), tick_step))
    ax.set_xticklabels([months[i] for i in range(0, len(months), tick_step)], rotation=45, ha="right", fontsize=7)

    y_labels = [f"{_short_label(c)}  ({meta_rows[i].get('temporal_pattern', '')})" for i, c in enumerate(companies)]
    ax.set_yticks(np.arange(len(companies)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Company (temporal pattern)")
    ax.set_title(
        f"Top {len(companies)} companies — monthly trade link counts (from q1_temporal_patterns.json)\n"
        "Rows ranked by total_links; blank months on axis show zero activity",
        fontsize=10,
        pad=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def export_plotly_html(
    mat: np.ndarray,
    companies: list[str],
    months: list[str],
    meta_rows: list[dict],
    out_path: Path,
) -> None:
    import plotly.graph_objects as go

    z = np.log1p(mat)
    hover = []
    for i, c in enumerate(companies):
        row_h: list[str] = []
        pat = meta_rows[i].get("temporal_pattern", "")
        tot = meta_rows[i].get("total_links", 0)
        for j, m in enumerate(months):
            cnt = int(mat[i, j]) if mat[i, j] == mat[i, j] else 0
            row_h.append(
                f"<b>{c}</b><br>Month {m}<br>Links: {cnt}<br>Pattern: {pat}<br>Total links: {tot}"
            )
        hover.append(row_h)

    # Colors aligned with phase1_sketches/q1.html heat legend (cream → teal).
    phase1_colorscale = [
        [0.0, "#e8dcc8"],
        [0.2, "#d9e5e2"],
        [0.45, "#b7d4cf"],
        [0.68, "#5ea8a1"],
        [0.88, "#0f766e"],
        [1.0, "#0a5c56"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=months,
            y=[_short_label(c, 50) for c in companies],
            hovertext=hover,
            hoverinfo="text",
            colorscale=phase1_colorscale,
            colorbar=dict(
                title=dict(text="log(1+links)", side="right", font=dict(size=11)),
                tickfont=dict(size=10),
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"Top {len(companies)} companies — monthly link counts "
                "(outputs/q1/q1_temporal_patterns.json)"
            ),
            font=dict(size=14, color="#14213d", family="Avenir Next, Segoe UI, PingFang SC, Noto Sans SC, sans-serif"),
        ),
        xaxis=dict(
            title="Month",
            tickangle=-45,
            tickfont=dict(size=9, color="#5c6b73"),
            gridcolor="rgba(20,33,61,0.06)",
        ),
        yaxis=dict(
            title="Company",
            tickfont=dict(size=8, color="#14213d"),
            autorange="reversed",
            gridcolor="rgba(20,33,61,0.06)",
        ),
        margin=dict(l=280, r=28, t=56, b=120),
        paper_bgcolor="#fffaf0",
        plot_bgcolor="rgba(255,250,240,0.65)",
        font=dict(family="Avenir Next, Segoe UI, PingFang SC, Noto Sans SC, sans-serif", color="#14213d"),
        height=max(640, 14 * len(companies)),
        width=max(960, 10 * len(months)),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def main() -> None:
    ensure_dirs()
    out_dir = FIG_DIR / "q1"
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = load_q1_patterns()
    companies = top_companies_from_q1(patterns, TOP_N)
    if len(companies) < TOP_N:
        print(f"警告：仅能取到 {len(companies)} 家公司。", file=sys.stderr)

    mat, months, meta_rows = build_heatmap_matrix(patterns, companies)

    stem = "q1_monthly_heatmap_top50"
    data_path = out_dir / f"{stem}_data.json"
    payload = {
        "description": "Top companies by total_links; cell = monthly link count from q1_temporal_patterns.json",
        "top_n": len(companies),
        "companies": companies,
        "months": months,
        "counts": mat.tolist(),
        "row_meta": meta_rows,
    }
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入：{data_path}")

    png_path = out_dir / f"{stem}.png"
    plot_png(mat, companies, months, meta_rows, png_path)
    print(f"已写入：{png_path}")

    html_path = out_dir / f"{stem}.html"
    export_plotly_html(mat, companies, months, meta_rows, html_path)
    print(f"已写入：{html_path}")


if __name__ == "__main__":
    main()
