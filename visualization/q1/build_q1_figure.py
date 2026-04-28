#!/usr/bin/env python3
"""Q1 单图（圆形两层弦图 + 时间滑块动态）。"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_vis_root = str(Path(__file__).resolve().parents[1])
if _vis_root not in sys.path:
    sys.path.insert(0, _vis_root)

from figures2d_common import FIG_DIR, build_company_view, ensure_dirs, load_data, load_q1_patterns


def _clear_q1_outputs() -> None:
    """Q1 仅保留一张图，先清空目录。"""
    q1_dir = FIG_DIR / "q1"
    for path in q1_dir.glob("*"):
        if path.is_file():
            path.unlink()


def _short_mode(mode: str) -> str:
    mapping = {
        "fish_intensive_hub": "FishHub",
        "fish_network_expander": "FishExp",
        "diversified_broker": "Broker",
        "high_volume_distributor": "HighVol",
        "niche_or_shell": "Niche",
        "general_trader": "General",
    }
    return mapping.get(str(mode), str(mode))


def _build_endpoint_events(
    company_view: pd.DataFrame,
    q1_patterns: list[dict],
) -> tuple[pd.DataFrame, list[str]]:
    """将原始图谱的公司月度活动映射到 L1/L2 层级，按月形成可累计事件。

    数据源已从 reliable_links（预测链路）改为 q1_temporal_patterns（原始图谱），
    使 Q1 时间轴真实反映主图的贸易行为，而非预测补全结果。
    每条记录附带 count 字段（该公司当月在原始图谱中的贸易次数），
    供 _build_frame_data 用 sum() 而非 size() 进行加权聚合。
    """
    cmap = company_view.set_index("company")[
        ["hierarchical_cluster", "cluster_path", "business_mode"]
    ].to_dict("index")

    records = []
    for item in q1_patterns:
        company = item.get("company")
        meta = cmap.get(company)
        if not meta:
            continue
        l1 = f"L1-C{int(meta['hierarchical_cluster'])}"
        l2 = f"L2-{meta['cluster_path']}|{_short_mode(meta['business_mode'])}"
        for month, count in item.get("monthly_counts", {}).items():
            if len(month) == 7:
                records.append(
                    {
                        "month": month,
                        "company": company,
                        "l1": l1,
                        "l2": l2,
                        "count": int(count),
                    }
                )

    events = pd.DataFrame(records)
    if events.empty:
        return events, []

    months = sorted(events["month"].unique().tolist())
    return events, months


def _select_nodes(events: pd.DataFrame) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """选取展示的内圈/外圈节点和弦线边。

    按原始图谱贸易次数（count 列之和）排序，确保高活跃簇优先展示。
    """
    l1_counts = events.groupby("l1")["count"].sum().sort_values(ascending=False)
    l1_nodes = l1_counts.head(8).index.tolist()

    filtered = events[events["l1"].isin(l1_nodes)].copy()
    l2_counts = filtered.groupby("l2")["count"].sum().sort_values(ascending=False)
    l2_nodes = l2_counts.head(22).index.tolist()

    filtered = filtered[filtered["l2"].isin(l2_nodes)]
    edge_counts = filtered.groupby(["l1", "l2"])["count"].sum().sort_values(ascending=False)
    edge_list = edge_counts.head(55).index.tolist()

    return l1_nodes, l2_nodes, edge_list


def _node_positions(l1_nodes: list[str], l2_nodes: list[str]) -> dict[str, tuple[float, float]]:
    """圆形布局：内圈 L1，外圈 L2。"""
    positions: dict[str, tuple[float, float]] = {}
    inner_r = 0.46
    outer_r = 0.90

    for i, node in enumerate(l1_nodes):
        angle = (2 * math.pi * i / max(1, len(l1_nodes))) - math.pi / 2
        positions[node] = (inner_r * math.cos(angle), inner_r * math.sin(angle))

    for i, node in enumerate(l2_nodes):
        angle = (2 * math.pi * i / max(1, len(l2_nodes))) - math.pi / 2
        positions[node] = (outer_r * math.cos(angle), outer_r * math.sin(angle))

    return positions


def _curve_xy(start: tuple[float, float], end: tuple[float, float], points: int = 40) -> tuple[list[float], list[float]]:
    """二次贝塞尔曲线，用于弦线。"""
    x0, y0 = start
    x1, y1 = end
    cx, cy = 0.0, 0.0
    t = np.linspace(0, 1, points)
    x = ((1 - t) ** 2) * x0 + 2 * (1 - t) * t * cx + (t**2) * x1
    y = ((1 - t) ** 2) * y0 + 2 * (1 - t) * t * cy + (t**2) * y1
    return x.tolist(), y.tolist()


def _build_frame_data(
    events: pd.DataFrame,
    month: str,
    l1_nodes: list[str],
    l2_nodes: list[str],
    edge_list: list[tuple[str, str]],
    positions: dict[str, tuple[float, float]],
) -> dict:
    """构建单个月份（累计到该月）的节点与边计数。"""
    active = events[events["month"] <= month]
    # 用原始图谱贸易次数（count 列）累计求和，反映真实活跃强度。
    l1_count = active.groupby("l1")["count"].sum().to_dict()
    l2_count = active.groupby("l2")["count"].sum().to_dict()
    edge_count = active.groupby(["l1", "l2"])["count"].sum().to_dict()

    # 节点大小映射
    l1_sizes = [12 + 28 * (l1_count.get(node, 0) / max(1, max(l1_count.values()) if l1_count else 1)) for node in l1_nodes]
    l2_sizes = [8 + 22 * (l2_count.get(node, 0) / max(1, max(l2_count.values()) if l2_count else 1)) for node in l2_nodes]

    # 边宽度映射
    edge_values = [edge_count.get(edge, 0) for edge in edge_list]
    edge_max = max(edge_values) if edge_values else 1
    edge_widths = [0.6 + 8.0 * (value / edge_max) if value > 0 else 0.0 for value in edge_values]
    edge_opacity = [0.15 + 0.60 * (value / edge_max) if value > 0 else 0.0 for value in edge_values]

    return {
        "month": month,
        "l1_sizes": l1_sizes,
        "l2_sizes": l2_sizes,
        "edge_widths": edge_widths,
        "edge_opacity": edge_opacity,
        "edge_values": edge_values,
    }


def build_q1_single_figure(output_path: Path) -> None:
    """生成 Q1 单张精美交互图（圆形弦图+时间拖动）。

    数据源改为原始图谱的公司时序画像（q1_temporal_patterns.json），
    时间轴与节点大小均反映主图中的真实贸易行为，修复了原先错误地
    使用 reliable_links（预测链路）作为 Q1 展示数据的问题。
    """
    ensure_dirs()
    _, company_df, delta_df, _ = load_data()
    q1_patterns = load_q1_patterns()
    company_view = build_company_view(company_df, delta_df)
    events, months = _build_endpoint_events(company_view, q1_patterns)
    if events.empty or not months:
        raise RuntimeError("No event data available for Q1 figure.")

    l1_nodes, l2_nodes, edge_list = _select_nodes(events)
    events = events[events["l1"].isin(l1_nodes) & events["l2"].isin(l2_nodes)].copy()
    positions = _node_positions(l1_nodes, l2_nodes)

    frames_state = [
        _build_frame_data(events, month, l1_nodes, l2_nodes, edge_list, positions)
        for month in months
    ]

    fig = go.Figure()

    # 背景环（内外圈）
    inner_circle = go.Scatter(
        x=[0.46 * math.cos(t) for t in np.linspace(0, 2 * math.pi, 200)],
        y=[0.46 * math.sin(t) for t in np.linspace(0, 2 * math.pi, 200)],
        mode="lines",
        line=dict(color="rgba(47,118,176,0.25)", width=2),
        hoverinfo="skip",
        showlegend=False,
    )
    outer_circle = go.Scatter(
        x=[0.90 * math.cos(t) for t in np.linspace(0, 2 * math.pi, 300)],
        y=[0.90 * math.sin(t) for t in np.linspace(0, 2 * math.pi, 300)],
        mode="lines",
        line=dict(color="rgba(88,169,106,0.25)", width=2),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_trace(inner_circle)
    fig.add_trace(outer_circle)

    # 边 traces（数量固定，frame 更新宽度/透明度）
    for (l1, l2) in edge_list:
        x, y = _curve_xy(positions[l1], positions[l2], points=42)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="rgba(226,109,90,0.1)", width=0.1),
                hovertemplate=f"{l1} ↔ {l2}<extra></extra>",
                showlegend=False,
            )
        )

    # 内圈节点
    fig.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in l1_nodes],
            y=[positions[node][1] for node in l1_nodes],
            mode="markers+text",
            text=l1_nodes,
            textposition="middle center",
            marker=dict(size=frames_state[-1]["l1_sizes"], color="#2f76b0", line=dict(color="white", width=1.2)),
            name="L1 cluster",
            hovertemplate="%{text}<extra>L1</extra>",
        )
    )

    # 外圈节点
    fig.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in l2_nodes],
            y=[positions[node][1] for node in l2_nodes],
            mode="markers+text",
            text=l2_nodes,
            textposition="top center",
            marker=dict(size=frames_state[-1]["l2_sizes"], color="#58a96a", line=dict(color="white", width=1.0)),
            textfont=dict(size=10),
            name="L2 cluster | business mode",
            hovertemplate="%{text}<extra>L2</extra>",
        )
    )

    # frame 构建：只更新边与节点样式
    edge_trace_start = 2
    l1_trace_idx = edge_trace_start + len(edge_list)
    l2_trace_idx = l1_trace_idx + 1
    frames = []
    for state in frames_state:
        frame_data = []
        for idx, value in enumerate(state["edge_values"]):
            width = state["edge_widths"][idx]
            opacity = state["edge_opacity"][idx]
            frame_data.append(
                go.Scatter(
                    line=dict(color=f"rgba(226,109,90,{opacity:.3f})", width=width),
                    hovertemplate=f"{edge_list[idx][0]} ↔ {edge_list[idx][1]}<br>count={value}<extra></extra>",
                )
            )
        frame_data.append(go.Scatter(marker=dict(size=state["l1_sizes"])))
        frame_data.append(go.Scatter(marker=dict(size=state["l2_sizes"])))
        frames.append(go.Frame(name=state["month"], data=frame_data, traces=list(range(edge_trace_start, l2_trace_idx + 1))))

    fig.frames = frames

    slider_steps = [
        dict(
            method="animate",
            label=month,
            args=[
                [month],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": False},
                    "transition": {"duration": 0},
                },
            ],
        )
        for month in months
    ]

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Q1 · Circular Two-Layer Chord of Business Patterns (Time-Draggable)",
            x=0.5,
            y=0.96,
            font=dict(size=22),
        ),
        width=1320,
        height=920,
        margin=dict(l=30, r=30, t=90, b=120),
        xaxis=dict(visible=False, range=[-1.15, 1.15]),
        yaxis=dict(visible=False, range=[-1.15, 1.15], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.12,
                y=-0.08,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {"frame": {"duration": 420, "redraw": False}, "transition": {"duration": 120}, "fromcurrent": True},
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"},
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=len(months) - 1,
                x=0.1,
                y=-0.04,
                len=0.82,
                currentvalue={"prefix": "Month: ", "font": {"size": 13}},
                steps=slider_steps,
            )
        ],
        annotations=[
            dict(
                x=0.22,
                y=0.94,
                xref="paper",
                yref="paper",
                text="Inner ring = L1 clusters",
                showarrow=False,
                font=dict(size=12, color="#2f76b0"),
            ),
            dict(
                x=0.78,
                y=0.94,
                xref="paper",
                yref="paper",
                text="Outer ring = L2 subclusters + business mode",
                showarrow=False,
                font=dict(size=12, color="#58a96a"),
            ),
        ],
    )

    # 初始化到最后一个月
    if frames:
        last = frames[-1]
        for i, trace_idx in enumerate(last.traces):
            trace_obj = last.data[i]
            if "line" in trace_obj:
                fig.data[trace_idx].line = trace_obj.line
            if "marker" in trace_obj:
                fig.data[trace_idx].marker = trace_obj.marker

    fig.write_html(output_path, include_plotlyjs=True)


def main() -> None:
    _clear_q1_outputs()
    output = FIG_DIR / "q1" / "q1_01_circular_hierarchical_chord_dynamic.html"
    build_q1_single_figure(output)
    print(f"Q1 done: {output}")


if __name__ == "__main__":
    main()
