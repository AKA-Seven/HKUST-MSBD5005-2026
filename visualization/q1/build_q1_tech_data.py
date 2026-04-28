#!/usr/bin/env python3
"""构建 Q1 三层圆环弦图时间数据（科技风版）。"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import json

_vis_root = str(Path(__file__).resolve().parents[1])
if _vis_root not in sys.path:
    sys.path.insert(0, _vis_root)

from figures2d_common import FIG_DIR, OUTPUTS_DIR_Q4, build_company_view, ensure_dirs, load_data


RING_RADII = {1: 180, 2: 320, 3: 470}
L1_BAND = (95, 210)
L2_BAND = (210, 350)
L3_BAND = (350, 500)


def _mode_short(mode: str) -> str:
    mapping = {
        "fish_intensive_hub": "FishHub",
        "fish_network_expander": "FishExp",
        "diversified_broker": "Broker",
        "high_volume_distributor": "HighVol",
        "niche_or_shell": "Niche",
        "general_trader": "General",
    }
    return mapping.get(str(mode), str(mode))


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _split_cluster_path_3(value: str) -> tuple[int, int, int]:
    """从 Cx-Sy-Tz 解析出三层簇编号。"""
    try:
        parts = str(value).split("-")
        l1 = int(parts[0].replace("C", ""))
        l2 = int(parts[1].replace("S", ""))
        l3 = int(parts[2].replace("T", ""))
        return l1, l2, l3
    except Exception:
        return 0, 0, 0


def _node_ids(meta: dict) -> tuple[str, str, str]:
    l1 = f"L1-C{int(meta['hierarchical_cluster'])}"
    l2 = f"L2-C{int(meta['hierarchical_cluster'])}-S{int(meta['hierarchical_subcluster'])}"
    l3 = (
        f"L3-C{int(meta['hierarchical_cluster'])}-S{int(meta['hierarchical_subcluster'])}"
        f"-T{int(meta['hierarchical_microcluster'])}|{_mode_short(meta['business_mode'])}"
    )
    return l1, l2, l3


def _build_events(company_view: pd.DataFrame, reliable_links: pd.DataFrame, edge_clusters: pd.DataFrame) -> tuple[dict, list[str]]:
    """生成按月快照事件：节点活跃度、层次边、交易边(含边簇颜色)。"""
    company_meta = company_view.set_index("company").to_dict("index")

    edge_map = {}
    for _, row in edge_clusters.iterrows():
        key = _pair_key(str(row["source"]), str(row["target"]))
        edge_map[key] = {
            "cluster": int(row["edge_cluster"]),
            "color": str(row["edge_cluster_color"]),
            "semantic": str(row["edge_semantic_label"]),
        }

    links = reliable_links.copy()
    links["month"] = links["arrivaldate"].astype(str).str.slice(0, 7)
    links = links[links["month"].str.len() == 7].copy()
    months = sorted(links["month"].unique().tolist())

    # 预选展示节点（按全年活跃）
    node_counter = Counter()
    for _, row in links.iterrows():
        for endpoint in ("source", "target"):
            company = row.get(endpoint)
            meta = company_meta.get(company)
            if not meta:
                continue
            n1, n2, n3 = _node_ids(meta)
            node_counter[n1] += 1
            node_counter[n2] += 1
            node_counter[n3] += 1

    l1_nodes = [node for node in node_counter if node.startswith("L1-")]
    l1_nodes = sorted(l1_nodes, key=lambda n: node_counter[n], reverse=True)[:8]
    l2_nodes = [node for node in node_counter if node.startswith("L2-") and any(node.split("-S")[0].replace("L2", "L1") == l1 for l1 in l1_nodes)]
    l2_nodes = sorted(l2_nodes, key=lambda n: node_counter[n], reverse=True)[:18]
    l3_nodes = [node for node in node_counter if node.startswith("L3-") and any(node.replace("L3", "L2").split("-T")[0] == l2 for l2 in l2_nodes)]
    l3_nodes = sorted(l3_nodes, key=lambda n: node_counter[n], reverse=True)[:30]

    allowed_nodes = set(l1_nodes + l2_nodes + l3_nodes)

    snapshots = {}
    for month in months:
        snapshots[month] = {
            "node_values": Counter(),
            "hier_edges": Counter(),
            "trade_edges": defaultdict(int),
            "trade_meta": {},
        }

    for _, row in links.iterrows():
        month = row["month"]
        state = snapshots[month]
        endpoints = []
        for endpoint in ("source", "target"):
            company = row.get(endpoint)
            meta = company_meta.get(company)
            if not meta:
                continue
            n1, n2, n3 = _node_ids(meta)
            if n1 not in allowed_nodes or n2 not in allowed_nodes or n3 not in allowed_nodes:
                continue
            state["node_values"][n1] += 1
            state["node_values"][n2] += 1
            state["node_values"][n3] += 1
            state["hier_edges"][(n1, n2)] += 1
            state["hier_edges"][(n2, n3)] += 1
            endpoints.append((company, n3))

        if len(endpoints) == 2:
            (c1, n3_a), (c2, n3_b) = endpoints
            key = _pair_key(c1, c2)
            meta = edge_map.get(key)
            if not meta:
                continue
            if n3_a == n3_b:
                continue
            trade_key = tuple(sorted((n3_a, n3_b)))
            state["trade_edges"][trade_key] += 1
            state["trade_meta"][trade_key] = meta

    # 用主图 JSON 节点坐标参与排序，让同层节点空间组织更稳定。
    # FIG_DIR = …/visualization/figures_2d → parent.parent = 项目根下的 MC2/
    gephi_path = FIG_DIR.parent.parent / "MC2" / "mc2_challenge_graph.json"
    company_geo: dict[str, tuple[float, float]] = {}
    if gephi_path.exists():
        gephi = json.loads(gephi_path.read_text(encoding="utf-8"))
        for node in gephi.get("nodes", []):
            node_id = node.get("id")
            gx = float(node.get("x", 0) or 0)
            gy = float(node.get("y", 0) or 0)
            company_geo[node_id] = (gx, gy)
    # 若文件不存在则跳过地理排序，后续 sort_by_geo 返回活跃度顺序作为兜底。

    node_geo_angle = {}
    for company, meta in company_meta.items():
        if company not in company_geo:
            continue
        n1, n2, n3 = _node_ids(meta)
        gx, gy = company_geo[company]
        ang = float(np.arctan2(gy, gx))
        node_geo_angle.setdefault(n1, []).append(ang)
        node_geo_angle.setdefault(n2, []).append(ang)
        node_geo_angle.setdefault(n3, []).append(ang)

    def sort_by_geo(nodes: list[str]) -> list[str]:
        def key_fn(n: str):
            arr = node_geo_angle.get(n)
            if not arr:
                return 10.0
            return float(np.mean(arr))
        return sorted(nodes, key=key_fn)

    l1_nodes = sort_by_geo(l1_nodes)
    l2_nodes = sort_by_geo(l2_nodes)
    l3_nodes = sort_by_geo(l3_nodes)

    # 三层包含关系布局：L1 扇区 -> L2 子扇区 -> L3 子点
    l2_parent = {}
    for node in l2_nodes:
        prefix = node.split("-S")[0].replace("L2", "L1")
        l2_parent[node] = prefix
    l3_parent = {}
    for node in l3_nodes:
        prefix = node.split("-T")[0].replace("L3", "L2")
        l3_parent[node] = prefix

    l1_to_l2 = defaultdict(list)
    l2_to_l3 = defaultdict(list)
    for l2 in l2_nodes:
        if l2_parent[l2] in l1_nodes:
            l1_to_l2[l2_parent[l2]].append(l2)
    for l3 in l3_nodes:
        if l3_parent[l3] in l2_nodes:
            l2_to_l3[l3_parent[l3]].append(l3)

    # 分配角度：按 L3 活跃度权重，保证包含关系扇区清晰。
    l3_weight = {node: max(1, node_counter.get(node, 1)) for node in l3_nodes}
    l2_weight = {l2: max(1, sum(l3_weight.get(l3, 1) for l3 in l2_to_l3.get(l2, []))) for l2 in l2_nodes}
    l1_weight = {l1: max(1, sum(l2_weight.get(l2, 1) for l2 in l1_to_l2.get(l1, []))) for l1 in l1_nodes}

    l1_nodes = sorted(l1_nodes, key=lambda n: float(np.mean(node_geo_angle.get(n, [0]))))
    for l1 in l1_nodes:
        l1_to_l2[l1] = sorted(l1_to_l2.get(l1, []), key=lambda n: float(np.mean(node_geo_angle.get(n, [0]))))
        for l2 in l1_to_l2[l1]:
            l2_to_l3[l2] = sorted(l2_to_l3.get(l2, []), key=lambda n: float(np.mean(node_geo_angle.get(n, [0]))))

    def allocate(items: list[str], weight_map: dict, start: float, end: float, gap: float) -> dict[str, tuple[float, float]]:
        spans = {}
        if not items:
            return spans
        total_gap = gap * max(0, len(items) - 1)
        width = max(0.001, (end - start) - total_gap)
        total_weight = max(1.0, float(sum(weight_map.get(item, 1) for item in items)))
        cursor = start
        for idx, item in enumerate(items):
            frac = float(weight_map.get(item, 1)) / total_weight
            seg = width * frac
            seg_start = cursor
            seg_end = cursor + seg
            spans[item] = (seg_start, seg_end)
            cursor = seg_end + (gap if idx < len(items) - 1 else 0.0)
        return spans

    start_angle = -np.pi / 2
    end_angle = start_angle + 2 * np.pi
    l1_spans = allocate(l1_nodes, l1_weight, start_angle, end_angle, gap=0.06)
    l2_spans = {}
    for l1 in l1_nodes:
        children = l1_to_l2.get(l1, [])
        if not children:
            continue
        a0, a1 = l1_spans[l1]
        part = allocate(children, l2_weight, a0, a1, gap=0.02)
        l2_spans.update(part)
    l3_angles = {}
    for l2 in l2_nodes:
        children = l2_to_l3.get(l2, [])
        if not children:
            continue
        a0, a1 = l2_spans.get(l2, (start_angle, start_angle + 0.01))
        if len(children) == 1:
            l3_angles[children[0]] = (a0 + a1) / 2
        else:
            for idx, child in enumerate(children):
                t = idx / max(1, len(children) - 1)
                l3_angles[child] = a0 + (a1 - a0) * t

    # 配色：点使用单一冷色；背景扇区和边使用高对比色。
    l1_palette = [
        "#38C9FF",
        "#6AE3FF",
        "#52B7FF",
        "#7ED4FF",
        "#35E5E8",
        "#64C2FF",
        "#7AB8FF",
        "#4FD7FF",
    ]
    l1_color = {node: l1_palette[idx % len(l1_palette)] for idx, node in enumerate(l1_nodes)}

    sectors_l1 = []
    sectors_l2 = []
    for l1 in l1_nodes:
        a0, a1 = l1_spans.get(l1, (0.0, 0.0))
        sectors_l1.append({"id": l1, "start": float(a0), "end": float(a1), "color": l1_color[l1]})
        for l2 in l1_to_l2.get(l1, []):
            b0, b1 = l2_spans.get(l2, (a0, a0))
            sectors_l2.append({"id": l2, "parent": l1, "start": float(b0), "end": float(b1), "color": l1_color[l1]})

    # 坐标布局：聚类点严格在三层圆周上
    all_nodes = l1_nodes + l2_nodes + l3_nodes
    nodes = []
    for layer, ring_nodes in ((1, l1_nodes), (2, l2_nodes), (3, l3_nodes)):
        radius = RING_RADII[layer]
        for node in ring_nodes:
            if layer == 1:
                a0, a1 = l1_spans.get(node, (0.0, 0.0))
                angle = (a0 + a1) / 2
            elif layer == 2:
                a0, a1 = l2_spans.get(node, (0.0, 0.0))
                angle = (a0 + a1) / 2
            else:
                angle = l3_angles.get(node, 0.0)
            x = radius * float(np.cos(angle))
            y = radius * float(np.sin(angle))
            nodes.append(
                {
                    "id": node,
                    "name": node,
                    "layer": layer,
                    "category": layer - 1,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "angle": float(angle),
                    "baseSize": 10 if layer == 1 else (8 if layer == 2 else 6),
                }
            )

    snapshots_out = []
    for month in months:
        state = snapshots[month]
        node_max = max(state["node_values"].values()) if state["node_values"] else 1
        hier_max = max(state["hier_edges"].values()) if state["hier_edges"] else 1
        trade_max = max(state["trade_edges"].values()) if state["trade_edges"] else 1

        hier_links = []
        for (source, target), count in state["hier_edges"].items():
            hier_links.append(
                {
                    "source": source,
                    "target": target,
                    "value": count,
                    "kind": "hierarchy",
                    "lineWidth": round(0.8 + 5.0 * count / hier_max, 3),
                    "opacity": round(0.20 + 0.45 * count / hier_max, 3),
                    "color": "#4FA8FF" if source.startswith("L1-") else "#5BD9A0",
                    "edgeCluster": -1,
                    "edgeSemantic": "hierarchy",
                }
            )

        trade_links = []
        for (source, target), count in state["trade_edges"].items():
            meta = state["trade_meta"].get((source, target), {"cluster": -1, "color": "#FF8A65", "semantic": "trade"})
            trade_links.append(
                {
                    "source": source,
                    "target": target,
                    "value": count,
                    "kind": "trade",
                    "lineWidth": round(1.2 + 10.0 * count / trade_max, 3),
                    "opacity": round(0.30 + 0.60 * count / trade_max, 3),
                    "color": meta["color"],
                    "edgeCluster": int(meta["cluster"]),
                    "edgeSemantic": str(meta["semantic"]),
                }
            )

        snapshots_out.append(
            {
                "month": month,
                "nodeValues": {node: int(state["node_values"].get(node, 0)) for node in all_nodes},
                "nodeMax": int(node_max),
                "links": hier_links + trade_links,
            }
        )

    data = {
        "months": months,
        "nodes": nodes,
        "categories": [
            {"name": "L1 cluster"},
            {"name": "L2 subcluster"},
            {"name": "L3 microcluster"},
        ],
        "rings": {
            "l1_band": {"r0": L1_BAND[0], "r": L1_BAND[1]},
            "l2_band": {"r0": L2_BAND[0], "r": L2_BAND[1]},
            "l3_band": {"r0": L3_BAND[0], "r": L3_BAND[1]},
            "l1_sectors": sectors_l1,
            "l2_sectors": sectors_l2,
        },
        "snapshots": snapshots_out,
        "meta": {
            "l1_count": len(l1_nodes),
            "l2_count": len(l2_nodes),
            "l3_count": len(l3_nodes),
        },
    }
    return data, months


def build_q1_tech_data(output_js: Path, output_html: Path) -> None:
    ensure_dirs()
    _, company_df, delta_df, reliable_links = load_data()
    edge_clusters = pd.read_csv(OUTPUTS_DIR_Q4 / "edge_clusters.csv")
    company_view = build_company_view(company_df, delta_df)
    data, _ = _build_events(company_view, reliable_links, edge_clusters)

    output_js.write_text("window.Q1_TECH_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")

    template_path = Path(__file__).resolve().parent / "q1_tech_chord_template.html"
    html_text = template_path.read_text(encoding="utf-8")
    html_text = html_text.replace("./vendor/echarts.min.js", "../../vendor/echarts.min.js")
    html_text = html_text.replace("./figures_2d/q1/q1_tech_data.js", "./q1_tech_data.js")
    output_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    q1_dir = FIG_DIR / "q1"
    for path in q1_dir.glob("*"):
        if path.is_file():
            path.unlink()

    output_js = q1_dir / "q1_tech_data.js"
    output_html = q1_dir / "q1_01_futuristic_circular_chord_timeline.html"
    build_q1_tech_data(output_js, output_html)
    print(f"Q1 tech figure written: {output_html}")


if __name__ == "__main__":
    main()
