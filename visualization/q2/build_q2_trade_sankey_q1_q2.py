#!/usr/bin/env python3
"""Q1 公司对基础流 + Q2 公司对预测流（不按时序类型聚合）。

- **Q1（灰）**：预测端点诱导子图内的 relationship 边，`company_a → company_b`，
  相同有向对合并为一条，粗细由 confidence 累加。
- **Q2（彩色）**：`reliable_links` 有向边，按 `(source, target, bundle)` 聚合条数；
  不经由「时序类型」节点。

布局：公司在图中各出现两次（左「出发」、右「到达」）。节点过多时保留全部预测端点并按度数截断至 NODE_CAP。

可选 hover：`q1_temporal_patterns.json` 仅用于提示文案，不参与分组。

输出：`visualization/figures_2d/q2/q2_q1_base_trade_q2_predictions_sankey.html`
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

_vis_root = Path(__file__).resolve().parents[1]
if str(_vis_root) not in sys.path:
    sys.path.insert(0, str(_vis_root))

from figures2d_common import FIG_DIR, OUTPUTS_DIR_Q1, OUTPUTS_DIR_Q2, ensure_dirs

FIG_Q2 = FIG_DIR / "q2"
OUT_HTML = FIG_Q2 / "q2_q1_base_trade_q2_predictions_sankey.html"

Q1_REL_CSV = OUTPUTS_DIR_Q1 / "q1_relationship_patterns.csv"
Q1_TEMPORAL_JSON = OUTPUTS_DIR_Q1 / "q1_temporal_patterns.json"

BASE_LINK_COLOR = "rgba(148, 163, 184, 0.44)"

BUNDLE_EDGE_COLORS: dict[str, str] = {
    "carp": "rgba(249, 115, 22, 0.9)",
    "herring": "rgba(234, 179, 8, 0.9)",
    "pollock": "rgba(6, 182, 212, 0.9)",
    "salmon_wgl": "rgba(236, 72, 153, 0.9)",
    "chub_mackerel": "rgba(132, 204, 22, 0.9)",
}

NODE_CAP = 72


def _short_label(name: str, max_len: int = 22) -> str:
    if len(name) <= max_len:
        return name
    half = (max_len - 1) // 2
    return name[:half] + "..." + name[-(max_len - half - 1) :]


def _pick_nodes(pred_nodes: set[str], q1_pairs: list[tuple[str, str, float, str]]) -> list[str]:
    selected = set(pred_nodes)
    if len(selected) >= NODE_CAP:
        return sorted(selected)

    deg: defaultdict[str, int] = defaultdict(int)
    for a, b, _, _ in q1_pairs:
        deg[a] += 1
        deg[b] += 1

    cand = set()
    for a, b, _, _ in q1_pairs:
        if a in selected and b not in selected:
            cand.add(b)
        if b in selected and a not in selected:
            cand.add(a)

    for c, _ in sorted(((x, deg[x]) for x in cand), key=lambda t: -t[1]):
        if len(selected) >= NODE_CAP:
            break
        selected.add(c)
    return sorted(selected)


def _temporal_hover_map(companies: set[str], json_path: Path) -> dict[str, str]:
    """仅用于 hover，不参与桑基分组。"""
    if not json_path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        import ijson  # type: ignore[import-untyped]
    except ImportError:
        ijson = None

    if ijson is not None:
        with json_path.open("rb") as f:
            for obj in ijson.items(f, "item"):
                c = str(obj.get("company", ""))
                if c in companies:
                    out[c] = str(obj.get("temporal_pattern") or "?")
        return out

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for r in data:
        c = str(r.get("company", ""))
        if c in companies:
            out[c] = str(r.get("temporal_pattern") or "?")
    return out


def main() -> None:
    ensure_dirs()
    FIG_Q2.mkdir(parents=True, exist_ok=True)

    rel_json = OUTPUTS_DIR_Q2 / "reliable_links.json"
    if not rel_json.exists():
        raise FileNotFoundError(rel_json)
    if not Q1_REL_CSV.exists():
        OUT_HTML.write_text(
            "<!DOCTYPE html><meta charset='utf-8'><title>Missing Q1 CSV</title>"
            "<body style='font-family:sans-serif;padding:2rem'>缺少 outputs/q1/q1_relationship_patterns.csv</body>",
            encoding="utf-8",
        )
        print(f"Stub -> {OUT_HTML}")
        return

    reliable_links: list[dict] = json.loads(rel_json.read_text(encoding="utf-8"))
    pred_nodes: set[str] = set()
    raw_preds: list[tuple[str, str, str]] = []
    for lk in reliable_links:
        s, t = lk.get("source"), lk.get("target")
        if not s or not t or s == t:
            continue
        s, t = str(s), str(t)
        pred_nodes.add(s)
        pred_nodes.add(t)
        raw_preds.append((s, t, str(lk.get("generated_by") or "unknown")))

    df = pd.read_csv(Q1_REL_CSV)
    ca = df["company_a"].astype(str)
    cb = df["company_b"].astype(str)
    mask = ca.isin(pred_nodes) & cb.isin(pred_nodes) & (ca != cb)
    sub = df.loc[mask]

    best: dict[tuple[str, str], tuple[float, str]] = {}
    for row in sub.itertuples(index=False):
        a = str(row.company_a)
        b = str(row.company_b)
        key = (a, b)
        conf = float(row.confidence) if pd.notna(row.confidence) else 0.0
        pat = str(row.relationship_pattern)
        if key not in best or conf > best[key][0]:
            best[key] = (conf, pat)

    q1_pairs: list[tuple[str, str, float, str]] = [
        (a, b, conf, pat) for (a, b), (conf, pat) in best.items()
    ]

    companies = _pick_nodes(pred_nodes, q1_pairs)
    idx = {c: i for i, c in enumerate(companies)}
    n = len(companies)

    hover_tm = _temporal_hover_map(set(companies), Q1_TEMPORAL_JSON)

    left_labels = [_short_label(c) for c in companies]
    node_labels = left_labels + left_labels[:]
    node_colors = ["#cbd5e1"] * n + ["#e2e8f0"] * n

    def lp(u: str, v: str) -> tuple[int, int]:
        return idx[u], n + idx[v]

    # Q1：有向对聚合
    q1_agg: defaultdict[tuple[str, str], tuple[float, str]] = defaultdict(lambda: (0.0, ""))
    for a, b, conf, pat in q1_pairs:
        if a not in idx or b not in idx:
            continue
        w_prev, _ = q1_agg[(a, b)]
        q1_agg[(a, b)] = (w_prev + 2.0 + conf * 24.0, pat)

    pred_agg: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for u, v, b in raw_preds:
        if u not in idx or v not in idx:
            continue
        pred_agg[(u, v, b)] += 1

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    colors: list[str] = []
    hovers: list[str] = []

    for (a, b), (w_sum, pat) in sorted(q1_agg.items(), key=lambda x: -x[1][0]):
        su, tu = lp(a, b)
        vw = max(2, int(round(w_sum)))
        sources.append(su)
        targets.append(tu)
        values.append(vw)
        colors.append(BASE_LINK_COLOR)
        ta = hover_tm.get(a, "?")
        tb = hover_tm.get(b, "?")
        hovers.append(
            f"<b>Q1</b> {pat}<br>"
            f"{_short_label(a, 48)} <small>({ta})</small><br>→ "
            f"{_short_label(b, 48)} <small>({tb})</small><br>weighted ~ {vw}"
        )

    n_q1_links = len(sources)
    scale_p = max(6.0, 4.0 * math.sqrt(max(pred_agg.values()) if pred_agg else 1))

    for (u, v, bundle), cnt in sorted(pred_agg.items(), key=lambda x: -x[1]):
        su, tu = lp(u, v)
        vw = max(8, int(round(scale_p * math.sqrt(float(cnt)))))
        sources.append(su)
        targets.append(tu)
        values.append(vw)
        colors.append(BUNDLE_EDGE_COLORS.get(bundle, "rgba(244, 63, 94, 0.88)"))
        tu_ = hover_tm.get(u, "?")
        tv = hover_tm.get(v, "?")
        hovers.append(
            f"<b>Q2</b> {bundle}<br>"
            f"{_short_label(u, 48)} <small>({tu_})</small><br>→ "
            f"{_short_label(v, 48)} <small>({tv})</small><br>count: {cnt}"
        )

    if not sources:
        OUT_HTML.write_text(
            "<!DOCTYPE html><meta charset='utf-8'><title>Empty Sankey</title>"
            "<body style='font-family:sans-serif;padding:2rem'>无可用边。</body>",
            encoding="utf-8",
        )
        print(f"Empty stub -> {OUT_HTML}")
        return

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=10,
                    thickness=12,
                    line=dict(color="#475569", width=0.4),
                    label=node_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=colors,
                    customdata=hovers,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )
        ]
    )

    n_pred_drawn = len(sources) - n_q1_links
    fig.update_layout(
        title=dict(
            text=(
                "公司对桑基（不按时序类型分组）· Q1 灰底聚合 + Q2 彩色预测<br>"
                f"<sup>公司槽位 {n}（双向分列）| Q1 边 {n_q1_links} | Q2 聚合带 {n_pred_drawn} "
                f"<small>(原始预测 {len(raw_preds)})</small></sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        font=dict(size=11, family="Segoe UI, PingFang SC, Microsoft YaHei, sans-serif"),
        template="plotly_white",
        height=max(480, min(900, 260 + n * 8)),
        width=min(1120, 600 + n * 5),
        margin=dict(l=36, r=36, t=92, b=56),
        annotations=[
            dict(
                text="灰：Q1 relationship_patterns（同向公司对合并）；彩色：Q2（按公司与 bundle 合并）",
                xref="paper",
                yref="paper",
                x=0.02,
                y=-0.06,
                showarrow=False,
                font=dict(size=11, color="#64748b"),
            ),
            dict(
                text="hover 中小字为 Q1 temporal_pattern，仅作参考，不参与布局聚合",
                xref="paper",
                yref="paper",
                x=0.02,
                y=-0.095,
                showarrow=False,
                font=dict(size=10, color="#94a3b8"),
            ),
        ],
    )

    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn", config={"responsive": True})
    print(f"Wrote {OUT_HTML} ({n} company slots, {len(sources)} link bundles)")


if __name__ == "__main__":
    main()
