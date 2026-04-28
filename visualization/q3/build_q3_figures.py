#!/usr/bin/env python3
"""Q3 图表：加入可靠链接后的新模式与异常。"""

from __future__ import annotations

import sys
from pathlib import Path

_vis_root = str(Path(__file__).resolve().parents[1])
if _vis_root not in sys.path:
    sys.path.insert(0, _vis_root)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from figures2d_common import FIG_DIR, build_company_view, ensure_dirs, load_data

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def build_q3(company_view: pd.DataFrame, reliable_links: pd.DataFrame) -> None:
    q3_dir = FIG_DIR / "q3"

    cols = [
        "predicted_link_count",
        "filled_gap_months_count",
        "extended_months_count",
        "burst_months_count",
        "new_partner_count",
        "new_hscode_count",
        "suspicious_revival_score",
    ]
    top = company_view.head(30).set_index("company")[cols]
    if not top.empty:
        scaled = (top - top.min()) / (top.max() - top.min() + 1e-9)
        plt.figure(figsize=(11, 8))
        sns.heatmap(scaled, cmap="magma", linewidths=0.2)
        plt.title("Q3-1 New Pattern/Anomaly Signals After Adding Reliable Links")
        plt.xlabel("Signal")
        plt.ylabel("Company")
        plt.tight_layout()
        plt.savefig(q3_dir / "q3_01_new_signal_heatmap.png", dpi=220)
        plt.close()

    links = reliable_links.copy()
    links["month"] = links["arrivaldate"].str.slice(0, 7)
    monthly = links.groupby(["month", "generated_by"]).size().reset_index(name="count")
    area = px.area(
        monthly,
        x="month",
        y="count",
        color="generated_by",
        title="Q3-2 Reliable Link Additions by Month and Bundle",
    )
    area.update_layout(xaxis_title="Month", yaxis_title="Added links")
    area.write_html(q3_dir / "q3_02_reliable_monthly_stream.html", include_plotlyjs=True)

    stage = company_view.copy()
    stage["pattern"] = np.select(
        [
            stage["extended_months_count"] > 0,
            stage["filled_gap_months_count"] > 0,
            stage["burst_months_count"] > 0,
        ],
        ["extended", "gap_filled", "burst"],
        default="other",
    )
    stage["confidence"] = stage["iuu_confidence"].astype(str)
    flow = stage.groupby(["pattern", "confidence"]).size().reset_index(name="value")
    node_names = pd.Index(pd.unique(flow[["pattern", "confidence"]].values.ravel("K")))
    node_id = {n: i for i, n in enumerate(node_names)}
    sankey = go.Figure(
        go.Sankey(
            node=dict(label=[str(x) for x in node_names], pad=14, thickness=14),
            link=dict(
                source=flow["pattern"].map(node_id),
                target=flow["confidence"].map(node_id),
                value=flow["value"],
            ),
        )
    )
    sankey.update_layout(title="Q3-3 Pattern Transition to Risk Confidence After Reliable Links")
    sankey.write_html(q3_dir / "q3_03_pattern_transition_sankey.html", include_plotlyjs=True)


def main() -> None:
    ensure_dirs()
    _, company_df, delta_df, reliable_links = load_data()
    company_view = build_company_view(company_df, delta_df)
    build_q3(company_view, reliable_links)
    print("Q3 figures done.")


if __name__ == "__main__":
    main()
