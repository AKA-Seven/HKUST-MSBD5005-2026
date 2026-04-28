#!/usr/bin/env python3
"""Q4 图表：疑似 IUU 公司识别与证据。"""

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
import plotly.graph_objects as go
import seaborn as sns

from figures2d_common import FIG_DIR, build_company_view, ensure_dirs, load_data

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def build_q4(company_view: pd.DataFrame, reliable_links: pd.DataFrame) -> None:
    q4_dir = FIG_DIR / "q4"
    top = company_view.head(20).copy()

    plt.figure(figsize=(10, 7))
    top_sorted = top.sort_values("iuu_score")
    plt.hlines(y=top_sorted["company"], xmin=0, xmax=top_sorted["iuu_score"], color="#6a8ca8", alpha=0.65)
    plt.scatter(top_sorted["iuu_score"], top_sorted["company"], s=80, c=top_sorted["anomaly_score"], cmap="Reds")
    plt.title("Q4-1 Top IUU Candidate Ranking (Lollipop)")
    plt.xlabel("IUU Composite Score")
    plt.ylabel("Company")
    plt.tight_layout()
    plt.savefig(q4_dir / "q4_01_iuu_lollipop.png", dpi=220)
    plt.close()

    radar_cols = [
        "anomaly_score",
        "suspicious_revival_score",
        "predicted_link_count",
        "new_partner_count",
        "new_hscode_count",
        "nearest_similarity",
    ]
    radar_df = top.head(8).copy()
    for col in radar_cols:
        low, high = radar_df[col].min(), radar_df[col].max()
        radar_df[f"{col}_norm"] = 0 if np.isclose(low, high) else (radar_df[col] - low) / (high - low)

    fig = go.Figure()
    theta = radar_cols + [radar_cols[0]]
    for _, row in radar_df.iterrows():
        r = [float(row[f"{col}_norm"]) for col in radar_cols]
        fig.add_trace(
            go.Scatterpolar(
                r=r + [r[0]],
                theta=theta,
                fill="toself",
                opacity=0.28,
                name=f"{row['company']} ({row['iuu_confidence']})",
            )
        )
    fig.update_layout(
        title="Q4-2 Top Candidate Evidence Radar",
        polar=dict(radialaxis=dict(range=[0, 1])),
    )
    fig.write_html(q4_dir / "q4_02_candidate_radar.html", include_plotlyjs=True)

    top_ids = set(top.head(10)["company"])
    links = reliable_links[(reliable_links["source"].isin(top_ids)) | (reliable_links["target"].isin(top_ids))].copy()
    pair = links.groupby(["source", "target"]).size().reset_index(name="count").sort_values("count", ascending=False).head(60)
    node_names = pd.Index(pd.unique(pair[["source", "target"]].values.ravel("K")))
    node_id = {n: i for i, n in enumerate(node_names)}
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=[str(x) for x in node_names],
                pad=15,
                thickness=14,
                color=["#ef7168" if name in top_ids else "#6ba9d1" for name in node_names],
            ),
            link=dict(
                source=pair["source"].map(node_id),
                target=pair["target"].map(node_id),
                value=pair["count"],
                color="rgba(101,169,215,0.36)",
            ),
        )
    )
    fig.update_layout(title="Q4-3 Top Candidate Predicted Relationship Flow")
    fig.write_html(q4_dir / "q4_03_candidate_relationship_flow.html", include_plotlyjs=True)

    export_cols = [
        "company",
        "iuu_score",
        "iuu_confidence",
        "anomaly_score",
        "suspicious_revival_score",
        "predicted_link_count",
        "new_partner_count",
        "new_hscode_count",
        "nearest_company",
        "nearest_similarity",
    ]
    top[export_cols].to_csv(q4_dir / "q4_top_candidates.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    _, company_df, delta_df, reliable_links = load_data()
    company_view = build_company_view(company_df, delta_df)
    build_q4(company_view, reliable_links)
    print("Q4 figures done.")


if __name__ == "__main__":
    main()
