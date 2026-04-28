#!/usr/bin/env python3
"""Q2 图表：预测链接可靠性评估。"""

from __future__ import annotations

import sys
from pathlib import Path

_vis_root = str(Path(__file__).resolve().parents[1])
if _vis_root not in sys.path:
    sys.path.insert(0, _vis_root)

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

from figures2d_common import FIG_DIR, ensure_dirs, load_data

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def build_q2(bundle_df: pd.DataFrame) -> None:
    q2_dir = FIG_DIR / "q2"

    plt.figure(figsize=(10.5, 6.5))
    palette = {"reliable": "#47d18c", "suspicious": "#e4ba52", "reject": "#ef7168"}
    sns.scatterplot(
        data=bundle_df,
        x="score",
        y="ml_link_probability",
        hue="label",
        size="link_count",
        sizes=(100, 900),
        palette=palette,
        alpha=0.85,
    )
    for _, row in bundle_df.iterrows():
        plt.text(row["score"] + 0.3, row["ml_link_probability"] + 0.002, row["bundle"], fontsize=8)
    plt.title("Q2-1 Bundle Reliability Bubble (Rule Score vs ML Probability)")
    plt.xlabel("Reliability Score")
    plt.ylabel("ML Link Probability")
    plt.tight_layout()
    plt.savefig(q2_dir / "q2_01_bundle_bubble.png", dpi=220)
    plt.close()

    pc = px.parallel_coordinates(
        bundle_df.sort_values("score", ascending=False),
        color="score",
        dimensions=[
            "seen_pair_ratio",
            "valid_hscode_ratio",
            "outside_date_ratio",
            "physical_field_ratio",
            "unique_pair_ratio",
            "ml_link_probability",
            "score",
        ],
        color_continuous_scale=px.colors.sequential.Plasma,
        title="Q2-2 Bundle Metric Comparison (Parallel Coordinates)",
    )
    pc.write_html(q2_dir / "q2_02_bundle_parallel_coordinates.html", include_plotlyjs=True)

    radar_cols = [
        "seen_pair_ratio",
        "valid_hscode_ratio",
        "physical_field_ratio",
        "unique_pair_ratio",
        "ml_link_probability",
        "fish_hscode_ratio",
    ]
    radar_ref = pd.concat([bundle_df.nlargest(5, "score"), bundle_df.nsmallest(1, "score")], ignore_index=True)
    fig = go.Figure()
    for _, row in radar_ref.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[float(row[c]) for c in radar_cols] + [float(row[radar_cols[0]])],
                theta=radar_cols + [radar_cols[0]],
                fill="toself",
                name=f"{row['bundle']} ({row['label']})",
                opacity=0.35,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        title="Q2-3 Top Bundle Reliability Radar",
    )
    fig.write_html(q2_dir / "q2_03_bundle_radar.html", include_plotlyjs=True)


def main() -> None:
    ensure_dirs()
    bundle_df, _, _, _ = load_data()
    build_q2(bundle_df)
    print("Q2 figures done.")


if __name__ == "__main__":
    main()
