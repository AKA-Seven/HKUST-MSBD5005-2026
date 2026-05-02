#!/usr/bin/env python3
"""Q2 bundle reliability chart — fish emoji markers, hover bubble + metrics.

Scatter uses 🐟 sized by link_count (color = reliability tier). No always-on labels;
hover shows 🫧 summary plus full metrics.

Output: visualization/figures_2d/q2/q2_bundle_reliability_bubble.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_vis_root = Path(__file__).resolve().parents[1]
if str(_vis_root) not in sys.path:
    sys.path.insert(0, str(_vis_root))

from figures2d_common import FIG_DIR, OUTPUTS_DIR_Q2, ensure_dirs

FIG_Q2 = FIG_DIR / "q2"
OUT_HTML = FIG_Q2 / "q2_bundle_reliability_bubble.html"

CSV_PATH = OUTPUTS_DIR_Q2 / "bundle_reliability.csv"

X_THRESHOLD = 0.75
Y_THRESHOLD = 0.80


LABEL_DISPLAY = {
    "reliable": "Reliable",
    "suspicious": "Suspicious",
    "reject": "Reject",
}


def _upper_right_focus_ranges(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Tight axis ranges emphasizing the upper-right (high temporal × high model quality) band."""
    xs = df["temporal_consistency_ratio"].astype(float)
    ys = df["ml_validation_auc"].astype(float)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    x_span = max(xmax - xmin, 0.06)
    pad_x_lo = max(0.014, 0.048 * x_span)
    pad_x_hi = max(0.008, 0.028 * x_span)
    xr_lo = max(0.0, xmin - pad_x_lo)
    xr_hi = min(1.02, xmax + pad_x_hi)

    xr_hi = max(xr_hi, X_THRESHOLD + 0.20)
    if xmax >= X_THRESHOLD:
        xr_lo = min(xr_lo, X_THRESHOLD - 0.065)

    y_span = ymax - ymin
    if y_span < 1e-9:
        mid = ymax
        yr_lo = max(Y_THRESHOLD - 0.055, mid - 0.032)
        yr_hi = min(1.008, mid + 0.016)
    else:
        pad_y = max(0.008, 0.11 * y_span)
        yr_lo = max(Y_THRESHOLD - 0.075, ymin - pad_y)
        yr_hi = min(1.02, ymax + pad_y * 1.05)

    yr_hi = max(yr_hi, Y_THRESHOLD + 0.048)
    yr_lo = min(yr_lo, Y_THRESHOLD - 0.025)
    return xr_lo, xr_hi, yr_lo, yr_hi


LABEL_FILL = {
    "reliable": "#165DFF",   # deep blue — trusted
    "suspicious": "#F59E0B", # amber — distinct from reliable/reject
    "reject": "#64748B",     # slate — de-emphasized / excluded
}

def _marker_sizes(link_counts: pd.Series) -> np.ndarray:
    lc = link_counts.astype(float)
    mn, mx = float(lc.min()), float(lc.max())
    if np.isclose(mx, mn):
        t = np.ones(len(lc)) * 0.5
    else:
        t = (lc - mn) / (mx - mn)
    # Diameter range tuned for ~11 points, print-friendly
    return 22.0 + t * 38.0


EMOJI_FONT = "Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji, sans-serif"


def main() -> None:
    ensure_dirs()
    FIG_Q2.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["repetition_type"] = df["repetition_type"].astype(str).str.strip()

    for col in ("temporal_consistency_ratio", "ml_validation_auc", "score", "link_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["marker_size"] = _marker_sizes(df["link_count"])

    auc_unique = df["ml_validation_auc"].dropna().unique()
    auc_note = ""
    if len(auc_unique) == 1:
        auc_note = (
            f" · Shared global link-model validation AUC = {float(auc_unique[0]):.4f}"
            " (all bundles); markers align on Y by construction."
        )

    hover_cols = [
        "bundle",
        "score",
        "link_count",
        "seen_pair_ratio",
        "valid_hscode_ratio",
        "endpoint_in_base_ratio",
        "temporal_consistency_ratio",
        "ml_validation_auc",
        "ml_link_probability",
        "fish_hscode_ratio",
        "outside_date_ratio",
        "exact_duplicate_ratio",
        "unique_pair_ratio",
        "bundle_type",
        "repetition_type",
    ]
    for c in hover_cols:
        if c not in df.columns:
            df[c] = np.nan

    xr_lo, xr_hi, yr_lo, yr_hi = _upper_right_focus_ranges(df)
    xw = xr_hi - xr_lo
    yh = yr_hi - yr_lo

    fig = go.Figure()

    label_order = ("reliable", "suspicious", "reject")
    for lab in label_order:
        sub = df[df["label"] == lab]
        if sub.empty:
            continue

        custom_cols = (
            "bundle",
            "score",
            "link_count",
            "seen_pair_ratio",
            "valid_hscode_ratio",
            "endpoint_in_base_ratio",
            "temporal_consistency_ratio",
            "ml_validation_auc",
            "ml_link_probability",
            "fish_hscode_ratio",
            "outside_date_ratio",
            "exact_duplicate_ratio",
            "unique_pair_ratio",
            "bundle_type",
            "repetition_type",
        )
        custom = sub[list(custom_cols)].to_numpy()

        tier_name = LABEL_DISPLAY.get(lab, lab)
        tier_color = LABEL_FILL[lab]

        fig.add_trace(
            go.Scatter(
                x=sub["temporal_consistency_ratio"],
                y=sub["ml_validation_auc"],
                mode="markers+text",
                name=LABEL_DISPLAY.get(lab, lab),
                legendgroup=lab,
                showlegend=False,
                marker=dict(
                    size=sub["marker_size"].astype(float).tolist(),
                    color="rgba(0,0,0,0)",
                    line=dict(width=0),
                    sizemode="diameter",
                ),
                text=["🐟"] * len(sub),
                textposition="middle center",
                textfont=dict(
                    size=sub["marker_size"].astype(float).tolist(),
                    family=EMOJI_FONT,
                    color=tier_color,
                ),
                customdata=custom,
                hovertemplate=(
                    "<span style='font-size:20px'>🫧</span> "
                    "<b>%{customdata[0]}</b> · score %{customdata[1]:.2f}<br>"
                    f"<span style='opacity:0.85'>Tier: <b>{tier_name}</b></span><br><br>"
                    "<b>Details</b><br>"
                    "Link count: %{customdata[2]:.0f}<br>"
                    "Seen-pair ratio: %{customdata[3]:.3f}<br>"
                    "Valid HS-code ratio: %{customdata[4]:.3f}<br>"
                    "Endpoint-in-base ratio: %{customdata[5]:.3f}<br>"
                    "Temporal consistency: %{customdata[6]:.3f}<br>"
                    "ML validation AUC: %{customdata[7]:.4f}<br>"
                    "Mean link probability: %{customdata[8]:.4f}<br>"
                    "Seafood HS ratio: %{customdata[9]:.3f}<br>"
                    "Outside-date ratio: %{customdata[10]:.3f}<br>"
                    "Exact-duplicate ratio: %{customdata[11]:.3f}<br>"
                    "Unique-pair ratio: %{customdata[12]:.3f}<br>"
                    "Bundle type: %{customdata[13]}<br>"
                    "Repetition type: %{customdata[14]}<br>"
                    "<extra></extra>"
                ),
                hoverlabel=dict(bgcolor="rgba(248,250,252,0.97)", bordercolor=tier_color),
            )
        )

    for lab in label_order:
        if df[df["label"] == lab].empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=LABEL_DISPLAY.get(lab, lab),
                legendgroup=lab,
                showlegend=True,
                hoverinfo="skip",
                marker=dict(
                    size=16,
                    color=LABEL_FILL[lab],
                    line=dict(
                        width=2.5 if lab == "reliable" else 1.2,
                        color="#FFFFFF"
                        if lab == "reliable"
                        else ("#1e293b" if lab == "suspicious" else "#94a3b8"),
                    ),
                ),
            )
        )

    fig.add_shape(
        type="line",
        x0=X_THRESHOLD,
        x1=X_THRESHOLD,
        y0=yr_lo,
        y1=yr_hi,
        line=dict(color="#CBD5E1", width=1.5, dash="dash"),
        layer="below",
    )
    if yr_lo - 1e-9 <= Y_THRESHOLD <= yr_hi + 1e-9:
        fig.add_shape(
            type="line",
            x0=xr_lo,
            x1=xr_hi,
            y0=Y_THRESHOLD,
            y1=Y_THRESHOLD,
            line=dict(color="#CBD5E1", width=1.5, dash="dash"),
            layer="below",
        )

    quadrant_style = dict(
        font=dict(size=11, color="#64748B", family="Segoe UI, Source Sans 3, sans-serif"),
        showarrow=False,
    )
    fig.add_annotation(
        x=xr_hi - 0.02 * xw,
        y=yr_hi - 0.035 * yh,
        xref="x",
        yref="y",
        xanchor="right",
        yanchor="top",
        text="Preferred",
        **quadrant_style,
    )
    fig.add_annotation(
        x=xr_lo + 0.02 * xw,
        y=yr_hi - 0.035 * yh,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="top",
        text="Strong model,<br>temporal risk",
        **quadrant_style,
    )
    fig.add_annotation(
        x=xr_hi - 0.02 * xw,
        y=yr_lo + 0.035 * yh,
        xref="x",
        yref="y",
        xanchor="right",
        yanchor="bottom",
        text="Temporal OK,<br>weaker signal",
        **quadrant_style,
    )
    fig.add_annotation(
        x=xr_lo + 0.02 * xw,
        y=yr_lo + 0.035 * yh,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="bottom",
        text="Low quality<br>(exclude)",
        **quadrant_style,
    )

    dtick_x = max(0.04, round(xw / 6.0, 3))
    dtick_y = max(0.012, round(yh / 6.0, 4))

    fig.update_xaxes(
        title=dict(text="Temporal consistency ratio", font=dict(size=12, color="#475569")),
        range=[xr_lo, xr_hi],
        dtick=dtick_x,
        tickformat=".2f",
        gridcolor="#E8EDF5",
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        mirror=True,
    )
    fig.update_yaxes(
        title=dict(text="ML validation AUC", font=dict(size=12, color="#475569")),
        range=[yr_lo, yr_hi],
        dtick=dtick_y,
        tickformat=".3f",
        gridcolor="#E8EDF5",
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        mirror=True,
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b style='color:#0B3D91;font-size:18px'>Q2 Bundle reliability — multi-metric overview</b>"
                "<br><sup style='color:#64748B;font-size:12px'>"
                "Fish color = reliability tier · Size ∝ link count · Hover for 🫧 summary + metrics"
                f"{auc_note}<br>"
                "View zoomed on the upper-right quality band.</sup>"
            ),
            x=0.5,
            xanchor="center",
            pad=dict(t=12),
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F7FAFF",
        font=dict(family="Segoe UI, Source Sans 3, sans-serif", color="#334155"),
        legend=dict(
            title=dict(text="Reliability", font=dict(size=12)),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            traceorder="normal",
            itemsizing="constant",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=72, r=220, t=120, b=64),
        width=1120,
        height=720,
        hovermode="closest",
    )

    fig.write_html(
        str(OUT_HTML),
        include_plotlyjs="cdn",
        config={
            "responsive": True,
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
