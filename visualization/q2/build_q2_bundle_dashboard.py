#!/usr/bin/env python3
"""Q2 bundle reliability dashboard: Fig 1 matrix, Fig 2 HX radar, Fig 3 3D embed, Fig 4 heatmap.

Writes final_sketches/q2.html (primary) and mirrors to visualization/figures_2d/q2/q2_bundle_dashboard.html.
Generate the 3D asset first: python visualization/q2/build_q2_dandelion_3d.py
"""

from __future__ import annotations

import html
import colorsys
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_vis_root = Path(__file__).resolve().parents[1]
_repo_root = Path(__file__).resolve().parents[2]
if str(_vis_root) not in sys.path:
    sys.path.insert(0, str(_vis_root))

from figures2d_common import FIG_DIR, OUTPUTS_DIR_Q1, OUTPUTS_DIR_Q2, ensure_dirs

ROOT = _repo_root
FIG_Q2 = FIG_DIR / "q2"
OUT_HTML_MIRROR = FIG_Q2 / "q2_bundle_dashboard.html"
OUT_HTML_FINAL = ROOT / "final_sketches" / "q2.html"

REL_PATH_3D_FROM_FINAL = "../visualization/figures_3d/q2/q2_dandelion_graph.html"
REL_PATH_3D_FROM_MIRROR = "../../figures_3d/q2/q2_dandelion_graph.html"

CSV_PATH = OUTPUTS_DIR_Q2 / "bundle_reliability.csv"
REL_PATH = OUTPUTS_DIR_Q2 / "reliable_links.json"
Q1_TEMP_PATH = OUTPUTS_DIR_Q1 / "q1_temporal_patterns.json"

BT_ORDER = ["novel_discovery", "mixed", "relationship_ext", "gap_filler"]
REP_ORDER = ["highly_repeated", "moderately_repeated", "diverse"]

MONTHS_2034 = [f"2034-{m:02d}" for m in range(1, 13)]

# Six axes for HX radar (hyper-dimensional metric profile on one HTML radar).
HX_RADAR_AXES: list[tuple[str, str]] = [
    ("Endpoint cover", "endpoint_in_base_ratio"),
    ("Temporal fit", "temporal_consistency_ratio"),
    ("ML p90", "ml_link_probability_p90"),
    ("Valid HS", "valid_hscode_ratio"),
    ("Seen pairs", "seen_pair_ratio"),
    ("Physical fields", "physical_field_ratio"),
]


def _norm_hs(h) -> str:
    if h is None:
        return ""
    try:
        return str(int(float(h)))
    except (ValueError, TypeError):
        return "".join(c for c in str(h) if c.isdigit())


def load_bundle_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["bundle_type"] = df["bundle_type"].astype(str).str.strip()
    df["repetition_type"] = df["repetition_type"].astype(str).str.strip()
    num_cols = [
        "score",
        "link_count",
        "node_count",
        "endpoint_in_base_ratio",
        "node_in_base_ratio",
        "seen_pair_ratio",
        "exact_duplicate_ratio",
        "valid_hscode_ratio",
        "fish_hscode_ratio",
        "outside_date_ratio",
        "physical_field_ratio",
        "unique_pair_ratio",
        "max_pair_repeat",
        "bad_country_count",
        "temporal_consistency_ratio",
        "ml_link_probability",
        "ml_link_probability_p90",
        "ml_validation_auc",
        "ml_training_pairs",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fmt_row_hover(r: pd.Series) -> str:
    """Rich hover line for one bundle row (bundle_reliability.csv)."""

    def fv(col: str, fmt: str = "{:.4f}") -> str:
        if col not in r.index or pd.isna(r[col]):
            return ""
        try:
            v = float(r[col])
            if pd.isna(v):
                return ""
            if fmt == "{:.4f}" and abs(v - round(v)) < 1e-9:
                return f"{col}: {int(round(v))}"
            return f"{col}: {fmt.format(v)}"
        except (TypeError, ValueError):
            return f"{col}: {r[col]}" if pd.notna(r[col]) else ""

    lines = [
        f"<b>{html.escape(str(r['bundle']))}</b> · {html.escape(str(r['label']))}",
        f"score: {float(r['score']):.2f}",
        fv("link_count", "{:.0f}"),
        fv("node_count", "{:.0f}"),
        "bundle_type: " + html.escape(str(r["bundle_type"])),
        "repetition_type: " + html.escape(str(r["repetition_type"])),
        fv("endpoint_in_base_ratio"),
        fv("temporal_consistency_ratio"),
        fv("ml_link_probability_p90"),
        fv("valid_hscode_ratio"),
        fv("seen_pair_ratio"),
        fv("fish_hscode_ratio"),
        fv("outside_date_ratio"),
        fv("node_in_base_ratio"),
        fv("physical_field_ratio"),
        fv("unique_pair_ratio"),
        fv("exact_duplicate_ratio"),
        fv("max_pair_repeat", "{:.0f}"),
        fv("bad_country_count", "{:.0f}"),
        fv("ml_link_probability"),
        fv("ml_validation_auc"),
        fv("ml_training_pairs", "{:.0f}"),
    ]
    return "<br>".join(line for line in lines if line)


def load_links() -> list[dict]:
    if not REL_PATH.exists():
        return []
    return json.loads(REL_PATH.read_text(encoding="utf-8"))


def load_q1_monthly() -> dict[str, dict[str, int]]:
    if not Q1_TEMP_PATH.exists():
        return {}
    data = json.loads(Q1_TEMP_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = {}
    for row in data:
        c = str(row.get("company", ""))
        mc = row.get("monthly_counts") or {}
        out[c] = {str(k): int(v) for k, v in mc.items()}
    return out


def build_fig1_matrix(df: pd.DataFrame, rng: np.random.Generator) -> tuple[go.Figure, list[str]]:
    """Bubble matrix only (bundle_type × repetition_type)."""
    smin, smax = float(df["score"].min()), float(df["score"].max())

    def bx(cat: str) -> float:
        return float(BT_ORDER.index(cat)) if cat in BT_ORDER else 1.5

    def by(cat: str) -> float:
        return float(REP_ORDER.index(cat)) if cat in REP_ORDER else 1.0

    df = df.copy()
    df["_bx"] = df["bundle_type"].map(bx)
    df["_by"] = df["repetition_type"].map(by)
    jitter_x = rng.uniform(-0.22, 0.22, len(df))
    jitter_y = rng.uniform(-0.18, 0.18, len(df))
    df["_jx"] = df["_bx"] + jitter_x
    df["_jy"] = df["_by"] + jitter_y

    lc = df["link_count"].astype(float)
    lc_rng = max(1e-9, float(lc.max() - lc.min()))
    sizes = 5 + (lc - lc.min()) / lc_rng * 22

    bundles_ordered = list(df.sort_values(["score"], ascending=False)["bundle"].astype(str))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["_jx"],
                y=df["_jy"],
                mode="markers",
                name="bundles",
                showlegend=False,
                marker=dict(
                    size=sizes,
                    color=df["score"],
                    colorscale="RdYlGn",
                    cmin=max(40.0, smin - 5),
                    cmax=min(100.0, smax + 5),
                    line=dict(width=0),
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Score"),
                        x=1.02,
                        len=0.55,
                        outlinewidth=0,
                        borderwidth=0,
                        thickness=12,
                    ),
                ),
                hovertext=[_fmt_row_hover(r) for _, r in df.iterrows()],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        ]
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(BT_ORDER))),
        ticktext=BT_ORDER,
        range=[-0.55, len(BT_ORDER) - 0.45],
        title="bundle_type",
        showgrid=True,
        gridcolor="rgba(19,34,56,0.06)",
        zeroline=False,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(REP_ORDER))),
        ticktext=REP_ORDER,
        range=[-0.55, len(REP_ORDER) - 0.45],
        title="repetition_type",
        showgrid=True,
        gridcolor="rgba(19,34,56,0.06)",
        zeroline=False,
    )

    fig.update_layout(
        title=dict(
            text="Fig 1 · Reliability bubble matrix<br><sup>Compact bubbles · hover for bundle name & full CSV metrics</sup>",
            font=dict(size=13),
        ),
        height=520,
        paper_bgcolor="rgba(255,251,243,0.35)",
        plot_bgcolor="#fffdfa",
        margin=dict(l=60, r=100, t=72, b=52),
        font=dict(size=11),
        legend=dict(borderwidth=0),
    )
    return fig, bundles_ordered


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(71,85,105,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _ratio_0_1(row: object, col: str) -> float:
    try:
        v = float(getattr(row, col))
    except (TypeError, ValueError, AttributeError):
        return 0.0
    if np.isnan(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _distinct_bundle_hex(i: int) -> str:
    """Golden-ratio hue spacing for maximum perceived separation."""
    h = (i * 0.618033988749895) % 1.0
    r_f, g_f, b_f = colorsys.hls_to_rgb(h, 0.46, 0.82)
    return f"#{int(r_f * 255):02x}{int(g_f * 255):02x}{int(b_f * 255):02x}"


def build_fig2_hx_radar(df: pd.DataFrame) -> go.Figure:
    """HX radar: six ratios per bundle — compact bottom-right dropdown: All or one bundle."""
    theta_labels = [a[0] for a in HX_RADAR_AXES]
    theta_closed = theta_labels + [theta_labels[0]]

    ordered = df.sort_values("score", ascending=False).reset_index(drop=True)
    n = len(ordered)

    fig = go.Figure()
    for i in range(n):
        row = ordered.iloc[i]
        bname = str(row["bundle"])
        stroke = _distinct_bundle_hex(i)
        fill_rgba = _hex_to_rgba(stroke, 0.26)

        hover_lines: list[str] = []
        for lbl, col in HX_RADAR_AXES:
            u = _ratio_0_1(row, col)
            pct = 100.0 * u
            raw_txt = ""
            try:
                raw_v = row[col]
                if pd.notna(raw_v):
                    raw_txt = f"<br>Raw: {float(raw_v):.4f}"
            except (TypeError, ValueError, KeyError):
                pass
            hover_lines.append(
                f"<b>{html.escape(bname)}</b> · {html.escape(str(row['label']).strip().lower())}<br>"
                f"<b>{html.escape(lbl)}</b><br>Scale 0–100: {pct:.1f}{raw_txt}"
            )

        r_closed = [100.0 * _ratio_0_1(row, col) for lbl, col in HX_RADAR_AXES]
        r_closed = r_closed + [r_closed[0]]
        hover_closed = hover_lines + [hover_lines[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta_closed,
                mode="lines",
                name=bname,
                line=dict(color=stroke, width=2),
                fill="toself",
                fillcolor=fill_rgba,
                opacity=0.82,
                hovertemplate="%{text}<extra></extra>",
                text=hover_closed,
                showlegend=False,
                legendgroup=bname,
                visible=True,
            )
        )

    vis_all = [True] * n
    menu_buttons: list[dict] = [
        dict(label="All", method="update", args=[{"visible": vis_all}, {}]),
    ]
    for i in range(n):
        bname = str(ordered.iloc[i]["bundle"])
        label = bname if len(bname) <= 16 else bname[:13] + "…"
        vis_one = [j == i for j in range(n)]
        menu_buttons.append(
            dict(label=label, method="update", args=[{"visible": vis_one}, {}]),
        )

    fig.update_layout(
        title=dict(
            text=(
                "Fig 2 · HX radar (bundle metric profile)<br>"
                "<sup>Default: all overlaid · compact menu bottom-right: All vs single bundle</sup>"
            ),
            font=dict(size=13),
        ),
        height=480,
        paper_bgcolor="rgba(255,251,243,0.5)",
        margin=dict(l=48, r=72, t=82, b=70),
        polar=dict(
            bgcolor="rgba(255,255,255,0.25)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                angle=90,
                tickfont=dict(size=10),
                gridcolor="rgba(232,223,208,0.55)",
                linecolor="rgba(232,223,208,0.35)",
                showline=False,
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
                gridcolor="rgba(19,34,56,0.08)",
                linecolor="rgba(19,34,56,0.08)",
            ),
        ),
        showlegend=False,
        updatemenus=[
            dict(
                type="dropdown",
                direction="up",
                showactive=True,
                active=0,
                x=1.024,
                xanchor="right",
                y=-0.068,
                yanchor="bottom",
                bgcolor="rgba(255,251,243,0.96)",
                bordercolor="rgba(19,34,56,0.14)",
                borderwidth=1,
                pad=dict(r=0, t=0, b=0),
                font=dict(size=8, color="#132238"),
                buttons=menu_buttons,
            )
        ],
    )
    return fig


def build_fig4_heatmap(
    links: list[dict],
    q1_monthly: dict[str, dict[str, int]],
    max_companies: int = 18,
) -> tuple[go.Figure, list[str], list[str]]:
    cnt_ep = Counter()
    link_rows_by_company: dict[str, list[dict]] = defaultdict(list)
    for row in links:
        for ep in (row.get("source"), row.get("target")):
            if ep:
                cnt_ep[str(ep)] += 1
        for ep in (row.get("source"), row.get("target")):
            if ep:
                link_rows_by_company[str(ep)].append(row)

    top_cos = [c for c, _ in cnt_ep.most_common(max_companies)]

    z = np.zeros((len(top_cos), len(MONTHS_2034)))
    first_hs = np.zeros(z.shape, dtype=bool)

    for ci, company in enumerate(top_cos):
        seen_hs: set[str] = set()
        rows_c = link_rows_by_company.get(company, [])
        by_month: dict[str, list[dict]] = defaultdict(list)
        for row in rows_c:
            month = str(row.get("arrivaldate") or "")[:7]
            if month in MONTHS_2034:
                by_month[month].append(row)

        for mo in MONTHS_2034:
            mi = MONTHS_2034.index(mo)
            month_new_hs = False
            rows_m = sorted(by_month.get(mo, []), key=lambda x: str(x.get("arrivaldate") or ""))
            bc: Counter[str] = Counter()
            for row in rows_m:
                hs = _norm_hs(row.get("hscode"))
                z[ci, mi] += 1
                bc[str(row.get("generated_by") or "?")] += 1
                if hs and hs not in seen_hs:
                    month_new_hs = True
                if hs:
                    seen_hs.add(hs)
            if month_new_hs:
                first_hs[ci, mi] = True

        mc = q1_monthly.get(company, {})
        for mi, mo in enumerate(MONTHS_2034):
            active = int(mc.get(mo, 0)) > 0
            if z[ci, mi] == 0 and active:
                z[ci, mi] = -0.25

    hover = []
    for ci, company in enumerate(top_cos):
        row_ht = []
        for mi, mo in enumerate(MONTHS_2034):
            v = z[ci, mi]
            rows_m = sorted(
                link_rows_by_company.get(company, []),
                key=lambda x: str(x.get("arrivaldate") or ""),
            )
            rows_m = [r for r in rows_m if str(r.get("arrivaldate") or "").startswith(mo)]
            bc = Counter(str(r.get("generated_by") or "?") for r in rows_m)
            bundle_lines = "<br>".join(
                f"• {html.escape(k)}: {v2}" for k, v2 in bc.most_common(12)
            )
            parts = [f"<b>{html.escape(company)}</b> · {mo}"]
            if v > 0:
                parts.append(f"Reliable links added: {int(v)}")
                if bundle_lines:
                    parts.append("By generating bundle:<br>" + bundle_lines)
                if first_hs[ci, mi]:
                    parts.append(
                        '<span style="color:#8f2841">First-seen HS this month for this company (among predicted links)</span>'
                    )
            elif v < 0:
                parts.append("Active in Q1 monthly profile · no reliable-link injection this month (grey cell)")
            else:
                parts.append("No recorded activity / no injection")
            row_ht.append("<br>".join(parts))
        hover.append(row_ht)

    colorscale = [
        [0.0, "#f8fafc"],
        [0.35, "#cbd5e1"],
        [0.55, "#bfdbfe"],
        [0.75, "#60a5fa"],
        [1.0, "#1d4ed8"],
    ]

    z_plot = z.copy()
    z_min, z_max = float(z_plot.min()), float(z_plot.max())

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_plot,
                x=MONTHS_2034,
                y=top_cos,
                colorscale=colorscale,
                zmin=z_min,
                zmax=max(z_max, 0.5),
                hoverinfo="text",
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                colorbar=dict(
                    title=dict(text="Intensity", font=dict(size=10)),
                    tickfont=dict(size=9),
                    outlinewidth=0,
                    borderwidth=0,
                    thickness=10,
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=(
                "Fig 4 · Completion footprint heatmap (top endpoints × months in 2034)<br>"
                "<sup>Hover for bundle breakdown · first-seen HS in tooltip</sup>"
            ),
            font=dict(size=11),
        ),
        font=dict(size=10),
        xaxis=dict(
            side="bottom",
            tickangle=-35,
            tickfont=dict(size=9),
            title_font=dict(size=10),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=7),
            title_font=dict(size=10),
        ),
        height=max(380, 18 * len(top_cos)),
        paper_bgcolor="rgba(255,251,243,0.5)",
        margin=dict(l=132, r=36, t=62, b=108),
    )
    return fig, top_cos, MONTHS_2034


def fig_to_json(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


DASHBOARD_CSS = """
:root {
  --ink: #132238;
  --muted: #62717a;
  --panel: rgba(255, 251, 243, 0.92);
  --line: rgba(19, 34, 56, 0.12);
  --shadow: 0 18px 44px rgba(19, 34, 56, 0.13);
  --accent: #1d6f5f;
  --accent-2: #b15e11;
  --accent-3: #8f2841;
  --bg-1: #f7f0e5;
  --bg-2: #efe4d1;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  font-family: "Avenir Next", "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
  background:
    radial-gradient(circle at top right, rgba(29, 111, 95, 0.16), transparent 24%),
    radial-gradient(circle at left bottom, rgba(177, 94, 17, 0.10), transparent 22%),
    linear-gradient(180deg, var(--bg-1), var(--bg-2));
}
.page { max-width: 1560px; margin: 0 auto; padding: 28px; }
.hero, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 24px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
}
.hero {
  padding: 24px 28px;
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 18px;
  margin-bottom: 16px;
}
.eyebrow {
  display: inline-flex;
  padding: 6px 10px;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  background: rgba(19, 34, 56, 0.06);
  border-radius: 999px;
  border: 1px solid var(--line);
  margin-bottom: 12px;
}
h1 { margin: 0 0 10px; font-size: 32px; line-height: 1.08; letter-spacing: -0.03em; }
h2 { margin: 0 0 8px; font-size: 20px; letter-spacing: -0.02em; }
p { margin: 0; line-height: 1.65; }
.muted { color: var(--muted); }
.meta-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.metric {
  padding: 14px;
  border-radius: 18px;
  background: rgba(19, 34, 56, 0.05);
  border: 1px solid rgba(19, 34, 56, 0.08);
}
.metric strong { display: block; font-size: 22px; margin-bottom: 4px; }
.nav {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 10px;
}
.nav a {
  padding: 8px 12px;
  text-decoration: none;
  color: var(--ink);
  background: rgba(255, 255, 255, 0.64);
  font-size: 13px;
  border-radius: 999px;
  border: 1px solid var(--line);
}
.nav a.active {
  background: rgba(29, 111, 95, 0.12);
  border-color: rgba(29, 111, 95, 0.32);
  color: var(--accent);
  font-weight: 700;
}
.dashboard-grid {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr);
  gap: 18px;
  align-items: start;
}
.sidebar-stack {
  display: grid;
  gap: 18px;
}
.main-flow {
  display: grid;
  gap: 18px;
}
.row-split {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 18px;
  align-items: stretch;
}
.panel { padding: 18px; }
.summary h3 { margin: 0 0 6px; font-size: 14px; }
.section-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.badge {
  padding: 6px 10px;
  background: rgba(177, 94, 17, 0.10);
  color: var(--accent-2);
  font-size: 12px;
  font-weight: 700;
  border-radius: 999px;
  border: 1px solid var(--line);
}
.panel-desc { margin-bottom: 14px; color: var(--muted); font-size: 13px; }
.chart-shell {
  position: relative;
  padding: 10px;
  border-radius: 20px;
  border: none;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,255,255,0.58)),
    repeating-linear-gradient(
      0deg,
      transparent 0,
      transparent 30px,
      rgba(19, 34, 56, 0.04) 30px,
      rgba(19, 34, 56, 0.04) 31px
    );
  box-shadow:
    inset 0 0 80px 44px rgba(255, 251, 243, 0.78),
    0 2px 14px rgba(19, 34, 56, 0.06);
}
.plot-host { width: 100%; min-height: 400px; }
.plot-host.short { min-height: 360px; }
.plot-host.wide { min-height: 440px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.chip {
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.72);
  font-size: 12px;
}
.summary.blk h3 { margin: 0 0 6px; font-size: 14px; }
.summary.blk p { font-size: 13px; color: var(--muted); margin: 0; }
.fig3d-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
  font-size: 13px;
}
.fig3d-toolbar label { display: flex; align-items: center; gap: 8px; color: var(--muted); }
.fig3d-toolbar input[type="range"] { width: 150px; accent-color: var(--accent); }
.btn-expand-3d {
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(29,111,95,0.14);
  color: var(--accent);
  font-weight: 700;
  font-size: 12px;
  cursor: pointer;
}
.preview-3d-outer {
  position: relative;
  padding: 0;
  overflow: hidden;
  border-radius: 20px;
}
.iframe-scale-wrap {
  overflow: auto;
  max-height: min(440px, 54vh);
  border-radius: 16px;
  background: #0f172a;
}
.iframe-scale-inner { transform-origin: center center; transition: transform 0.12s ease-out; }
.iframe-3d-preview {
  width: 100%;
  height: 380px;
  border: 0;
  display: block;
  background: #0f172a;
}
.preview-3d-hint {
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  color: rgba(255,251,243,0.9);
  background: rgba(19,34,56,0.5);
  padding: 6px 12px;
  border-radius: 999px;
  pointer-events: none;
  white-space: nowrap;
}
.panel-fig4-twin {
  max-height: min(78vh, 940px);
  overflow-y: auto;
}
.fig3d-modal {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 100000;
  align-items: center;
  justify-content: center;
}
.fig3d-modal.open { display: flex; }
.fig3d-modal-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(15,23,42,0.58);
  backdrop-filter: blur(5px);
}
.fig3d-modal-panel {
  position: relative;
  z-index: 1;
  width: min(96vw, 1420px);
  height: min(90vh, 900px);
  margin: 16px;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 28px 90px rgba(0,0,0,0.45);
}
.iframe-3d-full {
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
  background: #0f172a;
}
.fig3d-modal-close {
  position: absolute;
  top: 12px;
  right: 14px;
  z-index: 2;
  width: 42px;
  height: 42px;
  border: none;
  border-radius: 999px;
  background: rgba(255,251,243,0.94);
  font-size: 24px;
  line-height: 1;
  cursor: pointer;
  box-shadow: 0 2px 14px rgba(0,0,0,0.18);
}
@media (max-width: 1280px) {
  .dashboard-grid { grid-template-columns: 1fr; }
}
@media (max-width: 1100px) {
  .row-split { grid-template-columns: 1fr; }
}
@media (max-width: 900px) {
  .page { padding: 18px; }
  .hero { grid-template-columns: 1fr; }
  h1 { font-size: 26px; }
}
""".strip()


def sidebar_summary_html(df: pd.DataFrame) -> str:
    chunks: list[str] = []
    for lab, title, color in (
        ("reliable", "Reliable — primary adoption tier", "#176b5c"),
        ("suspicious", "Suspicious — review before use", "#c58435"),
        ("reject", "Reject — illustrative / hold-out", "#8f2841"),
    ):
        sub = df[df["label"].astype(str).str.lower() == lab].sort_values("score", ascending=False)
        names = ", ".join(html.escape(str(x)) for x in sub["bundle"].astype(str))
        chunks.append(
            f'<div class="summary blk"><h3 style="color:{color}">{html.escape(title)}</h3>'
            f"<p>{names or '—'}</p></div>"
        )
    return "".join(chunks)


def format_dashboard_page(
    *,
    summary_inner: str,
    n_bundle: int,
    n_rel: int,
    n_susp: int,
    n_rej: int,
    n_links: int,
    payload_json: str,
    iframe_src: str,
    nav: dict[str, str],
) -> str:
    sep = "&" if "?" in iframe_src else "?"
    src_preview = iframe_src + sep + "dashboard=1"
    src_modal = iframe_src
    sp = json.dumps(src_preview)
    sm = json.dumps(src_modal)
    js = (
        "(function(){var P="
        + payload_json
        + ";var cfg={responsive:true,displayModeBar:true,displaylogo:false};"
        + "Plotly.newPlot('fig1',P.fig1.data,P.fig1.layout,cfg);"
        + "Plotly.newPlot('fig2',P.fig2.data,P.fig2.layout,cfg);"
        + "Plotly.newPlot('fig4',P.fig4.data,P.fig4.layout,cfg);"
        + "var srcP="
        + sp
        + ",srcM="
        + sm
        + ";var prev=document.getElementById('fig3d-iframe-preview');"
        + "var full=document.getElementById('fig3d-iframe-full');var modal=document.getElementById('fig3d-modal');"
        + "if(prev)prev.src=srcP;"
        + "function openM(){if(full)full.src=srcM;if(modal){modal.classList.add('open');modal.setAttribute('aria-hidden','false');document.body.style.overflow='hidden';}}"
        + "function closeM(){if(modal){modal.classList.remove('open');modal.setAttribute('aria-hidden','true');document.body.style.overflow='';}if(full)full.src='about:blank';}"
        + "var btn=document.getElementById('fig3d-open-modal');if(btn)btn.addEventListener('click',function(e){e.stopPropagation();openM();});"
        + "var zr=document.getElementById('fig3d-zoom-range');var inner=document.getElementById('fig3d-scale-inner');"
        + "if(zr&&inner){zr.addEventListener('input',function(){inner.style.transform='scale('+(parseFloat(zr.value)/100)+')';});}"
        + "var bd=modal&&modal.querySelector('.fig3d-modal-backdrop');var xb=modal&&modal.querySelector('.fig3d-modal-close');"
        + "if(bd)bd.addEventListener('click',closeM);if(xb)xb.addEventListener('click',closeM);"
        + "document.addEventListener('keydown',function(e){if(e.key==='Escape'&&modal&&modal.classList.contains('open'))closeM();});"
        + "})();"
    )

    q1f = html.escape(nav["q1_final"], quote=True)
    q1p = html.escape(nav["q1_phase"], quote=True)
    q2p = html.escape(nav["q2_phase"], quote=True)
    q3p = html.escape(nav["q3"], quote=True)
    q4p = html.escape(nav["q4"], quote=True)
    vz = html.escape(nav["viz"], quote=True)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Q2 · Bundle reliability · Final board</title>
<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>{DASHBOARD_CSS}</style>
</head><body>
<div class="page">
  <section class="hero">
    <div>
      <div class="eyebrow">Final sketches · Q2 bundle reliability · pipeline-backed</div>
      <h1>FishFlow Q2 · Predicted-link reliability workspace</h1>
      <p class="muted">
        Bubble matrix (compact markers, hover for bundle &amp; metrics); Fig&nbsp;2 HX radar: <strong>bottom-right menu</strong> (<em>All</em> vs each bundle);
        Fig&nbsp;3–4 side by side — rotating Three.js preview (<code>?dashboard=1</code>) with zoom slider and fullscreen for full controls.
        <strong>Plotly details are in hover tooltips</strong>.
      </p>
    </div>
    <div class="meta-grid">
      <div class="metric"><strong>{n_bundle}</strong><div class="muted">prediction bundles</div></div>
      <div class="metric"><strong>{n_rel} / {n_susp} / {n_rej}</strong><div class="muted">reliable / suspicious / reject</div></div>
      <div class="metric"><strong>{n_links}</strong><div class="muted">merged reliable links</div></div>
      <div class="metric"><strong>2034</strong><div class="muted">heatmap window</div></div>
    </div>
  </section>

  <nav class="nav">
    <a href="{q1f}">Q1 Final board</a>
    <a href="{q1p}">Q1 Phase-1 sketch</a>
    <a href="{q2p}">Q2 Phase-1 sketch</a>
    <a class="active" href="#">Q2 Final board</a>
    <a href="{q3p}">Q3 Comparator</a>
    <a href="{q4p}">Q4 Suspicious Co.</a>
    <a href="{vz}">Visualization index</a>
  </nav>

  <div class="dashboard-grid">
    <aside class="sidebar-stack">
      <div class="panel">
        <h2>How to read</h2>
        <p class="panel-desc">
          Hover bubbles / radar — Fig&nbsp;1 has no point labels (hover only).
          Fig&nbsp;2: small dropdown at bottom-right — choose <strong>All</strong> or one bundle.
          Fig&nbsp;3: preview auto-rotates; scale slider magnifies; “Fullscreen detail” opens the full 3D UI.
        </p>
        <div class="chip-row">
          <span class="chip">Fig 1 · Small bubbles · hover only</span>
          <span class="chip">Fig 2 · Bottom-right · All / one</span>
          <span class="chip">Fig 3 · Rotate + scale + fullscreen</span>
          <span class="chip">Fig 4 · Heatmap beside 3D</span>
        </div>
      </div>
      <div class="panel">
        <div class="section-head">
          <div>
            <h2>Labels summary</h2>
            <p class="panel-desc">Grouped from <code>bundle_reliability.csv</code>.</p>
          </div>
          <span class="badge">Tier</span>
        </div>
        <div id="decisionSummary">{summary_inner}</div>
      </div>
    </aside>

    <div class="main-flow">
      <div class="row-split">
        <div class="panel">
          <div class="section-head">
            <div>
              <h2>Fig 1 · Bubble matrix</h2>
              <p class="panel-desc"><code>bundle_type</code> × <code>repetition_type</code>; markers scaled by <code>link_count</code> — no text labels; hover for bundle name &amp; full CSV metrics.</p>
            </div>
            <span class="badge">Fig 1</span>
          </div>
          <div class="chart-shell"><div id="fig1" class="plot-host"></div></div>
        </div>
        <div class="panel">
          <div class="section-head">
            <div>
              <h2>Fig 2 · HX radar</h2>
              <p class="panel-desc">Six ratios ×100; default all bundles overlaid. Bottom-right dropdown: <strong>All</strong> or pick one bundle (compact list).</p>
            </div>
            <span class="badge">Fig 2</span>
          </div>
          <div class="chart-shell"><div id="fig2" class="plot-host"></div></div>
        </div>
      </div>

      <div class="row-split">
        <div class="panel">
          <div class="section-head">
            <div>
              <h2>Fig 3 · 3D dandelion network</h2>
              <p class="panel-desc">
                Embedded viewer uses <code>?dashboard=1</code> for a clean rotating preview (wheel zoom / drag orbit inside frame).
                Fullscreen restores filters &amp; hints.
              </p>
            </div>
            <span class="badge">Fig 3</span>
          </div>
          <div class="fig3d-toolbar">
            <label title="Scale preview canvas"><span>Preview zoom</span><input type="range" id="fig3d-zoom-range" min="85" max="130" value="100" /></label>
            <button type="button" class="btn-expand-3d" id="fig3d-open-modal">Fullscreen detail</button>
          </div>
          <div class="chart-shell preview-3d-outer">
            <div class="iframe-scale-wrap">
              <div class="iframe-scale-inner" id="fig3d-scale-inner">
                <iframe id="fig3d-iframe-preview" class="iframe-3d-preview" title="Q2 3D dandelion preview" loading="lazy"></iframe>
              </div>
            </div>
            <div class="preview-3d-hint">Auto-rotate on · drag / wheel inside preview</div>
          </div>
        </div>

        <div class="panel panel-fig4-twin">
          <div class="section-head">
            <div>
              <h2>Fig 4 · Completion footprint</h2>
              <p class="panel-desc">Heavy endpoints × 2034 months; hover lists generating bundles and first-seen HS notes.</p>
            </div>
            <span class="badge">Fig 4</span>
          </div>
          <div class="chart-shell"><div id="fig4" class="plot-host wide"></div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="fig3d-modal" class="fig3d-modal" aria-hidden="true">
  <div class="fig3d-modal-backdrop"></div>
  <div class="fig3d-modal-panel">
    <button type="button" class="fig3d-modal-close" aria-label="Close">&times;</button>
    <iframe id="fig3d-iframe-full" class="iframe-3d-full" title="Q2 3D dandelion fullscreen"></iframe>
  </div>
</div>

<script>{js}</script>
</body></html>"""


def main() -> None:
    ensure_dirs()
    FIG_Q2.mkdir(parents=True, exist_ok=True)
    OUT_HTML_FINAL.parent.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    rng = np.random.default_rng(42)
    df = load_bundle_df()
    links = load_links()
    q1m = load_q1_monthly()

    fig1, _bundle_order_score = build_fig1_matrix(df, rng)
    fig2 = build_fig2_hx_radar(df)
    fig4, _companies2034, _months2034 = build_fig4_heatmap(links, q1m)

    n_bundle = len(df)
    n_rel = int((df["label"].astype(str).str.lower() == "reliable").sum())
    n_susp = int((df["label"].astype(str).str.lower() == "suspicious").sum())
    n_rej = int((df["label"].astype(str).str.lower() == "reject").sum())
    n_links = len(links)

    summary_inner = sidebar_summary_html(df)

    payload = {
        "fig1": fig_to_json(fig1),
        "fig2": fig_to_json(fig2),
        "fig4": fig_to_json(fig4),
    }

    payload_json = json.dumps(payload, separators=(",", ":"))

    nav_final = {
        "q1_final": "./q1.html",
        "q1_phase": "../phase1_sketches/q1.html",
        "q2_phase": "../phase1_sketches/q2.html",
        "q3": "../phase1_sketches/q3.html",
        "q4": "../phase1_sketches/q4.html",
        "viz": "../visualization/index.html",
    }
    nav_mirror = {
        "q1_final": "../../../final_sketches/q1.html",
        "q1_phase": "../../../phase1_sketches/q1.html",
        "q2_phase": "../../../phase1_sketches/q2.html",
        "q3": "../../../phase1_sketches/q3.html",
        "q4": "../../../phase1_sketches/q4.html",
        "viz": "../../index.html",
    }

    doc_final = format_dashboard_page(
        summary_inner=summary_inner,
        n_bundle=n_bundle,
        n_rel=n_rel,
        n_susp=n_susp,
        n_rej=n_rej,
        n_links=n_links,
        payload_json=payload_json,
        iframe_src=REL_PATH_3D_FROM_FINAL,
        nav=nav_final,
    )
    doc_mirror = format_dashboard_page(
        summary_inner=summary_inner,
        n_bundle=n_bundle,
        n_rel=n_rel,
        n_susp=n_susp,
        n_rej=n_rej,
        n_links=n_links,
        payload_json=payload_json,
        iframe_src=REL_PATH_3D_FROM_MIRROR,
        nav=nav_mirror,
    )

    OUT_HTML_FINAL.write_text(doc_final, encoding="utf-8")
    OUT_HTML_MIRROR.write_text(doc_mirror, encoding="utf-8")
    print(f"Wrote {OUT_HTML_FINAL}")
    print(f"Wrote {OUT_HTML_MIRROR}")


if __name__ == "__main__":
    main()
