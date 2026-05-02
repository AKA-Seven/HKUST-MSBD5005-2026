#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "q3"
FIG_DIR = PROJECT_ROOT / "visualization" / "figures_2d" / "q3"
FINAL_Q3_HTML = PROJECT_ROOT / "final_sketches" / "q3.html"

FINAL_Q3_CSS = """
    :root {
      --ink: #14213d;
      --muted: #61707c;
      --panel: rgba(251, 249, 242, 0.9);
      --line: rgba(20, 33, 61, 0.11);
      --shadow: 0 18px 44px rgba(20, 33, 61, 0.12);
      --accent: #196d69;
      --accent-2: #ad5c16;
      --accent-3: #812b49;
      --bg-1: #f5efe2;
      --bg-2: #ebdfca;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(25,109,105,0.14), transparent 24%),
        radial-gradient(circle at right bottom, rgba(129,43,73,0.10), transparent 22%),
        linear-gradient(180deg, var(--bg-1), var(--bg-2));
    }
    .page { max-width: 1200px; margin: 0 auto; padding: 28px; }
    .hero {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      padding: 24px 28px;
      margin-bottom: 22px;
    }
    @media (max-width: 900px) { .page { padding: 18px; } }
    .eyebrow {
      display: inline-flex;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(20,33,61,0.05);
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 12px;
      border: 1px solid var(--line);
    }
    h1 { margin: 0 0 10px; font-size: 32px; line-height: 1.08; letter-spacing: -0.03em; }
    .muted { color: var(--muted); }
    p.lead { margin: 0; line-height: 1.65; }
    .top-nav {
      display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px;
    }
    .top-nav a {
      padding: 8px 12px; border-radius: 999px; text-decoration: none; color: var(--ink);
      background: rgba(255,255,255,0.68); font-size: 13px;
      border: 1px solid var(--line);
    }
    .top-nav a.active {
      background: rgba(25,109,105,0.12);
      border-color: rgba(25,109,105,0.32);
      color: var(--accent);
      font-weight: 700;
    }
    .figures-stack {
      display: flex;
      flex-direction: column;
      gap: 28px;
      width: 100%;
      margin-bottom: 8px;
    }
    .panel {
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .panel h2 { margin: 0 0 8px; font-size: 20px; letter-spacing: -0.02em; }
    .panel-desc { margin: 0 0 14px; color: var(--muted); font-size: 13px; line-height: 1.65; }
    .chart-shell {
      padding: 12px;
      border-radius: 20px;
      border: 1px dashed rgba(20,33,61,0.14);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,255,255,0.58)),
        repeating-linear-gradient(0deg, transparent 0, transparent 30px, rgba(20,33,61,0.04) 30px, rgba(20,33,61,0.04) 31px);
    }
    .plot-slot {
      width: 100%;
      overflow: hidden;
      min-height: 120px;
    }
    .plot-slot .plotly-graph-div {
      width: 100% !important;
    }
"""

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.5)",
    font=dict(family='"Avenir Next", "Segoe UI", "PingFang SC", sans-serif', color="#14213d", size=12),
)

def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    delta_path = OUTPUT_DIR / "anomaly_delta.csv"
    analysis_path = OUTPUT_DIR / "graph_diff_analysis.json"
    relay_path = OUTPUT_DIR / "relay_chains.csv"
    bridge_path = OUTPUT_DIR / "bridge_companies.csv"

    if not delta_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {delta_path}")

    delta_df = pd.read_csv(delta_path)
    company_df = delta_df[['company', 'base_first_date', 'base_last_date']].copy()

    relay_df = pd.DataFrame()
    if relay_path.exists():
        relay_df = pd.read_csv(relay_path)
    else:
        json_relay = OUTPUT_DIR / "top_relay_chains.json"
        if json_relay.exists():
            with open(json_relay, 'r', encoding='utf-8') as f:
                data = json.load(f)
            relay_df = pd.DataFrame(data)

    bridge_df = pd.DataFrame()
    if bridge_path.exists():
        bridge_df = pd.read_csv(bridge_path)

    reliable_links = pd.DataFrame()
    if analysis_path.exists():
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            month_new_data = analysis_data.get("temporal_diff", {}).get("month_new_top20", {})
            if month_new_data:
                records = [{"month": k[:7], "count": v} for k, v in month_new_data.items()]
                reliable_links = pd.DataFrame(records)
            else:
                print("警告: graph_diff_analysis.json 中缺少 month_new_top20 数据")
        except Exception as e:
            print(f"警告: 读取 temporal_diff 失败: {e}")
    else:
        print("警告: 未找到 graph_diff_analysis.json，无法生成时间序列图")

    return company_df, delta_df, reliable_links, relay_df, bridge_df

def build_company_view(company_df, delta_df):
    view = company_df.merge(delta_df, on='company', how='left')

    for col in ['filled_gap_months', 'extended_months']:
        count_col = f"{col}_count"
        if count_col not in view.columns:
            view[count_col] = view[col].fillna('').astype(str).apply(lambda x: len(x.split(';')) if x and x != 'nan' else 0)

    if 'burst_months_count' not in view.columns:
        view['burst_months_count'] = view.get('burst_months', '').fillna('').astype(str).apply(lambda x: len(x.split(';')) if x and x != 'nan' else 0)

    def get_confidence(score):
        if pd.isna(score):
            return 'low'
        if score >= 50:
            return 'high'
        if score >= 10:
            return 'medium'
        return 'low'

    if 'suspicious_revival_score' in view.columns:
        view['suspicious_revival_score'] = pd.to_numeric(view['suspicious_revival_score'], errors='coerce').fillna(0)
        view['iuu_confidence'] = view['suspicious_revival_score'].apply(get_confidence)
    else:
        view['iuu_confidence'] = 'low'

    view = view.sort_values('suspicious_revival_score', ascending=False).reset_index(drop=True)
    return view

def _filter_relay_for_sankey(relay_df):
    if relay_df.empty:
        return pd.DataFrame()
    filtered = relay_df[relay_df['shared_partner_count'] >= 40].copy()
    if filtered.empty:
        filtered = relay_df.nlargest(20, 'shared_partner_count')
    succ_total = filtered.groupby('successor')['shared_partner_count'].sum().sort_values(ascending=False)
    top_succ = succ_total.head(20).index.tolist()
    return filtered[filtered['successor'].isin(top_succ)]

def build_relay_sankey_figure(relay_df):
    filtered = _filter_relay_for_sankey(relay_df)
    if filtered.empty:
        return None

    pred_total = filtered.groupby('predecessor')['shared_partner_count'].sum().sort_values(ascending=False)
    succ_total = filtered.groupby('successor')['shared_partner_count'].sum().sort_values(ascending=False)

    left_nodes = pred_total.index.tolist()
    right_nodes = succ_total.index.tolist()
    all_nodes = left_nodes + [n for n in right_nodes if n not in left_nodes]

    node_id = {name: i for i, name in enumerate(all_nodes)}

    n_left = len(left_nodes)
    node_x = [0.1] * n_left + [0.9] * (len(all_nodes) - n_left)
    left_y = np.linspace(0.05, 0.95, n_left) if n_left > 1 else [0.5]
    right_exclusive = [n for n in right_nodes if n not in left_nodes]
    n_right = len(right_exclusive)
    right_y = np.linspace(0.05, 0.95, n_right) if n_right > 1 else [0.5]
    node_y = list(left_y) + list(right_y)

    sources, targets, values, customdata = [], [], [], []
    for _, row in filtered.iterrows():
        pred, succ, cnt = row['predecessor'], row['successor'], row['shared_partner_count']
        if pred in node_id and succ in node_id:
            sources.append(node_id[pred])
            targets.append(node_id[succ])
            values.append(cnt)
            customdata.append(cnt)

    short_labels = []
    for name in all_nodes:
        short = name.replace("Limited Liability Company", "LLC").replace("GmbH & Co. KG", "").strip()
        if len(short) > 22:
            short = short[:19] + "..."
        short_labels.append(short)

    node_colors = []
    if n_left > 0:
        for i in range(n_left):
            intensity = i / (n_left - 1) if n_left > 1 else 0.5
            r = 255
            g = 165 + int(55 * intensity)
            b = 89 + int(60 * intensity)
            node_colors.append(f"rgb({r}, {g}, {b})")
    if n_right > 0:
        for i in range(n_right):
            intensity = i / (n_right - 1) if n_right > 1 else 0.5
            r = 93 - int(30 * intensity)
            g = 173 - int(30 * intensity)
            b = 226 - int(20 * intensity)
            node_colors.append(f"rgb({r}, {g}, {b})")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=20,
            line=dict(color="black", width=0.5),
            label=short_labels,
            x=node_x, y=node_y,
            color=node_colors,
            customdata=all_nodes,
            hovertemplate='%{customdata}<extra></extra>'
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color='rgba(100, 150, 200, 0.4)',
            customdata=customdata,
            hovertemplate='%{source.customdata} → %{target.customdata}<br>shared partners: %{customdata}<extra></extra>'
        )
    )])

    fig.update_layout(
        title=dict(text="Relay chains (shared partners filtered, successor top-20)", font=dict(size=15), x=0.5),
        height=720,
        autosize=True,
        margin=dict(l=56, r=56, t=54, b=40),
        **PLOT_THEME,
    )
    return fig

def draw_relay_sankey(relay_df, out_dir):
    if relay_df.empty:
        print("警告: 未找到接力数据，跳过桑基图")
        return
    fig = build_relay_sankey_figure(relay_df)
    if fig is None:
        print("警告: 过滤后无数据，跳过桑基图")
        return
    fig.write_html(out_dir / "q3_01_relay_sankey_clean.html", include_plotlyjs=True)
    try:
        fig.write_image(out_dir / "q3_01_relay_sankey_clean.png", scale=3)
    except Exception as e:
        print(f"PNG 导出失败（需安装 kaleido）: {e}")

def build_bridge_bubble_figure(bridge_df):
    if bridge_df.empty:
        return None
    bdf = bridge_df.copy()
    bdf['efficiency'] = bdf['bridge_scope'] / bdf['bridge_link_count']
    top30 = bdf.nlargest(30, 'bridge_scope').copy()
    avg_efficiency = bridge_df['bridge_scope'].sum() / bridge_df['bridge_link_count'].sum()
    x_max = float(top30['bridge_link_count'].max()) + 1
    xs = np.linspace(0.0, x_max, 100)
    ys = avg_efficiency * xs
    hover_cd = []
    for _, row in top30.iterrows():
        hover_cd.append(
            f"{row['company']}<br>new links: {row['bridge_link_count']}"
            f"<br>reachable: {row['bridge_scope']:.0f}<br>efficiency: {row['efficiency']:.1f}"
        )

    mx = float(top30['bridge_scope'].max()) or 1.0
    sizes = np.clip(top30['bridge_scope'].values / mx * 48 + 10, 10, 56)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=top30['bridge_link_count'],
        y=top30['bridge_scope'],
        mode='markers',
        customdata=hover_cd,
        hovertemplate='%{customdata}<extra></extra>',
        name='Companies',
        marker=dict(
            size=sizes,
            color=top30['efficiency'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(
                title=dict(text='Efficiency<br>(nodes/link)', font=dict(size=11)),
                tickfont=dict(size=10), len=0.75, thickness=14, x=1.02,
            ),
            line=dict(width=1, color='rgba(20,33,61,0.55)'),
        ),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines',
        line=dict(color='#61707c', width=2, dash='dash'),
        name=f'Average ({avg_efficiency:.1f} nodes/link)',
        hoverinfo='skip',
    ))

    lbl_x, lbl_y, lbl_t = [], [], []
    for _, row in top30.head(6).iterrows():
        lbl_x.append(row['bridge_link_count'])
        lbl_y.append(row['bridge_scope'])
        c = str(row['company'])
        lbl_t.append(c[:22] + '…' if len(c) > 23 else c)
    fig.add_trace(go.Scatter(
        x=lbl_x, y=lbl_y, mode='text', text=lbl_t,
        textposition='top center', textfont=dict(size=8, color='#14213d'),
        hoverinfo='skip', showlegend=False,
    ))

    fig.update_layout(
        title=dict(text='Bridge leverage: few new links vs. reach', x=0.5, font=dict(size=15)),
        xaxis=dict(title='New reliable links used', gridcolor='rgba(20,33,61,0.08)', zeroline=False),
        yaxis=dict(title='Reachable nodes (bridge scope)', gridcolor='rgba(20,33,61,0.08)', zeroline=False),
        height=500,
        margin=dict(l=72, r=100, t=48, b=88),
        legend=dict(
            orientation='h', yanchor='top', y=-0.14, xanchor='center', x=0.5,
            bgcolor='rgba(255,250,240,0.9)', borderwidth=0,
        ),
        **PLOT_THEME,
    )
    return fig

def build_timeline_figure(reliable_links):
    if reliable_links.empty:
        return None
    df = reliable_links.sort_values('month').copy()

    ymax = float(df['count'].max()) if len(df) else 1
    pad = ymax * 0.12 + 3

    fig = go.Figure(go.Scatter(
        x=df['month'].astype(str),
        y=df['count'],
        mode='lines+markers+text',
        line=dict(color='#196d69', width=3),
        marker=dict(size=11, color='#fff', line=dict(width=2, color='#196d69')),
        text=[str(int(v)) for v in df['count']],
        textposition='top center',
        textfont=dict(size=11, color='#14213d'),
        cliponaxis=False,
        hovertemplate='%{x}<br>new links: %{y}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Monthly new endorsed links merged into graph', x=0.5, font=dict(size=15)),
        xaxis=dict(title='Month', tickangle=-40, gridcolor='rgba(20,33,61,0.08)', zeroline=False),
        yaxis=dict(title='New links counted', gridcolor='rgba(20,33,61,0.08)', zeroline=False, range=[0, ymax + pad]),
        height=410,
        margin=dict(l=64, r=40, t=52, b=108),
        **PLOT_THEME,
    )
    return fig

def _heatmap_labels_and_matrix(company_view):
    top20 = company_view.head(20).copy()
    if top20.empty:
        return None, None, None
    signal_cols = ['dormancy_months', 'new_partner_count', 'new_hscode_count',
                   'filled_gap_months_count', 'extended_months_count', 'burst_months_count']
    col_labels = ['Dormancy', 'New partners', 'New HS codes', 'Filled gap mos.',
                  'Extended mos.', 'Burst mos.']
    col_map = dict(zip(signal_cols, col_labels))
    existing_cols = [c for c in signal_cols if c in top20.columns]
    if not existing_cols:
        return None, None, None
    labels_x = [col_map[c] for c in existing_cols]
    heat_arr = top20[existing_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    rows = []
    for c in top20['company'].astype(str):
        rows.append((c[:40] + '…') if len(c) > 41 else c)
    return rows, labels_x, heat_arr

def build_suspicion_heatmap_figure(company_view):
    ys, xs, zm = _heatmap_labels_and_matrix(company_view)
    if zm is None:
        return None
    text_m = zm.astype(float).astype(int).astype(str)
    fig = go.Figure(data=go.Heatmap(
        z=zm,
        x=xs,
        y=ys,
        colorscale='Reds',
        text=text_m,
        texttemplate='%{text}',
        textfont=dict(size=10, color='#1a1a1a'),
        hovertemplate='%{y}<br>%{x}: %{z}<extra></extra>',
        colorbar=dict(title=dict(text='Signal strength', font=dict(size=11)), len=0.55, thickness=14, x=1.03),
    ))
    fig.update_layout(
        title=dict(text='Top-20 revival-related signals', x=0.5, font=dict(size=14)),
        xaxis=dict(side='bottom', tickangle=-35, zeroline=False),
        yaxis=dict(autorange='reversed', tickfont=dict(size=10)),
        height=max(640, len(ys) * 30 + 200),
        margin=dict(l=220, r=100, t=54, b=136),
        **PLOT_THEME,
    )
    return fig

def _fig_embed_div(fig, div_id):
    dup = go.Figure(fig)
    dup.update_layout(title=None)

    fragment = pio.to_html(
        dup, full_html=False, include_plotlyjs=False,
        config={'responsive': True, 'displayModeBar': True},
        div_id=div_id,
    )
    return f'<div class="plot-slot">{fragment}</div>'

def _missing_chart_blurb(kind):
    return (
        '<div class="muted plot-slot" style="display:flex;align-items:center;justify-content:center;'
        'text-align:center;padding:24px 12px;"><p>This chart cannot be rendered because no “{kind}” data '
        'is available from the analytics export.</p></div>'
    ).format(kind=kind)

def write_final_q3_board(company_view, relay_df, bridge_df, reliable_links):
    FINAL_Q3_HTML.parent.mkdir(parents=True, exist_ok=True)

    f_sankey = build_relay_sankey_figure(relay_df)
    f_bridge = build_bridge_bubble_figure(bridge_df)
    f_time = build_timeline_figure(reliable_links)
    f_heat = build_suspicion_heatmap_figure(company_view)

    sankey_slot = _fig_embed_div(f_sankey, 'chart-sankey') if f_sankey else _missing_chart_blurb('relay chains')
    time_slot = _fig_embed_div(f_time, 'chart-timeline') if f_time else _missing_chart_blurb('monthly timeline')
    bridge_slot = _fig_embed_div(f_bridge, 'chart-bridge') if f_bridge else _missing_chart_blurb('bridge companies')
    heat_slot = _fig_embed_div(f_heat, 'chart-heatmap') if f_heat else _missing_chart_blurb('revival summary')

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Q3 · Graph completion · Final board</title>
  <style>{FINAL_Q3_CSS}</style>
  <script src="https://cdn.plot.ly/plotly-2.29.0.min.js" charset="utf-8"></script>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Final board · Q3 · Graph completion</div>
      <h1>What changes once the graph is completed</h1>
      <p class="lead muted">
        Trusted predicted links materially reshape how we read timing, relays, bridging power,
        and endpoint-level revival patterns. Each section below translates one analytic angle into an interactive graphic.
      </p>
    </section>

    <nav class="top-nav" aria-label="Final board navigation">
      <a href="./q1.html">Q1 · Temporal patterns</a>
      <a href="./q2.html">Q2 · Bundle reliability</a>
      <a href="./q3.html" class="active">Q3 · Graph completion</a>
      <a href="./q4.html">Q4 · Suspicious companies</a>
    </nav>

    <main class="figures-stack">
      <section class="panel">
        <h2>(1) Relay chains</h2>
        <p class="panel-desc">
          Predecessors hand trading relationships to successors; thicker bands mean more overlapping partners,
          signalling possible inheritance of a dormant firm’s counterparties by a revived actor.
        </p>
        <div class="chart-shell">{sankey_slot}</div>
      </section>

      <section class="panel">
        <h2>(2) Temporal pulse of endorsed links</h2>
        <p class="panel-desc">
          Counts show how aggressively new vetted ties land month by month once graph completion kicks in—the emphasis is on bursts and quiet periods rather than evenly spread noise.
        </p>
        <div class="chart-shell">{time_slot}</div>
      </section>

      <section class="panel">
        <h2>(3) Bridge leverage</h2>
        <p class="panel-desc">
          Bubble area amplifies reachable scope; colour encodes efficiency (nodes gained per endorsed link).
          Labels mark the widest bridges. Points drifting above the dashed trend beat the dataset-wide leverage average.
        </p>
        <div class="chart-shell">{bridge_slot}</div>
      </section>

      <section class="panel">
        <h2>(4) Revival signal matrix</h2>
        <p class="panel-desc">
          Rows are the highest-ranked suspicious-revival endpoints. Columns summarise complementary cues—extended dormancy, partner novelty,
          abrupt HS diversification, patched timeline gaps—so hotspots read as corroborating stories instead of lone metrics.
        </p>
        <div class="chart-shell">{heat_slot}</div>
      </section>
    </main>
  </div>
</body>
</html>"""

    FINAL_Q3_HTML.write_text(html_out, encoding='utf-8')
    print("Final board written:", FINAL_Q3_HTML)

def draw_bridge_bubble(bridge_df, out_dir):
    if bridge_df.empty:
        print("警告: 未找到桥接公司数据，跳过气泡图")
        return

    bridge_df = bridge_df.copy()
    bridge_df['efficiency'] = bridge_df['bridge_scope'] / bridge_df['bridge_link_count']
    top30 = bridge_df.nlargest(30, 'bridge_scope')

    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(
        top30['bridge_link_count'], top30['bridge_scope'],
        s=top30['bridge_scope'] / 40,
        c=top30['efficiency'], cmap='RdYlGn_r',
        alpha=0.7, edgecolors='black', linewidth=1
    )

    avg_efficiency = bridge_df['bridge_scope'].sum() / bridge_df['bridge_link_count'].sum()
    x_line = np.linspace(0, top30['bridge_link_count'].max() + 1, 100)
    y_line = avg_efficiency * x_line
    plt.plot(x_line, y_line, 'k--', alpha=0.5, label=f'Avg efficiency ({avg_efficiency:.1f} nodes/link)')

    for _, row in top30.head(10).iterrows():
        plt.annotate(row['company'][:20], (row['bridge_link_count'], row['bridge_scope']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    plt.xlabel('Number of New Links Used (bridge_link_count)')
    plt.ylabel('Number of Reachable Nodes (bridge_scope)')
    plt.title('Bridge Companies: Few Links, Massive Connectivity')
    plt.colorbar(scatter, label='Efficiency (nodes per link)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "q3_02_bridge_influence.png", dpi=200)
    plt.close()

def draw_timeline_line(reliable_links, out_dir):
    if reliable_links.empty:
        print("警告: 无可用的时间序列数据，跳过折线图")
        return
    df = reliable_links.sort_values('month')
    months = df['month']
    counts = df['count']

    plt.figure(figsize=(12, 5))
    plt.plot(months, counts, marker='o', linestyle='-', linewidth=2.5,
             markersize=8, color='#1f77b4', markerfacecolor='white', markeredgewidth=1.5)
    for x, y in zip(months, counts):
        plt.text(x, y + max(counts) * 0.02, str(y), ha='center', va='bottom', fontsize=9)

    plt.xlabel('Month (2034)', fontsize=12)
    plt.ylabel('Number of New Links Added', fontsize=12)
    plt.title('Temporal Trend of Graph Completion Links', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "q3_03_monthly_new_links.png", dpi=200)
    plt.close()

def draw_suspicious_heatmap(company_view, out_dir):
    top20 = company_view.head(20).copy()
    if top20.empty:
        return
    signal_cols = ['dormancy_months', 'new_partner_count', 'new_hscode_count',
                   'filled_gap_months_count', 'extended_months_count', 'burst_months_count']
    existing_cols = [c for c in signal_cols if c in top20.columns]
    heat_data = top20[existing_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    heat_data.index = top20['company']

    plt.figure(figsize=(12, 8))
    sns.heatmap(heat_data, annot=True, fmt='.0f', cmap='Reds', linewidths=0.5,
                cbar_kws={'label': 'Raw Value'}, annot_kws={'size': 8})
    plt.title('High-suspicion Companies: Key Revival Signals')
    plt.xlabel('Signal Type')
    plt.ylabel('Company (sorted by suspicious_revival_score)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / "q3_04_suspicious_heatmap.png", dpi=200)
    plt.close()

def main():
    ensure_dirs()
    company_df, delta_df, reliable_links, relay_df, bridge_df = load_data()
    company_view = build_company_view(company_df, delta_df)

    draw_relay_sankey(relay_df, FIG_DIR)
    draw_bridge_bubble(bridge_df, FIG_DIR)
    draw_timeline_line(reliable_links, FIG_DIR)
    draw_suspicious_heatmap(company_view, FIG_DIR)
    write_final_q3_board(company_view, relay_df, bridge_df, reliable_links)

    print("Q3 四张图表已生成至:", FIG_DIR)

if __name__ == "__main__":
    main()