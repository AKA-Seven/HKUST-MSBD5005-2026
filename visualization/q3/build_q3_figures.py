#!/usr/bin/env python3

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
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

# ================= 前后对比网络图（增大字体版本） =================
def load_network_data():
    anomaly_path = OUTPUT_DIR / "anomaly_delta.csv"
    bridge_path = OUTPUT_DIR / "bridge_companies.csv"
    relay_path = OUTPUT_DIR / "relay_chains.csv"
    
    anomaly_df = pd.read_csv(anomaly_path)
    score_col = 'suspicious_revival_score' if 'suspicious_revival_score' in anomaly_df.columns else 'score'
    high_df = anomaly_df[anomaly_df[score_col] > 50]
    if high_df.empty:
        high_df = anomaly_df.nlargest(3, score_col)
    highlighted = high_df['company'].tolist()
    if not highlighted:
        highlighted = anomaly_df['company'].tolist()[:1]

    bridge_df = pd.read_csv(bridge_path)
    top_bridges = bridge_df.nlargest(1, 'bridge_scope')['company'].tolist()

    relay_df = pd.read_csv(relay_path)
    top_relays = relay_df.nlargest(1, 'shared_partner_count')[['predecessor', 'successor']].values.tolist()

    normal_df = anomaly_df[anomaly_df[score_col] < 10]
    if len(normal_df) >= 2:
        normal = normal_df.head(3)['company'].tolist()
    else:
        normal = ["Serengeti Inc. Ltd Seabed", "Costa de la Felicidad Shipping"]
    return highlighted, top_bridges, top_relays, normal

def build_original_edges(highlighted, top_bridges, top_relays, normal):
    edges = []
    if len(normal) >= 2:
        edges.append((normal[0], normal[1]))
    if len(normal) >= 3:
        edges.append((normal[1], normal[2]))
    if highlighted:
        edges.append((highlighted[0], highlighted[0]))
    return edges

def build_trusted_edges(highlighted, top_bridges, top_relays, normal):
    bridge_edges = []
    relay_edges = []
    burst_edges = []
    if top_bridges and normal:
        b = top_bridges[0]
        for n in normal[:2]:
            bridge_edges.append((b, n))
    if top_relays:
        pred, succ = top_relays[0]
        relay_edges.append((pred, succ))
    if highlighted and normal:
        sh = highlighted[0]
        for n in normal:
            burst_edges.append((sh, n))
    return bridge_edges + relay_edges + burst_edges

def plot_network_comparison(original_edges, trusted_edges, highlighted, top_bridges, top_relays, normal, save_path):
    all_nodes = set()
    for u, v in original_edges:
        all_nodes.add(u); all_nodes.add(v)
    for u, v in trusted_edges:
        all_nodes.add(u); all_nodes.add(v)

    highlighted_set = set(highlighted) | set(top_bridges)
    for pred, succ in top_relays:
        highlighted_set.add(pred); highlighted_set.add(succ)

    node_color = {}
    for n in all_nodes:
        node_color[n] = '#ad5c16' if n in highlighted_set else '#7f8c8d'

    G_full = nx.Graph()
    G_full.add_edges_from(original_edges + trusted_edges)
    pos = nx.spring_layout(G_full, seed=42, k=1.8, iterations=80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), facecolor='white')
    fig.suptitle('Before / After Network Comparison\nQ3 Graph Completion Analysis',
                 fontsize=20, fontweight='bold', y=0.98, color='#14213d')

    # Before
    G_before = nx.Graph()
    G_before.add_edges_from(original_edges)
    if G_before.number_of_nodes() > 0:
        node_colors_before = [node_color[n] for n in G_before.nodes()]
        node_sizes_before = [2200 if n in highlighted_set else 1600 for n in G_before.nodes()]
        nx.draw_networkx_nodes(G_before, pos, ax=ax1, node_color=node_colors_before,
                               node_size=node_sizes_before, edgecolors='black', linewidths=1.8)
        nx.draw_networkx_edges(G_before, pos, ax=ax1, edge_color='#203553', width=3.0,
                               style='dashed', alpha=0.8)
        for node, (x, y) in pos.items():
            if node in G_before.nodes():
                ax1.text(x, y, node, fontsize=12, fontweight='bold', ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
    ax1.set_title("Before: Original Graph", fontsize=16, fontweight='bold', pad=20, color='#14213d')
    ax1.set_facecolor('white')
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlim(-1.8, 1.8); ax1.set_ylim(-1.5, 1.5)

    # After
    G_after = nx.Graph()
    G_after.add_edges_from(original_edges + trusted_edges)
    if G_after.number_of_nodes() > 0:
        node_colors_after = [node_color[n] for n in G_after.nodes()]
        node_sizes_after = [2200 if n in highlighted_set else 1600 for n in G_after.nodes()]
        nx.draw_networkx_nodes(G_after, pos, ax=ax2, node_color=node_colors_after,
                               node_size=node_sizes_after, edgecolors='black', linewidths=1.8)
        if original_edges:
            nx.draw_networkx_edges(G_after, pos, ax=ax2, edgelist=original_edges,
                                   edge_color='#203553', width=3.0, style='dashed', alpha=0.7)
        if trusted_edges:
            nx.draw_networkx_edges(G_after, pos, ax=ax2, edgelist=trusted_edges,
                                   edge_color='#196d69', width=4.5, style='solid', alpha=0.95)
        for node, (x, y) in pos.items():
            if node in G_after.nodes():
                ax2.text(x, y, node, fontsize=12, fontweight='bold', ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
    ax2.set_title("After: Completed Graph", fontsize=16, fontweight='bold', pad=20, color='#14213d')
    ax2.set_facecolor('white')
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_xlim(-1.8, 1.8); ax2.set_ylim(-1.5, 1.5)

    legend_elements = [
        mpatches.Patch(facecolor='#7f8c8d', edgecolor='black', label='Normal company'),
        mpatches.Patch(facecolor='#ad5c16', edgecolor='black', label='Highlighted actor'),
        plt.Line2D([0], [0], color='#203553', lw=3, linestyle='dashed', label='original structure'),
        plt.Line2D([0], [0], color='#196d69', lw=4, label='trusted added edge')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=13,
               bbox_to_anchor=(0.5, -0.02), frameon=False, facecolor='white', edgecolor='none')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Network comparison saved to {save_path}")

# ================= 静态热力图（无网格、大尺寸） =================
def build_suspicion_heatmap_static(company_view, save_path):
    """生成无网格、大尺寸表格型热力图，文字清晰"""
    top20 = company_view.head(20).copy()
    if top20.empty:
        return None
    signal_cols = ['dormancy_months', 'new_partner_count', 'new_hscode_count',
                   'filled_gap_months_count', 'extended_months_count', 'burst_months_count']
    col_labels = ['Dormancy', 'New partners', 'New HS codes', 'Filled gap mos.', 'Extended mos.', 'Burst mos.']
    existing_cols = [c for c in signal_cols if c in top20.columns]
    if not existing_cols:
        return None
    labels_x = col_labels[:len(existing_cols)]
    data = top20[existing_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # 截断公司名
    rows = [ (c[:35] + '…') if len(c) > 36 else c for c in top20['company'].astype(str) ]
    
    # 大幅增大图片尺寸：每列宽2.0英寸，每行高0.7英寸
    n_cols = len(labels_x)
    n_rows = len(rows)
    fig_width = n_cols * 2.0
    fig_height = n_rows * 0.7 + 1.2  # 额外空间给标题和颜色条
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    
    im = ax.imshow(data, cmap='Reds', aspect='auto', vmin=0, vmax=data.max() if data.size>0 else 1)
    
    # 显示数值文本，字体调大
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i, j])
            ax.text(j, i, str(val), ha='center', va='center', fontsize=11, color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.8))
    
    # 设置轴标签，去除网格线和刻度线
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(labels_x, fontsize=11, rotation=45, ha='right')
    ax.set_yticklabels(rows, fontsize=10)
    
    # 隐藏所有网格和边框
    ax.grid(False)
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('Signal strength', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title("Top-20 revival-related signals", fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Heatmap saved to {save_path}")
    return save_path

# ================= 原有数据处理和其他图表（桑基图、气泡图保持不变） =================
def load_data():
    delta_path = OUTPUT_DIR / "anomaly_delta.csv"
    analysis_path = OUTPUT_DIR / "graph_diff_analysis.json"
    relay_path = OUTPUT_DIR / "relay_chains.csv"
    bridge_path = OUTPUT_DIR / "bridge_companies.csv"

    if not delta_path.exists():
        raise FileNotFoundError(f"Missing data: {delta_path}")

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
        except Exception as e:
            print(f"Warning: reading temporal_diff failed: {e}")
    return company_df, delta_df, reliable_links, relay_df, bridge_df

def build_company_view(company_df, delta_df):
    view = company_df.merge(delta_df, on='company', how='left')
    for col in ['filled_gap_months', 'extended_months']:
        count_col = f"{col}_count"
        if count_col not in view.columns:
            view[count_col] = view[col].fillna('').astype(str).apply(lambda x: len(x.split(';')) if x and x != 'nan' else 0)
    if 'burst_months_count' not in view.columns:
        view['burst_months_count'] = view.get('burst_months', '').fillna('').astype(str).apply(lambda x: len(x.split(';')) if x and x != 'nan' else 0)
    if 'suspicious_revival_score' in view.columns:
        view['suspicious_revival_score'] = pd.to_numeric(view['suspicious_revival_score'], errors='coerce').fillna(0)
        view['iuu_confidence'] = view['suspicious_revival_score'].apply(lambda s: 'high' if s >= 50 else ('medium' if s >= 10 else 'low'))
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
    pred_total = filtered.groupby('predecessor')['shared_partner_count'].sum()
    succ_total = filtered.groupby('successor')['shared_partner_count'].sum()
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
        node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=short_labels, x=node_x, y=node_y, color=node_colors, customdata=all_nodes, hovertemplate='%{customdata}<extra></extra>'),
        link=dict(source=sources, target=targets, value=values, color='rgba(100,150,200,0.4)', customdata=customdata, hovertemplate='%{source.customdata} → %{target.customdata}<br>shared partners: %{customdata}<extra></extra>')
    )])
    fig.update_layout(title=dict(text="Relay chains (shared partners filtered, successor top-20)", font=dict(size=15), x=0.5), height=720, autosize=True, margin=dict(l=56, r=56, t=54, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)", font=dict(family='"Avenir Next", "Segoe UI", "PingFang SC", sans-serif', color="#14213d", size=12))
    return fig

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
    hover_cd = [f"{row['company']}<br>new links: {row['bridge_link_count']}<br>reachable: {row['bridge_scope']:.0f}<br>efficiency: {row['efficiency']:.1f}" for _, row in top30.iterrows()]
    mx = float(top30['bridge_scope'].max()) or 1.0
    sizes = np.clip(top30['bridge_scope'].values / mx * 48 + 10, 10, 56)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=top30['bridge_link_count'], y=top30['bridge_scope'], mode='markers', customdata=hover_cd, hovertemplate='%{customdata}<extra></extra>', name='Companies', marker=dict(size=sizes, color=top30['efficiency'], colorscale='RdYlGn_r', showscale=True, colorbar=dict(title=dict(text='Efficiency<br>(nodes/link)', font=dict(size=11)), tickfont=dict(size=10), len=0.75, thickness=14, x=1.02), line=dict(width=1, color='rgba(20,33,61,0.55)'))))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='#61707c', width=2, dash='dash'), name=f'Average ({avg_efficiency:.1f} nodes/link)', hoverinfo='skip'))
    lbl_x, lbl_y, lbl_t = [], [], []
    for _, row in top30.head(6).iterrows():
        lbl_x.append(row['bridge_link_count'])
        lbl_y.append(row['bridge_scope'])
        c = str(row['company'])
        lbl_t.append(c[:22] + '…' if len(c) > 23 else c)
    fig.add_trace(go.Scatter(x=lbl_x, y=lbl_y, mode='text', text=lbl_t, textposition='top center', textfont=dict(size=8, color='#14213d'), hoverinfo='skip', showlegend=False))
    fig.update_layout(title=dict(text='Bridge leverage: few new links vs. reach', x=0.5, font=dict(size=15)), xaxis=dict(title='New reliable links used', gridcolor='rgba(20,33,61,0.08)', zeroline=False), yaxis=dict(title='Reachable nodes (bridge scope)', gridcolor='rgba(20,33,61,0.08)', zeroline=False), height=500, margin=dict(l=72, r=100, t=48, b=88), legend=dict(orientation='h', yanchor='top', y=-0.14, xanchor='center', x=0.5, bgcolor='rgba(255,250,240,0.9)', borderwidth=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)", font=dict(family='"Avenir Next", "Segoe UI", "PingFang SC", sans-serif', color="#14213d", size=12))
    return fig

def _fig_embed_div(fig, div_id):
    dup = go.Figure(fig)
    dup.update_layout(title=None)
    fragment = pio.to_html(dup, full_html=False, include_plotlyjs=False, config={'responsive': True, 'displayModeBar': True}, div_id=div_id)
    return f'<div class="plot-slot">{fragment}</div>'

def _missing_chart_blurb(kind):
    return f'<div class="muted plot-slot" style="display:flex;align-items:center;justify-content:center;text-align:center;padding:24px 12px;"><p>Cannot render "{kind}" — data missing.</p></div>'

# ================= 最终 HTML 生成（顺序 1 → 2 → 3 → 4） =================
def write_final_q3_board(company_view, relay_df, bridge_df, reliable_links, network_img_path, heatmap_img_path):
    FINAL_Q3_HTML.parent.mkdir(parents=True, exist_ok=True)

    f_sankey = build_relay_sankey_figure(relay_df)
    f_bridge = build_bridge_bubble_figure(bridge_df)

    sankey_slot = _fig_embed_div(f_sankey, 'chart-sankey') if f_sankey else _missing_chart_blurb('relay chains')
    bridge_slot = _fig_embed_div(f_bridge, 'chart-bridge') if f_bridge else _missing_chart_blurb('bridge companies')
    
    # 静态图片
    img_rel_network = os.path.relpath(network_img_path, start=FINAL_Q3_HTML.parent)
    network_slot = f'<div class="plot-slot"><img src="{img_rel_network}" alt="Before/After Network Comparison" style="width:100%; border-radius:12px;"></div>'
    
    img_rel_heat = os.path.relpath(heatmap_img_path, start=FINAL_Q3_HTML.parent)
    heatmap_slot = f'<div class="plot-slot"><img src="{img_rel_heat}" alt="Revival signal matrix" style="width:100%; border-radius:12px;"></div>'

    CSS = """
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
      background: radial-gradient(circle at top left, rgba(25,109,105,0.14), transparent 24%),
                  radial-gradient(circle at right bottom, rgba(129,43,73,0.10), transparent 22%),
                  linear-gradient(180deg, var(--bg-1), var(--bg-2));
    }
    .page { max-width: 1400px; margin: 0 auto; padding: 28px; }
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
      transition: all 0.2s ease;
    }
    .top-nav a:hover {
      background: rgba(25,109,105,0.08);
      border-color: var(--accent);
    }
    .top-nav a.active {
      background: rgba(25,109,105,0.12);
      border-color: rgba(25,109,105,0.32);
      color: var(--accent);
      font-weight: 700;
    }
    .figures-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
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
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .panel:hover {
      transform: translateY(-4px);
      box-shadow: 0 24px 48px rgba(20,33,61,0.16);
    }
    .panel h2 { margin: 0 0 8px; font-size: 20px; letter-spacing: -0.02em; }
    .panel-desc { margin: 0 0 14px; color: var(--muted); font-size: 13px; line-height: 1.65; }
    .chart-shell {
      padding: 12px;
      border-radius: 20px;
      border: 1px dashed rgba(20,33,61,0.14);
      background: linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,255,255,0.58)),
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
    @media (max-width: 1000px) {
      .figures-grid { grid-template-columns: 1fr; gap: 24px; }
    }
    """

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Q3 · Graph completion · Final board</title>
  <style>{CSS}</style>
  <script src="https://cdn.plot.ly/plotly-2.29.0.min.js" charset="utf-8"></script>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Final board · Q3 · Graph completion</div>
      <h1>What changes once the graph is completed</h1>
      <p class="lead muted">Trusted predicted links materially reshape how we read timing, relays, bridging power, and endpoint-level revival patterns. Each section below translates one analytic angle into an interactive graphic.</p>
    </section>
    <nav class="top-nav">
      <a href="./q1.html">Q1 · Temporal patterns</a>
      <a href="./q2.html">Q2 · Bundle reliability</a>
      <a href="./q3.html" class="active">Q3 · Graph completion</a>
      <a href="./q4.html">Q4 · Suspicious companies</a>
    </nav>
    <main class="figures-grid">
      <!-- (1) 前后对比网络图 -->
      <section class="panel">
        <h2>(1) Before / After Network Comparison</h2>
        <p class="panel-desc">Same set of companies before and after graph completion. Added trusted edges (bridge, relay, shell burst) reveal hidden structural connections.</p>
        <div class="chart-shell">{network_slot}</div>
      </section>
      <!-- (2) 桥接气泡图 -->
      <section class="panel">
        <h2>(2) Bridge leverage</h2>
        <p class="panel-desc">Bubble area amplifies reachable scope; colour encodes efficiency (nodes gained per endorsed link). Labels mark the widest bridges.</p>
        <div class="chart-shell">{bridge_slot}</div>
      </section>
      <!-- (3) 继电器链桑基图 -->
      <section class="panel">
        <h2>(3) Relay chains</h2>
        <p class="panel-desc">Predecessors hand trading relationships to successors; thicker bands mean more overlapping partners, signalling possible inheritance of a dormant firm’s counterparties by a revived actor.</p>
        <div class="chart-shell">{sankey_slot}</div>
      </section>
      <!-- (4) 热力图 -->
      <section class="panel">
        <h2>(4) Revival signal matrix</h2>
        <p class="panel-desc">Rows are the highest-ranked suspicious-revival endpoints. Columns summarise complementary cues—extended dormancy, partner novelty, abrupt HS diversification, patched timeline gaps.</p>
        <div class="chart-shell">{heatmap_slot}</div>
      </section>
    </main>
  </div>
</body>
</html>"""
    FINAL_Q3_HTML.write_text(html_out, encoding='utf-8')
    print("Final board written:", FINAL_Q3_HTML)

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # 生成网络对比图
    highlighted, top_bridges, top_relays, normal = load_network_data()
    original = build_original_edges(highlighted, top_bridges, top_relays, normal)
    trusted = build_trusted_edges(highlighted, top_bridges, top_relays, normal)
    network_img_path = FIG_DIR / "Q3_before_after_network.png"
    plot_network_comparison(original, trusted, highlighted, top_bridges, top_relays, normal, save_path=network_img_path)
    
    # 生成热力图（静态表格样式，无网格）
    company_df, delta_df, reliable_links, relay_df, bridge_df = load_data()
    company_view = build_company_view(company_df, delta_df)
    heatmap_img_path = FIG_DIR / "Q3_suspicion_heatmap.png"
    build_suspicion_heatmap_static(company_view, heatmap_img_path)
    
    # 生成最终 HTML（顺序 1-2-3-4）
    write_final_q3_board(company_view, relay_df, bridge_df, reliable_links, network_img_path, heatmap_img_path)
    print("All done. Open", FINAL_Q3_HTML)

if __name__ == "__main__":
    main()