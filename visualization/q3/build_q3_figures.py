#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "q3"
FIG_DIR = PROJECT_ROOT / "visualization" / "figures_2d" / "q3"

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

def draw_relay_sankey(relay_df, out_dir):
    if relay_df.empty:
        print("警告: 未找到接力数据，跳过桑基图")
        return

    filtered = relay_df[relay_df['shared_partner_count'] >= 40].copy()
    if filtered.empty:
        filtered = relay_df.nlargest(20, 'shared_partner_count')

    succ_total = filtered.groupby('successor')['shared_partner_count'].sum().sort_values(ascending=False)
    top_succ = succ_total.head(20).index.tolist()
    filtered = filtered[filtered['successor'].isin(top_succ)]

    if filtered.empty:
        print("警告: 过滤后无数据，跳过桑基图")
        return

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
            hovertemplate='%{source.customdata} → %{target.customdata}<br>共享伙伴数: %{customdata}<extra></extra>'
        )
    )])

    fig.update_layout(
        title=dict(text="Relay Chains (Shared Partners ≥ 40, Sorted by Relevance)", font=dict(size=16), x=0.5),
        font=dict(size=10), height=700, width=1000,
        margin=dict(l=80, r=80, t=60, b=40)
    )
    fig.write_html(out_dir / "q3_01_relay_sankey_clean.html", include_plotlyjs=True)
    try:
        fig.write_image(out_dir / "q3_01_relay_sankey_clean.png", scale=3)
    except Exception as e:
        print(f"PNG 导出失败（需安装 kaleido）: {e}")

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

    print("Q3 四张图表已生成至:", FIG_DIR)

if __name__ == "__main__":
    main()