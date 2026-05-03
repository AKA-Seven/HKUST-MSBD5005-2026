import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import json
from math import pi
import html

# ========== 路径配置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "outputs", "q4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualization", "figures_2d", "q4")
FINAL_Q4_HTML = os.path.join(PROJECT_ROOT, "final_sketches", "q4.html")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FINAL_Q4_HTML), exist_ok=True)

# ========== 数据加载 ==========
df_rank = pd.read_csv(os.path.join(DATA_DIR, "suspicion_ranking.csv"), encoding='utf-8-sig')
with open(os.path.join(DATA_DIR, "suspects_high.json"), 'r', encoding='utf-8-sig') as f:
    suspects_high = pd.DataFrame(json.load(f))
df_clusters = pd.read_csv(os.path.join(DATA_DIR, "company_clusters.csv"), encoding='utf-8-sig')
df_edges = pd.read_csv(os.path.join(DATA_DIR, "edge_clusters.csv"), encoding='utf-8-sig')

df_rank['company'] = df_rank['company'].astype(str)
suspects_high['company'] = suspects_high['company'].astype(str)
df_clusters['company'] = df_clusters['company'].astype(str)

# ========== 添加缺失的列 ==========
if 'trusted_links' not in df_rank.columns:
    if 'bridge_link_count' in df_rank.columns:
        df_rank['trusted_links'] = df_rank['bridge_link_count'].fillna(0).astype(int)
    else:
        df_rank['trusted_links'] = 0

if 'signal_count' not in df_rank.columns:
    signal_cols_list = ['sig_iso_anomaly', 'sig_revival', 'sig_dormancy', 
                        'sig_q1_inconsistent', 'sig_q1_risk', 'sig_bridge', 'sig_relay']
    existing_signal_cols = [c for c in signal_cols_list if c in df_rank.columns]
    if existing_signal_cols:
        df_rank['signal_count'] = df_rank[existing_signal_cols].gt(0.5).sum(axis=1)
    else:
        df_rank['signal_count'] = 0

def safe_save_fig(fig, filepath, **kwargs):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, **kwargs)

# ========== 特征选择与降维 ==========
feature_cols = ['total_links', 'active_months', 'max_monthly_count', 'avg_monthly_count',
                'partner_count', 'fish_hscode_ratio', 'anomaly_score', 'bridge_scope',
                'sig_iso_anomaly', 'sig_revival', 'sig_dormancy', 'sig_bridge']
available = [f for f in feature_cols if f in df_rank.columns]
X = df_rank[available].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

scatter_color = {'HIGH': '#E63946', 'MEDIUM': '#F4A261', 'LOW': '#88C0A0'}

# ========== 辅助函数：生成 hover 文本 ==========
hover_text = []
for idx, row in df_rank.iterrows():
    hover_text.append(
        f"<b>{row['company']}</b><br>" +
        f"Composite Score: {row['composite_score']:.3f}<br>" +
        f"Confidence: {row['confidence_tier']}<br>" +
        f"Signal Count: {row['signal_count']}<br>" +
        f"Fish HS Ratio: {row['fish_hscode_ratio']:.2%}<br>" +
        f"Bridge Scope: {row['bridge_scope']:.0f}"
    )

# ========== 图1：3D PCA（Plotly）- 标题精简，轴标签正确 ==========
score_min = df_rank['composite_score'].min()
score_max = df_rank['composite_score'].max()
size_range = (2, 6)
def map_size(score):
    return size_range[0] + (size_range[1] - size_range[0]) * (score - score_min) / (score_max - score_min) if score_max > score_min else size_range[0]

fig1 = go.Figure()

low_mask = df_rank['confidence_tier'] == 'LOW'
if low_mask.any():
    sizes_low = [map_size(s) for s in df_rank.loc[low_mask, 'composite_score']]
    fig1.add_trace(go.Scatter3d(
        x=X_pca[low_mask, 0], y=X_pca[low_mask, 1], z=X_pca[low_mask, 2],
        mode='markers',
        marker=dict(size=sizes_low, color=scatter_color['LOW'], opacity=0.6, line=dict(width=0.5, color='darkgray')),
        text=[hover_text[i] for i, is_low in enumerate(low_mask) if is_low],
        hoverinfo='text', name='LOW (background)', showlegend=True
    ))

med_mask = df_rank['confidence_tier'] == 'MEDIUM'
if med_mask.any():
    sizes_med = [map_size(s) for s in df_rank.loc[med_mask, 'composite_score']]
    fig1.add_trace(go.Scatter3d(
        x=X_pca[med_mask, 0], y=X_pca[med_mask, 1], z=X_pca[med_mask, 2],
        mode='markers',
        marker=dict(size=sizes_med, color=scatter_color['MEDIUM'], opacity=0.8,
                    line=dict(width=0.8, color='black'), symbol='circle'),
        text=[hover_text[i] for i, is_med in enumerate(med_mask) if is_med],
        hoverinfo='text', name='MEDIUM Confidence'
    ))

high_mask = df_rank['confidence_tier'] == 'HIGH'
if high_mask.any():
    sizes_high = [map_size(s) for s in df_rank.loc[high_mask, 'composite_score']]
    fig1.add_trace(go.Scatter3d(
        x=X_pca[high_mask, 0], y=X_pca[high_mask, 1], z=X_pca[high_mask, 2],
        mode='markers',
        marker=dict(size=sizes_high, color=scatter_color['HIGH'], opacity=0.9,
                    line=dict(width=1.0, color='white'), symbol='circle'),
        text=[hover_text[i] for i, is_high in enumerate(high_mask) if is_high],
        hoverinfo='text', name='HIGH Confidence'
    ))

total_var = explained_var.sum() * 100
# 更简洁的标题，避免换行异常
title_text = (f"<b>3D PCA: Suspicious Companies</b><br>"
              f"Cumulative variance {total_var:.1f}% (PC1: {explained_var[0]*100:.1f}%, PC2: {explained_var[1]*100:.1f}%, PC3: {explained_var[2]*100:.1f}%)")

fig1.update_layout(
    # 去掉 title 字段，不显示标题
    autosize=True,
    width=None,
    height=600,
    scene=dict(
        xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
        yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
        zaxis_title=f'PC3 ({explained_var[2]*100:.1f}%)',
        camera=dict(
            eye=dict(x=2.2, y=1.8, z=1.8),
            center=dict(x=0, y=0, z=0)
        ),
        bgcolor='#FDF8F0',
        xaxis=dict(gridcolor='#9CAFBF', gridwidth=1.5, showbackground=True,
                   backgroundcolor='#FDF8F0', linecolor='#2C3E50', linewidth=2,
                   title_font=dict(size=12, color='#2C3E50'), tickfont=dict(size=9)),
        yaxis=dict(gridcolor='#9CAFBF', gridwidth=1.5, showbackground=True,
                   backgroundcolor='#FDF8F0', linecolor='#2C3E50', linewidth=2,
                   title_font=dict(size=12, color='#2C3E50'), tickfont=dict(size=9)),
        zaxis=dict(gridcolor='#9CAFBF', gridwidth=1.5, showbackground=True,
                   backgroundcolor='#FDF8F0', linecolor='#2C3E50', linewidth=2,
                   title_font=dict(size=12, color='#2C3E50'), tickfont=dict(size=9))
    ),
    paper_bgcolor='#F7F0E5',
    plot_bgcolor='#F7F0E5',
    # 左右边距设为 20，上下边距设为 30，避免留白过多
    margin=dict(l=20, r=20, b=30, t=30),
    legend=dict(
        title=dict(text='Confidence Level', font=dict(size=11)),
        font=dict(size=10),
        x=0.02, y=0.98,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='black',
        borderwidth=0.5
    ),
    hoverlabel=dict(bgcolor='white', font_size=11)
)

FIG1_HTML = os.path.join(OUTPUT_DIR, "fig1_3d_pca_enhanced.html")
fig1.write_html(FIG1_HTML)

# 添加自动旋转脚本
with open(FIG1_HTML, 'r', encoding='utf-8') as f:
    content = f.read()
rotate_script = """
<script>
(function() {
    let rafId = null;
    const speed = 0.02;
    function rotateScene() {
        const el = document.querySelector('.plotly-graph-div');
        if (el && el._fullLayout && el._fullLayout._scene) {
            const camera = el._fullLayout._scene._scene.getCamera();
            const eye = camera.eye;
            const newX = eye.x * Math.cos(speed) - eye.z * Math.sin(speed);
            const newZ = eye.x * Math.sin(speed) + eye.z * Math.cos(speed);
            camera.eye = {x: newX, y: eye.y, z: newZ};
            el._fullLayout._scene._scene.setCamera(camera);
        }
        rafId = requestAnimationFrame(rotateScene);
    }
    function waitForPlotly() {
        if (document.querySelector('.plotly-graph-div') && window.Plotly) {
            setTimeout(() => {
                if (rafId) cancelAnimationFrame(rafId);
                rafId = requestAnimationFrame(rotateScene);
            }, 500);
        } else {
            setTimeout(waitForPlotly, 200);
        }
    }
    window.addEventListener('load', waitForPlotly);
})();
</script>
"""
content = content.replace('</body>', rotate_script + '\n</body>')
with open(FIG1_HTML, 'w', encoding='utf-8') as f:
    f.write(content)

# ========== 图2：雷达图（静态）- 紧凑布局减少空白 ==========
signal_cols = ['sig_iso_anomaly', 'sig_revival', 'sig_dormancy', 'sig_q1_inconsistent',
               'sig_q1_risk', 'sig_bridge', 'sig_relay', 'sig_fish']
signal_labels = ['Isolation Forest', 'Revival Score', 'Dormancy', 'Q1-Q3 Inconsist.',
                 'Q1 Risk Level', 'Bridge Scope', 'Relay Successor', 'Fish HS Ratio']
existing_signal_cols = [c for c in signal_cols if c in df_rank.columns]
if existing_signal_cols and 'confidence_tier' in df_rank.columns:
    group_means = df_rank.groupby('confidence_tier')[existing_signal_cols].mean().reset_index()
    categories = [signal_labels[signal_cols.index(c)] for c in existing_signal_cols]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig3, ax = plt.subplots(figsize=(8, 6.2), subplot_kw=dict(polar=True), facecolor='white')
    colors_group = {'HIGH': '#e63946', 'MEDIUM': '#f4a261', 'LOW': '#2a9d8f'}
    for _, row in group_means.iterrows():
        tier = row['confidence_tier']
        values = row[existing_signal_cols].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, linestyle='-', label=tier, color=colors_group[tier])
        ax.fill(angles, values, alpha=0.15, color=colors_group[tier])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title('Average Signal Intensity by Confidence Tier\n(0–1 scale)', fontsize=13, pad=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 0.98), fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 0.82, 0.98], pad=0.3)
    FIG3_PNG = os.path.join(OUTPUT_DIR, "fig3_radar_signal_profiles.png")
    safe_save_fig(fig3, FIG3_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig3)
else:
    FIG3_PNG = None

# ========== 图3：Treemap（Plotly）- 去掉副标题，固定高度600，无滚动条（您提供的版本） ==========
if 'cluster_path_3' in df_clusters.columns:
    df_clusters[['L1', 'L2', 'L3']] = df_clusters['cluster_path_3'].str.split('-', expand=True)
    high_set = set(suspects_high['company'])
    df_clusters['is_high'] = df_clusters['company'].isin(high_set)

    l1_data = df_clusters.groupby('L1').agg(count=('company', 'size'), high_count=('is_high', 'sum')).reset_index()
    l1_data = l1_data[l1_data['count'] > 0]
    l1_data['high_ratio'] = l1_data['high_count'] / l1_data['count']
    l1_data['id'] = l1_data['L1']
    l1_data['parent'] = ''

    l2_data = df_clusters.groupby(['L1', 'L2']).agg(count=('company', 'size'), high_count=('is_high', 'sum')).reset_index()
    l2_data = l2_data[l2_data['count'] > 0]
    l2_data['high_ratio'] = l2_data['high_count'] / l2_data['count']
    l2_data['id'] = l2_data['L1'] + '-' + l2_data['L2']
    l2_data['parent'] = l2_data['L1']

    l3_data = df_clusters.groupby(['L1', 'L2', 'L3']).agg(count=('company', 'size'), high_count=('is_high', 'sum')).reset_index()
    l3_data = l3_data[l3_data['count'] > 0]
    l3_data['high_ratio'] = l3_data['high_count'] / l3_data['count']
    l3_data['id'] = l3_data['L1'] + '-' + l3_data['L2'] + '-' + l3_data['L3']
    l3_data['parent'] = l3_data['L1'] + '-' + l3_data['L2']

    treemap_df = pd.concat([
        l1_data[['id', 'parent', 'count', 'high_ratio']],
        l2_data[['id', 'parent', 'count', 'high_ratio']],
        l3_data[['id', 'parent', 'count', 'high_ratio']]
    ], ignore_index=True)

    fig4 = go.Figure(go.Treemap(
        ids=treemap_df['id'],
        labels=treemap_df['id'],
        parents=treemap_df['parent'],
        values=treemap_df['count'],
        branchvalues='total',
        marker=dict(
            colors=treemap_df['high_ratio'],
            colorscale='Reds',
            showscale=True,
            line=dict(width=1.5, color='white'),
            colorbar=dict(
                title=None,
                tickformat=".0%",
                len=0.45,
                thickness=12,
                x=1.02,
                xanchor='left',
                y=0.5,
                yanchor='middle',
                ticks='outside',
                ticklen=4,
                tickfont=dict(size=10)
            )
        ),
        textinfo='label+value+percent parent',
        hovertemplate='<b>%{label}</b><br>Total companies: %{value}<br>HIGH ratio: %{color:.2%}<br>%{percentParent:.1f}% of parent<extra></extra>'
    ))
    fig4.update_layout(
        title=dict(
            text='<b>Hierarchy of Company Clusters (Treemap)</b>',
            font=dict(size=18),
            x=0.5
        ),
        autosize=True,
        height=600,
        margin=dict(t=60, l=20, r=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#132238')
    )
    FIG4_HTML = os.path.join(OUTPUT_DIR, "fig4_treemap_clusters.html")
    fig4.write_html(FIG4_HTML, include_plotlyjs='cdn', config={'responsive': True})
else:
    FIG4_HTML = None

# ========== 准备数据摘要 ==========
total_companies = len(df_rank)
n_high = (df_rank['confidence_tier'] == 'HIGH').sum()
n_medium = (df_rank['confidence_tier'] == 'MEDIUM').sum()
n_low = (df_rank['confidence_tier'] == 'LOW').sum()
top_high_companies = df_rank[df_rank['confidence_tier'] == 'HIGH'].nlargest(5, 'composite_score')['company'].tolist()
top_high_str = ", ".join([f"<strong>{html.escape(c[:35])}</strong>" for c in top_high_companies]) if top_high_companies else "—"

# ========== 准备表格数据（滚动表格 + 行背景色） ==========
table_df = df_rank.sort_values('composite_score', ascending=False).head(50).copy()
table_df['short_name'] = table_df['company'].apply(lambda x: x[:35] + '…' if len(x) > 36 else x)
table_rows = []
for _, row in table_df.iterrows():
    conf = row['confidence_tier']
    if conf == 'HIGH':
        row_class = 'risk-high'
    elif conf == 'MEDIUM':
        row_class = 'risk-medium'
    else:
        row_class = 'risk-low'
    table_rows.append(f"""
        <div class="table-row {row_class}" data-company="{html.escape(row['company'])}">
            <div>{html.escape(row['short_name'])}</div>
            <div><strong>{row['composite_score']:.1f}</strong></div>
            <div>{row['signal_count']}</div>
            <div>{row['trusted_links']}</div>
            <div><span class="row-tag {conf.lower()}">{conf}</span></div>
        </div>
    """)
table_html = f"""
<div class="table-container" style="max-height: 500px; overflow-y: auto;">
    <div class="table">
        <div class="table-header">
            <div>Company</div><div>Score</div><div>Signals</div><div>Trusted Links</div><div>Confidence</div>
        </div>
        <div id="rankingTableBody">{''.join(table_rows)}</div>
    </div>
</div>
"""

company_data_json = df_rank[['company', 'composite_score', 'confidence_tier', 'bridge_scope', 
                             'sig_revival', 'sig_iso_anomaly', 'trusted_links', 'signal_count']].to_dict(orient='records')
for d in company_data_json:
    for k, v in d.items():
        if pd.isna(v):
            d[k] = 0 if isinstance(v, (int, float)) else ""
company_data_json = json.dumps(company_data_json, ensure_ascii=False)

# ========== 生成最终 Dashboard ==========
def write_final_q4_board():
    rel_path_fig1 = os.path.relpath(FIG1_HTML, start=os.path.dirname(FINAL_Q4_HTML))
    rel_path_fig3 = os.path.relpath(FIG3_PNG, start=os.path.dirname(FINAL_Q4_HTML)) if FIG3_PNG else None
    rel_path_fig4 = os.path.relpath(FIG4_HTML, start=os.path.dirname(FINAL_Q4_HTML)) if FIG4_HTML else None

    how_to_read = """
    <div class="panel">
        <h2>How to read the figures</h2>
        <p class="panel-desc">
            Four views reveal suspicious companies from different angles. All data derived from graph completion (Q3) and temporal patterns (Q1).
        </p>
        <div class="chip-row">
            <span class="chip">Fig 1 · 3D PCA (interactive, auto-rotate)</span>
            <span class="chip">Fig 2 · Suspicion ranking table (scrollable)</span>
            <span class="chip">Fig 3 · Radar signal profiles</span>
            <span class="chip">Fig 4 · Cluster treemap (full view)</span>
        </div>
        <div class="insight-text">
            💡 <strong>Key</strong>: Click any row in the ranking table to update evidence and narrative.<br>
            HIGH companies are well-separated in PCA and dominate treemap nodes.
        </div>
    </div>
    """

    signal_summary = f"""
    <div class="panel">
        <div class="section-head"><div><h2>Suspicion tier summary</h2><p class="panel-desc">Distribution of companies by confidence level.</p></div><span class="badge">Q4 scores</span></div>
        <div class="summary"><h3>Company counts</h3><p><strong style="color:#E63946">HIGH:</strong> {n_high} &nbsp;|&nbsp;<strong style="color:#F4A261">MEDIUM:</strong> {n_medium} &nbsp;|&nbsp;<strong style="color:#88C0A0">LOW:</strong> {n_low}</p></div>
        <div class="summary" style="margin-top:12px"><h3>Top‑5 HIGH‑confidence companies</h3><p>{top_high_str}</p></div>
        <div class="insight-text">🔍 HIGH companies exhibit strong signals across isolation forest, revival score, dormancy, bridge scope, and fish HS ratio.</div>
    </div>
    <div class="panel" id="evidencePanel">
        <div class="section-head"><div><h2>Evidence Card</h2><p class="panel-desc">Click a company in the table → see details.</p></div><span class="badge">Drill‑down</span></div>
        <div id="evidenceCard" class="card">Select a company from the table.</div>
    </div>
    <div class="panel" id="narrativePanel">
        <div class="section-head"><div><h2>Case Narrative</h2><p class="panel-desc">Why suspicious, what changed after completion, and limitations.</p></div><span class="badge">Explanation</span></div>
        <div id="caseNarrative" class="card">Select a company from the table.</div>
    </div>
    """

    fig1_slot = f'<div class="plot-slot" style="overflow: hidden; height: 600px; display: flex; justify-content: center;"><iframe src="{rel_path_fig1}" width="100%" height="600px" frameborder="0" style="display: block; border: none; overflow: hidden;" scrolling="no"></iframe></div>'
    fig2_slot = f'<div class="plot-slot">{table_html}</div>'
    if rel_path_fig3:
        fig3_slot = f'<div class="plot-slot"><img src="{rel_path_fig3}" alt="Radar chart" style="width:100%; border-radius:12px;"></div>'
    else:
        fig3_slot = '<div class="plot-slot"><p class="muted">Radar chart data not available.</p></div>'
    fig4_slot = f'<div class="plot-slot" style="overflow: visible;"><iframe src="{rel_path_fig4}" width="100%" height="600px" frameborder="0" style="border-radius:12px; overflow: hidden;" scrolling="no"></iframe></div>' if rel_path_fig4 else '<div class="plot-slot"><p class="muted">Treemap data not available.</p></div>'

    figures_grid = f"""
    <div class="grid-2x2">
        <div class="panel"><div class="section-head"><div><h2>Fig 1 · 3D PCA</h2><p class="panel-desc">Interactive 3D projection. Red = HIGH, orange = MEDIUM, green = LOW. (Auto‑rotating)</p></div><span class="badge">PCA</span></div><div class="chart-shell">{fig1_slot}</div></div>
        <div class="panel"><div class="section-head"><div><h2>Fig 2 · Suspicion Ranking Table</h2><p class="panel-desc">Click any row to see evidence & narrative. Scroll for more rows. Row background = risk level.</p></div><span class="badge">Table</span></div><div class="chart-shell">{fig2_slot}</div></div>
        <div class="panel"><div class="section-head"><div><h2>Fig 3 · Radar profiles</h2><p class="panel-desc">Average signal intensity per confidence tier (0–1).</p></div><span class="badge">Radar</span></div><div class="chart-shell">{fig3_slot}</div></div>
        <div class="panel"><div class="section-head"><div><h2>Fig 4 · Cluster treemap</h2><p class="panel-desc">Hierarchy of company clusters; colour = proportion of HIGH companies.</p></div><span class="badge">Treemap</span></div><div class="chart-shell">{fig4_slot}</div></div>
    </div>
    """

    hero_metrics = f"""
    <div class="meta-grid">
        <div class="metric"><strong>{total_companies}</strong><div class="muted">companies analyzed</div></div>
        <div class="metric"><strong>{n_high}</strong><div class="muted">HIGH confidence</div></div>
        <div class="metric"><strong>{n_medium}</strong><div class="muted">MEDIUM confidence</div></div>
        <div class="metric"><strong>{n_low}</strong><div class="muted">LOW confidence</div></div>
    </div>
    """

    nav_links = """
    <nav class="nav">
        <a href="./q1.html">Q1 · Temporal patterns</a>
        <a href="./q2.html">Q2 · Bundle reliability</a>
        <a href="./q3.html">Q3 · Graph completion</a>
        <a href="./q4.html" class="active">Q4 · Suspicious companies</a>
    </nav>
    """

    CSS = """
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
      border-radius: 999px;
      text-decoration: none;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.64);
      font-size: 13px;
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
      grid-template-columns: 300px minmax(0, 1fr);
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
    .grid-2x2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .panel { padding: 18px; }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .badge {
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(177, 94, 17, 0.10);
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 700;
      border: 1px solid var(--line);
    }
    .panel-desc { margin-bottom: 14px; color: var(--muted); font-size: 13px; }
    .chart-shell {
      padding: 10px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,255,255,0.58)),
                  repeating-linear-gradient(0deg, transparent 0, transparent 30px, rgba(19,34,56,0.04) 30px, rgba(19,34,56,0.04) 31px);
      box-shadow: inset 0 0 80px 44px rgba(255,251,243,0.78), 0 2px 14px rgba(19,34,56,0.06);
    }
    .plot-slot { width: 100%; min-height: auto; overflow: visible; }
    .plot-slot .plotly-graph-div { width: 100% !important; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .chip {
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.74);
      font-size: 13px;
      border: 1px solid var(--line);
    }
    .summary h3 { margin: 0 0 6px; font-size: 14px; }
    .insight-text { font-size: 13px; color: var(--accent); margin-top: 12px; }
    .card {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(23,35,63,0.10);
      background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(255,255,255,0.62));
    }
    .card h3 { margin: 0 0 8px; font-size: 18px; }
    .card p, .card li { color: var(--muted); font-size: 13px; line-height: 1.65; }
    .card ul { margin: 0; padding-left: 18px; }
    /* 表格样式 */
    .table-container {
        border-radius: 18px;
        border: 1px solid rgba(23,35,63,0.09);
    }
    .table {
      background: rgba(255,255,255,0.62);
    }
    .table-header, .table-row {
      display: grid;
      grid-template-columns: 1.45fr 0.7fr 0.8fr 0.9fr 0.9fr;
      gap: 10px;
      align-items: center;
      padding: 12px 14px;
      font-size: 13px;
    }
    .table-header {
      background: rgba(23,35,63,0.06);
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 11px;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .table-row {
      border-top: 1px solid rgba(23,35,63,0.06);
      cursor: pointer;
      transition: background 0.18s ease;
    }
    /* 风险等级整行背景色 */
    .table-row.risk-high { background-color: #fff3cd; } 
    .table-row.risk-medium { background-color: #e9ecef; }
    .table-row.risk-low { background-color: #d4edda; }
    .table-row:hover { filter: brightness(0.96); }
    /* 选中行高亮默认浅红 */
    .table-row.active { 
        background-color: #f8d7da !important; 
    }
    .row-tag {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      width: fit-content;
      background: rgba(255,255,255,0.8);
      border: 1px solid var(--line);
    }
    .row-tag.high { color: #842c4b; }
    .row-tag.mid { color: #b45f16; }
    .row-tag.low { color: #1c6b63; }
    @media (max-width: 1100px) {
      .dashboard-grid { grid-template-columns: 1fr; }
      .grid-2x2 { grid-template-columns: 1fr; }
    }
    @media (max-width: 900px) {
      .page { padding: 18px; }
      .hero { grid-template-columns: 1fr; }
      h1 { font-size: 26px; }
    }
    """

    js_code = f"""
    <script>
        const companyData = {company_data_json};
        let currentCompany = companyData.length ? companyData[0]['company'] : '';

        function updateEvidence(companyName) {{
            const company = companyData.find(c => c.company === companyName);
            if (!company) return;
            document.getElementById('evidenceCard').innerHTML = `
                <h3>${{company.company.substring(0,50)}} · Evidence Card</h3>
                <p><strong>Composite Score:</strong> ${{company.composite_score.toFixed(2)}}</p>
                <ul>
                    <li><strong>Bridge scope:</strong> ${{company.bridge_scope}}</li>
                    <li><strong>Revival signal:</strong> ${{(company.sig_revival*100).toFixed(0)}}%</li>
                    <li><strong>Isolation Forest anomaly:</strong> ${{(company.sig_iso_anomaly*100).toFixed(0)}}%</li>
                    <li><strong>Trusted links involved:</strong> ${{company.trusted_links}}</li>
                    <li><strong>Signal count:</strong> ${{company.signal_count}}</li>
                </ul>
            `;
            document.getElementById('caseNarrative').innerHTML = `
                <h3>Case Narrative</h3>
                <p><strong>Why suspicious:</strong> ${{company.sig_revival > 0.6 ? 'High revival signal, reconnected after graph completion.' : 'Moderate multi-signal anomaly.'}}</p>
                <p><strong>What changed after completion:</strong> Bridge scope expanded to ${{company.bridge_scope}} nodes, involving ${{company.trusted_links}} trusted edges.</p>
                <p><strong>Confidence and limitations:</strong> ${{company.confidence_tier}} confidence. Requires manual verification of business semantics.</p>
            `;
        }}

        function bindTableRows() {{
            document.querySelectorAll('.table-row').forEach(row => {{
                row.addEventListener('click', () => {{
                    const company = row.dataset.company;
                    if (company) {{
                        currentCompany = company;
                        updateEvidence(company);
                        document.querySelectorAll('.table-row').forEach(r => r.classList.remove('active'));
                        row.classList.add('active');
                    }}
                }});
            }});
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            bindTableRows();
            if (companyData.length) {{
                updateEvidence(companyData[0].company);
                const firstRow = document.querySelector('.table-row');
                if (firstRow) firstRow.classList.add('active');
            }}
        }});
    </script>
    """

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q4 · Suspicious companies · Final board</title>
    <style>{CSS}</style>
</head>
<body>
<div class="page">
    <section class="hero"><div><div class="eyebrow">Final board · Q4 · Suspicious companies</div><h1>Multidimensional detection of suspicious trade actors</h1><p class="muted">3D PCA separates HIGH‑confidence companies, radar confirms signal dominance, treemap shows cluster concentration — and the ranking table enables drill‑down.</p></div>{hero_metrics}</section>
    {nav_links}
    <div class="dashboard-grid">
        <aside class="sidebar-stack">{signal_summary}</aside>
        <main class="main-flow">{figures_grid}</main>
    </div>
</div>
{js_code}
</body>
</html>"""

    with open(FINAL_Q4_HTML, 'w', encoding='utf-8') as f:
        f.write(html_out)
    print(f"Final Q4 dashboard with optimized 3D PCA (smaller dots) and your treemap style written: {FINAL_Q4_HTML}")

def main():
    write_final_q4_board()
    print("All Q4 charts and dashboard generated.")

if __name__ == "__main__":
    main()