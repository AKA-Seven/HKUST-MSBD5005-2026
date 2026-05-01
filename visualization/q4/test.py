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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "outputs", "q4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualization", "figures_2d", "q4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_rank = pd.read_csv(os.path.join(DATA_DIR, "suspicion_ranking.csv"), encoding='utf-8-sig')
with open(os.path.join(DATA_DIR, "suspects_high.json"), 'r', encoding='utf-8-sig') as f:
    suspects_high = pd.DataFrame(json.load(f))
df_clusters = pd.read_csv(os.path.join(DATA_DIR, "company_clusters.csv"), encoding='utf-8-sig')
df_edges = pd.read_csv(os.path.join(DATA_DIR, "edge_clusters.csv"), encoding='utf-8-sig')

df_rank['company'] = df_rank['company'].astype(str)
suspects_high['company'] = suspects_high['company'].astype(str)
df_clusters['company'] = df_clusters['company'].astype(str)

def safe_save_fig(fig, filepath, **kwargs):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, **kwargs)

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

UNIFORM_SIZE_3D = 5
fig1 = go.Figure()
low_mask = df_rank['confidence_tier'] == 'LOW'
if low_mask.any():
    fig1.add_trace(go.Scatter3d(
        x=X_pca[low_mask, 0], y=X_pca[low_mask, 1], z=X_pca[low_mask, 2],
        mode='markers',
        marker=dict(size=UNIFORM_SIZE_3D, color=scatter_color['LOW'], opacity=0.5, line=dict(width=0)),
        text=[hover_text[i] for i, is_low in enumerate(low_mask) if is_low],
        hoverinfo='text', name='LOW (background)', showlegend=True
    ))
med_mask = df_rank['confidence_tier'] == 'MEDIUM'
if med_mask.any():
    fig1.add_trace(go.Scatter3d(
        x=X_pca[med_mask, 0], y=X_pca[med_mask, 1], z=X_pca[med_mask, 2],
        mode='markers',
        marker=dict(size=UNIFORM_SIZE_3D, color=scatter_color['MEDIUM'], opacity=0.7,
                    line=dict(width=0.6, color='black'), symbol='circle'),
        text=[hover_text[i] for i, is_med in enumerate(med_mask) if is_med],
        hoverinfo='text', name='MEDIUM Confidence'
    ))
high_mask = df_rank['confidence_tier'] == 'HIGH'
if high_mask.any():
    fig1.add_trace(go.Scatter3d(
        x=X_pca[high_mask, 0], y=X_pca[high_mask, 1], z=X_pca[high_mask, 2],
        mode='markers',
        marker=dict(size=UNIFORM_SIZE_3D, color=scatter_color['HIGH'], opacity=0.9,
                    line=dict(width=1.2, color='black'), symbol='circle'),
        text=[hover_text[i] for i, is_high in enumerate(high_mask) if is_high],
        hoverinfo='text', name='HIGH Confidence'
    ))
fig1.update_layout(
    title=dict(text='<b>3D PCA: Suspicious Companies Are Well‑Separated</b>', font=dict(size=20), x=0.5),
    scene=dict(
        xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
        yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
        zaxis_title=f'PC3 ({explained_var[2]*100:.1f}%)',
        camera=dict(eye=dict(x=2.0, y=1.6, z=1.1)),
        bgcolor='#F0F4F8',
        xaxis=dict(gridcolor='#9CAFBF', gridwidth=1.5, showbackground=True,
                   backgroundcolor='#F0F4F8', linecolor='#2C3E50', linewidth=2,
                   title_font=dict(size=14, color='#2C3E50')),
        yaxis=dict(gridcolor='#9CAFBF', gridwidth=1.5, showbackground=True,
                   backgroundcolor='#F0F4F8', linecolor='#2C3E50', linewidth=2,
                   title_font=dict(size=14, color='#2C3E50')),
        zaxis=dict(gridcolor='#9CAFBF', gridwidth=1.5, showbackground=True,
                   backgroundcolor='#F0F4F8', linecolor='#2C3E50', linewidth=2,
                   title_font=dict(size=14, color='#2C3E50'))
    ),
    width=1100, height=800, margin=dict(l=0, r=120, b=0, t=80), paper_bgcolor='white',
    legend=dict(title=dict(text='Confidence Level', font=dict(size=14)), font=dict(size=13),
                x=1.02, y=0.98, xanchor='left', bgcolor='rgba(255,255,255,0.85)',
                bordercolor='black', borderwidth=0.8, itemsizing='constant'),
    hoverlabel=dict(bgcolor='white', font_size=11)
)
fig1.write_html(os.path.join(OUTPUT_DIR, "fig1_3d_pca_enhanced.html"))

pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

def plot_confidence_ellipse(x, y, ax, color, n_std=2.0):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = mpatches.Ellipse(xy=(mean_x, mean_y), width=width, height=height,
                               angle=angle, edgecolor=color, facecolor='none', lw=2, linestyle='--')
    ax.add_patch(ellipse)

fig2d, ax = plt.subplots(figsize=(10, 8), facecolor='#F8F9FA')
ax.set_facecolor('#F8F9FA')
UNIFORM_SIZE_2D = 112
if low_mask.any():
    ax.scatter(X_pca2[low_mask, 0], X_pca2[low_mask, 1],
               s=UNIFORM_SIZE_2D, c='#88C0A0', alpha=0.5, edgecolors='none', label='LOW (background)')
if med_mask.sum() >= 3:
    x_med = X_pca2[med_mask, 0]
    y_med = X_pca2[med_mask, 1]
    ax.scatter(x_med, y_med, s=UNIFORM_SIZE_2D, c='#F4A261', alpha=0.7, edgecolors='black', linewidth=0.8, label='MEDIUM')
    plot_confidence_ellipse(x_med, y_med, ax, '#F4A261', n_std=2.0)
if high_mask.sum() >= 3:
    x_high = X_pca2[high_mask, 0]
    y_high = X_pca2[high_mask, 1]
    ax.scatter(x_high, y_high, s=UNIFORM_SIZE_2D, c='#E63946', alpha=0.9, edgecolors='black', linewidth=1.2, label='HIGH')
    plot_confidence_ellipse(x_high, y_high, ax, '#E63946', n_std=2.0)
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('2D PCA Projection with 95% Confidence Ellipses\n(All points uniform size = 112)', fontsize=14)
ax.legend(loc='best', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.4, color='#D1D5DB')
plt.tight_layout()
safe_save_fig(fig2d, os.path.join(OUTPUT_DIR, "fig1_2d_pca_ellipses.png"), dpi=300, bbox_inches='tight')
plt.close(fig2d)

signal_cols = ['sig_iso_anomaly', 'sig_revival', 'sig_dormancy', 'sig_q1_inconsistent',
               'sig_q1_risk', 'sig_bridge', 'sig_relay', 'sig_fish']
signal_labels = ['Isolation Forest', 'Revival Score', 'Dormancy', 'Q1-Q3 Inconsist.',
                 'Q1 Risk Level', 'Bridge Scope', 'Relay Successor', 'Fish HS Ratio']
group_means = df_rank.groupby('confidence_tier')[signal_cols].mean().reset_index()
categories = signal_labels
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig3, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor='white')
colors_group = {'HIGH': '#e63946', 'MEDIUM': '#f4a261', 'LOW': '#2a9d8f'}
for _, row in group_means.iterrows():
    tier = row['confidence_tier']
    values = row[signal_cols].values.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2.5, linestyle='-', label=tier, color=colors_group[tier])
    ax.fill(angles, values, alpha=0.15, color=colors_group[tier])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title('Average Signal Intensity by Confidence Tier', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=10)
plt.tight_layout()
safe_save_fig(fig3, os.path.join(OUTPUT_DIR, "fig3_radar_signal_profiles.png"), dpi=300, bbox_inches='tight')
plt.close(fig3)

if 'cluster_path_3' in df_clusters.columns:
    df_clusters[['L1', 'L2', 'L3']] = df_clusters['cluster_path_3'].str.split('-', expand=True)
    high_set = set(suspects_high['company'])
    df_clusters['is_high'] = df_clusters['company'].isin(high_set)

    l1_data = df_clusters.groupby('L1').agg(count=('company', 'size'), high_count=('is_high', 'sum')).reset_index()
    l1_data['high_ratio'] = l1_data['high_count'] / l1_data['count']
    l1_data['id'] = l1_data['L1']
    l1_data['parent'] = ''

    l2_data = df_clusters.groupby(['L1', 'L2']).agg(count=('company', 'size'), high_count=('is_high', 'sum')).reset_index()
    l2_data['high_ratio'] = l2_data['high_count'] / l2_data['count']
    l2_data['id'] = l2_data['L1'] + '-' + l2_data['L2']
    l2_data['parent'] = l2_data['L1']

    l3_data = df_clusters.groupby(['L1', 'L2', 'L3']).agg(count=('company', 'size'), high_count=('is_high', 'sum')).reset_index()
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
            colorbar=dict(title="Proportion of HIGH Companies", tickformat=".0%")
        ),
        textinfo='label+value+percent parent',
        hovertemplate='<b>%{label}</b><br>Total companies: %{value}<br>HIGH ratio: %{color:.2%}<br>%{percentParent:.1%} of parent<extra></extra>'
    ))
    fig4.update_layout(
        title=dict(
            text='<b>Hierarchy of Company Clusters (Treemap)</b><br><span style="font-size:14px">Rectangle size = number of companies | Color = proportion of HIGH companies (Reds)</span>',
            font=dict(size=16)
        ),
        width=1200, height=900, margin=dict(t=100, l=0, r=0, b=0)
    )
    fig4.write_html(os.path.join(OUTPUT_DIR, "fig4_treemap_clusters.html"))

print(f"q4 finished")
print(df_rank.groupby('confidence_tier')[['sig_fish', 'sig_dormancy', 'sig_revival']].mean())