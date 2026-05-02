# 可视化产物索引

本目录汇集 MC2 项目 Q1–Q4 全部图表脚本与产物。所有图均严格基于 `outputs/` 下的分析结果生成，无模拟数据。每题在 [`final_sketches/`](../final_sketches/) 下还有一个统一展示页（`q1.html`…`q4.html`）。

## 目录结构

```
visualization/
├── q1/  q2/  q3/  q4/        # 各题图表构建脚本（Python）
├── figures_2d/q1…q4/         # 2D 图表产物（HTML 交互版 + PNG 静态版）
└── figures_3d/q2/            # 3D 图表产物（HTML）
```

> **注意**：`figures_2d/q1/*.png`、与 HTML 同名的 `*_data.json`、`q1/data/` 等大文件已在 `.gitignore` 中排除，需本机跑构建脚本再生（细节见 [`q1/README.md`](q1/README.md)、[`q2/README.md`](q2/README.md)）。

---

## Q1 时序模式（`figures_2d/q1/`）

| 文件 | 类型 | 含义 |
|------|------|------|
| `q1_bubble_activity_partners.html` | 气泡散点图 | 横轴=活跃月份数，纵轴=贸易伙伴数，气泡大小=总贸易量；用以一眼区分**长期稳定贸易方**与**短期爆发型公司**，并按 5 类时序模式着色（stable / bursty / periodic / short_term / general）。 |
| `q1_monthly_heatmap_top50.html` | 月度热图 | 行=Top50 公司，列=2028-01～2034-12 的逐月时间轴，颜色深浅=当月贸易边数；揭示**活跃节奏**（持续型 / 季节型 / 突发型 / 短命型）和**集体停活窗口**。 |
| `q1_ridge_river_top50.html` | Ridge / River 流图 | 多条堆叠面积曲线沿时间轴展开，每条对应一家 Top50 公司的月度活动量；强调公司之间的**相对消长**与**生命周期错位**，便于发现"接力替代"的疑似时间点。 |

> Q1 还提供独立交互弦图：[`q1/q1_relationship_chord_time.html`](q1/q1_relationship_chord_time.html)（带时间轴的两层关系弦图，需先跑 `build_q1_relationship_chord_data.py` 生成数据）。

---

## Q2 Bundle 可靠性（`figures_2d/q2/` + `figures_3d/q2/`）

| 文件 | 类型 | 含义 |
|------|------|------|
| `figures_2d/q2/q2_bundle_dashboard.html` | 多视图仪表盘 | 把 12 个 bundle 的规则评分、ML 概率、端点覆盖率、HS 有效率、日期越界率等指标横向并列，配 reliable / suspicious / reject 标签；用作 **Q2 主答卷视图**。 |
| `figures_2d/q2/q2_bundle_reliability_bubble.html` | 可靠性气泡图 | 横轴=综合得分，纵轴=ML 概率，气泡大小=链接数，颜色=可靠性标签；直观看出**哪些 bundle 同时被规则与机器学习认可**。 |
| `figures_3d/q2/q2_dandelion_graph.html` | 3D 蒲公英图 | 每个 bundle 是花心，向外辐射自己新增的预测边；不同 bundle 用色相区分，可旋转 / 缩放，用于展示**可靠 bundle 在网络中的扩张范围**与重叠程度。 |

---

## Q3 图谱补全后的新模式（`figures_2d/q3/`）

详细规格见 [`q3/Q3.md`](q3/Q3.md)。

| 文件 | 类型 | 含义 |
|------|------|------|
| `q3_01_relay_sankey_clean.html` / `.png` | 桑基图 | 左=前驱公司，右=后继公司，连线宽度=共享贸易伙伴数；揭示**"接力传承"**模式——前驱沉寂 15–49 个月后由后继继承伙伴网络，疑似"换壳"经营。 |
| `q3_02_bridge_influence.png` | 桥接气泡图 | 横轴=新增链接数，纵轴=可达桥接范围，气泡大小=影响力；定位**靠少量补全边即撬动大范围网络的"杠杆桥接公司"**。 |
| `q3_03_monthly_new_links.png` | 月度趋势线 | 时间轴上展示补全前后每月新增可信链接数；标出**异常激增窗口**，用于解释"图谱补全后突然显现的活跃期"。 |
| `q3_04_suspicious_heatmap.png` | 可疑度热图 | 行=公司，列=月份，颜色=补全后新增的可疑度增量（suspicious_revival_score）；快速定位**沉寂—复苏型**可疑公司。 |

---

## Q4 可疑公司识别（`figures_2d/q4/`）

详细规格见 [`q4/Q4.md`](q4/Q4.md)。

| 文件 | 类型 | 含义 |
|------|------|------|
| `fig1_2d_pca_ellipses.png` | 2D PCA + 置信椭圆 | 公司在前两个主成分平面的位置，HIGH / MEDIUM / LOW 风险三组各画 95% 置信椭圆；显示**风险等级在特征空间中的可分性**。 |
| `fig1_3d_pca_enhanced.html` | 3D PCA 散点 | 把上图扩展到第三主成分，可旋转交互；HIGH 簇明显与主群分离，**佐证多维特征下可疑公司形成独立子空间**。 |
| `fig3_radar_signal_profiles.png` | 雷达画像图 | 多维信号轴（异常分、桥接度、接力度、节奏突变、HS 异常等）上画三组均值多边形；展示**HIGH 组在哪几个信号维度显著拉高**，构成"四重证据链"。 |
| `fig4_treemap_clusters.html` | 矩形树图 | 按聚类层级（风险等级 → 簇 → 公司）分块，矩形大小=可疑度得分，颜色=簇标签；用作**Top 可疑公司榜单的层级总览**。 |

---

## 重新生成全部图表

前置：本机已放置 `MC2/` 原始数据并运行过 `analysis/run_pipeline.py`（产生 `outputs/q1…q4`）。

```powershell
# Q1
conda run -n dv python visualization/q1/build_q1_bubble_scatter.py
conda run -n dv python visualization/q1/build_q1_monthly_heatmap.py
conda run -n dv python visualization/q1/build_q1_ridge_river.py
conda run -n dv python visualization/q1/build_q1_relationship_chord_data.py

# Q2
conda run -n dv python visualization/q2/build_q2_figures.py
conda run -n dv python visualization/q2/build_q2_bundle_dashboard.py
conda run -n dv python visualization/q2/build_q2_bundle_reliability_bubble.py
conda run -n dv python visualization/q2/build_q2_dandelion_3d.py
conda run -n dv python visualization/q2/build_q2_trade_sankey_q1_q2.py

# Q3 / Q4
conda run -n dv python visualization/q3/build_q3_figures.py
conda run -n dv python visualization/q4/build_q4_figures.py
```

跑完后用浏览器直接打开 `final_sketches/q*.html` 或 `figures_2d/`、`figures_3d/` 下的单图 HTML 即可（无需启动服务器）。
