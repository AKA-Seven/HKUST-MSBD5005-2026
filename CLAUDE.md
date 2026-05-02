# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

VAST Challenge 2023 Mini-Challenge 2（MC2）——基于 Oceanus 海洋贸易知识图谱，识别涉及非法、未报告和无管制捕捞（IUU fishing）的可疑公司。核心数据为 540 万条贸易边 + 12 组预测链接集（bundles）。

完整任务定义见 [`Tasks.md`](Tasks.md)。

## 常用命令

推荐使用 conda 环境 `dv`（与脚本注释一致），无需先 `activate`：

```powershell
# 完整分析流水线（Q1→Q4，写入 outputs/q1…q4）
conda run -n dv python analysis/run_pipeline.py

# 各题 2D / 3D 可视化（按需运行，每个脚本独立产出 HTML/PNG）
conda run -n dv python visualization/q1/build_q1_bubble_scatter.py
conda run -n dv python visualization/q1/build_q1_monthly_heatmap.py
conda run -n dv python visualization/q1/build_q1_ridge_river.py
conda run -n dv python visualization/q1/build_q1_relationship_chord_data.py

conda run -n dv python visualization/q2/build_q2_figures.py
conda run -n dv python visualization/q2/build_q2_bundle_dashboard.py
conda run -n dv python visualization/q2/build_q2_bundle_reliability_bubble.py
conda run -n dv python visualization/q2/build_q2_dandelion_3d.py
conda run -n dv python visualization/q2/build_q2_trade_sankey_q1_q2.py

conda run -n dv python visualization/q3/build_q3_figures.py
conda run -n dv python visualization/q4/build_q4_figures.py
```

Final 展示页直接用浏览器打开（无需服务器）：
- `final_sketches/q1.html` / `q2.html` / `q3.html` / `q4.html`

依赖见 [`requirements.txt`](requirements.txt)（pandas / numpy / scikit-learn / matplotlib / seaborn / plotly 等）。

## 数据准备

`MC2/` 已被 `.gitignore` 排除，需手动放入项目根目录：

```
MC2/
├── mc2_challenge_graph.json   # 主知识图谱（约 1.5 GB）
└── bundles/                   # 12 个预测链接集（carp / tuna / salmon / …）
```

## 整体架构

项目分为两个独立模块，通过 `outputs/` 目录中的 CSV/JSON 文件解耦：

```
analysis/（数据挖掘）→ outputs/（中间产物）→ visualization/ + final_sketches/（图表与终稿）
```

**注意**：原先用于驱动统一 3D 视图的 `visualization/build_3d_data.py`、`visualization/index.html`、`visualization/data/mc2_3d_data.js`、`visualization/figures2d_common.py`、`visualization/build_2d_advanced_figures.py` 已在最新版本中被移除——3D 可视化转向各题独立实现（如 `visualization/q2/build_q2_dandelion_3d.py` 输出 `figures_3d/q2/q2_dandelion_graph.html`）。

### 分析模块（analysis/）

入口为 [`run_pipeline.py`](analysis/run_pipeline.py)，按顺序执行 Q1→Q2→Q3→Q4。该脚本会自动把 `shared/`、`q1/`…`q4/` 加入 `sys.path`，所以子模块之间使用无包名直接 `import`。

| 步骤 | 文件 | 功能 |
|------|------|------|
| 公共 | `shared/config.py` | 全局路径、日期范围、HS 编码前缀、随机种子 |
| 公共 | `shared/load_data.py` | 加载主图 JSON 与 bundles |
| 公共 | `shared/build_index.py` | 节点 / 边对 / 日期 / HS 索引（避免全图重复遍历） |
| 公共 | `shared/export_results.py` | 统一写出 CSV / JSON |
| Q1 | `q1/extract_temporal_patterns.py` | 公司时序画像，分类为 5 种模式 |
| Q1 | `q1/extract_relationship_patterns.py` | 公司间关系画像 |
| Q1 | `q1/check_relay.py` | 中继链路辅助检查 |
| Q2 | `q2/evaluate_bundles.py` | 规则多维评分（端点覆盖、HS 有效性等） |
| Q2 | `q2/link_prediction.py` | 自监督链接预测（2028-2033 训练，2034 验证） |
| Q2 | `q2/score_bundles.py` | 综合规则分与 ML 分，打标签 reliable / suspicious / reject |
| Q2 | `q2/merge_links.py` | 去重并汇总 reliable 链接 |
| Q3 | `q3/detect_anomalies.py` | `compare_anomalies` / `detect_bridge_companies` / `detect_relay_chains` |
| Q3 | `q3/analyze_graph_diff.py` | 图谱补全前后差分分析 |
| Q4 | `q4/company_clustering.py` | 层次聚类 + IsolationForest |
| Q4 | `q4/edge_clustering.py` | 贸易边聚类并赋予语义标签 |
| Q4 | `q4/synthesize_suspicion.py` | 多信号融合的可疑度排名 |

### 可视化模块（visualization/ 与 final_sketches/）

- **`visualization/q1/`** — 4 套独立可视化构建：
  - `build_q1_bubble_scatter.py`：活动 / 伙伴气泡散点
  - `build_q1_monthly_heatmap.py`：Top50 公司月度热图
  - `build_q1_ridge_river.py`：Ridge / River 时序流
  - `build_q1_relationship_chord_data.py` + `q1_relationship_chord_time.html`：带时间轴的关系弦图
- **`visualization/q2/`** — bundle 可靠性多视图：
  - `build_q2_figures.py`：基础图表
  - `build_q2_bundle_dashboard.py`：总览 dashboard
  - `build_q2_bundle_reliability_bubble.py`：可靠性气泡
  - `build_q2_trade_sankey_q1_q2.py`：贸易 Sankey
  - `build_q2_dandelion_3d.py`：3D 蒲公英图（输出到 `figures_3d/q2/`）
- **`visualization/q3/`** — `build_q3_figures.py` 生成 relay sankey、桥接公司、月度新增链接、可疑热图等；说明见 [`visualization/q3/Q3.md`](visualization/q3/Q3.md)
- **`visualization/q4/`** — `build_q4_figures.py` 生成 PCA 椭圆、3D PCA、雷达画像、treemap；说明见 [`visualization/q4/Q4.md`](visualization/q4/Q4.md)
- **`final_sketches/q1.html` … `q4.html`** — 每题的最终展示页（静态 HTML，浏览器直开）；说明见 [`final_sketches/README.md`](final_sketches/README.md)

### 数据流与中间产物

| 输出文件 | 生成来源 | 消费者 |
|----------|---------|--------|
| `outputs/q1/q1_temporal_patterns.json` | Q1 分析 | Q4 公司聚类、Q1/Q3 可视化 |
| `outputs/q1/q1_relationship_patterns*.{csv,json}` | Q1 分析 | Q1 弦图 |
| `outputs/q2/bundle_reliability.csv` | Q2 评分 | Q2 图表 |
| `outputs/q2/reliable_links.json` | Q2 合并 | Q3 异常对比、Q4 边簇 |
| `outputs/q3/anomaly_delta.csv`、`bridge_companies.csv`、`relay_chains.csv`、`top_*.json` | Q3 分析 | Q3 图表、Q4 异常得分 |
| `outputs/q4/company_clusters.csv`、`suspicion_ranking.csv`、`suspects_*.json` | Q4 分析 | Q4 图表 |
| `outputs/q4/edge_clusters.csv` | Q4 边簇 | Q4 图表 |

注：`outputs/q1/` 与 `visualization/figures_2d/` 等大体积产物已在 `.gitignore` 中排除，需要本地通过 `run_pipeline.py` + 各 build 脚本再生。

## 核心数据结构

**主图边记录（JSON，主图与 bundles schema 一致）：**

```json
{
  "source": "公司A", "target": "公司B",
  "arrivaldate": "2030-03-15", "hscode": "302010",
  "weightkg": 1500.0, "valueofgoods_omu": 50000.0
}
```

**Q1 时序模式分类（5 类）：**
- `stable`：覆盖率 >55% 且变异系数 <0.4
- `bursty`：峰值/均值 ≥5 且覆盖率 <45%
- `periodic`：活跃月份间隔存在规律
- `short_term`：生命周期 <12 个月
- `general`：以上皆不满足

**Q2 bundle 综合评分：**

```
score = 20×endpoint_ratio + 25×seen_pair_ratio + 20×valid_hscode_ratio
      + 10×physical_field + 10×unique_pair + 10×fish_hscode_ratio
      + 15×ml_probability
      - 25×outside_date_ratio - 惩罚项
```

≥60 → reliable；50–59 → suspicious；<50 → reject。

**Q4 边簇语义标签（6 类）：** `fish_dense_bridge`、`novel_predicted_route`、`persistent_cross_bundle`、`historical_backbone`、`multi_tool_bridge`、`opportunistic_route`。

## 重要常量（`analysis/shared/config.py`）

修改任何阈值都应在此处统一调整：

- `DEFAULT_DATE_MIN/MAX`：`"2028-01-01"` ~ `"2034-12-30"`
- `FISH_HSCODE_PREFIXES`：`("301", "302", …, "1604", "1605")`
- `LINK_PREDICTION_TRAIN_END`：`"2033-12-31"`
- `LINK_PREDICTION_VALID_START/END`：`"2034-01-01"` / `"2034-12-31"`
- `LINK_PREDICTION_SAMPLE_SIZE`：`6000`
- `RANDOM_SEED`：`42`（所有随机操作统一种子）

## 开发约定

- **导入路径**：`run_pipeline.py` 启动时已把 `analysis/{shared,q1,q2,q3,q4}` 注入 `sys.path`，子模块之间直接 `from extract_temporal_patterns import …` 即可，不要写包名前缀。
- **大文件**：`MC2/mc2_challenge_graph.json` 约 1.5 GB，调试时避免反复整文件读入；优先复用 `build_index.py` 提供的索引。
- **生成产物勿手改**：`outputs/` 与 `visualization/figures_2d/`、`visualization/figures_3d/` 下的文件由脚本生成，提交前可重跑确保一致。
- **任务定义**：见 [`Tasks.md`](Tasks.md)（Q1–Q5 完整描述）。`template/` 与 `phase1_sketches/` 在最新版本中已被移除或不再用于生产。
