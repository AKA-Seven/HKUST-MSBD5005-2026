# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

VAST Challenge 2023 Mini-Challenge 2（MC2）——基于 Oceanus 海洋贸易知识图谱，识别涉及非法、未报告和无管制捕捞（IUU fishing）的可疑公司。核心数据为 540 万条贸易边 + 12 组预测链接集（bundles）。

## 常用命令

所有 Python 命令使用项目专用解释器：
```bash
# Python 路径
D:\AnacondaEnviroment\envs\python312\python.exe

# 完整分析流水线（Q1-Q4 全部）
python analysis/run_pipeline.py

# 生成 3D 可视化数据文件（输出到 visualization/data/mc2_3d_data.js）
python visualization/build_3d_data.py

# 生成全部 2D 图表
python visualization/build_2d_advanced_figures.py

# 单独生成各题 2D 图表
python visualization/q1/build_q1_figure.py
python visualization/q2/build_q2_figures.py
python visualization/q3/build_q3_figures.py
python visualization/q4/build_q4_figures.py
```

3D 交互页面直接用浏览器打开 `visualization/index.html`（无需服务器）。

## 数据准备

`MC2/` 目录已被 `.gitignore` 排除，需手动将原始数据放入项目根目录：
```
MC2/
├── mc2_challenge_graph.json   # 主知识图谱（约 1.5GB）
└── bundles/                   # 12 个预测链接集（carp/tuna/salmon 等）
```

## 整体架构

项目分为两个独立模块，通过 `outputs/` 目录中的 CSV/JSON 文件传递数据：

```
analysis/（数据挖掘）→ outputs/（中间产物）→ visualization/（图表渲染）
```

### 分析模块（analysis/）

入口为 `run_pipeline.py`，按顺序执行 Q1→Q2→Q3→Q4：

| 步骤 | 文件 | 功能 |
|------|------|------|
| 公共基础 | `shared/config.py` | 全局路径、日期范围、HS编码前缀等常量 |
| 公共基础 | `shared/build_index.py` | 构建节点/边对/HS编码的轻量索引（避免全图重复遍历） |
| Q1 | `q1/extract_temporal_patterns.py` | 从原始图提取公司时序画像，分类为 5 种模式 |
| Q2 | `q2/evaluate_bundles.py` | 规则多维评分（端点覆盖率、HS编码有效性等） |
| Q2 | `q2/link_prediction.py` | 自监督链接预测（2028-2033 训练，2034 验证） |
| Q2 | `q2/score_bundles.py` | 综合规则分和 ML 分，打标签 reliable/suspicious/reject |
| Q2 | `q2/merge_links.py` | 去重并汇总 reliable 链接 |
| Q3 | `q3/detect_anomalies.py` | 对比补全图谱前后的异常变化 |
| Q4 | `q4/company_clustering.py` | 层次聚类 + IsolationForest 异常检测 |
| Q4 | `q4/edge_clustering.py` | 贸易边聚类并赋予语义标签 |

### 可视化模块（visualization/）

- `build_3d_data.py`：读取全部 outputs 产物，聚合成 `data/mc2_3d_data.js`（3D页面所需的唯一数据源，**勿手动修改**）
- `index.html`：Three.js + 3d-force-graph 构建的交互式 3D 网络图，支持路线频度/生命周期/混合三种视图
- `q1/build_q1_figure.py`：生成带时间滑块的圆形两层弦图
- `q2-q4/build_*.py`：生成各题的气泡图、雷达图、Sankey 图、平行坐标等

### 数据流与中间产物

| 输出文件 | 生成来源 | 消费者 |
|----------|---------|--------|
| `outputs/q1/q1_temporal_patterns.json` | Q1 分析 | Q4 公司聚类、3D 数据构建 |
| `outputs/q2/bundle_reliability.csv` | Q2 评分 | Q2 图表、3D 筛选 |
| `outputs/q2/reliable_links.json` | Q2 合并 | Q3 异常对比、Q4 边簇 |
| `outputs/q3/anomaly_delta.csv` | Q3 分析 | Q3 图表、Q4 异常得分 |
| `outputs/q4/company_clusters.csv` | Q4 公司聚类 | Q4 图表、3D 节点属性 |
| `outputs/q4/edge_clusters.csv` | Q4 边簇 | Q4 图表 |

## 核心数据结构

**主图边记录**（JSON）：
```json
{
  "source": "公司A", "target": "公司B",
  "arrivaldate": "2030-03-15", "hscode": "302010",
  "weightkg": 1500.0, "valueofgoods_omu": 50000.0
}
```

**时序模式分类**（Q1，5类）：
- `stable`：覆盖率 >55% 且变异系数 <0.4
- `bursty`：峰值/均值 ≥5 且覆盖率 <45%
- `periodic`：活跃月份间隔存在规律
- `short_term`：生命周期 <12 个月
- `general`：不满足以上条件

**Bundle 综合评分**（Q2）：
```
score = 20×endpoint_ratio + 25×seen_pair_ratio + 20×valid_hscode_ratio
      + 10×physical_field + 10×unique_pair + 10×fish_hscode_ratio
      + 15×ml_probability
      - 25×outside_date_ratio - 惩罚项
```
≥60 → reliable；50-59 → suspicious；<50 → reject

**边簇语义标签**（Q4，6类）：`fish_dense_bridge`、`novel_predicted_route`、`persistent_cross_bundle`、`historical_backbone`、`multi_tool_bridge`、`opportunistic_route`

## 重要常量

在 `analysis/shared/config.py` 中定义，修改任何阈值应在此处统一修改：

- `DEFAULT_DATE_MIN/MAX`：`"2028-01-01"` ～ `"2034-12-30"`（有效日期范围）
- `FISH_HSCODE_PREFIXES`：`("301", "302", ..., "1604", "1605")`（海产品HS编码前缀）
- `LINK_PREDICTION_TRAIN_END`：`"2033-12-31"`（训练集截断）
- `LINK_PREDICTION_SAMPLE_SIZE`：`6000`（链接预测负样本采样数）
- `RANDOM_SEED`：`42`（所有随机操作统一种子）

## 任务定义

详细的分析任务要求见 `Tasks.md`（Q1-Q5）。`template/q1/` 下有可视化需求规格说明书和早期原型，`phase1_sketches/` 是开发阶段的早期草图，均非生产代码。
