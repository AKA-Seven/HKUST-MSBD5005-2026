# Outputs 数据产物说明

> 由 `analysis/run_pipeline.py` 自动生成，作为可视化模块（`visualization/`）的唯一数据源。

整个 outputs 目录按 Q1-Q4 拆分子目录。所有 CSV 文件使用 `utf-8-sig` 编码（带 BOM），可直接用 Excel 打开；JSON 文件使用 `utf-8` 无 BOM，缩进 2 格。

---

## 目录概览

```
outputs/
├── q1/   时序模式发现
│   ├── q1_temporal_patterns.json          公司时序画像（1371 家）
│   ├── q1_relationship_patterns.csv       公司对关系（2021 对）
│   └── q1_relationship_patterns_top.json  按模式分组 Top-50 预览
│
├── q2/   预测链接可靠性评估
│   ├── bundle_reliability.csv             12 个 bundle 综合评分
│   └── reliable_links.json                394 条可靠预测链接
│
├── q3/   图谱补全前后异常对比
│   ├── anomaly_delta.csv                  486 家公司复活评分
│   ├── top_anomaly_delta.json             Top-100 复活公司
│   ├── bridge_companies.csv               457 家桥接公司
│   ├── top_bridge_companies.json          Top-50 桥接公司
│   ├── relay_chains.csv                   300 条接力链
│   ├── top_relay_chains.json              Top-100 接力链
│   └── graph_diff_analysis.json           补全前后多维对比汇总
│
└── q4/   可疑公司识别
    ├── company_clusters.csv               1371 家公司聚类 + 异常分
    ├── top_company_anomalies.json         Top-100 异常公司
    ├── edge_clusters.csv                  363 条边聚类 + 语义标签
    ├── top_edge_clusters.json             Top-300 边
    ├── suspicion_ranking.csv              ★ Q4 最终交付：1371 家可疑度排名
    ├── suspects_high.json                 25 家 HIGH 置信度
    ├── suspects_medium.json               66 家 HIGH+MEDIUM 置信度
    └── top_suspects.json                  Top-50 综合排名
```

---

## Q1 — 时序模式发现

### `q1_temporal_patterns.json`

**生成者**：`analysis/q1/extract_temporal_patterns.py`

**内容**：1371 家代表性公司（Top-800 活跃 + bundle 涉及公司并集）的完整时序画像。

**每条记录字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `company` | string | 公司名（节点 ID） |
| `monthly_counts` | object | `{"YYYY-MM": int}` 月度交易次数 |
| `first_date` / `last_date` | string | YYYY-MM-DD 格式的活跃起止 |
| `total_links` | int | 主图中该公司涉及的总链接数 |
| `active_months` | int | 有交易的不同月份数 |
| `partner_count` | int | 不同贸易伙伴数 |
| `fish_hscode_ratio` | float | 海产品 HS 编码占比（0-1） |
| `temporal_pattern` | string | 5 类标签之一 |

**`temporal_pattern` 实际分布**：

| 模式 | 数量 | 占比 |
|------|------|------|
| stable | 1178 | 85.9% |
| general | 122 | 8.9% |
| short_term | 44 | 3.2% |
| periodic | 22 | 1.6% |
| bursty | 5 | 0.4% |

> 大多数代表性公司行为稳定（符合"高活跃度选取"的偏好），少数 short_term/bursty 公司是后续 Q3/Q4 重点关注的可疑候选。

### `q1_relationship_patterns.csv`

**生成者**：`analysis/q1/extract_relationship_patterns.py`

**内容**：通过倒排索引筛出的、共享伙伴 ≥ 2 的公司对，按时序关系打标签后分层采样输出 2021 对。

**关键字段**：

| 字段 | 含义 |
|------|------|
| `company_a` / `company_b` | 公司对（按字典序固定） |
| `relationship_pattern` | 5 类之一：synchronous / relay / substitution / short_term_collab / co_active |
| `confidence` | 0–1 置信度 |
| `month_overlap_count` / `month_overlap_jaccard` | 活跃月份重叠 |
| `shared_partner_count` / `shared_partner_jaccard` | 共同伙伴 |
| `gap_months` | 前后任间隔月数（同步则为 None） |
| `a_first_month` / `a_last_month` / `b_first_month` / `b_last_month` | 活跃区间 |
| `a_temporal_pattern` / `b_temporal_pattern` | 双方 Q1 单实体模式 |
| `overlap_month_sample` | 最多 6 个重叠月份样本 |

**实际分布**：synchronous 2000、short_term_collab 14、substitution 4、co_active 3。relay 数量为 0，说明在选定的活跃公司池内没有强接力对（接力链信号在 Q3 的 `relay_chains.csv` 中以更宽松条件重新挖掘）。

### `q1_relationship_patterns_top.json`

按 5 种模式分组的 Top-50 简要列表，供前端快速预览，不需要完整字段。

---

## Q2 — 预测链接可靠性评估

### `bundle_reliability.csv`

**生成者**：`analysis/q2/score_bundles.py`（先经 `evaluate_bundles` 和 `link_prediction`）

**内容**：12 个 bundle 的综合评分与分类。

**核心字段**：

| 字段 | 含义 |
|------|------|
| `bundle` | bundle 名称（carp / tuna / salmon 等） |
| `link_count` / `node_count` | 链接数 / 节点数 |
| `endpoint_in_base_ratio` | 端点在主图中的比例 |
| `seen_pair_ratio` | 公司对在主图中已存在的比例 |
| `valid_hscode_ratio` | 使用主图已知 HS 编码的比例 |
| `fish_hscode_ratio` | 海产品占比 |
| `outside_date_ratio` | 日期超出有效范围的比例（应为 0） |
| `temporal_consistency_ratio` | 预测日期落在端点公司活跃期内的比例 |
| `unique_pair_ratio` / `max_pair_repeat` | 重复程度 |
| `ml_link_probability` / `ml_link_probability_p90` | 自监督模型预测概率均值/p90 |
| `ml_validation_auc` | 留出集 AUC（典型 0.78–0.85） |
| `score` | 0-100 综合分 |
| `label` | reliable / suspicious / reject |
| `bundle_type` | gap_filler / relationship_ext / mixed / novel_discovery（按 seen_pair） |
| `repetition_type` | diverse / moderately_repeated / highly_repeated |

**分类结果**：

| 标签 | 数量 |
|------|------|
| reliable | 5 |
| suspicious | 6 |
| reject | 1 |

5 个 reliable bundle 是后续 Q3/Q4 的可信数据源。

### `reliable_links.json`

**生成者**：`analysis/q2/merge_links.py`

**内容**：从 5 个 reliable bundle 中收集、去重（删除主图重复 + bundle 间重复）后的 394 条新链接。

**每条记录字段**：

| 字段 | 含义 |
|------|------|
| `source` / `target` | 公司端点 |
| `arrivaldate` | YYYY-MM-DD 日期，全部为 2034 年 |
| `hscode` | HS 编码 |
| `valueofgoods_omu` | 货物价值（OMU 单位） |
| `volumeteu` | 体积（TEU 集装箱当量） |
| `weightkg` | 重量（千克） |
| `dataset` | 原始 bundle 标识 |
| `generated_by` | 来源 bundle 名称（用于 Q4 多工具桥接判断） |

是后续 Q3 异常分析、Q4 边聚类的输入。

---

## Q3 — 图谱补全前后异常对比

### `anomaly_delta.csv`

**生成者**：`analysis/q3/detect_anomalies.compare_anomalies()`

**内容**：每家受 reliable_links 影响的 486 家公司在补全前后的画像变化与可疑评分。

**核心字段**：

| 字段 | 含义 |
|------|------|
| `company` | 公司名 |
| `base_first_date` / `base_last_date` | 主图中的活跃起止日期 |
| `base_link_count` / `predicted_link_count` | 主图链接数 / 新增预测链接数 |
| `filled_gap_months` | 主图生命周期内被填补的空档月（;分隔） |
| `extended_months` | 超出主图最后日期的预测月（;分隔） |
| `burst_months` | 月交易量超过 mean+3σ 阈值的月 |
| `dormancy_months` | 从最后活跃到首个预测月的间隔（月） |
| `dormancy_weight` | 1.0 + min(4.0, dormancy_months/12) 的权重，最高 5× |
| `new_partner_count` / `new_hscode_count` | 新增贸易伙伴/HS 编码数 |
| `predicted_weightkg` / `predicted_value_omu` | 新增链接物理总量 |
| `suspicious_revival_score` | **0-100 综合评分**（见 analysis/README.md） |
| `q1_temporal_pattern` | 该公司在 Q1 中的时序模式 |
| `q1_inconsistency_reason` | Q1-Q3 不一致原因（触发额外加分） |

**评分分布**：max 72.0，mean 5.4，median 3.0，> 50 共 14 家，> 30 共 26 家，> 10 共 50 家。

**Top 10 公司画像高度一致**：全部 `q1_temporal_pattern = short_term`，dormancy_months 39–76 个月，标注"short_term company revived after long dormancy"——典型的壳公司周期性复活模式。

### `top_anomaly_delta.json`

CSV 的前 100 条 JSON 版，含完整字段，便于前端直接读取展示。

### `bridge_companies.csv`

**生成者**：`analysis/q3/detect_anomalies.detect_bridge_companies()`

**内容**：因新增可靠链接而成为结构性桥梁的 457 家公司。

**字段**：

| 字段 | 含义 |
|------|------|
| `company` | 桥接公司 |
| `bridge_scope` | 该公司因新链接获得的可达节点总数 |
| `bridge_link_count` | 参与的可靠链接数 |
| `bridge_partner_sample` | 桥接对端公司样本（最多 5 个，;分隔） |

**Top 5**：Coastal Cruisers Pic Shipping（11,470 节点 / 5 链接）、Zambezi Valley Marine biology（7,603 / 3）、Uttar Pradesh s CJSC（7,453 / 2）。

> Coastal Cruisers 仅 5 条新链接就额外连通 1.1 万个节点，处于网络关键切割点。

### `relay_chains.csv`

**生成者**：`analysis/q3/detect_anomalies.detect_relay_chains()`

**内容**：300 条"前任停活、接班接管"的接力关系（按共享伙伴数降序，最多输出 300）。

**字段**：

| 字段 | 含义 |
|------|------|
| `predecessor` | 前任公司（在 Q1 中 last_date < 2033-01-01） |
| `successor` | 接班公司（reliable_links 端点） |
| `predecessor_last_month` | 前任最后活跃月（YYYY-MM） |
| `gap_months_to_2034` | 前任停活到 2034-01 的月数差 |
| `shared_partner_count` | 在主图中两公司共享的贸易伙伴数 |
| `shared_partner_sample` | 共享伙伴样本（最多 4 个） |
| `predecessor_pattern` / `successor_pattern` | 双方 Q1 时序模式 |

**Top 1**：Portuguese Cuttle GmbH Freight → Fresh LC，共享 142 个贸易伙伴，前任停活 36 个月。

### `top_bridge_companies.json` / `top_relay_chains.json`

CSV 的 Top-50 / Top-100 JSON 版。

### `graph_diff_analysis.json`

**生成者**：`analysis/q3/analyze_graph_diff.py`（事后分析脚本，不在主管道中）

**内容**：补全前后的多维对比汇总，分 7 个 section：

- `summary` — 关键变化数字（新增链接 394、新公司对 58、桥接公司 457、接力链 300、HIGH 嫌疑 14 等）
- `network_diff` — 节点/边/链接数、密度、度分布
- `company_diff` — 486 家受影响公司的复活类型、Q1 模式分布、Top 20 高可疑公司
- `temporal_diff` — 按年/月统计的链接增量分布（394 条全在 2034 年）
- `cargo_diff` — HS 编码变化、海产品比例、新引入 50 个 HS 码
- `pair_diff` — 58 个全新贸易对、305 个被强化的已知对
- `bridge_companies_top20` / `relay_chains` — 网络模式 Top 列表

供报告或 PPT 直接引用关键数字。

---

## Q4 — 可疑公司识别

### `company_clusters.csv`

**生成者**：`analysis/q4/company_clustering.cluster_and_detect_companies()`

**内容**：1371 家公司的特征画像、聚类标签、异常评分。

**字段（按类别分组）**：

| 类别 | 字段 |
|------|------|
| 行为统计 | total_links, active_months, max_monthly_count, avg_monthly_count, **activity_cv**, partner_count, hscode_count, fish_hscode_ratio, total_weightkg, total_value_omu |
| Q1 信号 | q1_temporal_pattern, q1_pattern_risk |
| Q3 信号 | revival_score, dormancy_months, dormancy_weight, has_extended_months, q1_inconsistent |
| 网络信号 | bridge_scope, is_relay_successor |
| 业务模式 | business_mode（8 类） |
| 异常检测 | isolation_label（normal/anomaly）, anomaly_score（0-1） |
| 聚类结果 | dbscan_cluster, hierarchical_cluster (L1), hierarchical_subcluster (L2), hierarchical_microcluster (L3), cluster_path (C{l1}-S{l2}), cluster_path_3 (C-S-T) |
| 相似度 | nearest_company, nearest_similarity（cosine） |

**业务模式分布**：

| 模式 | 数量 |
|------|------|
| general_trader | 973 |
| niche_or_shell | 214 |
| fish_network_expander | 79 |
| fish_intensive_hub | 30 |
| **dormant_revival** | **26** |
| **short_lived** | **24** |
| high_volume_distributor | 14 |
| diversified_broker | 11 |

异常公司（IsolationForest 标记 anomaly）共 110 家。

### `edge_clusters.csv`

**生成者**：`analysis/q4/edge_clustering.cluster_edges()`

**内容**：363 条贸易边的聚合特征、语义标签、KMeans 聚类。

**字段**：

| 字段 | 含义 |
|------|------|
| `source` / `target` | 公司对（已排序） |
| `predicted_count` / `base_count` / `total_count` | 预测 / 基础 / 总链接数 |
| `fish_ratio` | 该对中海产品占比 |
| `bundle_count` | 来自多少个 reliable bundle |
| `month_span` | 涉及的不同月数 |
| `avg_weightkg` / `avg_value_omu` | 平均物理量 |
| `source_l1/l2/l3` / `target_l1/l2/l3` | 双方在 company_clusters 中的三层聚类 |
| `source_business_mode` / `target_business_mode` | 双方业务模式 |
| `months` | 涉及月份（;分隔） |
| `edge_semantic_label` | 6 类语义标签 |
| `edge_cluster` / `edge_cluster_color` | KMeans 簇 ID + 预设色 |
| `edge_width_score` | log1p(total_count)，供前端线宽映射 |

**语义标签分布**：

| 标签 | 数量 |
|------|------|
| historical_backbone | 256 |
| opportunistic_route | 89 |
| **multi_tool_bridge** | **18** |

multi_tool_bridge 是最强的多工具汇聚证据（同一对被 ≥ 2 个 reliable bundle 同时预测）。

### `suspicion_ranking.csv` ★ Q4 最终交付

**生成者**：`analysis/q4/synthesize_suspicion.py`

**内容**：1371 家公司的多信号综合可疑度排名 + 可读证据链。**这是回答 Q4 "Identify Suspicious Companies" 的核心产物**。

**字段（在 company_clusters.csv 基础上新增）**：

| 字段 | 含义 |
|------|------|
| 全部 company_clusters 字段 | 见上 |
| `bridge_link_count` / `bridge_partner_sample` | 从 bridge_companies 补入的字段 |
| `relay_predecessors` | 该公司作为接班方时的前任列表（最多 3 个，;分隔） |
| `sig_iso_anomaly` | 信号 1：IsolationForest 异常分（0-1） |
| `sig_revival` | 信号 2：复活评分归一化 |
| `sig_dormancy` | 信号 3：停活时长（sigmoid 压缩） |
| `sig_q1_inconsistent` | 信号 4：Q1-Q3 不一致（0/1） |
| `sig_q1_risk` | 信号 5：Q1 时序风险级别 |
| `sig_bridge` | 信号 6：桥接范围（log 归一化） |
| `sig_relay` | 信号 7：接力接班标志（0/1） |
| `sig_fish` | 信号 8：海产品比例 |
| **`composite_score`** | **0–1 加权综合分** |
| **`signal_count`** | 触发（>0.3）的信号数 |
| **`confidence_tier`** | **HIGH / MEDIUM / LOW** |
| **`evidence_chain`** | **可读的证据链字符串**，`|` 分隔 |

**置信度分层结果**：

| 层级 | 数量 | 判别条件 |
|------|------|---------|
| HIGH | 25 | composite ≥ 0.55 且 signal_count ≥ 3 |
| MEDIUM | 41 | composite ≥ 0.30 且 signal_count ≥ 2 |
| LOW | 1305 | 其他 |

最高分 0.7531，前 10 名全部触发 6–7 个信号。

**证据链示例**（最高分公司）：
> `IsolationForest anomaly (score=0.82) | dormant 74mo then revived (revival_score=51), Q1 pattern=short_term; short_term company revived after long dormancy | structural bridge: 6,463 nodes newly reachable via 4 new link(s) | fish HS code ratio=18.3% | business_mode=dormant_revival`

### `suspects_high.json`

**内容**：25 家 HIGH 置信度公司的完整记录。**Q4 答题的核心嫌疑名单**。

### `suspects_medium.json`

**内容**：HIGH + MEDIUM 共 66 家公司（截前 100 实际只有 66 条），扩展嫌疑池。

### `top_suspects.json`

**内容**：综合排名 Top 50，无论置信度如何。供 PPT 焦点展示。

### `top_company_anomalies.json` / `top_edge_clusters.json`

CSV 的截断 JSON 版，便于前端快速加载。

---

## 数据流总结

```
MC2/ 原始数据
    ↓ (analysis 模块逐级处理)
outputs/q1/ → outputs/q2/ → outputs/q3/ → outputs/q4/
    ↓ (visualization 模块统一读取)
visualization/data/mc2_3d_data.js + 各 q*/build_*.py 生成 2D 图表
    ↓
浏览器交互可视化
```

### 主要消费关系

| 输出文件 | 主要消费者 |
|----------|----------|
| `q1_temporal_patterns.json` | Q3 detect_anomalies、Q4 company_clustering、3D 节点属性 |
| `bundle_reliability.csv` | Q2 气泡图/雷达图、3D bundle 筛选 |
| `reliable_links.json` | Q3 全部三类检测、Q4 edge_clustering、3D 新链接图层 |
| `anomaly_delta.csv` | Q3 散点图、Q4 company_clustering 信号注入 |
| `bridge_companies.csv` / `relay_chains.csv` | Q4 synthesize_suspicion 信号注入、3D 桥接高亮 |
| `company_clusters.csv` | Q4 平行坐标、Sunburst 三层聚类图 |
| `suspicion_ranking.csv` ★ | **Q4 最终展示页：Top 嫌疑公司表 + 证据链文本卡片** |
| `graph_diff_analysis.json` | Q3 报告页面的关键数字直接引用 |

---

## 重新生成

完整重跑（约 5–10 分钟，取决于硬件）：

```bash
D:\AnacondaEnviroment\envs\python312\python.exe analysis/run_pipeline.py
```

事后单独跑 Q3 图谱差异分析（可选，不在主管道中）：

```bash
D:\AnacondaEnviroment\envs\python312\python.exe analysis/q3/analyze_graph_diff.py
```

所有输出会被覆盖。如果要保留历史版本，请先备份整个 `outputs/` 目录。
