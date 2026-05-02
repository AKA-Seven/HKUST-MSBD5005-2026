# Analysis 模块说明

> VAST Challenge 2023 MC2 — 海洋贸易知识图谱挖掘与 IUU 渔业可疑公司识别

本目录包含整个分析流水线的全部代码。运行入口：

```bash
python analysis/run_pipeline.py
```

---

## 整体架构

```
                       ┌─────────────────────┐
                       │  shared/load_data   │  读取 1.5GB 主图 + 12 bundles
                       └──────────┬──────────┘
                                  ↓
    ┌─────────────────────────────┴─────────────────────────────┐
    │                                                           │
    ↓                                                           ↓
┌───────┐    ┌───────────────────────────────┐    ┌──────────────────────────────┐
│  Q1   │ ←  │  shared/build_index           │ →  │  Q2 evaluate_bundles         │
│ 时序   │    │  ─ base_nodes / base_pairs   │    │      link_prediction         │
│ 画像   │    │  ─ base_exact_edges          │    │      score_bundles           │
└───┬───┘    │  ─ company_date_ranges       │    │      merge_links             │
    │        └───────────────────────────────┘    └──────────────┬───────────────┘
    │                                                            │
    │                                                            ↓
    │                                                     reliable_links
    │                                                            │
    └─────────────┬──────────────────────────────────────────────┤
                  ↓                                              ↓
        ┌────────────────────┐                         ┌──────────────────────┐
        │  Q3 detect_anomalies                          │  Q4 company_clustering
        │  ─ compare_anomalies │                         │      edge_clustering   │
        │  ─ detect_bridges    │ ──────[Q1 + Q3 信号]──→ │      synthesize_       │
        │  ─ detect_relays     │                         │      suspicion         │
        └──────────┬───────────┘                         └──────────┬─────────────┘
                   ↓                                                ↓
                              shared/export_results
                                       ↓
                                outputs/{q1,q2,q3,q4}
```

数据流核心理念：**单次遍历 1.5GB 主图**——所有需要逐边扫描的步骤（Q1 画像、Q3 公司画像、Q4 特征构建）都使用单次 `for link in base_graph` 配合受影响公司白名单，避免对超大图做多次 IO。

---

## 顶层文件

### `run_pipeline.py`

整个分析流水线的入口，按顺序执行：

```
1. load_base_graph()            ← 加载主图（约 540 万条边）
2. load_bundles()               ← 加载 12 个预测链接集
3. build_base_index()           ← 构建轻量索引（base_pairs / hscodes / company_date_ranges）
4. evaluate_all_bundles()       ← Q2 规则多维评估
5. score_all_bundle_links()     ← Q2 自监督链接预测
6. score_all_bundles()          ← Q2 综合评分 + reliable/suspicious/reject
7. collect_reliable_links()     ← Q2 去重得到 394 条可靠预测链接
8. extract_temporal_patterns()  ← Q1 公司时序画像
9. extract_relationship_patterns() ← Q1 公司对关系模式
10. compare_anomalies()         ← Q3 公司级复活评分（含 Q1 联动）
11. detect_bridge_companies()   ← Q3 桥接公司检测
12. detect_relay_chains()       ← Q3 接力链检测
13. cluster_and_detect_companies() ← Q4 IsolationForest + 层次聚类（融合 Q1/Q3 信号）
14. synthesize_suspicion()      ← Q4 多信号综合可疑评分
15. cluster_edges()             ← Q4 贸易边语义聚类
16. export_outputs()            ← 写入 outputs/{q1,q2,q3,q4}/
```

后续步骤会复用前置步骤的结果（Q3 调用使用 Q1 的 `temporal_patterns`，Q4 同时使用 Q1 + Q3 的输出），**严格按顺序执行**。

---

## `shared/` — 公共基础模块

### `config.py`
所有全局常量：项目路径、输出目录、有效日期范围（`DEFAULT_DATE_MIN/MAX`）、海产品 HS 编码前缀（`FISH_HSCODE_PREFIXES`）、链接预测训练/验证日期切分（`LINK_PREDICTION_TRAIN_END` 等）、随机种子（`RANDOM_SEED=42`）。修改任何阈值都应在此处统一改动。

### `load_data.py`
两个 IO 函数：`load_base_graph()` 读取主图，`load_bundles()` 批量读取 `MC2/bundles/` 下全部 JSON。返回的是普通 Python dict，主图大小约 1.5GB，加载耗时较长，**调用方应避免重复读取**。

### `build_index.py`
对主图构建轻量索引：

- `build_base_index(base_graph)` → 单次遍历主图，输出
  - `base_nodes`：节点 ID 集合
  - `base_pairs`：贸易公司对集合（带方向）
  - `base_exact_edges`：(source, target, date, hscode) 四元组集合，用于精确去重
  - `base_hscodes`：所有出现过的 HS 编码集合
  - `pair_counts`：公司对出现次数（Counter）
  - `date_min` / `date_max`：日期范围
  - `company_date_ranges`：每家公司 → [first_date, last_date]，供 Q2 时间一致性评估

- `month_key(date)` / `normalize_hscode(value)`：把 `YYYY-MM-DD` 转月份键、统一 HS 编码格式。

### `export_results.py`
统一管理所有产物的输出。`export_outputs()` 接收 9 类结果（bundle 评分、可靠链接、Q3 异常 delta、桥接公司、接力链、公司聚类、边聚类、可疑公司排名、Q1 时序模式、Q1 关系模式），按 Q1–Q4 子目录写入 CSV/JSON。CSV 写入时使用 `utf-8-sig`（带 BOM）以便 Excel 直接打开。

---

## `q1/` — 时序模式发现

### `extract_temporal_patterns.py`

**目标**：识别每家代表性公司的时序行为模式。

**采样策略（三类并集，约 3010 家）**：
1. 按贸易总活跃度选取 Top-800 公司（高交易量主体）
2. 并入 12 个 bundle 中涉及的全部公司端点（与 Q2/Q3/Q4 一致）
3. **新增**：并入早退场前任候选（`last_date < EARLY_STOP_BEFORE` 且活跃度 ≥ `MIN_PREDECESSOR_ACTIVITY`）

> **设计动机**：前两类采样偏向高活跃度公司，会把"短暂活跃后停业"的潜在 relay 前任排除在外，导致 Q1 relay 对为 0。第三类专门为 relay 模式兜底，保证已停业的合法前任公司进入分析池。

**关键常量**：
- `TOP_ACTIVE_COMPANIES = 800`
- `EARLY_STOP_BEFORE = "2033-07-01"`（早于此日期停活视为前任候选，与 Q3 的 2033-01-01 留半年缓冲）
- `MIN_PREDECESSOR_ACTIVITY = 20`（避免只有 1-2 次交易的极端噪声节点）

**模式分类（5 类）**：

| 模式 | 判别规则 |
|------|----------|
| `stable` | 月度覆盖率 ≥ 60%，月度 CV < 1.2 |
| `bursty` | 最大月交易量 / 均值 ≥ 5 且覆盖率 < 45% |
| `periodic` | 活跃月份间隔均值 1.5–4.5，方差 ≤ 4（季度节奏） |
| `short_term` | 生命周期 < 12 个月 |
| `general` | 不满足以上任一特征 |

**实现要点**：单次遍历主图边列表，使用白名单过滤；输出按公司名排序。

### `extract_relationship_patterns.py`

**目标**：识别公司对之间的时序关系模式（synchronous/relay/substitution/short_term_collab/co_active）。

**候选生成（避免 O(n²) 全量枚举）**：
1. 为每家选定公司构建合作伙伴集合
2. 倒排索引：partner → 与之贸易的选定公司列表
3. 跳过超级枢纽伙伴（被 > 40 家公司交易），避免组合爆炸
4. 对共享伙伴 ≥ 2 的公司对生成候选

**模式分类优先级**：relay > substitution > synchronous > short_term_collab > co_active

**关键阈值**：
- `MIN_SHARED_PARTNERS = 2`：候选筛选下限
- `RELAY_GAP_MAX = 60`：接力允许的最大间隔月数（5 年，与 Q3 实测 36–76 月对齐）
- `RELAY_PARTNER_COUNT_MIN = 3`：接力条件的共享伙伴绝对数下限（对小公司更友好）
- `RELAY_JACCARD_MIN = 0.05`：接力条件的 Jaccard 下限（对大公司提供独立判据）
- `MIN_CONFIDENCE = 0.25`：co_active 置信度过滤

**relay 判别逻辑（满足以下全部条件）**：
1. 双方时间段完全错位（一方 last < 另一方 first），且间隔 ≤ 60 个月
2. 共享伙伴绝对数 ≥ 3（小公司判据）**或** Jaccard ≥ 0.05（大公司判据）

置信度公式：`0.55 × gap_score + 0.45 × max(min(jaccard×3, 1), min(shared/20, 1))`，其中 `gap_score = max(0, 1 − gap/61)`。

**分层采样**：每种模式最多保留 `PER_PATTERN_LIMIT` 条（relay/substitution/synchronous 各 2000、short_term_collab 1500 等），避免单一模式独占输出。

---

## `q2/` — 预测链接可靠性评估

### `evaluate_bundles.py`

**目标**：对每个 bundle 计算 14 项规则化可靠性指标，**只算事实，不做主观判断**。

**关键指标**：
- `endpoint_in_base_ratio`：链接端点是否在主图中的比例
- `seen_pair_ratio`：预测公司对在主图中已存在的比例
- `exact_duplicate_ratio`：完全重复边的比例
- `valid_hscode_ratio`：使用主图已知 HS 编码的比例
- `fish_hscode_ratio`：海产品 HS 编码占比
- `outside_date_ratio`：日期超出主图范围的比例（应为 0）
- `physical_field_ratio`：完整物理字段（重量/体积/价值）的比例
- `unique_pair_ratio`：唯一公司对占比（衡量重复程度）
- `max_pair_repeat`：单对最大重复次数
- `bad_country_count`：异常国家代码（"-27"）的节点数
- `temporal_consistency_ratio`：预测日期落在公司在主图中已知活跃期内的比例

### `link_prediction.py`

**目标**：自监督训练链接预测模型，给每个 bundle 输出额外的 ML 概率分。

**实现思路**：
1. 按时间切分主图：用 ≤ 2033-12-31 的边作为训练图结构，2034 边作为正样本
2. 构建 8 维特征：共同邻居、Jaccard、Adamic-Adar、资源分配、度乘积、min/max 度、训练期对内重复次数
3. **2034 真实边作正样本，随机未见对作负样本**，按 80/20 拆训练/留出集
4. `LogisticRegression(class_weight=balanced)` + `StandardScaler`
5. 在留出集上计算 AUC（典型 0.78–0.85），避免循环验证乐观偏差
6. 用训练好的模型预测每条 bundle 链接的概率（mean + p90）

### `score_bundles.py`

**目标**：把规则指标 + ML 分综合成 0–100 评分，并给出 reliable/suspicious/reject 标签。

**评分公式**：
```
score = 20×endpoint_in_base + 25×seen_pair + 20×valid_hscode 
      + 10×physical_field + 10×unique_pair + 5×fish_hscode
      + 15×ml_link_probability + 10×temporal_consistency
      − 25×outside_date_ratio − 8(if bad_country>0) − 罚分(if max_pair_repeat>25)
```

**分类逻辑**：
- **reliable**：score ≥ 60 且同时满足 outside_date_ratio=0、bad_country_count=0、seen_pair≥0.80、valid_hscode≥0.80、max_pair_repeat≤40、temporal_consistency≥0.70
- **suspicious**：50 ≤ score < 60，或某项硬约束未达
- **reject**：score < 50

附加分类标签：
- `bundle_type`：gap_filler / relationship_ext / mixed / novel_discovery（按 seen_pair_ratio）
- `repetition_type`：diverse / moderately_repeated / highly_repeated（按 unique_pair_ratio）

### `merge_links.py`

**目标**：从可靠 bundle 中收集所有链接，去掉与主图重复的边和 bundle 间重复，得到最终 `reliable_links.json`（394 条）。

每条新链接附加 `generated_by` 字段标记来源 bundle，供 Q4 边聚类使用。

---

## `q3/` — 图谱补全前后异常对比

### `detect_anomalies.py`

整个 Q3 的核心，包含三套互补的检测：

**1. `compare_anomalies()` — 公司级复活评分**

为每家受可靠链接影响的公司计算 0–100 的 `suspicious_revival_score`：

```
filled_score    = min(15, 5 × filled_gap_count)        ← 空档月被填补
extended_score  = min(15, 3 × extended_count) × dormancy_weight   ← 复活月（最高 75）
burst_score     = min(20, 5 × burst_count)             ← 突发月
partner_score   = min(15, 3 × new_partners)
hscode_score    = min(10, 3 × new_hscodes)
injection_score = min(15, 15 × pred/base_links)        ← 链接注入比
q1_bonus        = 0/8/15                                ← Q1 模式不一致额外加分
```

**Dormancy 权重**：从公司在主图中的最后活跃日期到首个预测月的间隔月数 →`dormancy_weight = 1.0 + min(4.0, months/12)`，最高 5×。停活越久，复活的可疑度越高。

**Q1 联动加分**：
- short_term 公司沉寂 > 12 个月又复活 → +15
- stable 公司出现 burst 月 → +8
- general/bursty 沉寂 > 24 个月又复活 → +8

**2. `detect_bridge_companies()` — 网络桥接检测**

对每条可靠链接 (U, V)：
- U 侧桥接范围 = `|nb_base[U] − nb_base[V]|`，即 V 通过新链接获得的、原本只与 U 相连的节点数
- V 侧同理

公司 C 的 `bridge_scope` = 所有相关链接中其对端新开放的联通公司数总和。`BRIDGE_MIN_SCOPE = 5` 过滤微弱桥接。

**3. `detect_relay_chains()` — 接力链检测**

定义：
- **前任 A**：在 Q1 中 last_date < 2033-01-01 的公司（已停活 ≥ 1 年）
- **接班 B**：可靠链接的端点（在 2034 获得新预测交易）
- **证据**：A 与 B 在主图中共享 ≥ 2 个共同贸易伙伴

输出 (predecessor, successor, shared_partner_count, gap_months_to_2034, predecessor_pattern, successor_pattern) 关系，按共享伙伴数降序，最多 300 条。

### `analyze_graph_diff.py`

事后分析脚本（不在主管道中），生成图谱补全前后的对比 JSON：网络度量、公司变化、时序分布、HS 编码、贸易对、桥接接力等多维统计。输出至 `outputs/q3/graph_diff_analysis.json`。

### `check_scores.py` / `test_scoring.py`

调试和验证脚本，用于快速检查评分分布和列名完整性。

---

## `q4/` — 可疑公司识别

### `company_clustering.py`

**目标**：对受预测链接影响的公司做无监督异常检测、聚类，并融合 Q1/Q3 外部信号。

**特征矩阵（17 维）**：

| 类别 | 字段 |
|------|------|
| 行为统计（10） | total_links, active_months, max_monthly_count, avg_monthly_count, **activity_cv**, partner_count, hscode_count, fish_hscode_ratio, total_weightkg, total_value_omu |
| Q1 信号（1） | q1_pattern_risk（short_term=3, bursty=2, general/periodic=1, stable=0） |
| Q3 复活信号（4） | revival_score, dormancy_months, has_extended_months, q1_inconsistent |
| 网络信号（2） | bridge_scope, is_relay_successor |

**聚类与异常检测**：
- `IsolationForest(contamination=0.08)` → 标记 `isolation_label`，归一化为 0–1 的 `anomaly_score`
- `DBSCAN(eps=1.8, min_samples=4)` → 发现行为相似的小团体（−1 表示离群）
- 三层 `AgglomerativeClustering`：
  - L1：8 个一级簇
  - L2：每个一级簇内拆 2–4 个子簇
  - L3：每个二级簇内拆 2–3 个微簇
  - 输出 `cluster_path = "C{l1}-S{l2}-T{l3}"` 供前端 sunburst 等可视化
- `cosine_similarity` → 找最相似公司（`nearest_company` + `nearest_similarity`）

**业务模式标签（`business_mode`，8 类）**：
diversified_broker, high_volume_distributor, fish_intensive_hub, fish_network_expander, **dormant_revival**（新增）, **short_lived**（新增）, general_trader, niche_or_shell

修复后 `niche_or_shell` 从 77% 降到 15%，新增 `dormant_revival` 与 `short_lived` 精准对应高可疑集群。

### `edge_clustering.py`

**目标**：对可靠链接形成的公司对做特征聚类与语义标签。

**边特征（7 维）**：predicted_count, base_count, fish_ratio, bundle_count, month_span, avg_weightkg, avg_value_omu

**语义标签（基于真实数据分布校准）**：

| 标签 | 条件（按优先级） |
|------|-----------------|
| `multi_tool_bridge` | bundle_count ≥ 2 |
| `fish_dense_bridge` | base=0 AND fish≥0.5 AND pred≥2 |
| `novel_predicted_route` | base=0 AND pred≥2 |
| `historical_backbone` | base≥10 AND base>pred |
| `persistent_cross_bundle` | month_span ≥ 3 |
| `opportunistic_route` | 其他 |

**KMeans 聚类**：根据数据量自适应 2–6 簇（`n_clusters = min(6, max(2, len(rows)//35 + 2))`），输出 `edge_cluster` + `edge_cluster_color`（预设 6 色），`edge_width_score = log1p(total_count)` 供前端线宽映射。

### `synthesize_suspicion.py`

**Q4 的核心交付**：把 Q1/Q3/Q4 全部信号融合成一个综合可疑评分 + 证据链。

**8 路信号 → 加权综合分**：

| 信号 | 权重 | 来源 |
|------|------|------|
| `sig_iso_anomaly` | 0.25 | Q4 IsolationForest 异常分 |
| `sig_revival` | 0.25 | Q3 suspicious_revival_score / 100 |
| `sig_dormancy` | 0.15 | Q3 dormancy_months（sigmoid 压缩） |
| `sig_q1_inconsistent` | 0.10 | Q3 q1_inconsistency_reason 标志 |
| `sig_q1_risk` | 0.08 | Q1 temporal_pattern 风险级别 |
| `sig_bridge` | 0.10 | Q3 bridge_scope（log 归一化） |
| `sig_relay` | 0.05 | Q3 接力接班标志 |
| `sig_fish` | 0.02 | 海产品比例 |

**置信度分层**：
- **HIGH**：composite_score ≥ 0.55 且 ≥ 3 个信号 > 0.3
- **MEDIUM**：composite_score ≥ 0.30 且 ≥ 2 个信号 > 0.3
- **LOW**：其余

**证据链**：根据触发的信号自动拼接人类可读字符串，例如：
> `IsolationForest anomaly (score=0.82) | dormant 74mo then revived (revival_score=51), Q1 pattern=short_term; short_term company revived after long dormancy | structural bridge: 6,463 nodes newly reachable via 4 new link(s) | fish HS code ratio=18.3%`

---

## 关键设计决策

### 1. 单次遍历优化
主图 540 万条边、1.5GB，多次遍历会带来分钟级 IO 延迟。所有边级统计步骤都用「先确定白名单 → 单次 for link in links → 按白名单累加」的模式。

### 2. Q1 → Q3 → Q4 信号传递
Q3 的 `compare_anomalies` 接收 Q1 的 `temporal_patterns`，触发 Q1 不一致加分；Q4 的 `cluster_and_detect_companies` 同时接收 Q1 + Q3 全部输出，构造 17 维特征；`synthesize_suspicion` 在 Q4 内部做最终融合。

### 3. 评分归一化与可解释性
- 单维信号统一归到 0–1
- 每个分项独立可读（CSV 中保留 `sig_*` 八列）
- 综合分加权和也在 0–1，与置信度分层正交
- 证据链用 `|` 分隔的可读串，不仅给分还给原因

### 4. 阈值校准基于真实数据
- bundle 评分阈值（reliable ≥ 60）参考实测分数分布
- edge 语义标签阈值（multi_tool_bridge 从 ≥3 降到 ≥2）按 394 条链接的实际分散度调整
- bridge_scope ≥ 5 过滤掉单公司桥接的噪声

### 5. 随机性可复现
所有随机操作（IsolationForest、KMeans、DBSCAN、LogisticRegression、负样本采样）统一使用 `RANDOM_SEED = 42`。

---

## 与上下游的接口

### 上游：`MC2/`
需要手动准备：
```
MC2/
├── mc2_challenge_graph.json   # 主图
└── bundles/                   # 12 个预测链接 JSON
```

### 下游：`outputs/`
全部产物按 Q1–Q4 子目录组织，详见 `outputs/README.md`。可视化模块（`visualization/`）以这些 CSV/JSON 为唯一数据源，不再回读原始主图。
