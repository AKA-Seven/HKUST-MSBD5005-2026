# VAST Challenge 2023 Mini-Challenge 2 — MC2 贸易图谱分析

本项目围绕 **IUU（非法、未报告和无管制捕捞）** 相关贸易知识图谱：评估 12 组预测链路（bundles）的可靠性，识别可疑公司与异常模式，并配套 2D/3D 可视化。

原始数据约 **540 万条边**，外加 **12 个 bundle JSON**。仓库根目录即为工程根目录（不再使用旧的 `lch/` 子目录）；原始数据位于 **`MC2/`**。

任务定义与成功标准见 [`Tasks.md`](Tasks.md)。

---

## 目录结构

```
├── MC2/                      # 原始数据（mc2_challenge_graph.json、bundles/*.json）
├── analysis/
│   ├── shared/               # config, load_data, build_index, export_results
│   ├── q1/ … q4/             # 各题分析脚本
│   └── run_pipeline.py       # 流水线入口（自动把 shared/q1–q4 加入 sys.path）
├── outputs/
│   ├── q1/ … q4/             # 分析产物（CSV / JSON）
├── visualization/
│   ├── figures2d_common.py
│   ├── build_3d_data.py    # 生成 3D 页面用的 mc2_3d_data.js
│   ├── build_2d_advanced_figures.py
│   ├── q1/ … q4/           # 各题图表构建脚本
│   ├── figures_2d/         # 2D 图表输出（可按脚本再生）
│   ├── data/mc2_3d_data.js # 由 build_3d_data.py 生成，勿手改
│   ├── vendor/             # 离线前端库（echarts、three.js）
│   └── index.html          # 3D 主页面
├── phase1_sketches/          # 早期原型草稿（非生产）
├── template/q1/              # Q1 规格与模拟数据模板（参考）
└── Tasks.md                  # 题目说明
```

**导入约定**：`analysis/q*/` 各模块通过 `sys.path` 指向 `analysis/shared/`；`visualization/q*/` 脚本将 `visualization/` 加入路径以导入 `figures2d_common`。新增代码时请沿用同一模式。

---

## 环境与 Python

在本机环境中推荐使用：

```
D:\AnacondaEnviroment\envs\python312\python.exe
```

若 conda 未加入 PATH，请勿假设直接输入 `python` 或 `conda run` 可用；请将命令中的解释器路径替换为你的实际路径。

依赖主要包括：`pandas`、`numpy`、`scikit-learn`、`matplotlib`、`seaborn`、`plotly` 等（用于分析与 2D 出图）。

---

## 常用命令

在项目根目录执行（请先 `cd` 到本仓库根目录）。

**完整分析流水线（写入 `outputs/q1/` … `outputs/q4/`）：**

```powershell
& "D:\AnacondaEnviroment\envs\python312\python.exe" analysis/run_pipeline.py
```

**生成 3D 可视化数据（写入 `visualization/data/mc2_3d_data.js`）：**

```powershell
& "D:\AnacondaEnviroment\envs\python312\python.exe" visualization/build_3d_data.py
```

然后在浏览器中打开 **`visualization/index.html`**。页面若依赖 CDN（如 D3、Three.js、3d-force-graph），需要联网；本地 `vendor/` 用于部分离线脚本。

**分别生成 Q1–Q4 的 2D 图表：**

```powershell
& "D:\AnacondaEnviroment\envs\python312\python.exe" visualization/q1/build_q1_figure.py
& "D:\AnacondaEnviroment\envs\python312\python.exe" visualization/q1/build_q1_tech_data.py
& "D:\AnacondaEnviroment\envs\python312\python.exe" visualization/q2/build_q2_figures.py
& "D:\AnacondaEnviroment\envs\python312\python.exe" visualization/q3/build_q3_figures.py
& "D:\AnacondaEnviroment\envs\python312\python.exe" visualization/q4/build_q4_figures.py
```

**一次性生成全部 Q1–Q4 图表：**

```powershell
& "D:\AnacondaEnviroment\envs\python312\python.exe" visualization/build_2d_advanced_figures.py
```

---

## Q1–Q4 对应关系

| 问题 | 分析代码 | 产物 (`outputs/`) | 可视化脚本 | 图表产物 |
|------|---------|-------------------|-----------|---------|
| **Q1** 时序模式 | `analysis/q1/extract_temporal_patterns.py` | `q1/q1_temporal_patterns.json` | `visualization/q1/build_q1_figure.py`<br>`visualization/q1/build_q1_tech_data.py` | `visualization/figures_2d/q1/` |
| **Q2** 预测链路评估 | `analysis/q2/evaluate_bundles.py`<br>`link_prediction.py`<br>`score_bundles.py`<br>`merge_links.py` | `q2/bundle_reliability.csv`<br>`q2/reliable_links.json` | `visualization/q2/build_q2_figures.py` | `visualization/figures_2d/q2/` |
| **Q3** 图谱补全异常 | `analysis/q3/detect_anomalies.py` | `q3/anomaly_delta.csv`<br>`q3/top_anomaly_delta.json` | `visualization/q3/build_q3_figures.py` | `visualization/figures_2d/q3/` |
| **Q4** 可疑公司与边簇 | `analysis/q4/company_clustering.py`<br>`edge_clustering.py` | `q4/company_clusters.csv`<br>`q4/edge_clusters.csv`<br>`q4/top_company_anomalies.json`<br>`q4/top_edge_clusters.json` | `visualization/q4/build_q4_figures.py` | `visualization/figures_2d/q4/` |
| **共用** | `analysis/shared/*.py`、`analysis/run_pipeline.py` | — | `visualization/figures2d_common.py`<br>`visualization/build_3d_data.py` | `visualization/data/mc2_3d_data.js` |

可视化流水线用到的字段说明见 [`analysis/mining_usage_audit.md`](analysis/mining_usage_audit.md)。

---

## 分析流水线模块概览

| 模块 | 对应问题 | 功能 |
|------|---------|------|
| `load_data.py` | 共用 | 加载主图 JSON 与 bundles |
| `build_index.py` | 共用 | 节点、边对、日期、HS 等索引 |
| `extract_temporal_patterns.py` | Q1 | 采样公司月度画像与模式分类 |
| `evaluate_bundles.py` | Q2 | 规则多维评分 |
| `link_prediction.py` | Q2 | 自监督链路预测（训练 2028–2033，验证 2034） |
| `score_bundles.py` | Q2 | 综合分数与 reliable/suspicious/reject |
| `merge_links.py` | Q2/Q3 | 汇总可靠预测边 |
| `detect_anomalies.py` | Q3 | 补全前后异常变化、suspicious_revival_score |
| `company_clustering.py` | Q4 | 层次聚类 + IsolationForest |
| `edge_clustering.py` | Q4 | 贸易边聚类 |
| `export_results.py` | 共用 | 写出 CSV/JSON |

**边记录 schema（主图与 bundles 一致）：**

```json
{
  "source": "公司A",
  "target": "公司B",
  "arrivaldate": "2030-03-15",
  "hscode": "302010",
  "weightkg": 1500.0,
  "valueofgoods_omu": 50000.0
}
```

**`analysis/shared/config.py` 要点：**

- 数据根目录：**`MC2/`**（`mc2_challenge_graph.json`、`bundles/`）
- 日期兜底：2028-01-01 ~ 2034-12-30
- 鱼类相关 HS 前缀：301–308、1604、1605
- 链路预测采样：6000，随机种子 42

---

## 数据流（3D）

```text
MC2/ 原始 JSON
  → analysis/run_pipeline.py
  → outputs/q1 … outputs/q4/
  → visualization/build_3d_data.py
  → visualization/data/mc2_3d_data.js
  → visualization/index.html
```

更细的交互与视图说明见 [`visualization/README.md`](visualization/README.md)。

---

## 重要约束

- **`visualization/data/mc2_3d_data.js`** 必须由脚本生成，请勿手动编辑。
- **`MC2/mc2_challenge_graph.json`** 体积约 1.5 GB，调试时不要反复整文件读入内存；管线侧应避免无谓的全量加载。
- **`outputs/`**、**`visualization/figures_2d/`**、**`visualization/data/mc2_3d_data.js`** 可在有原始数据的前提下通过上述命令再生；归档或提交时可按需忽略以减小体积。

---

## 推送到 GitHub

本仓库已在本地初始化：`main` 分支含首次提交；**`MC2/` 已通过 `.gitignore` 排除**（需在本地自行放置原始数据）。

在 GitHub 网页新建空仓库（不要勾选添加 README），然后在项目根目录执行（将 `YOUR_USER` / `YOUR_REPO` 换成你的仓库地址）：

```powershell
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

若使用 SSH：`git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git`

首次推送前请确认已登录 GitHub（HTTPS 凭据或 SSH key）。

---

## 许可证与赛程

挑战赛数据与条款以 **VAST Challenge 2023** 官方要求为准。
