# Q1 可视化：版本控制与本地构建

本文说明 Q1 相关产物在 Git 中的取舍，以及如何在本机重新生成被忽略的数据文件。

## 纳入版本库的内容

| 路径 | 说明 |
|------|------|
| `analysis/q1/*.py` | Q1 分析脚本（始终跟踪） |
| `visualization/q1/build_*.py` | Q1 图表构建脚本 |
| `visualization/q1/q1_relationship_chord_time.html` | 弦图页面（静态 HTML + 内联/外链脚本占位） |
| `visualization/figures_2d/q1/*.html` | Plotly 等导出的交互式 HTML 页面 |
| `final_sketches/q1.html` | Q1 四图看板（若仓库中包含 `final_sketches/`） |

原则：**保留全部 `.py` 与 Q1 相关的 `.html`**，便于评审代码与交互原型；**不提交由脚本写出的大数据或光栅图**。

## 被 `.gitignore` 忽略的内容（Q1）

在仓库根目录 `.gitignore` 中约定：

1. **`outputs/q1/`**  
   分析产物：`q1_temporal_patterns.json`、`q1_relationship_patterns.csv`、`q1_relationship_patterns_top.json` 等。  
   由 `analysis/run_pipeline.py`（或单独跑 Q1 分析脚本）在本地生成。

2. **`visualization/figures_2d/q1/*.png`**  
   Matplotlib 等导出的预览图。

3. **`visualization/figures_2d/q1/*_data.json`**  
   与 HTML 同名的汇总/中间 JSON（如 `q1_ridge_river_top50_data.json`），体积可能较大。

4. **`visualization/q1/data/`**  
   弦图数据：`q1_relationship_chord.json`、`q1_relationship_chord_data.js` 等，由 `build_q1_relationship_chord_data.py` 生成，`q1_relationship_chord_time.html` 运行时加载。

> **曾提交过上述路径时**：执行 `git rm -r --cached outputs/q1/` 以及需要对已跟踪的 `visualization/**` 忽略文件执行 `git rm --cached <文件>`，再提交一次，之后才会仅受 `.gitignore` 约束。

## 从零生成本地 Q1 数据与图表

前置：将原始数据放入 `MC2/`（参见仓库根目录 `README` / `CLAUDE.md`）。

1. **分析产物（写入 `outputs/q1/`）**  
   - 全流水线：`python analysis/run_pipeline.py`  
   - 或仅运行 Q1 分析目录下对应脚本（见 `analysis/README.md`、`outputs/README.md` 中 Q1 小节）。

2. **弦图数据**  
   ```bash
   python visualization/q1/build_q1_relationship_chord_data.py
   ```

3. **2D 图（HTML + PNG + `_data.json`）**  
   ```bash
   python visualization/q1/build_q1_ridge_river.py
   python visualization/q1/build_q1_monthly_heatmap.py
   python visualization/q1/build_q1_bubble_scatter.py
   ```

4. **看板**  
   在资源生成后，用浏览器打开 `final_sketches/q1.html`（路径相对于仓库根目录；看板内 iframe 指向 `visualization/` 下各 HTML）。

## 与 `outputs/README.md` 的关系

`outputs/README.md` 仍详细描述 **Q1 各文件的字段与统计含义**；本文只补充 **哪些文件在 Git 中不跟踪** 以及如何 **复现**。二者配合使用即可。
