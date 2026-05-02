# Q2 可视化：版本控制与本地构建

说明 Q2 图表脚本的职责、Git 中保留/忽略的文件，以及与 `final_sketches/q2.html` 看板的关系。

## 纳入版本库的内容

| 路径 | 说明 |
|------|------|
| `analysis/q2/*.py` | Q2 分析脚本（评分、合并链接等） |
| `visualization/q2/*.py` | Q2 图表构建脚本 |
| `visualization/figures_2d/q2/*.html` | 由脚本写出的 2D 交互页（如 bundle dashboard 镜像） |
| `visualization/figures_3d/q2/*.html` | 3D 蒲公英网络页面 |
| `final_sketches/q2.html` | Q2 最终看板（Plotly + 内嵌 3D，由 `build_q2_bundle_dashboard.py` 生成） |

原则：**只跟踪 `visualization/q2/` 下的 `.py`（以及本 README）**；该目录内若出现缓存或非源码文件，见下文 `.gitignore`。**跟踪 Q2 相关的 `.html`** 成品与镜像，便于直接打开演示。

## 被 `.gitignore` 忽略的内容（Q2）

仓库根目录 `.gitignore` 中约定（详见该文件 Q2 小节）：

1. **`outputs/q2/`**  
   分析产物：`bundle_reliability.csv`、`reliable_links.json` 等，由 `analysis/run_pipeline.py`（或单独跑 Q2 步骤）生成。

2. **`visualization/figures_2d/q2/*.png`**、**`visualization/figures_2d/q2/*_data.json`**  
   导出的光栅图或与 HTML 配套的大数据 JSON（若脚本生成）。

3. **`visualization/q2/` 下除 `*.py` 与 `README.md` 以外的杂项**  
   例如误放的临时文件、日志等（通过“目录忽略 + 白名单放行 .py/README”实现；`__pycache__/` 仍由全局规则忽略）。

> 若某文件曾被提交后又加入忽略规则，需对该文件执行 `git rm --cached <路径>` 再提交。

## 推荐的本地构建顺序

前置：`MC2/` 数据就位，且已生成 `outputs/q2/`（至少包含 `bundle_reliability.csv`、`reliable_links.json`；看板脚本还会读取 `outputs/q1/q1_temporal_patterns.json` 等）。

在 conda 环境 **`dv`** 下（项目根目录）：

```powershell
conda activate dv
```

1. **3D 蒲公英图（必须先于看板）**

   ```text
   python visualization/q2/build_q2_dandelion_3d.py
   ```

   输出示例：`visualization/figures_3d/q2/q2_dandelion_graph.html`。

2. **Q2 最终看板（Fig1–4 合一页）**

   ```text
   python visualization/q2/build_q2_bundle_dashboard.py
   ```

   写入 **`final_sketches/q2.html`**，并镜像到 `visualization/figures_2d/q2/q2_bundle_dashboard.html`。

3. **其他可选脚本**（独立图，不写入 final 看板时按需运行）

   - `build_q2_bundle_reliability_bubble.py`：可靠性气泡图 HTML。  
   - `build_q2_trade_sankey_q1_q2.py`：Q1/Q2 桑基等。  
   - `build_q2_figures.py`：若仓库内已补齐其引用的子模块，可作为串联入口；否则请以单脚本为准。

## 与 `outputs/README.md`、`final_sketches/README.md` 的关系

- **`outputs/README.md`**：Q2 中间产物的字段与业务含义。  
- **`final_sketches/README.md`**：如何在浏览器中打开 Q1/Q2 看板，以及 Q3/Q4 占位与接入手势。

本文仅说明 **可视化构建** 与 **Git 取舍**。
