# Final boards（Q1–Q4 导航看板）

本目录为 **英文** 叙事型看板：统一顶栏在 Q1–Q4 之间切换；当前 **Q1、Q2 已接入** 真实图表，**Q3、Q4 为占位页**，便于后续接入同一导航壳。

## 如何预览

在项目根目录下用浏览器直接打开（无需服务器）：

| 文件 | 内容 |
|------|------|
| `final_sketches/q1.html` | Q1：四图预览（热力、岭线、气泡、弦图）+ 全屏灯箱 |
| `final_sketches/q2.html` | Q2：Bundle 可靠性（Plotly 矩阵/雷达/热力 + 内嵌 3D） |
| `final_sketches/q3.html` | Q3：占位，说明「补全前后对比」叙事方向 |
| `final_sketches/q4.html` | Q4：占位，说明「聚类 / 边语义 / 异常主体」叙事方向 |

若 iframe 或 3D 子页空白，请先在本地按 `visualization/q1/README.md`、`visualization/q2/README.md` 生成对应 HTML 与数据。

## Q1 接入要点

- 看板通过 **相对路径** 引用 `visualization/` 下已生成的页面（如 `figures_2d/q1/`、`q1_relationship_chord_time.html`）。  
- **缩放预览** 由 CSS `transform: scale` 包住 iframe；**Expand / 点击预览** 打开灯箱加载同源 HTML。  
- 后续改图：保持相对路径层级不变，或同步修改 `q1.html` 内 `data-src` / `src`。

## Q2 接入要点

- **`final_sketches/q2.html`** 主要由 `visualization/q2/build_q2_bundle_dashboard.py` **覆写生成**；手工改 `q2.html` 可能在下次构建时被冲掉。持久文案/结构改动应进该 Python 模板逻辑。  
- 3D 区块依赖 **`visualization/figures_3d/q2/q2_dandelion_graph.html`**（`?dashboard=1` 用于嵌入预览）。  
- Plotly 使用 CDN；离线环境需自备 `plotly` 脚本或使用内联方案。

## Q3 / Q4 后续接入提示

1. **导航**  
   四个页面已共享同一组 `top-nav` 链接；新页只需复制 **hero + `top-nav` + 主内容区** 结构，并把当前题的 `<a class="active">` 切到对应文件。

2. **版式**  
   - Q1 采用 **左说明 / 中两图 / 右两图** 的 `workspace` 网格；新图可沿用 **`.panel` + `.chart-shell`**，或对齐 Q2 的 **`dashboard-grid`（侧栏 + 主列）**。  
   - 保持 `:root` 墨色、米色底、青绿主色与圆角卡片风格，视觉上与 Q1/Q2 一致即可。

3. **内容边界**  
   与现有一致：**看板正文只解释图与数据含义**，不写数据来源路径、不把 CLI 写进 hero（细节放在 `visualization/q*/README.md`）。

4. **数据依赖**  
   - Q3：预期消费 `outputs/q3/` 等补全前后对比产物（见 `outputs/README.md`）。  
   - Q4：预期消费 `outputs/q4/company_clusters.csv`、`edge_clusters.csv` 等。

接入完成后，将 `q3.html` / `q4.html` 中的占位主面板替换为 iframe、Plotly 容器或与 `build_q3_*` / `build_q4_*` 生成页一致的引用即可。
