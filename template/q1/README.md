# Q1 可视化模拟实现

本目录根据《可视化需求规格说明书（编程落地版，MD格式）.md》生成了两套可直接打开的模拟可视化：

- `q1_2d_embedding_cluster.html`：企业双维度嵌入时序聚类图，包含 2D 散点、KDE 聚类密度晕染、时间轴、自动播放、hover 提示、图例显隐。
- `q1_3d_trade_terrain.html`：企业关系 3D 时序地形图，包含纯 3D 曲面地形、整数坐标网格拟合、可信度颜色映射、视角操控、时间轴。
- `q1_mock_data.js`：两套可视化共用的确定性模拟数据。

## 数据字段对应

`q1_mock_data.js` 中的 `embeddingData` 严格包含 2D 规格要求字段：

- `company_id`
- `time_slice`
- `emb_relation`
- `emb_feature`
- `cluster_id`
- `cluster_name`

`q1_mock_data.js` 中的 `terrainData` 严格包含 3D 规格要求字段：

- `company_id`
- `related_company_id`
- `time_slice`
- `pos_x`
- `pos_y`
- `rel_embedding`
- `confidence`
- `relation_strength`

## 打开方式

可直接在浏览器中打开两个 HTML 文件。若浏览器或网络环境限制 CDN 脚本加载，可在本目录启动本地静态服务后访问：

```bash
python -m http.server 8000
```

然后访问：

- `http://localhost:8000/q1_2d_embedding_cluster.html`
- `http://localhost:8000/q1_3d_trade_terrain.html`

两个页面使用 `q1_mock_data.js` 中的月度时间切片驱动动画，过渡过程均通过线性插值实现。
