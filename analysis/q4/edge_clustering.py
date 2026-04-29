import sys
from collections import Counter, defaultdict
from pathlib import Path

_shared = str(Path(__file__).resolve().parents[1] / "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from build_index import normalize_hscode
from config import FISH_HSCODE_PREFIXES, RANDOM_SEED


EDGE_CLUSTER_COLORS = [
    "#FF1744",
    "#FF6D00",
    "#FFC400",
    "#D500F9",
    "#FF3D00",
    "#FF9100",
]


def _pair_key(source: str, target: str) -> tuple[str, str]:
    """使用无向公司对作为边聚类单元。"""
    return tuple(sorted((source, target)))


def _edge_cluster_label(row: dict) -> str:
    """根据边特征生成可解释的边簇语义标签。

    阈值基于 394 条可靠链接分散后的实际分布校准：
      - 单对最多约 10 余条链接，月跨度最多 12 个月，bundle 最多 3-4 个。
    """
    fish   = row["fish_ratio"]
    pred   = row["predicted_count"]
    base   = row["base_count"]
    span   = row["month_span"]
    nbundles = row["bundle_count"]

    # 多个可靠 bundle 同时预测该对：多工具汇聚证据最强
    if nbundles >= 2:
        return "multi_tool_bridge"
    # 新出现贸易对（原图完全没有），海产品占比高
    if base == 0 and fish >= 0.5 and pred >= 2:
        return "fish_dense_bridge"
    # 新出现贸易对，多个月份出现
    if base == 0 and pred >= 2:
        return "novel_predicted_route"
    # 原图中已有大量历史，预测链接只是补充
    if base >= 10 and base > pred:
        return "historical_backbone"
    # 跨多月持续出现（即使单 bundle）
    if span >= 3:
        return "persistent_cross_bundle"
    return "opportunistic_route"


def cluster_edges(base_graph: dict, reliable_links: list[dict], company_clusters: list[dict]) -> list[dict]:
    """基于 mining + 原始图谱构建边簇输出。"""
    company_map = {row["company"]: row for row in company_clusters}
    involved_companies = {
        company
        for link in reliable_links
        for company in (link.get("source"), link.get("target"))
        if company in company_map
    }

    base_pair_counts = Counter()
    for link in base_graph.get("links", []):
        source = link.get("source")
        target = link.get("target")
        if not source or not target:
            continue
        if source in involved_companies and target in involved_companies:
            base_pair_counts[_pair_key(source, target)] += 1

    agg = defaultdict(
        lambda: {
            "source": "",
            "target": "",
            "predicted_count": 0,
            "weight_sum": 0.0,
            "value_sum": 0.0,
            "fish_count": 0,
            "bundle_set": set(),
            "months": [],
        }
    )
    for link in reliable_links:
        source = link.get("source")
        target = link.get("target")
        if source not in involved_companies or target not in involved_companies:
            continue
        key = _pair_key(source, target)
        row = agg[key]
        row["source"], row["target"] = key
        row["predicted_count"] += 1
        row["weight_sum"] += float(link.get("weightkg") or 0)
        row["value_sum"] += float(link.get("valueofgoods_omu") or 0)
        hs = normalize_hscode(link.get("hscode"))
        if hs.startswith(FISH_HSCODE_PREFIXES):
            row["fish_count"] += 1
        bundle = link.get("generated_by")
        if bundle:
            row["bundle_set"].add(bundle)
        month = str(link.get("arrivaldate") or "")[:7]
        if len(month) == 7:
            row["months"].append(month)

    rows = []
    for key, row in agg.items():
        source, target = key
        months = sorted(set(row["months"]))
        month_span = 0
        if months:
            month_span = len(months)
        predicted_count = row["predicted_count"]
        fish_ratio = row["fish_count"] / predicted_count if predicted_count else 0.0
        base_count = base_pair_counts.get(key, 0)
        bundle_count = len(row["bundle_set"])
        avg_weight = row["weight_sum"] / predicted_count if predicted_count else 0.0
        avg_value = row["value_sum"] / predicted_count if predicted_count else 0.0

        source_meta = company_map[source]
        target_meta = company_map[target]
        out = {
            "source": source,
            "target": target,
            "predicted_count": predicted_count,
            "base_count": base_count,
            "total_count": base_count + predicted_count,
            "fish_ratio": round(fish_ratio, 4),
            "bundle_count": bundle_count,
            "month_span": month_span,
            "avg_weightkg": round(avg_weight, 2),
            "avg_value_omu": round(avg_value, 2),
            "source_l1": int(source_meta["hierarchical_cluster"]),
            "source_l2": int(source_meta["hierarchical_subcluster"]),
            "source_l3": int(source_meta.get("hierarchical_microcluster", 0)),
            "target_l1": int(target_meta["hierarchical_cluster"]),
            "target_l2": int(target_meta["hierarchical_subcluster"]),
            "target_l3": int(target_meta.get("hierarchical_microcluster", 0)),
            "source_business_mode": source_meta["business_mode"],
            "target_business_mode": target_meta["business_mode"],
            "months": ";".join(months),
        }
        out["edge_semantic_label"] = _edge_cluster_label(out)
        rows.append(out)

    if not rows:
        return []

    features = np.array(
        [
            [
                row["predicted_count"],
                row["base_count"],
                row["fish_ratio"],
                row["bundle_count"],
                row["month_span"],
                row["avg_weightkg"],
                row["avg_value_omu"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    scaled = StandardScaler().fit_transform(features)
    cluster_count = min(6, max(2, len(rows) // 35 + 2))
    if len(rows) < cluster_count:
        cluster_count = max(1, len(rows))
    if cluster_count > 1:
        model = KMeans(n_clusters=cluster_count, random_state=RANDOM_SEED, n_init="auto")
        labels = model.fit_predict(scaled)
    else:
        labels = np.zeros(len(rows), dtype=int)

    for idx, row in enumerate(rows):
        cid = int(labels[idx])
        row["edge_cluster"] = cid
        row["edge_cluster_color"] = EDGE_CLUSTER_COLORS[cid % len(EDGE_CLUSTER_COLORS)]
        # 用于前端直接映射线宽：由链接数驱动
        row["edge_width_score"] = round(np.log1p(row["total_count"]), 4)

    return sorted(rows, key=lambda x: x["total_count"], reverse=True)
