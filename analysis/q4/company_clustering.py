import sys
from collections import Counter, defaultdict
from pathlib import Path

_shared = str(Path(__file__).resolve().parents[1] / "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from build_index import month_key, normalize_hscode
from config import FISH_HSCODE_PREFIXES, RANDOM_SEED


def _empty_company_features() -> dict:
    """公司行为画像的原始累加容器。"""
    return {
        "monthly_counts": Counter(),
        "partners": set(),
        "hscodes": Counter(),
        "fish_hscode_count": 0,
        "total_links": 0,
        "total_weightkg": 0.0,
        "total_value_omu": 0.0,
    }


def affected_companies_from_bundles(bundles: dict[str, dict]) -> set[str]:
    """只聚焦预测链接涉及的公司，避免输出过多无关节点。"""
    companies = set()
    for bundle_graph in bundles.values():
        for link in bundle_graph.get("links", []):
            if link.get("source"):
                companies.add(link["source"])
            if link.get("target"):
                companies.add(link["target"])
    return companies


def build_company_feature_table(base_graph: dict, companies: set[str]) -> list[dict]:
    """从主图中抽取公司级行为特征，供聚类和异常检测使用。"""
    raw_profiles = defaultdict(_empty_company_features)

    for link in base_graph.get("links", []):
        source = link.get("source")
        target = link.get("target")
        month = month_key(link.get("arrivaldate"))
        hscode = normalize_hscode(link.get("hscode"))

        for company, partner in ((source, target), (target, source)):
            if company not in companies:
                continue

            profile = raw_profiles[company]
            if month:
                profile["monthly_counts"][month] += 1
            if partner:
                profile["partners"].add(partner)
            if hscode:
                profile["hscodes"][hscode] += 1
                if hscode.startswith(FISH_HSCODE_PREFIXES):
                    profile["fish_hscode_count"] += 1
            profile["total_links"] += 1
            profile["total_weightkg"] += float(link.get("weightkg") or 0)
            profile["total_value_omu"] += float(link.get("valueofgoods_omu") or 0)

    rows = []
    for company in sorted(companies):
        profile = raw_profiles.get(company, _empty_company_features())
        monthly_values = list(profile["monthly_counts"].values())
        total_links = profile["total_links"]
        active_months = len(monthly_values)
        max_monthly_count = max(monthly_values) if monthly_values else 0
        avg_monthly_count = (sum(monthly_values) / active_months) if active_months else 0.0

        rows.append(
            {
                "company": company,
                "total_links": total_links,
                "active_months": active_months,
                "max_monthly_count": max_monthly_count,
                "avg_monthly_count": round(avg_monthly_count, 4),
                "partner_count": len(profile["partners"]),
                "hscode_count": len(profile["hscodes"]),
                "fish_hscode_ratio": round(profile["fish_hscode_count"] / total_links, 4) if total_links else 0.0,
                "total_weightkg": round(profile["total_weightkg"], 2),
                "total_value_omu": round(profile["total_value_omu"], 2),
            }
        )

    return rows


def _feature_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """把公司画像转成机器学习矩阵。"""
    feature_names = [
        "total_links",
        "active_months",
        "max_monthly_count",
        "avg_monthly_count",
        "partner_count",
        "hscode_count",
        "fish_hscode_ratio",
        "total_weightkg",
        "total_value_omu",
    ]
    matrix = np.array([[row[name] for name in feature_names] for row in rows], dtype=float)
    return matrix, feature_names


def _business_mode_label(row: dict) -> str:
    """基于公司行为特征给出可解释的商业模式标签。"""
    fish_ratio = float(row.get("fish_hscode_ratio", 0))
    partner_count = float(row.get("partner_count", 0))
    hscode_count = float(row.get("hscode_count", 0))
    max_monthly = float(row.get("max_monthly_count", 0))
    total_links = float(row.get("total_links", 0))

    if fish_ratio >= 0.35 and max_monthly >= 1200:
        return "fish_intensive_hub"
    if fish_ratio >= 0.2 and partner_count >= 1000:
        return "fish_network_expander"
    if partner_count >= 2200 and hscode_count >= 1500:
        return "diversified_broker"
    if total_links >= 100000 and max_monthly >= 2000:
        return "high_volume_distributor"
    if hscode_count < 300 and partner_count < 400:
        return "niche_or_shell"
    return "general_trader"


def _second_level_clusters(scaled: np.ndarray, level1_labels: np.ndarray) -> np.ndarray:
    """在每个一级层次簇内部做二级子聚类，形成二层商业模式结构。"""
    sub_labels = np.full(len(level1_labels), -1, dtype=int)
    for level1 in np.unique(level1_labels):
        indices = np.where(level1_labels == level1)[0]
        cluster_size = len(indices)
        if cluster_size <= 2:
            sub_labels[indices] = 0
            continue

        # 每个一级簇拆成 2-4 个子簇，避免过细导致不可解释。
        sub_cluster_count = min(4, max(2, cluster_size // 45 + 1))
        if sub_cluster_count >= cluster_size:
            sub_cluster_count = max(1, cluster_size - 1)

        if sub_cluster_count <= 1:
            sub_labels[indices] = 0
            continue

        model = AgglomerativeClustering(n_clusters=sub_cluster_count)
        local_labels = model.fit_predict(scaled[indices])
        sub_labels[indices] = local_labels
    return sub_labels


def _third_level_clusters(scaled: np.ndarray, level1_labels: np.ndarray, level2_labels: np.ndarray) -> np.ndarray:
    """在二级簇内部继续拆分三级簇，形成三层层次结构。"""
    lvl3_labels = np.full(len(level1_labels), -1, dtype=int)
    unique_paths = sorted(set((int(a), int(b)) for a, b in zip(level1_labels, level2_labels)))
    for l1, l2 in unique_paths:
        indices = np.where((level1_labels == l1) & (level2_labels == l2))[0]
        cluster_size = len(indices)
        if cluster_size <= 3:
            lvl3_labels[indices] = 0
            continue

        # 在每个二级簇内拆成 2-3 个微簇，用于最外层圈展示。
        sub_cluster_count = min(3, max(2, cluster_size // 25 + 1))
        if sub_cluster_count >= cluster_size:
            sub_cluster_count = max(1, cluster_size - 1)
        if sub_cluster_count <= 1:
            lvl3_labels[indices] = 0
            continue

        model = AgglomerativeClustering(n_clusters=sub_cluster_count)
        local_labels = model.fit_predict(scaled[indices])
        lvl3_labels[indices] = local_labels

    return lvl3_labels


def cluster_and_detect_companies(
    base_graph: dict,
    bundles: dict[str, dict],
    extra_companies: set[str] | None = None,
) -> list[dict]:
    """对公司做无监督异常检测和行为相似聚类。

    默认只处理 bundle 中涉及的公司；传入 extra_companies 可扩展至
    原始图谱中更广泛的代表性公司集合（用于 Q1 时序分析覆盖）。
    """
    companies = affected_companies_from_bundles(bundles)
    if extra_companies:
        companies = companies | extra_companies
    rows = build_company_feature_table(base_graph, companies)
    if not rows:
        return []

    matrix, _feature_names = _feature_matrix(rows)
    scaled = StandardScaler().fit_transform(matrix)

    # Isolation Forest 输出越低越异常，这里转换成 0-1 的 anomaly_score 方便排序。
    isolation = IsolationForest(contamination=0.08, random_state=RANDOM_SEED)
    isolation_labels = isolation.fit_predict(scaled)
    decision_scores = isolation.decision_function(scaled)
    max_score = decision_scores.max()
    min_score = decision_scores.min()
    score_range = max(max_score - min_score, 1e-9)
    anomaly_scores = (max_score - decision_scores) / score_range

    # DBSCAN 用来发现行为相似的小团体，-1 表示离群点。
    dbscan_labels = DBSCAN(eps=1.8, min_samples=4).fit_predict(scaled)

    # 层次聚类提供稳定的分组编号，便于后续可视化做 small multiples。
    cluster_count = min(8, len(rows))
    if cluster_count >= 2:
        hierarchical_labels = AgglomerativeClustering(n_clusters=cluster_count).fit_predict(scaled)
    else:
        hierarchical_labels = np.zeros(len(rows), dtype=int)
    hierarchical_sub_labels = _second_level_clusters(scaled, hierarchical_labels)
    hierarchical_micro_labels = _third_level_clusters(scaled, hierarchical_labels, hierarchical_sub_labels)

    similarities = cosine_similarity(scaled)
    for index, row in enumerate(rows):
        similarity_row = similarities[index].copy()
        similarity_row[index] = -1
        nearest_index = int(np.argmax(similarity_row))
        row["isolation_label"] = "anomaly" if isolation_labels[index] == -1 else "normal"
        row["anomaly_score"] = round(float(anomaly_scores[index]), 4)
        row["dbscan_cluster"] = int(dbscan_labels[index])
        row["hierarchical_cluster"] = int(hierarchical_labels[index])
        row["hierarchical_subcluster"] = int(hierarchical_sub_labels[index])
        row["hierarchical_microcluster"] = int(hierarchical_micro_labels[index])
        row["cluster_path"] = f"C{int(hierarchical_labels[index])}-S{int(hierarchical_sub_labels[index])}"
        row["cluster_path_3"] = (
            f"C{int(hierarchical_labels[index])}-S{int(hierarchical_sub_labels[index])}-T{int(hierarchical_micro_labels[index])}"
        )
        row["business_mode"] = _business_mode_label(row)
        row["nearest_company"] = rows[nearest_index]["company"] if len(rows) > 1 else ""
        row["nearest_similarity"] = round(float(similarity_row[nearest_index]), 4) if len(rows) > 1 else 0.0

    return sorted(rows, key=lambda item: item["anomaly_score"], reverse=True)
