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


_Q1_PATTERN_RISK = {
    "short_term": 3,   # 最高风险：历史上短暂注册即停业
    "bursty":     2,   # 中高风险：突发性交易
    "general":    1,
    "periodic":   1,
    "stable":     0,   # 最低风险：稳定合规
    "":           1,
}


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


def build_company_feature_table(
    base_graph: dict,
    companies: set[str],
    q1_index: dict[str, dict] | None = None,
    q3_index: dict[str, dict] | None = None,
    bridge_index: dict[str, int] | None = None,
    relay_successors: set[str] | None = None,
) -> list[dict]:
    """从主图中抽取公司级行为特征，并融合 Q1/Q3 外部信号。

    q1_index   : company → Q1 temporal pattern 记录
    q3_index   : company → anomaly_delta 记录（含 suspicious_revival_score）
    bridge_index: company → bridge_scope（桥接可达节点数）
    relay_successors: 作为接力接班方的公司集合
    """
    q1_index       = q1_index or {}
    q3_index       = q3_index or {}
    bridge_index   = bridge_index or {}
    relay_successors = relay_successors or set()

    raw_profiles = defaultdict(_empty_company_features)

    for link in base_graph.get("links", []):
        source = link.get("source")
        target = link.get("target")
        month  = month_key(link.get("arrivaldate"))
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
        total_links    = profile["total_links"]
        active_months  = len(monthly_values)
        max_monthly    = max(monthly_values) if monthly_values else 0
        avg_monthly    = (sum(monthly_values) / active_months) if active_months else 0.0

        # 变异系数：衡量月活跃度的波动程度
        if active_months > 1 and avg_monthly > 0:
            variance = sum((v - avg_monthly) ** 2 for v in monthly_values) / active_months
            cv = variance ** 0.5 / avg_monthly
        else:
            cv = 0.0

        # Q1 时序模式信号
        q1 = q1_index.get(company, {})
        q1_pattern = q1.get("temporal_pattern", "")
        q1_pattern_risk = _Q1_PATTERN_RISK.get(q1_pattern, 1)

        # Q3 复活信号
        q3 = q3_index.get(company, {})
        revival_score  = float(q3.get("suspicious_revival_score", 0.0))
        dormancy_months = int(q3.get("dormancy_months", 0))
        dormancy_weight = float(q3.get("dormancy_weight", 1.0))
        has_extended   = 1 if q3.get("extended_months") else 0
        q1_inconsistent = 1 if q3.get("q1_inconsistency_reason") else 0

        # Q3 网络模式信号
        bridge_scope   = int(bridge_index.get(company, 0))
        is_relay_succ  = 1 if company in relay_successors else 0

        row = {
            "company": company,
            # ── 行为特征（原有）──
            "total_links":        total_links,
            "active_months":      active_months,
            "max_monthly_count":  max_monthly,
            "avg_monthly_count":  round(avg_monthly, 4),
            "activity_cv":        round(cv, 4),
            "partner_count":      len(profile["partners"]),
            "hscode_count":       len(profile["hscodes"]),
            "fish_hscode_ratio":  round(profile["fish_hscode_count"] / total_links, 4) if total_links else 0.0,
            "total_weightkg":     round(profile["total_weightkg"], 2),
            "total_value_omu":    round(profile["total_value_omu"], 2),
            # ── Q1 信号 ──
            "q1_temporal_pattern": q1_pattern,
            "q1_pattern_risk":     q1_pattern_risk,
            # ── Q3 信号 ──
            "revival_score":       round(revival_score, 1),
            "dormancy_months":     dormancy_months,
            "dormancy_weight":     round(dormancy_weight, 2),
            "has_extended_months": has_extended,
            "q1_inconsistent":     q1_inconsistent,
            # ── Q3 网络信号 ──
            "bridge_scope":        bridge_scope,
            "is_relay_successor":  is_relay_succ,
        }
        row["business_mode"] = _business_mode_label(row)
        rows.append(row)

    return rows


def _feature_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """把公司画像转成机器学习矩阵（含 Q1/Q3 外部信号）。"""
    feature_names = [
        # 行为特征
        "total_links",
        "active_months",
        "max_monthly_count",
        "avg_monthly_count",
        "activity_cv",
        "partner_count",
        "hscode_count",
        "fish_hscode_ratio",
        "total_weightkg",
        "total_value_omu",
        # Q1 信号
        "q1_pattern_risk",
        # Q3 信号
        "revival_score",
        "dormancy_months",
        "has_extended_months",
        "q1_inconsistent",
        # 网络信号
        "bridge_scope",
        "is_relay_successor",
    ]
    matrix = np.array([[row[name] for name in feature_names] for row in rows], dtype=float)
    return matrix, feature_names


def _business_mode_label(row: dict) -> str:
    """基于公司行为特征给出可解释的商业模式标签。

    判别顺序从"最特殊"到"最普通"，降低 niche_or_shell 的误伤率。
    """
    fish_ratio    = float(row.get("fish_hscode_ratio", 0))
    partner_count = float(row.get("partner_count", 0))
    hscode_count  = float(row.get("hscode_count", 0))
    max_monthly   = float(row.get("max_monthly_count", 0))
    total_links   = float(row.get("total_links", 0))
    active_months = float(row.get("active_months", 0))
    revival_score = float(row.get("revival_score", 0))
    dormancy      = float(row.get("dormancy_months", 0))
    q1_risk       = int(row.get("q1_pattern_risk", 1))

    # 大型多品类经纪商
    if partner_count >= 2000 and hscode_count >= 1000:
        return "diversified_broker"
    # 大规模高频分发商
    if total_links >= 50000 and max_monthly >= 1500:
        return "high_volume_distributor"
    # 海产品密集枢纽（鱼类比例高 + 有一定规模）
    if fish_ratio >= 0.3 and partner_count >= 200:
        return "fish_intensive_hub"
    # 鱼类扩张者（中规模，鱼类比例适中）
    if fish_ratio >= 0.15 and partner_count >= 100:
        return "fish_network_expander"
    # 复活嫌疑公司：停活后重新出现，历史交易少
    if revival_score >= 30 or (dormancy >= 12 and q1_risk >= 2 and total_links < 200):
        return "dormant_revival"
    # 短暂活跃 / 周期短（注册即用）
    if active_months <= 6 and total_links < 500:
        return "short_lived"
    # 中等活跃的普通贸易商
    if total_links >= 500 or partner_count >= 50:
        return "general_trader"
    # 小规模 / 低活跃（默认兜底，覆盖率应大幅降低）
    return "niche_or_shell"


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
    temporal_patterns: list[dict] | None = None,
    anomaly_delta: list[dict] | None = None,
    bridge_companies: list[dict] | None = None,
    relay_chains: list[dict] | None = None,
) -> list[dict]:
    """对公司做无监督异常检测和行为相似聚类（融合 Q1/Q3 外部信号）。

    temporal_patterns : Q1 时序模式列表（每项含 company, temporal_pattern）
    anomaly_delta     : Q3 公司级复活评分列表
    bridge_companies  : Q3 桥接公司列表
    relay_chains      : Q3 接力链列表
    """
    # 构建外部信号索引
    q1_index: dict[str, dict] = {r["company"]: r for r in (temporal_patterns or [])}
    q3_index: dict[str, dict] = {r["company"]: r for r in (anomaly_delta or [])}
    bridge_index: dict[str, int] = {
        r["company"]: int(r["bridge_scope"])
        for r in (bridge_companies or [])
    }
    relay_successors: set[str] = {
        r["successor"] for r in (relay_chains or [])
    }

    companies = affected_companies_from_bundles(bundles)
    if extra_companies:
        companies = companies | extra_companies
    rows = build_company_feature_table(
        base_graph, companies,
        q1_index=q1_index,
        q3_index=q3_index,
        bridge_index=bridge_index,
        relay_successors=relay_successors,
    )
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
