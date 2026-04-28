def _bundle_type(seen_pair_ratio: float) -> str:
    """按预测公司对的已知比例区分 bundle 的预测性质。

    gap_filler        : seen_pair ≥ 0.99，所有对均已知，只补充已有关系的交易记录。
    relationship_ext  : 0.80 ≤ seen_pair < 0.99，主要填补已知关系，少量新连接。
    mixed             : 0.50 ≤ seen_pair < 0.80，已知与新关系混合。
    novel_discovery   : seen_pair < 0.50，主要预测原图中从未出现的新公司对。
    """
    if seen_pair_ratio >= 0.99:
        return "gap_filler"
    if seen_pair_ratio >= 0.80:
        return "relationship_ext"
    if seen_pair_ratio >= 0.50:
        return "mixed"
    return "novel_discovery"


def _repetition_type(unique_pair_ratio: float) -> str:
    """按链接多样性评估预测集的重复程度。

    diverse            : unique_pair ≥ 0.80，每条链接基本对应不同公司对。
    moderately_repeated: 0.35 ≤ unique_pair < 0.80，有一定重复。
    highly_repeated    : unique_pair < 0.35，大量同一对公司被重复预测。
    """
    if unique_pair_ratio >= 0.80:
        return "diverse"
    if unique_pair_ratio >= 0.35:
        return "moderately_repeated"
    return "highly_repeated"


def score_bundle(metrics: dict) -> dict:
    """把可靠性指标转换为 0-100 分，并给出 reliable/suspicious/reject 标签。"""
    score = 0.0

    # 与主图越一致，越像是在补全缺失贸易记录。
    score += 20 * metrics["endpoint_in_base_ratio"]
    score += 25 * metrics["seen_pair_ratio"]
    score += 20 * metrics["valid_hscode_ratio"]
    score += 10 * metrics["physical_field_ratio"]
    score += 10 * metrics["unique_pair_ratio"]

    # 海产品 HS 编码与任务主题相关性（权重从 10 降至 5，为时间一致性腾出空间）。
    score += 5 * metrics["fish_hscode_ratio"]

    # 自监督链接预测分数（留出集 AUC 修正后更可靠）。
    score += 15 * metrics.get("ml_link_probability", 0.0)

    # 时间一致性：预测日期落在双端公司已知活跃期内的比例。
    score += 10 * metrics.get("temporal_consistency_ratio", 0.0)

    # 超出主图日期范围、国家字段缺失、重复集中度过高都会降低可信度。
    score -= 25 * metrics["outside_date_ratio"]
    if metrics["bad_country_count"] > 0:
        score -= 8
    if metrics["max_pair_repeat"] > 50:
        score -= 10
    elif metrics["max_pair_repeat"] > 25:
        score -= 5

    score = round(max(0.0, min(100.0, score)), 2)

    if (
        score >= 60
        and metrics["outside_date_ratio"] == 0
        and metrics["bad_country_count"] == 0
        and metrics["seen_pair_ratio"] >= 0.80
        and metrics["valid_hscode_ratio"] >= 0.80
        and metrics["max_pair_repeat"] <= 40
        and metrics.get("temporal_consistency_ratio", 1.0) >= 0.70
    ):
        label = "reliable"
    elif score >= 50:
        label = "suspicious"
    else:
        label = "reject"

    bundle_type = _bundle_type(metrics["seen_pair_ratio"])
    repetition_type = _repetition_type(metrics["unique_pair_ratio"])

    return {
        **metrics,
        "score": score,
        "label": label,
        "bundle_type": bundle_type,
        "repetition_type": repetition_type,
    }


def score_all_bundles(bundle_metrics: list[dict]) -> list[dict]:
    """批量打分，并按分数从高到低排序，便于可视化展示。"""
    scored = [score_bundle(metrics) for metrics in bundle_metrics]
    return sorted(scored, key=lambda row: row["score"], reverse=True)


def reliable_bundle_names(bundle_scores: list[dict]) -> set[str]:
    """取出最终允许加入图谱的预测集名称。"""
    return {row["bundle"] for row in bundle_scores if row["label"] == "reliable"}
