def score_bundle(metrics: dict) -> dict:
    """把可靠性指标转换为 0-100 分，并给出 reliable/suspicious/reject 标签。"""
    score = 0.0

    # 与主图越一致，越像是在补全缺失贸易记录。
    score += 20 * metrics["endpoint_in_base_ratio"]
    score += 25 * metrics["seen_pair_ratio"]
    score += 20 * metrics["valid_hscode_ratio"]
    score += 10 * metrics["physical_field_ratio"]
    score += 10 * metrics["unique_pair_ratio"]

    # 海产品 HS 编码不是唯一标准，但能帮助识别与任务主题更相关的预测集。
    score += 10 * metrics["fish_hscode_ratio"]

    # 自监督链接预测分数：越像 2034 真实会出现的边，越可信。
    score += 15 * metrics.get("ml_link_probability", 0.0)

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
    ):
        label = "reliable"
    elif score >= 50:
        label = "suspicious"
    else:
        label = "reject"

    return {
        **metrics,
        "score": score,
        "label": label,
    }


def score_all_bundles(bundle_metrics: list[dict]) -> list[dict]:
    """批量打分，并按分数从高到低排序，便于可视化展示。"""
    scored = [score_bundle(metrics) for metrics in bundle_metrics]
    return sorted(scored, key=lambda row: row["score"], reverse=True)


def reliable_bundle_names(bundle_scores: list[dict]) -> set[str]:
    """取出最终允许加入图谱的预测集名称。"""
    return {row["bundle"] for row in bundle_scores if row["label"] == "reliable"}
