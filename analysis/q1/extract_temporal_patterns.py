"""提取原始知识图谱中公司的时序行为画像，作为 Q1 可视化的专属数据源。

选取策略（取并集，总量通常 1000–1500 家）：
  - 按贸易总活跃度 Top-800 的公司，确保覆盖高交易量实体。
  - 12 个 bundle 中涉及的所有公司，保持与其他模块（Q2-Q4）的一致性。

时序模式分类（temporal_pattern）：
  - stable     : 长期持续活跃，覆盖率高且波动小。
  - bursty     : 交易量呈脉冲式爆发，最大月交易量 >> 均值。
  - periodic   : 活跃月份间隔存在规律性（约季度周期）。
  - short_term : 生命周期 < 12 个月的短暂公司。
  - general    : 不满足以上任一特征的一般型公司。
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_shared = str(Path(__file__).resolve().parents[1] / "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)

from build_index import month_key, normalize_hscode
from config import FISH_HSCODE_PREFIXES


# 按活跃度选取的公司数量上限。
TOP_ACTIVE_COMPANIES = 800


def _bundle_companies(bundles: dict[str, dict]) -> set[str]:
    """收集所有 bundle 中的公司端点。"""
    companies: set[str] = set()
    for graph in bundles.values():
        for link in graph.get("links", []):
            if link.get("source"):
                companies.add(link["source"])
            if link.get("target"):
                companies.add(link["target"])
    return companies


def select_representative_companies(
    base_graph: dict,
    bundles: dict[str, dict],
    top_n: int = TOP_ACTIVE_COMPANIES,
) -> set[str]:
    """从主图中按总贸易次数选取最活跃的公司，并合并 bundle 涉及公司。

    只遍历一次边列表，避免对 1.5 GB 图做多次 IO。
    """
    activity: Counter[str] = Counter()
    for link in base_graph.get("links", []):
        src = link.get("source")
        tgt = link.get("target")
        if src:
            activity[src] += 1
        if tgt:
            activity[tgt] += 1

    top = {company for company, _ in activity.most_common(top_n)}
    return top | _bundle_companies(bundles)


def _classify_pattern(
    monthly_counts: dict[str, int],
    first_date: str | None,
    last_date: str | None,
) -> str:
    """按四类时序特征对公司活动模式打标签，供 Q1 分析框架使用。"""
    if not monthly_counts or not first_date or not last_date:
        return "short_term"

    active_months = len(monthly_counts)
    values = list(monthly_counts.values())
    mean = sum(values) / active_months
    max_val = max(values)

    try:
        y0, m0 = int(first_date[:4]), int(first_date[5:7])
        y1, m1 = int(last_date[:4]), int(last_date[5:7])
        lifespan = (y1 - y0) * 12 + (m1 - m0) + 1
    except (ValueError, IndexError):
        return "short_term"

    if lifespan < 12:
        return "short_term"

    coverage = active_months / lifespan
    variance = sum((v - mean) ** 2 for v in values) / active_months
    cv = math.sqrt(variance) / max(mean, 1.0)

    # 突发型：最大月交易量远超均值且活跃覆盖稀疏。
    if max_val / max(mean, 1.0) >= 5 and coverage < 0.45:
        return "bursty"

    # 稳定型：长期持续活跃且月度波动较小。
    if coverage >= 0.60 and cv < 1.2:
        return "stable"

    # 周期型：判断活跃月份间隔是否存在规律性（约季度节奏）。
    sorted_months = sorted(monthly_counts.keys())
    if len(sorted_months) >= 8:
        gaps: list[int] = []
        for i in range(1, len(sorted_months)):
            y_a, m_a = int(sorted_months[i - 1][:4]), int(sorted_months[i - 1][5:7])
            y_b, m_b = int(sorted_months[i][:4]), int(sorted_months[i][5:7])
            gaps.append((y_b - y_a) * 12 + (m_b - m_a))
        gap_mean = sum(gaps) / len(gaps)
        gap_var = sum((g - gap_mean) ** 2 for g in gaps) / len(gaps)
        # 平均间隔 1.5–4.5 个月且方差较小，认为具有周期性。
        if 1.5 <= gap_mean <= 4.5 and gap_var <= 4.0:
            return "periodic"

    return "general"


def extract_temporal_patterns(
    base_graph: dict,
    bundles: dict[str, dict],
) -> list[dict]:
    """提取代表性公司在原始图谱中的完整月度时序画像。

    只遍历一次主图边列表（约 540 万条），避免重复 IO。
    返回列表按公司名称排序，供 export_results.py 写出为 JSON。
    """
    companies = select_representative_companies(base_graph, bundles)

    # 使用 defaultdict 避免对每家公司单独初始化。
    raw: dict[str, dict] = defaultdict(
        lambda: {
            "monthly_counts": Counter(),
            "partners": set(),
            "fish_links": 0,
            "total_links": 0,
            "first_date": None,
            "last_date": None,
        }
    )

    for link in base_graph.get("links", []):
        src = link.get("source")
        tgt = link.get("target")
        date = link.get("arrivaldate")
        hscode = normalize_hscode(link.get("hscode"))
        month = month_key(date)
        is_fish = hscode.startswith(FISH_HSCODE_PREFIXES)

        for company, partner in ((src, tgt), (tgt, src)):
            if company not in companies:
                continue
            p = raw[company]
            p["total_links"] += 1
            if month:
                p["monthly_counts"][month] += 1
            if partner:
                p["partners"].add(partner)
            if is_fish:
                p["fish_links"] += 1
            if date:
                p["first_date"] = (
                    date if p["first_date"] is None else min(p["first_date"], date)
                )
                p["last_date"] = (
                    date if p["last_date"] is None else max(p["last_date"], date)
                )

    result: list[dict] = []
    for company in sorted(companies):
        p = raw.get(company)
        if not p:
            continue
        total = p["total_links"]
        monthly_counts = dict(p["monthly_counts"])
        first_date = p["first_date"]
        last_date = p["last_date"]
        result.append(
            {
                "company": company,
                "monthly_counts": monthly_counts,
                "first_date": first_date,
                "last_date": last_date,
                "total_links": total,
                "active_months": len(monthly_counts),
                "partner_count": len(p["partners"]),
                "fish_hscode_ratio": (
                    round(p["fish_links"] / total, 4) if total else 0.0
                ),
                "temporal_pattern": _classify_pattern(
                    monthly_counts, first_date, last_date
                ),
            }
        )

    return result
