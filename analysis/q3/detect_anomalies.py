import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

_shared = str(Path(__file__).resolve().parents[1] / "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)

from build_index import month_key, normalize_hscode


def _empty_profile() -> dict:
    """公司画像只保留可解释字段，避免为可视化输出过多原始边。"""
    return {
        "first_date": None,
        "last_date": None,
        "monthly_counts": Counter(),
        "partners": set(),
        "hscodes": Counter(),
        "total_links": 0,
        "total_weightkg": 0.0,
        "total_value_omu": 0.0,
    }


def _update_profile(profile: dict, link: dict, company: str) -> None:
    """把一条贸易边累加到某个公司的画像中。"""
    source = link.get("source")
    target = link.get("target")
    date = link.get("arrivaldate")
    month = month_key(date)
    partner = target if company == source else source

    if date:
        profile["first_date"] = date if profile["first_date"] is None else min(profile["first_date"], date)
        profile["last_date"] = date if profile["last_date"] is None else max(profile["last_date"], date)
    if month:
        profile["monthly_counts"][month] += 1
    if partner:
        profile["partners"].add(partner)

    hscode = normalize_hscode(link.get("hscode"))
    if hscode:
        profile["hscodes"][hscode] += 1

    profile["total_links"] += 1
    profile["total_weightkg"] += float(link.get("weightkg") or 0)
    profile["total_value_omu"] += float(link.get("valueofgoods_omu") or 0)


def build_profiles_for_companies(base_graph: dict, companies: set[str]) -> dict[str, dict]:
    """只为受预测链接影响的公司建画像，避免对 3 万多个节点全部输出。"""
    profiles = defaultdict(_empty_profile)

    for link in base_graph.get("links", []):
        source = link.get("source")
        target = link.get("target")
        if source in companies:
            _update_profile(profiles[source], link, source)
        if target in companies:
            _update_profile(profiles[target], link, target)

    return dict(profiles)


def build_prediction_profiles(reliable_links: list[dict]) -> dict[str, dict]:
    """为新增可靠链接单独建画像，用来和主图画像对比。"""
    profiles = defaultdict(_empty_profile)
    for link in reliable_links:
        source = link.get("source")
        target = link.get("target")
        if source:
            _update_profile(profiles[source], link, source)
        if target:
            _update_profile(profiles[target], link, target)
    return dict(profiles)


def _monthly_threshold(monthly_counts: Counter) -> float:
    """用均值 + 3 倍标准差识别新增链接造成的交易突增。"""
    values = list(monthly_counts.values())
    if not values:
        return 5.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return max(5.0, mean + 3 * sqrt(variance))


def compare_company_profile(company: str, before: dict | None, predicted: dict) -> dict:
    """比较加边前后某公司的变化，输出可解释的异常特征。"""
    before = before or _empty_profile()
    before_months = set(before["monthly_counts"])
    predicted_months = set(predicted["monthly_counts"])
    threshold = _monthly_threshold(before["monthly_counts"])

    filled_gap_months = []
    extended_months = []
    burst_months = []

    for month, count in predicted["monthly_counts"].items():
        inside_life_span = before["first_date"] and before["last_date"] and before["first_date"][:7] <= month <= before["last_date"][:7]
        if inside_life_span and month not in before_months:
            filled_gap_months.append(month)
        if before["last_date"] and month > before["last_date"][:7]:
            extended_months.append(month)
        if before["monthly_counts"].get(month, 0) + count > threshold:
            burst_months.append(month)

    new_partners = predicted["partners"] - before["partners"]
    new_hscodes = set(predicted["hscodes"]) - set(before["hscodes"])

    # 分数越高，说明新增可靠链接越明显改变了该公司的时间/关系模式。
    suspicious_revival_score = (
        3 * len(filled_gap_months)
        + 2 * len(extended_months)
        + 2 * len(burst_months)
        + min(5, len(new_partners))
        + min(5, len(new_hscodes))
    )

    return {
        "company": company,
        "base_first_date": before["first_date"],
        "base_last_date": before["last_date"],
        "base_link_count": before["total_links"],
        "predicted_link_count": predicted["total_links"],
        "filled_gap_months": ";".join(sorted(filled_gap_months)),
        "extended_months": ";".join(sorted(extended_months)),
        "burst_months": ";".join(sorted(burst_months)),
        "new_partner_count": len(new_partners),
        "new_hscode_count": len(new_hscodes),
        "predicted_weightkg": round(predicted["total_weightkg"], 2),
        "predicted_value_omu": round(predicted["total_value_omu"], 2),
        "suspicious_revival_score": suspicious_revival_score,
    }


def compare_anomalies(base_graph: dict, reliable_links: list[dict]) -> list[dict]:
    """对比加入可靠预测链接前后的公司画像变化。"""
    affected_companies = {
        company
        for link in reliable_links
        for company in (link.get("source"), link.get("target"))
        if company
    }

    base_profiles = build_profiles_for_companies(base_graph, affected_companies)
    prediction_profiles = build_prediction_profiles(reliable_links)

    rows = [
        compare_company_profile(company, base_profiles.get(company), predicted)
        for company, predicted in prediction_profiles.items()
    ]

    return sorted(rows, key=lambda row: row["suspicious_revival_score"], reverse=True)
