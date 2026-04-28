import sys
from collections import Counter
from pathlib import Path

_shared = str(Path(__file__).resolve().parents[1] / "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)

from build_index import normalize_hscode
from config import FISH_HSCODE_PREFIXES


def _safe_ratio(numerator: int, denominator: int) -> float:
    """避免除零，所有比例统一返回 0 到 1。"""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _is_fish_hscode(value) -> bool:
    """根据 HS 编码前缀粗略判断是否为海产品相关商品。"""
    return normalize_hscode(value).startswith(FISH_HSCODE_PREFIXES)


def evaluate_bundle(bundle_name: str, bundle_graph: dict, base_index: dict) -> dict:
    """计算单个预测链接集的可靠性指标，只计算事实，不在这里做主观分类。"""
    links = bundle_graph.get("links", [])
    nodes = bundle_graph.get("nodes", [])
    base_nodes = base_index["base_nodes"]
    base_pairs = base_index["base_pairs"]
    base_exact_edges = base_index["base_exact_edges"]
    base_hscodes = base_index["base_hscodes"]
    date_min = base_index["date_min"]
    date_max = base_index["date_max"]

    endpoints = [endpoint for link in links for endpoint in (link.get("source"), link.get("target"))]
    endpoint_in_base = sum(1 for endpoint in endpoints if endpoint in base_nodes)
    node_in_base = sum(1 for node in nodes if node.get("id") in base_nodes)

    seen_pair = 0
    exact_duplicate = 0
    valid_hscode = 0
    fish_hscode = 0
    outside_date = 0
    physical_fields = 0
    pair_counter = Counter()

    for link in links:
        source = link.get("source")
        target = link.get("target")
        date = link.get("arrivaldate")
        hscode = normalize_hscode(link.get("hscode"))
        pair = (source, target)
        pair_counter[pair] += 1

        if pair in base_pairs:
            seen_pair += 1
        if (source, target, date, hscode) in base_exact_edges:
            exact_duplicate += 1
        if hscode in base_hscodes:
            valid_hscode += 1
        if _is_fish_hscode(hscode):
            fish_hscode += 1
        if date_min and date_max and (not date or date < date_min or date > date_max):
            outside_date += 1
        if all(key in link and link.get(key) is not None for key in ("valueofgoods_omu", "volumeteu", "weightkg")):
            physical_fields += 1

    bad_country_count = sum(
        1
        for node in nodes
        if str(node.get("shpcountry")) == "-27" or str(node.get("rcvcountry")) == "-27"
    )

    return {
        "bundle": bundle_name,
        "link_count": len(links),
        "node_count": len(nodes),
        "endpoint_in_base_ratio": _safe_ratio(endpoint_in_base, len(endpoints)),
        "node_in_base_ratio": _safe_ratio(node_in_base, len(nodes)),
        "seen_pair_ratio": _safe_ratio(seen_pair, len(links)),
        "exact_duplicate_ratio": _safe_ratio(exact_duplicate, len(links)),
        "valid_hscode_ratio": _safe_ratio(valid_hscode, len(links)),
        "fish_hscode_ratio": _safe_ratio(fish_hscode, len(links)),
        "outside_date_ratio": _safe_ratio(outside_date, len(links)),
        "physical_field_ratio": _safe_ratio(physical_fields, len(links)),
        "unique_pair_ratio": _safe_ratio(len(pair_counter), len(links)),
        "max_pair_repeat": max(pair_counter.values()) if pair_counter else 0,
        "bad_country_count": bad_country_count,
    }


def evaluate_all_bundles(bundles: dict[str, dict], base_index: dict) -> list[dict]:
    """批量评估所有预测链接集。"""
    return [
        evaluate_bundle(bundle_name, bundle_graph, base_index)
        for bundle_name, bundle_graph in sorted(bundles.items())
    ]
