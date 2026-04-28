#!/usr/bin/env python3
"""Build compact data for the MC2 3D dynamic visualization.

The browser should not load the 5.4M-edge base graph directly. This builder
keeps the visual dataset focused on predicted-link companies, anomaly-ranked
companies, and their monthly activity summaries.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MC2_ROOT = PROJECT_ROOT / "MC2"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
BASE_GRAPH_PATH = MC2_ROOT / "mc2_challenge_graph.json"
BUNDLE_DIR = MC2_ROOT / "bundles"
OUTPUT_JS = PROJECT_ROOT / "visualization" / "data" / "mc2_3d_data.js"

MAX_COMPANIES = 420
MAX_BASE_CONTEXT_LINKS = 1400
MAX_PREDICTED_LINKS = 2600
FISH_HSCODE_PREFIXES = ("301", "302", "303", "304", "305", "306", "307", "308", "1604", "1605")


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def to_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def month_key(date_text: str | None) -> str:
    return str(date_text or "")[:7]


def normalize_hscode(value) -> str:
    return str(value or "").strip()


def is_fish_hscode(value) -> bool:
    return normalize_hscode(value).startswith(FISH_HSCODE_PREFIXES)


def split_months(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in str(value).split(";") if item]


def numeric_bundle_rows(rows: list[dict]) -> list[dict]:
    numeric_fields = {
        "link_count",
        "node_count",
        "endpoint_in_base_ratio",
        "node_in_base_ratio",
        "seen_pair_ratio",
        "exact_duplicate_ratio",
        "valid_hscode_ratio",
        "fish_hscode_ratio",
        "outside_date_ratio",
        "physical_field_ratio",
        "unique_pair_ratio",
        "max_pair_repeat",
        "bad_country_count",
        "ml_link_probability",
        "ml_link_probability_p90",
        "ml_validation_auc",
        "ml_training_pairs",
        "score",
    }
    typed_rows = []
    for row in rows:
        typed = dict(row)
        for field in numeric_fields:
            if field in typed:
                typed[field] = to_float(typed[field])
        typed_rows.append(typed)
    return typed_rows


def numeric_company_rows(rows: list[dict]) -> dict[str, dict]:
    typed = {}
    numeric_fields = {
        "total_links",
        "active_months",
        "max_monthly_count",
        "avg_monthly_count",
        "partner_count",
        "hscode_count",
        "fish_hscode_ratio",
        "total_weightkg",
        "total_value_omu",
        "anomaly_score",
        "dbscan_cluster",
        "hierarchical_cluster",
        "nearest_similarity",
    }
    for row in rows:
        company = row.get("company")
        if not company:
            continue
        item = dict(row)
        for field in numeric_fields:
            if field in item:
                item[field] = to_float(item[field])
        typed[company] = item
    return typed


def numeric_anomaly_rows(rows: list[dict]) -> dict[str, dict]:
    typed = {}
    numeric_fields = {
        "base_link_count",
        "predicted_link_count",
        "new_partner_count",
        "new_hscode_count",
        "predicted_weightkg",
        "predicted_value_omu",
        "suspicious_revival_score",
    }
    for row in rows:
        company = row.get("company")
        if not company:
            continue
        item = dict(row)
        for field in numeric_fields:
            if field in item:
                item[field] = to_float(item[field])
        item["filled_gap_months"] = split_months(item.get("filled_gap_months"))
        item["extended_months"] = split_months(item.get("extended_months"))
        item["burst_months"] = split_months(item.get("burst_months"))
        typed[company] = item
    return typed


def collect_predicted_links(bundle_scores: dict[str, dict]) -> tuple[list[dict], set[str]]:
    predicted_links = []
    companies = set()

    for path in sorted(BUNDLE_DIR.glob("*.json")):
        bundle_name = path.stem
        bundle_info = bundle_scores.get(bundle_name, {})
        graph = read_json(path)
        for link in graph.get("links", []):
            source = link.get("source")
            target = link.get("target")
            if not source or not target:
                continue
            companies.update((source, target))
            predicted_links.append(
                {
                    "id": f"pred::{bundle_name}::{len(predicted_links)}",
                    "source": source,
                    "target": target,
                    "date": link.get("arrivaldate"),
                    "month": month_key(link.get("arrivaldate")),
                    "hscode": normalize_hscode(link.get("hscode")),
                    "isFish": is_fish_hscode(link.get("hscode")),
                    "bundle": bundle_name,
                    "label": bundle_info.get("label", "unknown"),
                    "bundleScore": to_float(bundle_info.get("score")),
                    "mlProbability": to_float(bundle_info.get("ml_link_probability")),
                    "weightkg": to_float(link.get("weightkg")),
                    "value_omu": to_float(link.get("valueofgoods_omu")),
                    "volumeteu": to_float(link.get("volumeteu")),
                    "kind": "prediction",
                }
            )

    predicted_links.sort(
        key=lambda item: (
            item["label"] != "reliable",
            -item["bundleScore"],
            -item["mlProbability"],
            -item["weightkg"],
        )
    )
    return predicted_links[:MAX_PREDICTED_LINKS], companies


def select_companies(
    company_rows: dict[str, dict],
    anomaly_rows: dict[str, dict],
    predicted_companies: set[str],
) -> list[str]:
    scores = Counter()

    for company in predicted_companies:
        scores[company] += 2
    for rank, (company, row) in enumerate(company_rows.items()):
        scores[company] += max(0, 220 - rank) + 200 * to_float(row.get("anomaly_score"))
    for company, row in anomaly_rows.items():
        scores[company] += 80 + 25 * to_float(row.get("suspicious_revival_score"))

    selected = [company for company, _score in scores.most_common(MAX_COMPANIES)]
    return selected


def aggregate_base_activity(companies: set[str]) -> tuple[list[dict], list[dict], list[str]]:
    base_graph = read_json(BASE_GRAPH_PATH)
    monthly = defaultdict(lambda: {"count": 0, "weightkg": 0.0, "value_omu": 0.0, "fishCount": 0, "hscodes": set()})
    context_links = Counter()
    months = set()

    for link in base_graph.get("links", []):
        source = link.get("source")
        target = link.get("target")
        month = month_key(link.get("arrivaldate"))
        if not month:
            continue

        hscode = normalize_hscode(link.get("hscode"))
        weight = to_float(link.get("weightkg"))
        value = to_float(link.get("valueofgoods_omu"))

        for company in (source, target):
            if company not in companies:
                continue
            key = (company, month)
            monthly[key]["count"] += 1
            monthly[key]["weightkg"] += weight
            monthly[key]["value_omu"] += value
            if hscode:
                monthly[key]["hscodes"].add(hscode)
            if is_fish_hscode(hscode):
                monthly[key]["fishCount"] += 1
            months.add(month)

        if source in companies and target in companies and source != target:
            context_links[tuple(sorted((source, target)))] += 1

    events = []
    for (company, month), values in monthly.items():
        events.append(
            {
                "id": f"base::{company}::{month}",
                "company": company,
                "month": month,
                "kind": "base",
                "count": values["count"],
                "weightkg": round(values["weightkg"], 2),
                "value_omu": round(values["value_omu"], 2),
                "fishCount": values["fishCount"],
                "hscodeCount": len(values["hscodes"]),
            }
        )

    base_links = [
        {
            "id": f"base::{idx}",
            "source": pair[0],
            "target": pair[1],
            "count": count,
            "kind": "baseContext",
        }
        for idx, (pair, count) in enumerate(context_links.most_common(MAX_BASE_CONTEXT_LINKS))
    ]

    return events, base_links, sorted(months)


def aggregate_predicted_events(predicted_links: Iterable[dict]) -> tuple[list[dict], list[str]]:
    monthly = defaultdict(lambda: {"count": 0, "weightkg": 0.0, "value_omu": 0.0, "fishCount": 0, "hscodes": set(), "bundles": set(), "labels": set()})
    months = set()

    for link in predicted_links:
        month = link.get("month")
        if not month:
            continue
        for company in (link["source"], link["target"]):
            key = (company, month)
            monthly[key]["count"] += 1
            monthly[key]["weightkg"] += to_float(link.get("weightkg"))
            monthly[key]["value_omu"] += to_float(link.get("value_omu"))
            monthly[key]["bundles"].add(link["bundle"])
            monthly[key]["labels"].add(link["label"])
            if link.get("hscode"):
                monthly[key]["hscodes"].add(link["hscode"])
            if link.get("isFish"):
                monthly[key]["fishCount"] += 1
            months.add(month)

    events = []
    for (company, month), values in monthly.items():
        labels = sorted(values["labels"])
        events.append(
            {
                "id": f"pred::{company}::{month}",
                "company": company,
                "month": month,
                "kind": "prediction",
                "count": values["count"],
                "weightkg": round(values["weightkg"], 2),
                "value_omu": round(values["value_omu"], 2),
                "fishCount": values["fishCount"],
                "hscodeCount": len(values["hscodes"]),
                "bundles": sorted(values["bundles"]),
                "label": labels[0] if labels else "unknown",
            }
        )
    return events, sorted(months)


def build_companies(selected: list[str], company_rows: dict[str, dict], anomaly_rows: dict[str, dict]) -> list[dict]:
    companies = []
    for company in selected:
        cluster = company_rows.get(company, {})
        anomaly = anomaly_rows.get(company, {})
        companies.append(
            {
                "id": company,
                "company": company,
                "total_links": to_float(cluster.get("total_links")),
                "active_months": to_float(cluster.get("active_months")),
                "partner_count": to_float(cluster.get("partner_count")),
                "hscode_count": to_float(cluster.get("hscode_count")),
                "fish_hscode_ratio": to_float(cluster.get("fish_hscode_ratio")),
                "total_weightkg": to_float(cluster.get("total_weightkg")),
                "total_value_omu": to_float(cluster.get("total_value_omu")),
                "anomaly_score": to_float(cluster.get("anomaly_score")),
                "dbscan_cluster": to_int(cluster.get("dbscan_cluster"), 999),
                "hierarchical_cluster": to_int(cluster.get("hierarchical_cluster"), 999),
                "nearest_company": cluster.get("nearest_company", ""),
                "nearest_similarity": to_float(cluster.get("nearest_similarity")),
                "base_first_date": anomaly.get("base_first_date", ""),
                "base_last_date": anomaly.get("base_last_date", ""),
                "base_link_count": to_float(anomaly.get("base_link_count")),
                "predicted_link_count": to_float(anomaly.get("predicted_link_count")),
                "filled_gap_months": anomaly.get("filled_gap_months", []),
                "extended_months": anomaly.get("extended_months", []),
                "burst_months": anomaly.get("burst_months", []),
                "new_partner_count": to_float(anomaly.get("new_partner_count")),
                "new_hscode_count": to_float(anomaly.get("new_hscode_count")),
                "suspicious_revival_score": to_float(anomaly.get("suspicious_revival_score")),
            }
        )
    return companies


def build_similarity_links(companies: list[dict], selected: set[str]) -> list[dict]:
    links = []
    for company in companies:
        nearest = company.get("nearest_company")
        if nearest and nearest in selected and nearest != company["id"]:
            links.append(
                {
                    "id": f"sim::{company['id']}::{nearest}",
                    "source": company["id"],
                    "target": nearest,
                    "similarity": company.get("nearest_similarity", 0),
                    "kind": "similarity",
                }
            )
    return links


def build_routes(base_links: list[dict], predicted_links: list[dict]) -> list[dict]:
    """把公司间关系聚合成航线频率图需要的 route。"""
    routes = {}

    for link in base_links:
        key = (link["source"], link["target"])
        routes[key] = {
            "id": f"route::base::{link['source']}::{link['target']}",
            "source": link["source"],
            "target": link["target"],
            "kind": "base",
            "label": "base",
            "count": to_float(link.get("count")),
            "baseCount": to_float(link.get("count")),
            "predictedCount": 0,
            "weightkg": 0.0,
            "value_omu": 0.0,
            "score": 0.0,
            "mlProbability": 0.0,
            "bundles": [],
            "months": [],
        }

    for link in predicted_links:
        key = (link["source"], link["target"])
        if key not in routes:
            routes[key] = {
                "id": f"route::pred::{link['source']}::{link['target']}",
                "source": link["source"],
                "target": link["target"],
                "kind": "prediction",
                "label": link.get("label", "unknown"),
                "count": 0,
                "baseCount": 0,
                "predictedCount": 0,
                "weightkg": 0.0,
                "value_omu": 0.0,
                "score": 0.0,
                "mlProbability": 0.0,
                "bundles": [],
                "months": [],
            }
        route = routes[key]
        route["kind"] = "mixed" if route["baseCount"] else "prediction"
        route["label"] = link.get("label", route["label"])
        route["count"] += 1
        route["predictedCount"] += 1
        route["weightkg"] += to_float(link.get("weightkg"))
        route["value_omu"] += to_float(link.get("value_omu"))
        route["score"] = max(route["score"], to_float(link.get("bundleScore")))
        route["mlProbability"] = max(route["mlProbability"], to_float(link.get("mlProbability")))
        if link.get("bundle") and link["bundle"] not in route["bundles"]:
            route["bundles"].append(link["bundle"])
        if link.get("month") and link["month"] not in route["months"]:
            route["months"].append(link["month"])

    sorted_routes = sorted(
        routes.values(),
        key=lambda item: (
            item["label"] != "reliable",
            -item["predictedCount"],
            -item["baseCount"],
            -item["score"],
            -item["weightkg"],
        ),
    )
    for route in sorted_routes:
        route["weightkg"] = round(route["weightkg"], 2)
        route["value_omu"] = round(route["value_omu"], 2)
        route["months"] = sorted(route["months"])
        route["bundles"] = sorted(route["bundles"])
    return sorted_routes[:2400]


def build_lifecycle_flows(companies: list[dict]) -> list[dict]:
    """构造 D 方案的推断生命周期流：first -> last -> gap/extended/burst。"""
    flows = []
    for company in companies:
        score = to_float(company.get("suspicious_revival_score"))
        anomaly = to_float(company.get("anomaly_score"))
        if score <= 0 and anomaly < 0.55:
            continue

        stages = []
        if company.get("base_first_date"):
            stages.append({"stage": "firstSeen", "month": company["base_first_date"][:7]})
        if company.get("base_last_date"):
            stages.append({"stage": "lastSeen", "month": company["base_last_date"][:7]})
        for month in company.get("filled_gap_months", []):
            stages.append({"stage": "gapFilled", "month": month})
        for month in company.get("extended_months", []):
            stages.append({"stage": "extended", "month": month})
        for month in company.get("burst_months", []):
            stages.append({"stage": "burst", "month": month})

        stages = [stage for stage in stages if stage["month"]]
        stages.sort(key=lambda stage: stage["month"])
        if len(stages) < 2:
            continue

        for index in range(1, len(stages)):
            flows.append(
                {
                    "id": f"life::{company['id']}::{index}",
                    "company": company["id"],
                    "sourceStage": stages[index - 1]["stage"],
                    "sourceMonth": stages[index - 1]["month"],
                    "targetStage": stages[index]["stage"],
                    "targetMonth": stages[index]["month"],
                    "value": max(1, score + 2 * anomaly),
                    "anomaly_score": anomaly,
                    "suspicious_revival_score": score,
                }
            )

    return sorted(flows, key=lambda item: (-item["value"], item["company"]))[:1200]


def build_dataset() -> dict:
    bundle_rows = numeric_bundle_rows(read_csv_rows(OUTPUTS_ROOT / "q2" / "bundle_reliability.csv"))
    bundle_scores = {row["bundle"]: row for row in bundle_rows}
    company_rows = numeric_company_rows(read_csv_rows(OUTPUTS_ROOT / "q4" / "company_clusters.csv"))
    anomaly_rows = numeric_anomaly_rows(read_csv_rows(OUTPUTS_ROOT / "q3" / "anomaly_delta.csv"))

    predicted_links, predicted_companies = collect_predicted_links(bundle_scores)
    selected_companies = select_companies(company_rows, anomaly_rows, predicted_companies)
    selected_set = set(selected_companies)

    base_events, base_context_links, base_months = aggregate_base_activity(selected_set)
    predicted_events, predicted_months = aggregate_predicted_events(
        link for link in predicted_links if link["source"] in selected_set or link["target"] in selected_set
    )
    companies = build_companies(selected_companies, company_rows, anomaly_rows)
    similarity_links = build_similarity_links(companies, selected_set)

    filtered_predicted_links = [
        link for link in predicted_links if link["source"] in selected_set and link["target"] in selected_set
    ]
    routes = build_routes(base_context_links, filtered_predicted_links)
    lifecycle_flows = build_lifecycle_flows(companies)

    months = sorted(set(base_months) | set(predicted_months))
    reliable_bundles = [row["bundle"] for row in bundle_rows if row.get("label") == "reliable"]

    return {
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "months": months,
            "reliableBundles": reliable_bundles,
            "companyCount": len(companies),
            "eventCount": len(base_events) + len(predicted_events),
            "predictedLinkCount": len(filtered_predicted_links),
            "routeCount": len(routes),
            "lifecycleFlowCount": len(lifecycle_flows),
        },
        "bundles": bundle_rows,
        "companies": companies,
        "events": base_events + predicted_events,
        "links": base_context_links + filtered_predicted_links,
        "routes": routes,
        "lifecycleFlows": lifecycle_flows,
        "similarityLinks": similarity_links,
    }


def main() -> None:
    dataset = build_dataset()
    OUTPUT_JS.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JS.open("w", encoding="utf-8") as file:
        file.write("window.MC2_3D_DATA = ")
        json.dump(dataset, file, ensure_ascii=False, separators=(",", ":"))
        file.write(";\n")

    print(
        "Built visualization data:",
        f"{dataset['meta']['companyCount']} companies,",
        f"{dataset['meta']['eventCount']} events,",
        f"{dataset['meta']['predictedLinkCount']} predicted links,",
        f"{dataset['meta']['routeCount']} routes,",
        f"{dataset['meta']['lifecycleFlowCount']} lifecycle flows.",
    )


if __name__ == "__main__":
    main()
