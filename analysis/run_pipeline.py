import sys
from pathlib import Path

_analysis_root = Path(__file__).resolve().parent
for _sub in ("shared", "q1", "q2", "q3", "q4"):
    _p = str(_analysis_root / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_index import build_base_index
from company_clustering import cluster_and_detect_companies
from synthesize_suspicion import synthesize_suspicion
from detect_anomalies import compare_anomalies, detect_bridge_companies, detect_relay_chains
from edge_clustering import cluster_edges
from evaluate_bundles import evaluate_all_bundles
from export_results import export_outputs
from extract_relationship_patterns import extract_relationship_patterns
from extract_temporal_patterns import extract_temporal_patterns
from link_prediction import score_all_bundle_links
from load_data import load_base_graph, load_bundles
from merge_links import collect_reliable_links
from score_bundles import score_all_bundles


def merge_metric_rows(base_rows: list[dict], extra_rows: list[dict], key: str = "bundle") -> list[dict]:
    """按 bundle 名称合并规则指标和机器学习指标。"""
    extra_by_key = {row[key]: row for row in extra_rows}
    merged_rows = []
    for row in base_rows:
        extra = extra_by_key.get(row[key], {})
        merged = dict(row)
        for extra_key, extra_value in extra.items():
            if extra_key != key:
                merged[extra_key] = extra_value
        merged_rows.append(merged)
    return merged_rows


def main() -> None:
    """MC2 预测链接挖掘主流程。"""
    print("Loading MC2 base graph and predicted bundles...")
    base_graph = load_base_graph()
    bundles = load_bundles()

    print("Building base graph index...")
    base_index = build_base_index(base_graph)

    print("Evaluating predicted bundles...")
    bundle_metrics = evaluate_all_bundles(bundles, base_index)

    print("Training self-supervised link prediction model...")
    link_prediction_scores = score_all_bundle_links(base_graph, bundles)
    bundle_metrics = merge_metric_rows(bundle_metrics, link_prediction_scores)

    bundle_scores = score_all_bundles(bundle_metrics)

    print("Collecting reliable predicted links...")
    reliable_links = collect_reliable_links(bundles, bundle_scores, base_index)

    print("Extracting Q1 temporal patterns from base graph...")
    temporal_patterns = extract_temporal_patterns(base_graph, bundles)
    extra_companies = {item["company"] for item in temporal_patterns}

    print("Extracting Q1 inter-company relationship patterns...")
    relationship_patterns = extract_relationship_patterns(base_graph, temporal_patterns)

    print("Comparing anomaly changes after adding reliable links...")
    anomaly_delta = compare_anomalies(base_graph, reliable_links, temporal_patterns)

    print("Detecting bridge companies introduced by reliable links...")
    bridge_companies = detect_bridge_companies(base_graph, reliable_links)

    print("Detecting relay chains activated by reliable links...")
    relay_chains = detect_relay_chains(base_graph, reliable_links, temporal_patterns)

    print("Running unsupervised company anomaly detection and clustering...")
    company_clusters = cluster_and_detect_companies(
        base_graph, bundles,
        extra_companies=extra_companies,
        temporal_patterns=temporal_patterns,
        anomaly_delta=anomaly_delta,
        bridge_companies=bridge_companies,
        relay_chains=relay_chains,
    )

    print("Synthesizing multi-signal suspicion ranking...")
    suspicion_ranking = synthesize_suspicion(
        company_clusters,
        anomaly_delta=anomaly_delta,
        bridge_companies=bridge_companies,
        relay_chains=relay_chains,
    )

    print("Clustering trade edges with mining + base graph context...")
    edge_clusters = cluster_edges(base_graph, reliable_links, company_clusters)

    print("Exporting mining outputs...")
    export_outputs(
        bundle_scores=bundle_scores,
        reliable_links=reliable_links,
        anomaly_delta=anomaly_delta,
        bridge_companies=bridge_companies,
        relay_chains=relay_chains,
        company_clusters=company_clusters,
        edge_clusters=edge_clusters,
        suspicion_ranking=suspicion_ranking,
        temporal_patterns=temporal_patterns,
        relationship_patterns=relationship_patterns,
    )

    reliable_count = sum(1 for row in bundle_scores if row["label"] == "reliable")
    print(f"Done: {reliable_count} reliable bundles, {len(reliable_links)} reliable new links.")


if __name__ == "__main__":
    main()
