import csv
import json
from pathlib import Path

from config import OUTPUT_DIR, OUTPUT_DIR_Q1, OUTPUT_DIR_Q2, OUTPUT_DIR_Q3, OUTPUT_DIR_Q4


def ensure_output_dir(output_dir: Path = OUTPUT_DIR) -> None:
    """确保输出目录（含各题子目录）存在。"""
    for d in (output_dir, OUTPUT_DIR_Q1, OUTPUT_DIR_Q2, OUTPUT_DIR_Q3, OUTPUT_DIR_Q4):
        d.mkdir(parents=True, exist_ok=True)


def export_csv(rows: list[dict], path: Path) -> None:
    """把字典列表输出为 CSV，方便后续可视化直接读取。"""
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_json(data, path: Path) -> None:
    """输出 JSON，保留新增链接等结构化结果。"""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def export_outputs(
    bundle_scores: list[dict],
    reliable_links: list[dict],
    anomaly_delta: list[dict],
    company_clusters: list[dict],
    edge_clusters: list[dict],
    temporal_patterns: list[dict] | None = None,
) -> None:
    """统一导出挖掘结果（各题产物写入对应子目录）。"""
    ensure_output_dir()
    # Q1 产物
    if temporal_patterns is not None:
        export_json(temporal_patterns, OUTPUT_DIR_Q1 / "q1_temporal_patterns.json")
    # Q2 产物
    export_csv(bundle_scores, OUTPUT_DIR_Q2 / "bundle_reliability.csv")
    export_json(reliable_links, OUTPUT_DIR_Q2 / "reliable_links.json")
    # Q3 产物
    export_csv(anomaly_delta, OUTPUT_DIR_Q3 / "anomaly_delta.csv")
    export_json(anomaly_delta[:100], OUTPUT_DIR_Q3 / "top_anomaly_delta.json")
    # Q4 产物
    export_csv(company_clusters, OUTPUT_DIR_Q4 / "company_clusters.csv")
    export_json(company_clusters[:100], OUTPUT_DIR_Q4 / "top_company_anomalies.json")
    export_csv(edge_clusters, OUTPUT_DIR_Q4 / "edge_clusters.csv")
    export_json(edge_clusters[:300], OUTPUT_DIR_Q4 / "top_edge_clusters.json")
