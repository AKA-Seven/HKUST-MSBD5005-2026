#!/usr/bin/env python3
"""2D 图表公共数据加载与预处理。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR    = PROJECT_ROOT / "outputs"
OUTPUTS_DIR_Q1 = OUTPUTS_DIR / "q1"
OUTPUTS_DIR_Q2 = OUTPUTS_DIR / "q2"
OUTPUTS_DIR_Q3 = OUTPUTS_DIR / "q3"
OUTPUTS_DIR_Q4 = OUTPUTS_DIR / "q4"
FIG_DIR = PROJECT_ROOT / "visualization" / "figures_2d"


def ensure_dirs() -> None:
    """创建 Q1-Q4 输出目录。"""
    for name in ("q1", "q2", "q3", "q4"):
        (FIG_DIR / name).mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取 mining 输出（各题产物从对应子目录读取）。"""
    bundle_df = pd.read_csv(OUTPUTS_DIR_Q2 / "bundle_reliability.csv")
    company_df = pd.read_csv(OUTPUTS_DIR_Q4 / "company_clusters.csv")
    delta_df = pd.read_csv(OUTPUTS_DIR_Q3 / "anomaly_delta.csv")
    reliable_links = pd.DataFrame(json.loads((OUTPUTS_DIR_Q2 / "reliable_links.json").read_text(encoding="utf-8")))
    return bundle_df, company_df, delta_df, reliable_links


def load_q1_patterns() -> list[dict]:
    """读取 Q1 专属的原始图谱公司时序画像。

    由 run_pipeline.py 通过 extract_temporal_patterns 模块生成，
    文件中每条记录包含 monthly_counts（原始图谱数据）、temporal_pattern 等字段。
    """
    path = OUTPUTS_DIR_Q1 / "q1_temporal_patterns.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_month_list(cell: str) -> int:
    """把 ; 分隔的月份字符串转成数量。"""
    if pd.isna(cell) or not str(cell).strip():
        return 0
    return len([item for item in str(cell).split(";") if item.strip()])


def build_company_view(company_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    """合并公司画像和异常变化指标。"""
    merged = company_df.merge(delta_df, on="company", how="left")

    for column in ("filled_gap_months", "extended_months", "burst_months"):
        merged[f"{column}_count"] = merged[column].apply(parse_month_list)

    merged["predicted_link_count"] = merged["predicted_link_count"].fillna(0)
    merged["new_partner_count"] = merged["new_partner_count"].fillna(0)
    merged["new_hscode_count"] = merged["new_hscode_count"].fillna(0)
    merged["suspicious_revival_score"] = merged["suspicious_revival_score"].fillna(0)
    merged["hierarchical_cluster"] = merged["hierarchical_cluster"].fillna(-1).astype(int)
    merged["hierarchical_subcluster"] = merged["hierarchical_subcluster"].fillna(-1).astype(int)
    merged["cluster_path"] = merged.get("cluster_path", "").fillna("")
    merged["business_mode"] = merged.get("business_mode", "unknown").fillna("unknown")

    def norm(series: pd.Series) -> pd.Series:
        low = float(series.min())
        high = float(series.max())
        if np.isclose(high, low):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - low) / (high - low)

    merged["iuu_score"] = (
        0.33 * norm(merged["anomaly_score"])
        + 0.28 * norm(merged["suspicious_revival_score"])
        + 0.17 * norm(merged["new_partner_count"] + merged["new_hscode_count"])
        + 0.12 * norm(merged["predicted_link_count"])
        + 0.10 * norm(merged["nearest_similarity"].fillna(0))
    )
    merged["iuu_confidence"] = pd.cut(
        merged["iuu_score"],
        bins=[-0.01, 0.35, 0.65, 1.01],
        labels=["low", "medium", "high"],
    )
    return merged.sort_values("iuu_score", ascending=False).reset_index(drop=True)
