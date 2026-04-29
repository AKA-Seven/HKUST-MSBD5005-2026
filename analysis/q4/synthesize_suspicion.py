"""Q4 可疑公司识别：多信号综合评分与证据链生成。

信号来源：
  S1  IsolationForest 异常分      (company_clusters.anomaly_score)
  S2  Q3 复活可疑评分              (anomaly_delta.suspicious_revival_score)
  S3  停活时长                     (dormancy_months ≥ 12 → 加权)
  S4  Q1-Q3 行为模式不一致         (q1_inconsistent flag)
  S5  Q1 时序模式高风险            (short_term → 3, bursty → 2, other → 0-1)
  S6  桥接网络控制力               (bridge_scope 归一化)
  S7  接力链接班方                 (relay_successor flag)
  S8  海产品关联度                 (fish_hscode_ratio)

置信度分层：
  HIGH   : composite_score ≥ 0.55 且 signal_count ≥ 3
  MEDIUM : composite_score ≥ 0.30 且 signal_count ≥ 2
  LOW    : 其余（有一定信号但证据不足）
"""

from __future__ import annotations

import math

# ── 权重配置 ──────────────────────────────────────────────────────────────
_W = {
    "iso_anomaly":     0.25,   # S1：行为统计异常
    "revival":         0.25,   # S2：停活复活评分
    "dormancy":        0.15,   # S3：休眠时长
    "q1_inconsistent": 0.10,   # S4：模式不一致
    "q1_risk":         0.08,   # S5：时序模式风险级别
    "bridge":          0.10,   # S6：网络桥接控制力
    "relay":           0.05,   # S7：接力接班
    "fish":            0.02,   # S8：海产品关联（辅助）
}
assert abs(sum(_W.values()) - 1.0) < 1e-6, "权重之和必须为 1"

_MAX_DORMANCY_MONTHS = 72      # 6 年以上按满分计
_MAX_BRIDGE_SCOPE    = 12000   # 归一化桥接范围上限
_MAX_REVIVAL_SCORE   = 100.0


def _norm(value: float, max_val: float) -> float:
    """线性归一化到 [0, 1]，超出上限截断。"""
    if max_val <= 0:
        return 0.0
    return min(1.0, value / max_val)


def _sigmoid(x: float, center: float = 0.5, steepness: float = 6.0) -> float:
    """S 形压缩，使中间段更有区分力。"""
    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))
    except OverflowError:
        return 0.0 if x < center else 1.0


def _compute_signal_scores(row: dict) -> dict[str, float]:
    """把单条公司记录的各字段转换为 [0,1] 信号分。"""
    # S1: IsolationForest 异常分（已在 company_clusters 中归一化为 0-1）
    s1 = float(row.get("anomaly_score", 0.0))

    # S2: 复活评分（0-100 → 0-1）
    s2 = _norm(float(row.get("revival_score", 0.0)), _MAX_REVIVAL_SCORE)

    # S3: 停活时长（非线性：12 个月以下贡献低，超 36 个月快速上升）
    dormancy = float(row.get("dormancy_months", 0.0))
    s3 = _norm(max(0.0, dormancy - 6.0), _MAX_DORMANCY_MONTHS - 6.0)  # 6 个月内视为正常过渡
    s3 = _sigmoid(s3, center=0.3, steepness=5.0)

    # S4: Q1-Q3 行为模式不一致（布尔 → 0/1）
    s4 = 1.0 if int(row.get("q1_inconsistent", 0)) else 0.0

    # S5: Q1 时序风险级别（0-3 → 0-1）
    s5 = _norm(float(row.get("q1_pattern_risk", 0.0)), 3.0)

    # S6: 桥接范围（对数压缩后归一化，避免超大节点垄断排名）
    bridge = float(row.get("bridge_scope", 0.0))
    s6 = _norm(math.log1p(bridge), math.log1p(_MAX_BRIDGE_SCOPE))

    # S7: 接力接班方（布尔 → 0/1）
    s7 = 1.0 if int(row.get("is_relay_successor", 0)) else 0.0

    # S8: 海产品比例
    s8 = float(row.get("fish_hscode_ratio", 0.0))

    return {
        "sig_iso_anomaly":     round(s1, 4),
        "sig_revival":         round(s2, 4),
        "sig_dormancy":        round(s3, 4),
        "sig_q1_inconsistent": round(s4, 4),
        "sig_q1_risk":         round(s5, 4),
        "sig_bridge":          round(s6, 4),
        "sig_relay":           round(s7, 4),
        "sig_fish":            round(s8, 4),
    }


def _composite_score(signals: dict[str, float]) -> float:
    """加权求和得到综合可疑分（0-1）。"""
    mapping = {
        "sig_iso_anomaly":     _W["iso_anomaly"],
        "sig_revival":         _W["revival"],
        "sig_dormancy":        _W["dormancy"],
        "sig_q1_inconsistent": _W["q1_inconsistent"],
        "sig_q1_risk":         _W["q1_risk"],
        "sig_bridge":          _W["bridge"],
        "sig_relay":           _W["relay"],
        "sig_fish":            _W["fish"],
    }
    return sum(signals[k] * w for k, w in mapping.items())


def _signal_count(signals: dict[str, float], threshold: float = 0.3) -> int:
    """计算触发（>阈值）的信号数量，用于置信度分层。"""
    return sum(1 for v in signals.values() if v > threshold)


def _confidence_tier(composite: float, n_signals: int) -> str:
    if composite >= 0.55 and n_signals >= 3:
        return "HIGH"
    if composite >= 0.30 and n_signals >= 2:
        return "MEDIUM"
    return "LOW"


def _build_evidence_chain(row: dict, signals: dict[str, float]) -> str:
    """拼接人类可读的证据链字符串。"""
    parts = []

    if signals["sig_iso_anomaly"] >= 0.5:
        parts.append(
            f"IsolationForest anomaly (score={row.get('anomaly_score', 0):.2f})"
        )

    revival = float(row.get("revival_score", 0))
    dormancy = int(row.get("dormancy_months", 0))
    if revival >= 20 or dormancy >= 12:
        pattern = row.get("q1_temporal_pattern", "")
        reason  = row.get("q1_inconsistency_reason", "")
        base = f"dormant {dormancy}mo then revived (revival_score={revival:.0f})"
        if pattern:
            base += f", Q1 pattern={pattern}"
        if reason:
            base += f"; {reason}"
        parts.append(base)

    bridge = int(row.get("bridge_scope", 0))
    if bridge >= 500:
        bc = int(row.get("bridge_link_count", 0))
        parts.append(
            f"structural bridge: {bridge:,} nodes newly reachable via {bc} new link(s)"
        )

    if int(row.get("is_relay_successor", 0)):
        parts.append("relay successor: inherits predecessor's trade network")

    fish = float(row.get("fish_hscode_ratio", 0))
    if fish >= 0.15:
        parts.append(f"fish HS code ratio={fish:.1%}")

    bm = row.get("business_mode", "")
    if bm in ("dormant_revival", "short_lived"):
        parts.append(f"business_mode={bm}")

    return " | ".join(parts) if parts else "no strong individual signal"


def synthesize_suspicion(
    company_clusters: list[dict],
    anomaly_delta: list[dict] | None = None,
    bridge_companies: list[dict] | None = None,
    relay_chains: list[dict] | None = None,
) -> list[dict]:
    """多信号融合，输出可疑公司排名与证据链。

    所有外部信号已经在 company_clustering.py 中写入 company_clusters 行，
    此处直接读取，不需要重新连接索引。bridge_link_count 等额外字段通过
    bridge_companies 列表补充。
    """
    # bridge_link_count 在 company_clusters 中没有，需要从原始列表补入
    bridge_extra: dict[str, dict] = {r["company"]: r for r in (bridge_companies or [])}
    relay_pred_index: dict[str, list[str]] = {}
    for r in (relay_chains or []):
        relay_pred_index.setdefault(r["successor"], []).append(r["predecessor"])

    results = []
    for row in company_clusters:
        company = row["company"]

        # 合并 bridge 额外字段
        be = bridge_extra.get(company, {})
        enriched = dict(row)
        enriched["bridge_link_count"]    = int(be.get("bridge_link_count", 0))
        enriched["bridge_partner_sample"] = be.get("bridge_partner_sample", "")
        enriched["relay_predecessors"]   = ";".join(relay_pred_index.get(company, [])[:3])

        # 计算信号分
        sigs = _compute_signal_scores(enriched)
        composite = _composite_score(sigs)
        n_sig = _signal_count(sigs)
        tier = _confidence_tier(composite, n_sig)
        evidence = _build_evidence_chain(enriched, sigs)

        results.append({
            **enriched,
            **sigs,
            "composite_score":  round(composite, 4),
            "signal_count":     n_sig,
            "confidence_tier":  tier,
            "evidence_chain":   evidence,
        })

    # 主排序：composite_score 降序；同分则 signal_count 降序
    results.sort(key=lambda r: (-r["composite_score"], -r["signal_count"]))
    return results


def high_confidence_suspects(ranking: list[dict]) -> list[dict]:
    """取出 HIGH 置信度公司，供快速查看。"""
    return [r for r in ranking if r["confidence_tier"] == "HIGH"]


def medium_confidence_suspects(ranking: list[dict]) -> list[dict]:
    """取出 HIGH + MEDIUM 置信度公司。"""
    return [r for r in ranking if r["confidence_tier"] in ("HIGH", "MEDIUM")]
