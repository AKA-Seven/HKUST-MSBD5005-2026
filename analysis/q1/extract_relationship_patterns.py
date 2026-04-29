"""提取公司对之间的时序关系模式，作为 Q1 实体间关系分析的数据源。

模式分类（relationship_pattern）：
  - synchronous       : 两公司长期同步活跃，月度重叠率高（月 Jaccard ≥ 0.40）。
  - relay             : A 公司退场后 B 公司在 0–6 个月内接手，且共享合作伙伴。
  - substitution      : B 公司承接 A 公司原有贸易关系，伙伴重叠度高但时间段错位。
  - short_term_collab : 两公司仅在短暂窗口（1–8 个月）内共同活跃后分道扬镳。
  - co_active         : 存在共同合作伙伴但未呈现上述典型时序特征。

候选对生成策略：
  对全部选定公司构建合作伙伴集合，通过倒排索引找出共享伙伴数 ≥ 2 的公司对，
  再按时序指标分类过滤，避免 O(n²) 全量枚举。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_shared = str(Path(__file__).resolve().parents[1] / "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)


# ── 候选对筛选参数 ────────────────────────────────────────────────────────────
MIN_SHARED_PARTNERS = 2   # 共享合作伙伴数量下限
HUB_PARTNER_CAP = 40      # 倒排索引中超过此值的超级枢纽伙伴跳过（避免组合爆炸）
RELAY_GAP_MAX = 60        # 接力模式允许的最大间隔月数（5 年，与 Q3 实测的 36–76 月对齐）
RELAY_PARTNER_COUNT_MIN = 3  # 共享伙伴绝对数下限（绝对计数比 Jaccard 对小公司更友好）
RELAY_JACCARD_MIN = 0.05  # 共享伙伴 Jaccard 下限（对大公司提供独立判据）
MIN_CONFIDENCE = 0.25     # co_active 对的置信度下限（低于此则过滤）

# 每种模式最多保留的对数（分层采样，防止单一模式独占输出）
PER_PATTERN_LIMIT: dict[str, int] = {
    "relay": 2000,
    "substitution": 2000,
    "synchronous": 2000,
    "short_term_collab": 1500,
    "co_active": 500,
}


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _month_to_int(month: str) -> int:
    """YYYY-MM → 月序整数，用于计算月份差。"""
    try:
        return int(month[:4]) * 12 + int(month[5:7])
    except (ValueError, IndexError):
        return 0


def _month_gap(last_month: str, next_month: str) -> int:
    """返回 last_month 到 next_month 之间间隔的完整月数（不含端点）。
    正数 = 有间隔，0 = 相邻月，负数 = 有重叠。
    """
    if not last_month or not next_month:
        return -999
    return _month_to_int(next_month) - _month_to_int(last_month) - 1


def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return round(len(a & b) / union, 4) if union else 0.0


# ── 构建公司画像索引 ──────────────────────────────────────────────────────────

def _build_company_profiles(temporal_patterns: list[dict]) -> dict[str, dict]:
    """将 q1_temporal_patterns 列表转为以公司名为键的快速查找字典。"""
    profiles: dict[str, dict] = {}
    for item in temporal_patterns:
        company = item.get("company")
        if not company:
            continue
        months = set(item.get("monthly_counts", {}).keys())
        first = item.get("first_date", "")
        last = item.get("last_date", "")
        profiles[company] = {
            "months": months,
            "first_month": first[:7] if first else "",
            "last_month": last[:7] if last else "",
            "active_count": len(months),
            "temporal_pattern": item.get("temporal_pattern", "general"),
        }
    return profiles


# ── 合作伙伴集合构建 ──────────────────────────────────────────────────────────

def _build_partner_sets(
    base_graph: dict,
    selected: set[str],
) -> dict[str, set[str]]:
    """单次遍历主图，为每家选定公司收集全量合作伙伴（对端可不在 selected 内）。"""
    partners: dict[str, set[str]] = defaultdict(set)
    for link in base_graph.get("links", []):
        src = link.get("source")
        tgt = link.get("target")
        if not src or not tgt:
            continue
        if src in selected:
            partners[src].add(tgt)
        if tgt in selected:
            partners[tgt].add(src)
    return dict(partners)


# ── 候选对生成 ────────────────────────────────────────────────────────────────

def _candidate_pairs(partner_sets: dict[str, set[str]]) -> dict[tuple[str, str], int]:
    """倒排索引：共同贸易伙伴 → 候选公司对及共享计数。

    超级枢纽伙伴（被 >HUB_PARTNER_CAP 家选定公司交易）被跳过，
    避免组合爆炸且这类伙伴对区分公司关系意义不大。
    """
    # 倒排：partner → [selected companies that trade with it]
    inverted: dict[str, list[str]] = defaultdict(list)
    for company, partners in partner_sets.items():
        for partner in partners:
            inverted[partner].append(company)

    shared_count: dict[tuple[str, str], int] = defaultdict(int)
    for companies in inverted.values():
        if len(companies) < 2 or len(companies) > HUB_PARTNER_CAP:
            continue
        companies_sorted = sorted(companies)
        for i in range(len(companies_sorted)):
            for j in range(i + 1, len(companies_sorted)):
                shared_count[(companies_sorted[i], companies_sorted[j])] += 1

    return {pair: cnt for pair, cnt in shared_count.items() if cnt >= MIN_SHARED_PARTNERS}


# ── 模式分类 ─────────────────────────────────────────────────────────────────

def _classify_pair(
    months_a: set[str],
    months_b: set[str],
    first_a: str,
    last_a: str,
    first_b: str,
    last_b: str,
    partner_jaccard: float,
    shared_partner_count: int,
) -> tuple[str, float]:
    """对公司对打时序关系标签，返回 (pattern, confidence)。

    分类优先级（最具信息量的特殊模式优先）：
      relay > substitution > synchronous > short_term_collab > co_active

    relay 判别条件（满足任一即可，对大小公司都友好）：
      - 双方时间不重叠（前者完全停活后，后者才启动），且
      - 间隔在 RELAY_GAP_MAX 月以内（5 年），且
      - 共享伙伴绝对数 ≥ 3（小公司）或 Jaccard ≥ 0.05（大公司）
    """
    if not months_a or not months_b:
        return "co_active", 0.0

    overlap_count = len(months_a & months_b)
    month_jaccard = _jaccard(months_a, months_b)

    # 确定哪家公司先退场（用于接力/替代判断）。
    # 仅当时间段完全错位（一方 last < 另一方 first）才视为接力候选。
    gap = -999
    if last_a and first_b and last_a < first_b:
        gap = _month_gap(last_a, first_b)
    elif last_b and first_a and last_b < first_a:
        gap = _month_gap(last_b, first_a)

    # 接力型：时间错位 + 业务承接
    has_partner_evidence = (
        partner_jaccard >= RELAY_JACCARD_MIN
        or shared_partner_count >= RELAY_PARTNER_COUNT_MIN
    )
    if 0 <= gap <= RELAY_GAP_MAX and has_partner_evidence:
        # gap 越小越像直接接班，5 年以上的间隔置信度递减但不归零
        gap_score = max(0.0, 1.0 - gap / (RELAY_GAP_MAX + 1))
        partner_score = max(min(partner_jaccard * 3, 1.0),
                            min(shared_partner_count / 20.0, 1.0))
        confidence = round(0.55 * gap_score + 0.45 * partner_score, 3)
        return "relay", confidence

    # 替代型：伙伴高度重叠但活跃期基本不重叠（与 relay 的差别：无明显时间间隔）
    if partner_jaccard >= 0.18 and month_jaccard < 0.20:
        confidence = round(min(partner_jaccard * 2.5, 1.0), 3)
        return "substitution", confidence

    # 同步型：月度重叠率高，且存在实质性伙伴关联
    if month_jaccard >= 0.40 and partner_jaccard >= 0.12:
        confidence = round(0.60 * month_jaccard + 0.40 * partner_jaccard, 3)
        return "synchronous", confidence

    # 短期协同型：短暂共同活跃后分离
    if 1 <= overlap_count <= 8 and month_jaccard < 0.30:
        confidence = round(0.30 + 0.50 * overlap_count / 8, 3)
        return "short_term_collab", confidence

    # 共同活跃型：有共同伙伴但无显著时序特征
    return "co_active", round(partner_jaccard, 3)


# ── 主函数 ────────────────────────────────────────────────────────────────────

def extract_relationship_patterns(
    base_graph: dict,
    temporal_patterns: list[dict],
) -> list[dict]:
    """提取公司对之间的时序关系模式。

    输入：
      base_graph        – 原始主图（load_base_graph() 结果）
      temporal_patterns – Q1 单实体时序画像列表（extract_temporal_patterns() 结果）

    输出：
      按置信度降序排列的关系对列表，截取前 MAX_PAIRS_OUTPUT 条。
    """
    profiles = _build_company_profiles(temporal_patterns)
    selected = set(profiles.keys())
    print(f"  [relationship] 选定公司数: {len(selected)}")

    partner_sets = _build_partner_sets(base_graph, selected)
    candidate_map = _candidate_pairs(partner_sets)
    print(f"  [relationship] 候选对数（共享伙伴≥{MIN_SHARED_PARTNERS}）: {len(candidate_map)}")

    rows: list[dict] = []
    for (company_a, company_b), shared_cnt in candidate_map.items():
        prof_a = profiles.get(company_a)
        prof_b = profiles.get(company_b)
        if not prof_a or not prof_b:
            continue

        ps_a = partner_sets.get(company_a, set())
        ps_b = partner_sets.get(company_b, set())
        partner_jaccard = _jaccard(ps_a, ps_b)

        pattern, confidence = _classify_pair(
            prof_a["months"],
            prof_b["months"],
            prof_a["first_month"],
            prof_a["last_month"],
            prof_b["first_month"],
            prof_b["last_month"],
            partner_jaccard,
            shared_cnt,
        )

        if pattern == "co_active" and confidence < MIN_CONFIDENCE:
            continue

        overlap_months = sorted(prof_a["months"] & prof_b["months"])

        # 确定时间方向（谁先谁后）用于 relay/substitution 语义
        if prof_a["last_month"] and prof_b["first_month"] and prof_a["last_month"] < prof_b["first_month"]:
            gap_months = _month_gap(prof_a["last_month"], prof_b["first_month"])
        elif prof_b["last_month"] and prof_a["first_month"] and prof_b["last_month"] < prof_a["first_month"]:
            gap_months = _month_gap(prof_b["last_month"], prof_a["first_month"])
        else:
            gap_months = None  # 时间段有重叠，无明确先后

        rows.append({
            "company_a": company_a,
            "company_b": company_b,
            "relationship_pattern": pattern,
            "confidence": confidence,
            "month_overlap_count": len(overlap_months),
            "month_overlap_jaccard": _jaccard(prof_a["months"], prof_b["months"]),
            "shared_partner_count": shared_cnt,
            "shared_partner_jaccard": partner_jaccard,
            "gap_months": gap_months,
            "a_first_month": prof_a["first_month"],
            "a_last_month": prof_a["last_month"],
            "b_first_month": prof_b["first_month"],
            "b_last_month": prof_b["last_month"],
            "a_active_months": prof_a["active_count"],
            "b_active_months": prof_b["active_count"],
            "a_temporal_pattern": prof_a["temporal_pattern"],
            "b_temporal_pattern": prof_b["temporal_pattern"],
            "overlap_month_sample": ";".join(overlap_months[:6]),
        })

    # 按模式分层采样，防止高频模式（如大量 stable 对产生的同步）独占输出
    by_pattern: dict[str, list[dict]] = {}
    for row in rows:
        by_pattern.setdefault(row["relationship_pattern"], []).append(row)

    result: list[dict] = []
    for pattern, limit in PER_PATTERN_LIMIT.items():
        group = sorted(by_pattern.get(pattern, []), key=lambda r: r["confidence"], reverse=True)
        result.extend(group[:limit])
        print(f"  [relationship]   {pattern}: 共 {len(by_pattern.get(pattern, []))} 对，输出 {min(len(group), limit)} 对")

    result.sort(key=lambda r: r["confidence"], reverse=True)
    print(f"  [relationship] 总输出对数: {len(result)}")
    return result
