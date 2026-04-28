"""补全图谱前后的公司级和网络级异常对比（Q3）。

修复内容：
  1. suspicious_revival_score 改为 0-100 归一化，判别力大幅提升。
  2. 停活时间跨度加权：复活前休眠越久，extended_months 贡献分越高（最高 5×）。
  3. Q1 temporal_pattern 联动：行为与历史模式不一致时额外加分。
  4. 新增 detect_bridge_companies()：检测因新增链接成为结构性桥梁的公司。
  5. 新增 detect_relay_chains()：检测可靠链接激活的"前任→接班"接力关系。
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

_shared = str(Path(__file__).resolve().parents[1] / "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)

from build_index import month_key, normalize_hscode

# ── 超参数 ────────────────────────────────────────────────────────────────────
MAX_DORMANCY_WEIGHT = 4.0   # 停活超过 4 年不再额外加权（dormancy_weight ∈ [1, 5]）
BRIDGE_MIN_SCOPE = 5        # 桥接公司：最小新增单侧联通公司数
MAX_RELAY_OUTPUT = 300      # 接力链输出上限
RELAY_STOP_BEFORE = "2033-01-01"  # 前任公司必须在此日期前停活


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _month_to_int(month: str) -> int:
    try:
        return int(month[:4]) * 12 + int(month[5:7])
    except Exception:
        return 0


# ── 公司画像构建 ──────────────────────────────────────────────────────────────

def _empty_profile() -> dict:
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
    """只为受预测链接影响的公司建画像，避免对全量节点输出。"""
    profiles: dict[str, dict] = defaultdict(_empty_profile)
    for link in base_graph.get("links", []):
        src = link.get("source")
        tgt = link.get("target")
        if src in companies:
            _update_profile(profiles[src], link, src)
        if tgt in companies:
            _update_profile(profiles[tgt], link, tgt)
    return dict(profiles)


def build_prediction_profiles(reliable_links: list[dict]) -> dict[str, dict]:
    """为新增可靠链接单独建画像，用来和主图画像对比。"""
    profiles: dict[str, dict] = defaultdict(_empty_profile)
    for link in reliable_links:
        src = link.get("source")
        tgt = link.get("target")
        if src:
            _update_profile(profiles[src], link, src)
        if tgt:
            _update_profile(profiles[tgt], link, tgt)
    return dict(profiles)


# ── 公司级异常评分（重构） ────────────────────────────────────────────────────

def _monthly_threshold(monthly_counts: Counter) -> float:
    values = list(monthly_counts.values())
    if not values:
        return 5.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return max(5.0, mean + 3 * sqrt(variance))


def _q1_inconsistency_bonus(
    q1_pattern: str,
    dormancy_months: int,
    extended_count: int,
    burst_count: int,
) -> tuple[int, str]:
    """Q1 时序模式与预测行为不一致时的额外加分及原因。"""
    if q1_pattern == "short_term" and dormancy_months > 12 and extended_count > 0:
        return 15, "short_term company revived after long dormancy"
    if q1_pattern == "stable" and burst_count > 0:
        return 8, "stable company shows unexpected burst"
    if q1_pattern in ("general", "bursty") and dormancy_months > 24 and extended_count > 0:
        return 8, "company revived after 2+ year dormancy"
    return 0, ""


def compare_company_profile(
    company: str,
    before: dict | None,
    predicted: dict,
    q1_profile: dict | None = None,
) -> dict:
    """比较加边前后某公司的变化，输出 0-100 归一化的 suspicious_revival_score。

    评分组成（各项有上限，总量收束到 [0, 100]）：
      filled_score    = min(15, 5 × filled_gap_count)
      extended_score  = min(15, 3 × extended_count) × dormancy_weight    [max 75]
      burst_score     = min(20, 5 × burst_count)
      partner_score   = min(15, 3 × new_partners)   [去掉原 min(5,...) 截断]
      hscode_score    = min(10, 3 × new_hscodes)
      injection_score = min(15, 15 × pred_links / max(1, base_links))
      q1_bonus        = 0 / 8 / 15（模式不一致时额外加分）
    """
    before = before or _empty_profile()
    before_months = set(before["monthly_counts"])
    threshold = _monthly_threshold(before["monthly_counts"])

    filled_gap_months, extended_months, burst_months = [], [], []

    for month, count in predicted["monthly_counts"].items():
        inside_lifespan = (
            before["first_date"]
            and before["last_date"]
            and before["first_date"][:7] <= month <= before["last_date"][:7]
        )
        if inside_lifespan and month not in before_months:
            filled_gap_months.append(month)
        if before["last_date"] and month > before["last_date"][:7]:
            extended_months.append(month)
        if before["monthly_counts"].get(month, 0) + count > threshold:
            burst_months.append(month)

    new_partners = predicted["partners"] - before["partners"]
    new_hscodes = set(predicted["hscodes"]) - set(before["hscodes"])

    # 停活时间跨度：从 last_date 到首个预测月的月数
    dormancy_months = 0
    if extended_months and before["last_date"]:
        earliest_ext = min(extended_months)
        dormancy_months = max(0, _month_to_int(earliest_ext) - _month_to_int(before["last_date"][:7]))
    dormancy_weight = 1.0 + min(MAX_DORMANCY_WEIGHT, dormancy_months / 12)

    # 分项计分
    filled_score = min(15.0, 5.0 * len(filled_gap_months))
    extended_score = min(15.0, 3.0 * len(extended_months)) * dormancy_weight
    burst_score = min(20.0, 5.0 * len(burst_months))
    partner_score = min(15.0, 3.0 * len(new_partners))
    hscode_score = min(10.0, 3.0 * len(new_hscodes))
    injection_ratio = predicted["total_links"] / max(1, before["total_links"])
    injection_score = min(15.0, 15.0 * min(1.0, injection_ratio))

    q1_pattern = (q1_profile or {}).get("temporal_pattern", "")
    q1_bonus, q1_reason = _q1_inconsistency_bonus(
        q1_pattern, dormancy_months, len(extended_months), len(burst_months)
    )

    raw = (filled_score + extended_score + burst_score
           + partner_score + hscode_score + injection_score + q1_bonus)
    suspicious_revival_score = round(min(100.0, raw), 1)

    return {
        "company": company,
        "base_first_date": before["first_date"],
        "base_last_date": before["last_date"],
        "base_link_count": before["total_links"],
        "predicted_link_count": predicted["total_links"],
        "filled_gap_months": ";".join(sorted(filled_gap_months)),
        "extended_months": ";".join(sorted(extended_months)),
        "burst_months": ";".join(sorted(burst_months)),
        "dormancy_months": dormancy_months,
        "dormancy_weight": round(dormancy_weight, 2),
        "new_partner_count": len(new_partners),
        "new_hscode_count": len(new_hscodes),
        "predicted_weightkg": round(predicted["total_weightkg"], 2),
        "predicted_value_omu": round(predicted["total_value_omu"], 2),
        "suspicious_revival_score": suspicious_revival_score,
        "q1_temporal_pattern": q1_pattern,
        "q1_inconsistency_reason": q1_reason,
    }


# ── 邻居索引（共用） ─────────────────────────────────────────────────────────

def _build_neighbor_index(
    base_graph: dict,
    target_nodes: set[str],
) -> dict[str, set[str]]:
    """只为 target_nodes 建邻居集合，单次遍历主图。"""
    neighbors: dict[str, set[str]] = defaultdict(set)
    for link in base_graph.get("links", []):
        s = link.get("source")
        t = link.get("target")
        if not s or not t or s == t:
            continue
        if s in target_nodes:
            neighbors[s].add(t)
        if t in target_nodes:
            neighbors[t].add(s)
    return dict(neighbors)


# ── 网络级：桥接公司检测 ──────────────────────────────────────────────────────

def detect_bridge_companies(
    base_graph: dict,
    reliable_links: list[dict],
) -> list[dict]:
    """检测因新增可靠链接而成为结构性桥梁的公司。

    对每条可靠链接 (U, V)：
      U 侧桥接范围 = |base_neighbors[U] - base_neighbors[V]| — V 端新增单侧可达数
      V 侧桥接范围 = |base_neighbors[V] - base_neighbors[U]| — U 端新增单侧可达数
    公司 C 的总 bridge_scope = 其参与的所有可靠链接中，对端新开放的联通公司总数。
    """
    endpoints: set[str] = {
        ep
        for link in reliable_links
        for ep in (link.get("source"), link.get("target"))
        if ep
    }
    neighbors = _build_neighbor_index(base_graph, endpoints)

    bridge_scope: dict[str, int] = defaultdict(int)
    bridge_partners: dict[str, list[str]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()

    for link in reliable_links:
        u = link.get("source")
        v = link.get("target")
        if not u or not v or u == v:
            continue
        pair = tuple(sorted((u, v)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        nb_u = neighbors.get(u, set())
        nb_v = neighbors.get(v, set())
        u_excl = nb_u - nb_v - {u, v}   # 仅与 U 相连，现在可经由 V 扩展网络
        v_excl = nb_v - nb_u - {u, v}   # 仅与 V 相连，现在可经由 U 扩展网络

        if v_excl:
            bridge_scope[u] += len(v_excl)
            bridge_partners[u].append(v)
        if u_excl:
            bridge_scope[v] += len(u_excl)
            bridge_partners[v].append(u)

    rows = [
        {
            "company": company,
            "bridge_scope": scope,
            "bridge_link_count": len(bridge_partners[company]),
            "bridge_partner_sample": ";".join(bridge_partners[company][:5]),
        }
        for company, scope in sorted(bridge_scope.items(), key=lambda x: -x[1])
        if scope >= BRIDGE_MIN_SCOPE
    ]
    return rows


# ── 网络级：接力链检测 ────────────────────────────────────────────────────────

def detect_relay_chains(
    base_graph: dict,
    reliable_links: list[dict],
    temporal_patterns: list[dict],
) -> list[dict]:
    """检测因可靠链接激活的接力关系：前任公司停活后，接班公司接手原有业务网络。

    接力定义：
    - 前任 (A)：在 Q1 画像中 last_date < RELAY_STOP_BEFORE 的公司（已停活）。
    - 接班 (B)：可靠链接的端点之一（在 2034 年获得新预测交易的公司）。
    - 证据：A 与 B 在主图中共享 ≥ 2 个共同贸易伙伴，表明 B 接管了 A 的原有业务关系。
    """
    q1_by_company = {item["company"]: item for item in temporal_patterns}

    reliable_endpoints: set[str] = {
        ep
        for link in reliable_links
        for ep in (link.get("source"), link.get("target"))
        if ep
    }

    # 前任候选：在 Q1 中有记录、且早于阈值停活、且不在可靠链接中
    predecessors = {
        c: prof
        for c, prof in q1_by_company.items()
        if (prof.get("last_date") or "") < RELAY_STOP_BEFORE
        and c not in reliable_endpoints
    }

    if not predecessors or not reliable_endpoints:
        return []

    all_candidates = set(predecessors.keys()) | reliable_endpoints
    neighbors = _build_neighbor_index(base_graph, all_candidates)

    rows: list[dict] = []
    for pred_company, pred_prof in predecessors.items():
        nb_pred = neighbors.get(pred_company, set())
        if len(nb_pred) < 2:
            continue
        pred_last = (pred_prof.get("last_date") or "")[:7]

        for succ_company in reliable_endpoints:
            nb_succ = neighbors.get(succ_company, set())
            shared = nb_pred & nb_succ
            if len(shared) < 2:
                continue

            succ_prof = q1_by_company.get(succ_company, {})
            # 停活到 2034 可靠链接出现的月数差（量化前任缺席时长）
            gap_to_2034 = max(0, _month_to_int("2034-01") - _month_to_int(pred_last)) if pred_last else 0

            rows.append({
                "predecessor": pred_company,
                "successor": succ_company,
                "predecessor_last_month": pred_last,
                "gap_months_to_2034": gap_to_2034,
                "shared_partner_count": len(shared),
                "shared_partner_sample": ";".join(sorted(shared)[:4]),
                "predecessor_pattern": pred_prof.get("temporal_pattern", ""),
                "successor_pattern": succ_prof.get("temporal_pattern", ""),
            })

    rows.sort(key=lambda r: (-r["shared_partner_count"], r["gap_months_to_2034"]))
    return rows[:MAX_RELAY_OUTPUT]


# ── 主入口 ────────────────────────────────────────────────────────────────────

def compare_anomalies(
    base_graph: dict,
    reliable_links: list[dict],
    temporal_patterns: list[dict] | None = None,
) -> list[dict]:
    """对比加入可靠预测链接前后的公司画像变化，输出增强的异常指标。"""
    q1_by_company = {item["company"]: item for item in (temporal_patterns or [])}

    affected_companies: set[str] = {
        company
        for link in reliable_links
        for company in (link.get("source"), link.get("target"))
        if company
    }

    base_profiles = build_profiles_for_companies(base_graph, affected_companies)
    prediction_profiles = build_prediction_profiles(reliable_links)

    rows = [
        compare_company_profile(
            company,
            base_profiles.get(company),
            predicted,
            q1_by_company.get(company),
        )
        for company, predicted in prediction_profiles.items()
    ]

    return sorted(rows, key=lambda r: r["suspicious_revival_score"], reverse=True)
