"""
Q3 补全前后图谱差异全面分析
输出：analysis_graph_diff.json（供后续可视化使用）
"""
import sys, json, csv
from pathlib import Path
from collections import Counter, defaultdict
from math import sqrt, log

_root = Path(__file__).resolve().parents[2]
_shared = str(_root / "analysis" / "shared")
_q3 = str(_root / "analysis" / "q3")
for p in (_shared, _q3):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_index import month_key, normalize_hscode
from load_data import load_base_graph

OUT_FILE = _root / "outputs" / "q3" / "graph_diff_analysis.json"

# ── 加载数据 ───────────────────────────────────────────────────────────────
print("Loading base graph...")
base_graph = load_base_graph()
base_links = base_graph.get("links", [])
base_nodes_list = base_graph.get("nodes", [])

print("Loading reliable links...")
reliable_links = json.loads((_root / "outputs" / "q2" / "reliable_links.json").read_text("utf-8"))

print(f"Base links: {len(base_links):,}   Reliable links: {len(reliable_links)}")

# ── 工具 ──────────────────────────────────────────────────────────────────
def safe_div(a, b, default=0.0):
    return a / b if b else default

def build_neighbor_sets(links):
    nb = defaultdict(set)
    for lk in links:
        s, t = lk.get("source"), lk.get("target")
        if s and t and s != t:
            nb[s].add(t)
            nb[t].add(s)
    return nb

def degree_stats(nb):
    degs = [len(v) for v in nb.values()]
    if not degs:
        return {}
    degs.sort()
    n = len(degs)
    mean = sum(degs) / n
    variance = sum((d - mean)**2 for d in degs) / n
    return {
        "count": n,
        "max": max(degs),
        "mean": round(mean, 2),
        "std": round(sqrt(variance), 2),
        "median": degs[n // 2],
        "top10_threshold": degs[int(n * 0.9)],
    }

# ── 1. 网络级整体结构对比 ─────────────────────────────────────────────────
print("\n[1] Network-level structure...")
combined_links = base_links + reliable_links

nb_before = build_neighbor_sets(base_links)
nb_after  = build_neighbor_sets(combined_links)

nodes_before = set(nb_before.keys())
nodes_after  = set(nb_after.keys())
new_nodes    = nodes_after - nodes_before

# 边去重
def edge_set(links):
    return {(lk.get("source"), lk.get("target")) for lk in links
            if lk.get("source") and lk.get("target")}

edges_before = edge_set(base_links)
edges_after  = edge_set(combined_links)
new_edges    = edges_after - edges_before

# 度分布对比
deg_before = degree_stats(nb_before)
deg_after  = degree_stats(nb_after)

# 密度：2E / (N*(N-1))
n_b, e_b = len(nodes_before), len(edges_before)
n_a, e_a = len(nodes_after),  len(edges_after)
density_before = 2 * e_b / max(1, n_b * (n_b - 1))
density_after  = 2 * e_a / max(1, n_a * (n_a - 1))

network_diff = {
    "node_count_before": n_b,
    "node_count_after":  n_a,
    "new_nodes":         len(new_nodes),
    "edge_count_before": e_b,
    "edge_count_after":  e_a,
    "new_edges":         len(new_edges),
    "link_count_before": len(base_links),
    "link_count_after":  len(combined_links),
    "new_links":         len(reliable_links),
    "density_before":    round(density_before, 8),
    "density_after":     round(density_after,  8),
    "density_change_pct": round((density_after - density_before) / max(density_before, 1e-10) * 100, 4),
    "degree_before":     deg_before,
    "degree_after":      deg_after,
    "new_node_samples":  sorted(new_nodes)[:10],
}

# ── 2. 连通分量变化（近似：用共同邻居判断是否有桥接效果）───────────────────
print("[2] Component bridging analysis (approximate)...")
# 用 bridge_companies 数据近似评估连通性变化
bridge_path = _root / "outputs" / "q3" / "bridge_companies.csv"
bridge_companies = []
if bridge_path.exists():
    with open(bridge_path, encoding="utf-8-sig", newline="") as f:
        bridge_companies = list(csv.DictReader(f))

top_bridges = [
    {
        "company": r["company"],
        "bridge_scope": int(r["bridge_scope"]),
        "bridge_link_count": int(r["bridge_link_count"]),
        "bridge_partner_sample": r.get("bridge_partner_sample", ""),
    }
    for r in bridge_companies[:20]
]

# ── 3. 公司级变化统计 ─────────────────────────────────────────────────────
print("[3] Company-level changes...")
anomaly_path = _root / "outputs" / "q3" / "anomaly_delta.csv"
anomaly_rows = []
if anomaly_path.exists():
    with open(anomaly_path, encoding="utf-8-sig", newline="") as f:
        anomaly_rows = list(csv.DictReader(f))

scores = [float(r["suspicious_revival_score"]) for r in anomaly_rows]
dormancy_vals = [int(r.get("dormancy_months", 0)) for r in anomaly_rows]
q1_pattern_dist = Counter(r.get("q1_temporal_pattern", "unknown") for r in anomaly_rows)

# 按复活类型分桶
revival_types = {"dormant_revived": 0, "extended_active": 0, "new_burst": 0, "gap_filled": 0, "new_company": 0}
for r in anomaly_rows:
    dm = int(r.get("dormancy_months", 0))
    ext = r.get("extended_months", "")
    burst = r.get("burst_months", "")
    filled = r.get("filled_gap_months", "")
    base_lc = int(r.get("base_link_count", 0))
    if base_lc == 0:
        revival_types["new_company"] += 1
    elif dm >= 12 and ext:
        revival_types["dormant_revived"] += 1
    elif ext:
        revival_types["extended_active"] += 1
    elif burst:
        revival_types["new_burst"] += 1
    elif filled:
        revival_types["gap_filled"] += 1
    else:
        revival_types["extended_active"] += 1  # 轻微延伸或新伙伴

# 高可疑公司详情
top_suspicious = []
for r in sorted(anomaly_rows, key=lambda x: float(x["suspicious_revival_score"]), reverse=True)[:20]:
    top_suspicious.append({
        "company": r["company"],
        "score": float(r["suspicious_revival_score"]),
        "dormancy_months": int(r.get("dormancy_months", 0)),
        "dormancy_weight": float(r.get("dormancy_weight", 1.0)),
        "base_link_count": int(r.get("base_link_count", 0)),
        "predicted_link_count": int(r.get("predicted_link_count", 0)),
        "new_partner_count": int(r.get("new_partner_count", 0)),
        "extended_months": r.get("extended_months", ""),
        "burst_months": r.get("burst_months", ""),
        "q1_temporal_pattern": r.get("q1_temporal_pattern", ""),
        "q1_inconsistency_reason": r.get("q1_inconsistency_reason", ""),
    })

company_diff = {
    "affected_companies": len(anomaly_rows),
    "score_max": max(scores) if scores else 0,
    "score_mean": round(sum(scores)/len(scores), 2) if scores else 0,
    "score_median": sorted(scores)[len(scores)//2] if scores else 0,
    "score_distribution": {
        "gt50": sum(1 for s in scores if s > 50),
        "gt30": sum(1 for s in scores if s > 30),
        "gt10": sum(1 for s in scores if s > 10),
        "gt5":  sum(1 for s in scores if s > 5),
        "le5":  sum(1 for s in scores if s <= 5),
    },
    "dormancy_mean": round(sum(dormancy_vals)/len(dormancy_vals), 1) if dormancy_vals else 0,
    "dormancy_max": max(dormancy_vals) if dormancy_vals else 0,
    "companies_with_dormancy_gt12": sum(1 for d in dormancy_vals if d > 12),
    "revival_type_counts": revival_types,
    "q1_pattern_distribution": dict(q1_pattern_dist),
    "top_suspicious": top_suspicious,
}

# ── 4. 时序分布变化 ────────────────────────────────────────────────────────
print("[4] Temporal distribution...")
month_before = Counter(month_key(lk.get("arrivaldate")) for lk in base_links)
month_after  = Counter(month_key(lk.get("arrivaldate")) for lk in combined_links)

# 新增链接的月份分布
month_new = Counter(month_key(lk.get("arrivaldate")) for lk in reliable_links)

# 年度汇总
def year_agg(month_counter):
    yc = Counter()
    for m, cnt in month_counter.items():
        if m and len(m) >= 4:
            yc[m[:4]] += cnt
    return dict(sorted(yc.items()))

temporal_diff = {
    "year_before": year_agg(month_before),
    "year_after":  year_agg(month_after),
    "year_new":    year_agg(month_new),
    "month_new_top20": dict(month_new.most_common(20)),
    "new_link_date_range": {
        "min": min((lk.get("arrivaldate","") for lk in reliable_links if lk.get("arrivaldate")), default=""),
        "max": max((lk.get("arrivaldate","") for lk in reliable_links if lk.get("arrivaldate")), default=""),
    },
}

# ── 5. HS 编码与货物类型变化 ─────────────────────────────────────────────
print("[5] HS code / cargo changes...")
from config import FISH_HSCODE_PREFIXES

def is_fish(hscode):
    return normalize_hscode(hscode).startswith(FISH_HSCODE_PREFIXES)

hscode_before = Counter(normalize_hscode(lk.get("hscode")) for lk in base_links)
hscode_new    = Counter(normalize_hscode(lk.get("hscode")) for lk in reliable_links)

fish_before = sum(v for k, v in hscode_before.items() if is_fish(k))
fish_new    = sum(v for k, v in hscode_new.items()    if is_fish(k))

# 新出现的 HS 编码
known_hscodes = set(hscode_before.keys())
truly_new_hscodes = {k: v for k, v in hscode_new.items() if k and k not in known_hscodes}

cargo_diff = {
    "unique_hscodes_before": len(hscode_before),
    "unique_hscodes_after":  len(set(hscode_before.keys()) | set(hscode_new.keys())),
    "new_hscodes_introduced": len(truly_new_hscodes),
    "truly_new_hscode_samples": dict(list(truly_new_hscodes.items())[:10]),
    "fish_links_before": fish_before,
    "fish_links_new":    fish_new,
    "fish_ratio_before": round(safe_div(fish_before, len(base_links)) * 100, 2),
    "fish_ratio_new":    round(safe_div(fish_new, len(reliable_links)) * 100, 2),
    "top_new_hscodes":   dict(hscode_new.most_common(10)),
}

# ── 6. 贸易对变化 ─────────────────────────────────────────────────────────
print("[6] Trading pair changes...")
pairs_before = Counter()
for lk in base_links:
    s, t = lk.get("source"), lk.get("target")
    if s and t:
        pairs_before[tuple(sorted((s, t)))] += 1

pairs_new_cnt = Counter()
for lk in reliable_links:
    s, t = lk.get("source"), lk.get("target")
    if s and t:
        pairs_new_cnt[tuple(sorted((s, t)))] += 1

known_pairs_reinforced = {p: c for p, c in pairs_new_cnt.items() if p in pairs_before}
novel_pairs = {p: c for p, c in pairs_new_cnt.items() if p not in pairs_before}

pair_diff = {
    "unique_pairs_before": len(pairs_before),
    "unique_pairs_after":  len(pairs_before) + len(novel_pairs),
    "novel_pairs_added":   len(novel_pairs),
    "known_pairs_reinforced": len(known_pairs_reinforced),
    "novel_pair_ratio_in_new_links": round(safe_div(sum(novel_pairs.values()), len(reliable_links)) * 100, 1),
    "top_novel_pairs": [
        {"source": p[0], "target": p[1], "count": c}
        for p, c in sorted(novel_pairs.items(), key=lambda x: -x[1])[:10]
    ],
    "top_reinforced_pairs": [
        {"source": p[0], "target": p[1], "base_count": pairs_before[p], "new_count": c}
        for p, c in sorted(known_pairs_reinforced.items(), key=lambda x: -x[1])[:10]
    ],
}

# ── 7. 接力链统计 ─────────────────────────────────────────────────────────
relay_path = _root / "outputs" / "q3" / "relay_chains.csv"
relay_summary = {"count": 0, "top10": []}
if relay_path.exists():
    with open(relay_path, encoding="utf-8-sig", newline="") as f:
        relay_rows = list(csv.DictReader(f))
    relay_summary["count"] = len(relay_rows)
    relay_summary["avg_shared_partners"] = round(
        sum(int(r["shared_partner_count"]) for r in relay_rows) / max(1, len(relay_rows)), 1
    )
    relay_summary["avg_gap_months"] = round(
        sum(int(r["gap_months_to_2034"]) for r in relay_rows) / max(1, len(relay_rows)), 1
    )
    relay_summary["top10"] = [
        {
            "predecessor": r["predecessor"],
            "successor":   r["successor"],
            "shared_partner_count": int(r["shared_partner_count"]),
            "gap_months_to_2034": int(r["gap_months_to_2034"]),
            "predecessor_pattern": r.get("predecessor_pattern", ""),
            "successor_pattern":   r.get("successor_pattern", ""),
        }
        for r in relay_rows[:10]
    ]

# ── 汇总输出 ──────────────────────────────────────────────────────────────
result = {
    "summary": {
        "base_links": len(base_links),
        "reliable_links_added": len(reliable_links),
        "link_increase_pct": round(len(reliable_links) / len(base_links) * 100, 4),
        "new_nodes_introduced": len(new_nodes),
        "novel_pairs_added": len(novel_pairs),
        "bridge_companies": len(bridge_companies),
        "relay_chains": relay_summary["count"],
        "high_suspicion_companies_gt50": company_diff["score_distribution"]["gt50"],
    },
    "network_diff": network_diff,
    "company_diff": company_diff,
    "temporal_diff": temporal_diff,
    "cargo_diff": cargo_diff,
    "pair_diff": pair_diff,
    "bridge_companies_top20": top_bridges,
    "relay_chains": relay_summary,
}

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[OK] Analysis saved to {OUT_FILE}")

# ── 控制台摘要 ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY: Knowledge Graph Before vs After Completion")
print("="*60)
s = result["summary"]
nd = network_diff
cd = company_diff
print(f"\n[Network]")
print(f"  Nodes:  {nd['node_count_before']:,} → {nd['node_count_after']:,}  (+{nd['new_nodes']})")
print(f"  Edges:  {nd['edge_count_before']:,} → {nd['edge_count_after']:,}  (+{nd['new_edges']})")
print(f"  Links:  {nd['link_count_before']:,} → {nd['link_count_after']:,}  (+{nd['new_links']}, +{s['link_increase_pct']:.4f}%)")
print(f"  Density: {nd['density_before']:.8f} → {nd['density_after']:.8f}  ({nd['density_change_pct']:+.4f}%)")
print(f"  Degree mean: {nd['degree_before']['mean']} → {nd['degree_after']['mean']}")

print(f"\n[Companies Affected]")
print(f"  Affected: {cd['affected_companies']}  (score>50: {cd['score_distribution']['gt50']}, >30: {cd['score_distribution']['gt30']})")
print(f"  Revival types: {cd['revival_type_counts']}")
print(f"  Dormancy max: {cd['dormancy_max']} months, mean: {cd['dormancy_mean']} months")

print(f"\n[Temporal]")
tn = temporal_diff["year_new"]
print(f"  New links by year: {tn}")

print(f"\n[Cargo]")
cgd = cargo_diff
print(f"  Fish ratio: {cgd['fish_ratio_before']:.1f}% → new links {cgd['fish_ratio_new']:.1f}%")
print(f"  New HS codes introduced: {cgd['new_hscodes_introduced']}")

print(f"\n[Trading Pairs]")
pd = pair_diff
print(f"  Novel pairs: {pd['novel_pairs_added']}  Known reinforced: {pd['known_pairs_reinforced']}")

print(f"\n[Network Patterns]")
print(f"  Bridge companies: {s['bridge_companies']}")
print(f"  Relay chains: {s['relay_chains']}  (avg gap {relay_summary.get('avg_gap_months','?')} months)")
