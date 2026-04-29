"""快速验证新版 detect_anomalies.py 的评分分布，不需要加载主图。"""
import sys
from pathlib import Path
import json

_root = Path(__file__).resolve().parents[2]
_shared = str(_root / "analysis" / "shared")
_q3 = str(_root / "analysis" / "q3")
for p in (_shared, _q3):
    if p not in sys.path:
        sys.path.insert(0, p)

from detect_anomalies import (
    compare_anomalies,
    build_profiles_for_companies,
    build_prediction_profiles,
    compare_company_profile,
)

# 加载 reliable_links（Q2 产物，不需要主图）
reliable_links_path = _root / "outputs" / "q2" / "reliable_links.json"
q1_path = _root / "outputs" / "q1" / "q1_temporal_patterns.json"

print(f"Loading reliable_links from {reliable_links_path}...")
reliable_links = json.loads(reliable_links_path.read_text(encoding="utf-8"))
print(f"  {len(reliable_links)} reliable links")

temporal_patterns = []
if q1_path.exists():
    temporal_patterns = json.loads(q1_path.read_text(encoding="utf-8"))
    print(f"  {len(temporal_patterns)} Q1 temporal patterns")
else:
    print("  Q1 patterns not found, skipping Q1 bonus")

# 模拟"before 画像为空"的极端情况（公司首次出现于可靠链接）
print("\n=== 模拟：全为新公司（base 画像为空）===")
pred_profiles = build_prediction_profiles(reliable_links)
results_new = [
    compare_company_profile(c, None, prof, None)
    for c, prof in list(pred_profiles.items())[:200]
]
scores_new = [r["suspicious_revival_score"] for r in results_new]
print(f"Max: {max(scores_new):.1f}  Mean: {sum(scores_new)/len(scores_new):.2f}  >20: {sum(1 for s in scores_new if s>20)}")

# 简化版主图：从 reliable_links 自己提取 source/target 模拟 before 画像
# （实际运行时 before 来自主图；此处用 reliable_links 自身相互对比，是下界估计）
print("\n=== 实际字段检查（取前 5 条）===")
for r in sorted(results_new, key=lambda x: -x["suspicious_revival_score"])[:5]:
    print(
        f"  {r['company'][:35]:35s}  score={r['suspicious_revival_score']:5.1f}"
        f"  ext={len(r['extended_months'].split(';')) if r['extended_months'] else 0}"
        f"  partners={r['new_partner_count']}"
        f"  dormancy={r['dormancy_months']}"
    )

print("\n=== 列名检查 ===")
if results_new:
    print(list(results_new[0].keys()))
