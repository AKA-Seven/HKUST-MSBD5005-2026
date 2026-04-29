import csv, json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
q3_dir = root / "outputs" / "q3"

# ── anomaly_delta ──────────────────────────────────────────────────────────
print("=" * 60)
print("anomaly_delta.csv")
with open(q3_dir / "anomaly_delta.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

scores = [float(r["suspicious_revival_score"]) for r in rows]
print(f"  Rows     : {len(rows)}")
print(f"  Max score: {max(scores):.1f}")
print(f"  Mean     : {sum(scores)/len(scores):.2f}")
srt = sorted(scores)
print(f"  Median   : {srt[len(srt)//2]:.1f}")
print(f"  >50      : {sum(1 for s in scores if s > 50)}")
print(f"  >30      : {sum(1 for s in scores if s > 30)}")
print(f"  >10      : {sum(1 for s in scores if s > 10)}")
print(f"  Columns  : {list(rows[0].keys())}")
print()
print("  Top 10:")
for r in sorted(rows, key=lambda x: float(x["suspicious_revival_score"]), reverse=True)[:10]:
    print(
        f"    {r['company'][:38]:38s} score={r['suspicious_revival_score']:5}  "
        f"dormancy={r.get('dormancy_months','?'):3}  "
        f"q1={r.get('q1_temporal_pattern','')}"
    )

# ── bridge_companies ───────────────────────────────────────────────────────
bridge_path = q3_dir / "bridge_companies.csv"
print()
print("=" * 60)
if bridge_path.exists():
    with open(bridge_path, newline="", encoding="utf-8-sig") as f:
        bridge_rows = list(csv.DictReader(f))
    print(f"bridge_companies.csv  ({len(bridge_rows)} rows)")
    for r in bridge_rows[:5]:
        print(f"    {r['company'][:40]:40s}  scope={r['bridge_scope']}  links={r['bridge_link_count']}")
else:
    print("bridge_companies.csv  NOT FOUND")

# ── relay_chains ───────────────────────────────────────────────────────────
relay_path = q3_dir / "relay_chains.csv"
print()
print("=" * 60)
if relay_path.exists():
    with open(relay_path, newline="", encoding="utf-8-sig") as f:
        relay_rows = list(csv.DictReader(f))
    print(f"relay_chains.csv  ({len(relay_rows)} rows)")
    for r in relay_rows[:5]:
        print(
            f"    {r['predecessor'][:28]:28s} -> {r['successor'][:28]:28s}"
            f"  shared={r['shared_partner_count']}  gap_m={r['gap_months_to_2034']}"
        )
else:
    print("relay_chains.csv  NOT FOUND")
