import csv, json
from collections import Counter

with open('outputs/q1/q1_relationship_patterns.csv', encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))

print('=== Pattern distribution (after fix) ===')
print(dict(Counter(r['relationship_pattern'] for r in rows)))
print()

relays = [r for r in rows if r['relationship_pattern'] == 'relay']
print(f'=== Relay pairs ({len(relays)}) ===')
for r in sorted(relays, key=lambda x: float(x['confidence']), reverse=True):
    print(f"  {r['company_a'][:35]:35s} -> {r['company_b'][:35]:35s}")
    print(f"    conf={r['confidence']}  gap={r['gap_months']}mo  shared={r['shared_partner_count']}  jaccard={r['shared_partner_jaccard']}")
    print(f"    A: {r['a_first_month']}->{r['a_last_month']} ({r['a_temporal_pattern']})  B: {r['b_first_month']}->{r['b_last_month']} ({r['b_temporal_pattern']})")
    print()

print('=== Substitution top 5 (also benefits) ===')
subs = [r for r in rows if r['relationship_pattern'] == 'substitution']
for r in sorted(subs, key=lambda x: float(x['confidence']), reverse=True)[:5]:
    print(f"  {r['company_a'][:30]:30s} -> {r['company_b'][:30]:30s}  conf={r['confidence']}  shared={r['shared_partner_count']}")

print()
print('=== Q1 temporal_patterns total ===')
data = json.load(open('outputs/q1/q1_temporal_patterns.json', encoding='utf-8'))
print(f'Total companies: {len(data)}  (was 1371)')
print('Pattern dist:', dict(Counter(d['temporal_pattern'] for d in data)))
