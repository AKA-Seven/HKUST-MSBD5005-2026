# Output artefacts (Q1–Q4)

Intermediate tabular (**CSV**) and structured (**JSON**) products from the analysis pipeline summarized in **`analysis/README.md`**. Together they characterize company-level behaviour patterns, graded prediction bundles, anomalies after graph expansion, fused suspicion scores, and supporting cluster statistics.

Representative magnitudes (**row counts**, **score ranges**) quoted below stem from **one calibrated run** against the bundled challenge horizon (2034‑only predictions, etc.). Regenerating the pipeline yields the same schemas; raw counts may drift if inputs change.

CSV files are written with BOM‑prefixed UTF‑8 (**utf‑8-sig**) for spreadsheet tools; UTF‑8 JSON prefers two-space indentation.

---

## Folder layout

| Path | Role |
|------|------|
| `q1/` | Temporal portraits and pairwise relationship patterns |
| `q2/` | Bundle QA metrics and merged predictions accepted as plausible |
| `q3/` | Revival scoring, bridging, relay summaries, auxiliary graph-diff rollup |
| `q4/` | Clustered companies, clustered edges, final suspicion leaderboard |

Representative filenames:

```
q1 / q1_temporal_patterns.json
    q1_relationship_patterns.csv
    q1_relationship_patterns_top.json

q2 / bundle_reliability.csv · reliable_links.json

q3 / anomaly_delta.csv · bridge_companies.csv · relay_chains.csv
    top_* .json excerpts · graph_diff_analysis.json

q4 / company_clusters.csv · edge_clusters.csv · suspicion_ranking.csv
    suspects_* .json · top_* .json
```

---

## Q1 — Temporal and pairwise cues

### `q1_temporal_patterns.json`

One record per analysed company (**~1.37k** illustrative count), typically mixing high-activity exporters, endpoints appearing in predictive bundles, and dormant predecessors kept for relay alignment.

Key fields:

| Field | Meaning |
|-------|---------|
| `company` | Base graph node identifier |
| `monthly_counts` | `{ "YYYY-MM": count }` month-by-month traffic |
| `first_date`, `last_date` | Alive window |
| `total_links`, `active_months`, `partner_count`, `fish_hscode_ratio` | Summaries |
| `temporal_pattern` | `stable \| bursty \| periodic \| short_term \| general` |

Approximate empirical label mix on the reference profile: **`stable`** dominant (~86 %); residual mass split across **`general` (~9 %), `short_term` (~3 %), `periodic` (~2 %), `bursty` (<1 %). Skew arises from purposely biasing sampling toward vigorous traders.

---

### `q1_relationship_patterns.csv`

Thousands of pairwise rows after inverted-index pruning; each retains:

| Field | Meaning |
|-------|---------|
| `company_a`, `company_b` | Lexicographically ordered pair |
| `relationship_pattern` | `synchronous`, `relay`, `substitution`, `short_term_collab`, `co_active` |
| `confidence` | Merged heuristic confidence |
| `month_overlap_*` / shared-partner overlaps | Temporal or structural coexistence summaries |
| `gap_months` | Calendar gap separating predecessor vs successor arcs when relay-like |
| `a_first_month`, `a_last_month`, etc. | Month bounds |
| `a_temporal_pattern`, `b_temporal_pattern` | Propagated Q1 monadic labels |

Example observed frequency snapshot: synchronous dominates; relay-heavy pairs may be scarce here because exhaustive relay corroboration reappears in **Q3** with looser predicates.

---

### `q1_relationship_patterns_top.json`

Compact **`Top-K`** excerpts bucketed per relationship pattern — convenient for thumbnails without decompressing CSV.

---

## Q2 — Bundle reliability and pooled predictions

### `bundle_reliability.csv`

**12** rows summarising predictive programmes (`carp`, `tuna`, …). Representative columns mirror analysis weights:

| Bucket | Typical fields |
|--------|----------------|
| Integrity | endpoint coverage, duplicated pair fractions, unseen vs seen pairs |
| Commodity | valid HS prevalence, fisheries share |
| Calendar | stray-date fractions, temporal alignment with exporter lifetimes |
| ML boost | logistic scores, percentile markers, withheld-set AUC |
| Outcome | `score`, `label` (`reliable` / `suspicious` / `reject`), categorical `bundle_type`, `repetition_type` |

Representative tally on the reference tuning: **`reliable` = 5**, **`suspicious` = 6**, **`reject` ≈ 1** — illustrative for threshold interpretation.

---

### `reliable_links.json`

**394** fused directed predictions after subtracting overlaps with historical edges and collapsing duplicates induced by overlapping bundles (`generated_by`, `dataset` trace provenance).

Per-edge attributes align with originating records: routed ship date (**2034‑only** cohort), Harmonised commodity code, logistical measures when present (`weightkg`, TEU surrogates, monetised valuations).

---

## Q3 — After expansion analytics

### `anomaly_delta.csv`

Roughly **486** materially touched firms; columns expand behavioural storytelling:

Prominent entries include dormant span (`dormancy_months`), dormant multiplier (`dormancy_weight`), enumerated gap months reconstructed inside earlier lifetimes, extrapolated bursts, partner and HS diversification tallies, aggregated **`suspicious_revival_score` ∈ [0,100]**, and textual mismatch notes between Q3 and Q1 labels.

Approximate empirical distribution snapshots (reference tuning): arithmetic mean **≈ 5**, median **≈ 3**, high tail **above 50 (~14 firms)**, intermediary band **between 10–50 (~few dozen)**.

---

### Ancillary previews

| File | Description |
|------|-------------|
| `top_anomaly_delta.json` | First **100** anomalies — JSON mirror richer than CSV truncation |
| `bridge_companies.csv` · `top_bridge_companies.json` | **457** illustrative structural bridges prioritising widest **bridge_scope** (sample tops reach **five-digit** reachable-node counts with handfuls of bridging edges only) |
| `relay_chains.csv` · `top_relay_chains.json` | **300** ordered predecessor→successor stories sorted by corroborative partner multiplicity (shared evidence often **tens→hundreds**) |
| `graph_diff_analysis.json` | Seven thematic blocks (`summary`, coarse network deltas, revived firm histograms, year/month injection curves, commodity shifts, illustrative pair deltas, illustrative bridge/top-relay excerpts) consolidating headline counts (e.g. **394** novel predicted trades, tens of revived firms above suspicious cut-offs) |

---

## Q4 — Cluster semantics and leaderboard

### `company_clusters.csv`

Aligns **`~1371`** profiled exporters with behavioural statistics, bridging / relay overlays, categorical **business_mode** stereotype (**dormant_revival**, **short_lived**, **fish_intensive_hub**, …), anomaly score, hierarchical cluster breadcrumbs (multi-level **`C{S}- …`** paths), cosine nearest analogue.

Illustrative business-mode multiplicity (counts vary after retuning): dormant-revival arcs **≈ 26**, short-lived façades **≈ 24**, dominant general traders **`~970+`**, residual niche-shell balance shrinks materially once dormant clusters are peeled.

IsolationForest anomalies cluster around **≈ 110** rows (`isolation_label`).

---

### `edge_clusters.csv`

**363** pairwise aggregates joining predicted vs realised traffic, semantic tagging (`historical_backbone`, `multi_tool_bridge`, …), cluster colour identifiers, logarithmic **`edge_width_score`** surrogates for plotting emphasis.

Rough skew: backbone lanes dominate volumetric mass; flagged multi-bundle bridges remain **~few tens — low double-digit** illustrative — strongest multi-source corroborations.

---

### `suspicion_ranking.csv` — primary ranking table

Extends clustered rows by eight harmonised **`sig_*` signals**, composite (**`composite_score` ∈ [0 , 1]**) multiplicity (`signal_count`), confidence tier (`HIGH`/`MEDIUM`/`LOW`), and narrative **`evidence_chain`** strings.

Snapshot confidence tallies (**reference calibration**):

| Tier | Rough count |
|------|---------------|
| `HIGH` | ~25 exporters |
| `MEDIUM` | ~41 |
| `LOW` | Remaining bulk |

Front runners often combine **six or seven simultaneous strong cues**.

---

### JSON companions (`suspects_high.json`, …)

Dedicated extracts isolating maximal-confidence cohort subsets, condensed **Top‑N** leaderboard slices (**e.g.** `top_suspects.json` ≈50 rows) preserving essential signals for slide decks — schema mirrors parent CSV lines.

---

## Cross-stage consumption map

| Upstream artefact | Downstream use |
|-------------------|----------------|
| `q1_temporal_patterns.json` | Conditions Q3 scoring priors · supplies Q4 categorical risk |
| Bundle scores **`reliable_links.json`** | Feeds anomaly expansion, pairwise relay mining, fused suspicion |
| `anomaly_delta.csv`, bridge & relay summaries | Signals for clustering & sceptical scoring tiers |
| `company_clusters.csv`, `edge_clusters.csv` | Diagnostics and narrative scaffolding around `suspicion_ranking.csv` |
