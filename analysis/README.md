# Analysis pipeline overview

Documentation for **VAST Challenge 2023 Mini-Challenge 2**: mining a large maritime trade knowledge graph to prioritise firms that may relate to illegal, unreported, or unregulated (IUU) fishing. This document summarises methods, thresholds, and how results connect across stages Q1–Q4.

---

## Flow of stages

Rough processing order:

1. Load base graph (~5.4M directed trade edges) and twelve predicted “bundle” edge sets  
2. Build a lightweight index (nodes, directed pairs, exact edges, HS codes, per-company date ranges) — **single pass** over edges where possible  
3. **Q2** — Evaluate each bundle with rule metrics and self-supervised link prediction; score bundles; merge into a consolidated reliable link list  
4. **Q1** — Temporal portraits of representative companies and pairwise relationship patterns (can run logically after bundles are defined so sampling aligns with downstream tasks)  
5. **Q3** — Revival-style anomaly scoring, structural bridges, and relay chains on top of merged reliable edges and Q1 patterns  
6. **Q4** — Unsupervised company clustering/isolation scoring, semantic edge clustering, then a fused suspicion score with human-readable evidence strings  

Across heavy edge scans (Q1 company profiles, Q3 company-level features, parts of Q4), the guiding idea is **one linear pass over edges** whenever feasible, restricting aggregation to explicit company whitelists, to avoid repeated full-graph I/O.

---

## Shared primitives

### Configuration

Centralised constants define: calendar bounds for valid shipments, Harmonised System (HS) prefixes treated as fisheries-related **for this study**, temporal cut-offs for supervised link prediction (train up to end of 2033; hold-out 2034), negative-sample size for link prediction, and a fixed RNG seed (**42**) for repeatable sampling and models.

### Index construction

One pass yields:

| Structure | Role |
|-----------|------|
| Node set | All company identifiers |
| Directed pairs | Directed trade lanes as ordered pairs |
| Exact edges | Quadruples (source, target, date, HS) for duplication checks |
| HS set | Codes observed in-base |
| Pair counts | How often each directed pair appears |
| Calendar span | Minimum and maximum dates in-base |
| Per-company date ranges | First and last active dates for temporal consistency scoring (Q2) |

Helpers normalise month keys and HS string formats.

---

## Q1 — Temporal and pairwise patterns

### Company-level temporal patterns

**Goal.** Label each sampled company’s activity over time into one of five coarse behaviours.

**Company pool.** Union of: (i) roughly the top cohort by aggregate activity, (ii) all endpoints appearing in predicted bundles (aligned with later stages), (iii) **early-exit predecessors** — firms inactive before a fixed calendar date but with sufficient history so dormant—successor relays are not structurally invisible.

Motivation for (iii): naïvely taking only high-activity entities under-represents actors that went quiet early but are plausible “predecessors” in relay stories.

**Pattern classes** (illustrative cut-offs baked into classifier code)

| Pattern | Typical rule intuition |
|---------|-------------------------|
| `stable` | High fraction of calendar months covered; mild month-to-month spread |
| `bursty` | Peak month strongly exceeds typical intensity while coverage stays narrow |
| `periodic` | Active months spaced with fairly regular quarterly-ish rhythm |
| `short_term` | Active span strictly below one year |
| `general` | Residual |

**Representative preprocessing constants** (modifiable in central config):

- Thousands of busiest firms PLUS all bundle-touching endpoints PLUS **early retirees** inactive before **`2033‑07‑01`** with **`≥ 20`** historical edges — so dormant predecessors are not systematically dropped.

Implementation uses whitelist filtering and single-pass accumulation over edges.

### Pair-level relationship patterns

**Goal.** Classify ordered company pairs into **synchronous**, **relay**, **substitution**, **short-term collaboration**, **co-active**, avoiding an all-pairs quadratic scan.

**Candidate generation**

1. Neighbour lists for each selected firm  
2. Inverted index: partner → firms that traded with it  
3. Exclude hyper-hub partners touching too many sampled firms  
4. Enumerate pairs that share at least a minimum partner count  

Relay requires strict non-overlap of calendar windows plus partner overlap; confidence blends gap length and overlap strength. Outputs are capped per pattern class so one label does not dominate the file size.

Priority order among labels follows: relay beats substitution beats synchronous beats short_term_collab beats co_active.

**Relay heuristics**

- Temporal windows must remain disjoint with capped calendar gap (**≤ ~60 months**).  
- Overlap counts require either enough shared counterparties (**≥ 3**) **or** a minimal Jaccard overlap on joint partners (**≥ ~0.05**) as a hedge for large corporates.

---

## Q2 — Bundle reliability

### Rule layer

Bundles are assessed with reproducible proportions only (no subjective override): endpoints present in-base, historically seen directed pairs before prediction, HS validity vs base vocabulary, fisheries HS share, date validity, completeness of logistics fields, pair uniqueness and duplication, extreme country-code anomalies, temporal alignment of predictions with endpoints’ alive windows, etc.

### Self-supervised link prediction

Rough recipe: constrain structure to edges on or before a training cutoff; treat the next calendar year’s edges among base nodes as positives; sample unseen pairs as negatives; extract low-dimensional topological features (neighbour overlaps, spectral-style counts, degrees, repetition counts restricted to train period); logistic regression with scaled features; evaluate on held-out accuracy via AUC. Model scores summarise each predicted edge and each bundle distribution (e.g., mean ML probability and high quantiles).

### Composite score

Weighted combination (coefficients apply to normalised features):

```
score =
  20·endpoint_in_base + 25·seen_pair_ratio + 20·valid_hscode_ratio
  + 10·physical_fields + 10·unique_pair + 5·fish_hscode_ratio
  + 15·ml_link_probability + 10·temporal_consistency_ratio
  − 25·outside_date_ratio − penalties(geography anomalies, large pair-duplication bursts)
```

`reliable` requires both a sufficiently high headline score **and** hard feasibility checks (`outside_date_ratio == 0`, pair-duplication caps, temporal consistency thresholds, HS coverage floors…). Nearby scores without satisfying these fall into **`suspicious`**; weaker totals **`reject`**.

Secondary tags summarise **bundle type** (novel vs augmentative…) and **repetition type** across diversity axes.

### Merged predictions

Across bundles marked reliable, predictions are pooled, duplicates to the base graph removed, duplicates across bundles fused, yielding one deduplicated set of hypothetical new shipments (metadata includes which originating bundle justified each retained edge).

---

## Q3 — Graph completion vs base

### Revival-style anomaly score (`compare_anomalies`)

Per firm touched by merged predictions, summands (each capped internally) loosely follow:

```
filled_gap    — months inside historical window newly covered by predictions  
extended_gap  — first predicted months stretching beyond prior last shipment  
burst_months — months whose counts exceed smoothed dispersion  
partner / HSnovelty — diversification of counterparties / commodities injected  
prediction_density — injected mass vs historical mass  
bonus_q1_conflict — uplift when dormant “short-lived” personas suddenly burst, etc.

dormancy_weight = 1 + min(4 , dormancy_months / 12)   -- caps around 5×
```

Net index **suspicious_revival_score ∈ [0 , 100]**; dormant gaps inflate weight but never alone guarantee maxima.

Representative extremes on a tuned run peaked **near the low‑70s** with modest means/medians elsewhere—useful as relative ordering, not calibrated legal evidence.

### Bridge firms

**Bridge_scope** aggregates, for participating firms, how many counterparties became newly reachable purely because of supplementary predicted bridging edges versus the original graph. Narrow noise is trimmed with a modest minimum scope floor.

### Relay chains

“Predecessor” firms dormant before **`2033-01-01`**, “successor” firms receiving credible predicted edges afterward, corroborated by **≥ 2** shared historical partners unless filtered. Rank ordering favours widest overlap; tabular listings are capped (~300 illustrative rows typical).

### Optional richer graph-diff summary

Separate analysis aggregates network metrics before/after completion, summaries of anomalies, timelines, commodities, bridges, and relays in one JSON artefact parallel to CSV-level outputs.

---

## Q4 — Suspicion synthesis

### Company table

Roughly seventeen engineered descriptors layer:

- Base-trade statistics (intensity dispersion, diversification, fisheries exposure, physical aggregates).  
- Ordinal **Q1** risk overlays.  
- **Q3** revival deltas, dormant spans, bridging magnitudes and relay Booleans.

Unsupervised stage: **`IsolationForest`** contamination ~8 % ⇒ **`anomaly_score ∈ [0,1]`** plus **DBSCAN** micro-communities. Hierarchical clustering (`Agglomerative`, cosine) yields nested cluster paths for sunburst-like navigation; cosine neighbours highlight closest twin firms.

Interpretable stereotypes (**business modes**): diversified brokers, dormant-revival arcs, ephemeral shelf companies, fisheries hubs, residual general traders among others—the mix is purposely kept readable for auditors.

### Edge table

KMeans clustering (automatic **2–6** clusters sizing with row mass) overlays semantic tags such as `multi_tool_bridge`, `fish_dense_bridge`, `novel_predicted_route`, `historical_backbone`, `persistent_cross_bundle`, `opportunistic_route` based on calibrated thresholds over predicted vs history counts.

### Combined suspicion score (`synthesize_suspicion`)

| Component | Typical weight band |
|-----------|-----------------------|
| `sig_iso_anomaly` — isolation anomaly | ~0.25 |
| `sig_revival` — scaled revival metric | ~0.25 |
| `sig_dormancy` — dormant span via sigmoid | ~0.15 |
| `sig_q1_inconsistent` — Q1 vs Q3 tension | ~0.10 |
| `sig_q1_risk` — temporal risk tier | ~0.08 |
| `sig_bridge` — log-scaled bridging reach | ~0.10 |
| `sig_relay` — relay successor flag | ~0.05 |
| `sig_fish` — fisheries intensity | ~0.02 |

**Confidence bands** (**HIGH / MEDIUM / LOW**) require simultaneous consideration of **`composite_score ∈ [0,1]`** and **how many** component signals concurrently exceed calibrated mid-thresholds. Natural-language **`evidence_chain`** snippets pipe active modules with ` | ` delimiters for presentation.

---

## Design principles (methodological)

1. **Heavy IO once** — full edge list traversed sparingly via whitelisting.  
2. **Composable signals** — Q1 feeds Q3; Q3 and clusters feed fused Q4 judgments.  
3. **Interpretability first** — every component score kept as its own numeric column alongside aggregate indices.  
4. **Evidence strings** — final narrative lines join active factors for briefing.  
5. **Data-driven calibration** — cut-offs for labels (e.g. bundle tiers, semantic edge buckets, bridge-floor filters) were checked against empirical distributions rather than assumed a priori.  
6. **Reproducibility** — deterministic seed for stochastic steps.
