# Visualization overview

This folder holds **figure builders** and **rendered views** for Questions 1–4 of the MC2 trade-graph study. Everything is driven by **intermediate analysis artefacts** (the same aggregates used downstream for anomaly and suspicion scoring)—there is no illustrative or fabricated data layered on top.

The intent is layered storytelling: **time and actors** (Q1), **trust in predicted bundles** (Q2), **structural shocks after linking** (Q3), and **integrated suspicion** (Q4). Encoding choices emphasize **comparison** (before/after, bundle vs bundle, cohort vs cohort) and **ranking**, not ornamental chart junk.

Below, **visual idea** summarizes the design rationale; **typical outputs** describe what each deliverable communicates when read against the analysis notebooks.

---

## Q1 · Temporal behaviour of companies

**Visual idea.** Portraits should separate *who trades steadily*, *who spikes*, *who lives briefly on the ledger*, and *which pairs of firms overlap or relay* in calendar space. Colour carries the five **temporal pattern** classes; axes and glyphs emphasise magnitude (volume), breadth (months, partners), and synchrony.

**Typical outputs**

| Output | Encoding | Question it answers |
|--------|-----------|---------------------|
| **Activity vs partners (bubble)** | x = active span, y = distinct partners, size ∝ shipment mass, hue = temporal class | Which firms are habitual large movers vs explosive one-season actors |
| **Monthly heatmap** | Rows = ranked companies; columns = months; colour intensity = monthly edge counts | Joint rhythm across the cohort; quiet corridors and resurgence windows |
| **Ridge / river** | Stacked smoothed monthly curves per company along one timeline | Relative rise and decay; misaligned lifelines hint at takeover or relay timing |
| **Chord (time-sliced)** | Arc segments = firms; ribbons = directional flow in a chosen month; optional time control | Directed partner concentration among top movers for a focal month |

---

## Q2 · Reliability of predicted link bundles

**Visual idea.** Twelve bundle programmes need a **compact audit**: rule coverage, geography of endpoints, duplication, HS realism, temporal fit, plus model scores—without flattening nuance into a single karma number. Scatter and 3‑D views show **overlap of programmes** when they annex similar regions of the firm–firm fabric.

**Typical outputs**

| Output | Encoding | Question it answers |
|--------|-----------|---------------------|
| **Multi‑panel dashboard** | Parallel facets per bundle for key QA ratios; tier styling (reliable / borderline / reject) | At a glance: which bundles are safe to merge and which deserve manual review |
| **Reliability bubble** | x vs y encode structural / ML summaries; size ∝ predicted mass; hue = tier | Concordance between hand-crafted rules and learned link probability |
| **3‑D bundle expansion** | Each viable bundle as hub; chords = added predicted edges toward endpoints; colour = bundle identity | Extent of **footprint overlap** across programmes in one navigable scene |
| **Sankey bridging Q1 and Q2 (where included)** | Flow from temporal cohorts toward bundle-adoption lanes | Narrative stitch from trading style to uptake of predicted arcs |

---

## Q3 · After the graph is completed

**Visual idea.** Once accepted predictions are stitched in, emphasis shifts from *bundle QA* to **network surgery**: relays (dormant predecessor → revived successor sharing partners), bridges (few new stitches that unlock many nodes), and **firm-level suspicion** overlays. Figures stress **sparse leverage**—a handful of lanes moving mass—and **temporal coherence** of revival.

**Typical outputs**

| Output | Encoding | Question it answers |
|--------|-----------|---------------------|
| **Relay Sankey** | Left predecessors, right successors, band width ∝ corroborative shared partners | Hand-offs resembling shell rotation along shared intermediary contacts |
| **Bridge leverage bubble** | x = predicted links mobilised by a firm, y = breadth of reachable partners, hue = leverage efficiency | Entities that widen reach disproportionately from few added edges |
| **Before / after network sketch** | Baseline vs augmented sketch (distinguished line styles where applicable) | Qualitative topology shift when trusted predictions land |
| **Suspicion matrix** | Rows = top flagged firms; columns = diagnostic dimensions or months; darker = stronger signals | Consolidated revival-anomaly tableau for triage |

---

## Q4 · Suspicious companies and evidence

**Visual idea.** The final tier projects companies into **risk space** (projection plus isolation scores), summarizes **signals on spokes** (radar overlays by confidence tier), and shows **nested structure** (treemap: tier → cluster → entity). Goal is transparent **multi-evidence justification**, not a black-box leaderboard.

**Typical outputs**

| Output | Encoding | Question it answers |
|--------|-----------|---------------------|
| **2D PCA ellipses** | Projected plane with confidence blobs for LOW / MEDIUM / HIGH cohorts | Degree of separability between nominal traders and escalation bands |
| **3‑D PCA scatter** | Third axis plus rotation—colour by tier | Persistence of stratification beyond first two latent factors |
| **Radar strip (tier means)** | Spokes for normalised behavioural and structural signals averaged per tier | Which dimensions drive HIGH-tier divergence from the masses |
| **Cluster treemap** | Nested rectangles sized by suspicion mass; hue for cluster ancestry | Hierarchy of hotspots within each risk enclosure |

Standalone tables and textual **evidence chains** beside these views restate quantitative triggers in investigator language.

---

## How to read artefacts together

The analytic path runs Q1→Q4, but visually the story **loops**: Q1 conveys **movement**, Q2 establishes **trust in newly proposed metal**, Q3 exposes **stress lines** when that metal joins the backbone, Q4 folds prior cues into **actionable dossiers**. Use palette and legend cues consistently—the same categorical risk colouring recurs wherever tiers appear.
