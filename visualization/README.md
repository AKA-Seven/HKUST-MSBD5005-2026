# MC2 Route Frequency & Lifecycle Flow

This folder contains the new standalone 3D visualization for MC2. It does not depend on the older `viz/` prototype. The main visual style is inspired by airline route-frequency maps: companies are stations, trade/predicted relationships are routes, and animated particles indicate flow strength.

## Run Order

From the project root (use your conda Python if paths differ):

```powershell
& "D:\AnacondaEnviroment\envs\python312\python.exe" analysis/run_pipeline.py
& "D:\AnacondaEnviroment\envs\python312\python.exe" visualization/build_3d_data.py
```

Then open:

```text
visualization/index.html
```

The page loads `visualization/data/mc2_3d_data.js`, so it can be opened directly in a browser. Internet access is required for the CDN libraries: D3, Three.js, and 3d-force-graph.

## Data Flow

```text
MC2 raw JSON (MC2/)
  -> analysis/run_pipeline.py
  -> outputs/q1 … outputs/q4/
  -> visualization/build_3d_data.py
  -> visualization/data/mc2_3d_data.js
  -> visualization/index.html
```

The browser does not load the full 5.4M-edge graph. The builder keeps a focused subset:

- Top anomaly-ranked companies from `outputs/q4/company_clusters.csv`.
- Companies affected by predicted links.
- Company-month activity summaries from the base graph.
- Aggregated route-frequency links from base context and predicted bundle links.
- Similarity links from nearest-company behavior clustering.
- Lifecycle flows inferred from first seen, last seen, gap-filled, extended, and burst months.

## View Modes

- `Route Frequency`: companies are arranged like airport stations. Arcs show company-to-company route frequency. Brighter routes and denser particles indicate stronger predicted or repeated activity.
- `Lifecycle Flow`: implements the D-style view. It shows inferred stage transitions such as first seen -> last seen -> gap filled -> extended -> burst.
- `Hybrid`: overlays route frequency with the strongest lifecycle anomaly flows.

## Layouts

- `Cluster`: stations are grouped by behavior cluster, useful for finding suspicious groups.
- `Anomaly`: station height is driven by anomaly score.
- `Fish`: station height is driven by fish-related HS code ratio.

Camera presets:

- `Map`: route-frequency overview.
- `Side`: emphasizes arc height and anomaly separation.
- `Flow`: better angle for lifecycle-flow mode.
- Mouse drag and scroll are available for free rotation and zoom.

## Interaction

- Use the time slider or `Play` to reveal routes up to a selected month.
- Filter by reliable/suspicious/reject bundle sets or by one specific bundle.
- Raise the bundle score threshold to focus on high-confidence predictions.
- Raise the anomaly threshold to focus on unusual companies.
- Search a company name to focus on its route neighborhood.
- Click stations, routes, or lifecycle flows to inspect company profile, prediction source, route months, score, and anomaly evidence.

## Interpretation Notes

The visualization is designed to support visual analytics, not to prove illegal fishing by itself. Stronger candidates are companies that combine several signals:

- High anomaly score.
- Non-zero suspicious revival score.
- Filled activity gaps, extended activity after apparent dormancy, or monthly bursts.
- High similarity to another company with a compatible time gap.
- Reliable predicted links that add new HS codes, partners, or temporal continuity.

Use the route-frequency view to find repeated or high-confidence predicted connections, then use the lifecycle-flow view to explain whether those connections extend a company's activity, fill a gap, or create a burst after dormancy.
