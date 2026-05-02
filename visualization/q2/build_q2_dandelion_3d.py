#!/usr/bin/env python3
"""Q2+Q1：Three.js 3D dandelion (vendor three.min.js).

- Nodes sized & tinted by total weighted degree (quiet periphery vs vivid hubs).
- ML bundle tubes use a light-blue → purple spectrum (reliable_links only).
- See HTML output for interaction notes.

输出：visualization/figures_3d/q2/q2_dandelion_graph.html
"""

from __future__ import annotations

import base64
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_vis_root = Path(__file__).resolve().parents[1]
if str(_vis_root) not in sys.path:
    sys.path.insert(0, str(_vis_root))

from figures2d_common import OUTPUTS_DIR_Q1, OUTPUTS_DIR_Q2

ROOT = Path(__file__).resolve().parents[2]
FIG3_Q2 = ROOT / "visualization" / "figures_3d" / "q2"
OUT_HTML = FIG3_Q2 / "q2_dandelion_graph.html"

Q1_REL_CSV = OUTPUTS_DIR_Q1 / "q1_relationship_patterns.csv"
Q1_TEMPORAL_JSON = OUTPUTS_DIR_Q1 / "q1_temporal_patterns.json"

NODE_CAP_3D = 140

OP_ML_BUNDLE = 1.0
OP_Q1 = 0.72

PATTERN_COLORS: dict[str, str] = {
    "stable": "#38bdf8",
    "bursty": "#f9a8d4",
    "periodic": "#a5b4fc",
    "short_term": "#94a3b8",
    "general": "#22d3ee",
    "unknown": "#7dd3fc",
}

Q1_EDGE_HEX = "#bae6fd"

# ML bundle spectrum: light blue → purple (by sorted bundle name index)
_SPECTRUM_LO = (0x93, 0xC5, 0xFD)
_SPECTRUM_HI = (0x7C, 0x3A, 0xED)


def _spectrum_ml_bundle_hex(index: int, n_bundles: int) -> str:
    """0..n-1 → light blue .. purple."""
    if n_bundles <= 1:
        t = 0.5
    else:
        t = index / float(n_bundles - 1)
    lo, hi = _SPECTRUM_LO, _SPECTRUM_HI
    r = int(lo[0] + (hi[0] - lo[0]) * t)
    g = int(lo[1] + (hi[1] - lo[1]) * t)
    b = int(lo[2] + (hi[2] - lo[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _blend_pattern_by_strength(hex_fg: str, strength: float) -> str:
    """Low strength → dull slate tint; high → full pattern RGB."""
    strength = max(0.0, min(1.0, strength))
    dr, dg, db = 28, 38, 52
    try:
        hx = hex_fg.replace("#", "").strip()
        fr = int(hx[0:2], 16)
        fg = int(hx[2:4], 16)
        fb = int(hx[4:6], 16)
    except (ValueError, IndexError):
        fr, fg, fb = 125, 211, 252
    mix = 0.12 + 0.88 * strength
    r = int(dr + (fr - dr) * mix)
    g = int(dg + (fg - dg) * mix)
    b = int(db + (fb - db) * mix)
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def _short(name: str, n: int = 28) -> str:
    if len(name) <= n:
        return name
    return name[: n // 2 - 1] + "…" + name[-(n // 2 - 2) :]


def _fibonacci_sphere_dirs(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    pts = []
    for i in range(count):
        y = 1 - (2 * i + 1) / max(count, 1)
        rr = math.sqrt(max(0.0, 1.0 - y * y))
        th = golden * (i + 1)
        pts.append([math.cos(th) * rr, y, math.sin(th) * rr])
    return np.array(pts, dtype=float)


def _dandelion_xyz(deg: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    n = len(deg)
    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.zeros((1, 3))

    dirs = _fibonacci_sphere_dirs(n)
    dmax = float(deg.max()) if deg.max() > 0 else 1.0
    dn = np.clip(deg.astype(float) / dmax, 0, 1)
    radial = 2.35 * (0.1 + 0.9 * np.power(1.0 - dn, 1.38))
    xyz = dirs * radial[:, np.newaxis]

    hub = dn >= np.quantile(dn, min(0.94, 1 - 4 / max(n, 5)))
    xyz[hub] *= 0.62
    xyz += (rng.rand(n, 3) - 0.5) * 0.09
    return xyz


def _temporal_map(companies: list[str], json_path: Path) -> dict[str, str]:
    cset = set(companies)
    out: dict[str, str] = {}
    if not json_path.exists():
        return {c: "unknown" for c in companies}

    try:
        import ijson  # type: ignore[import-untyped]
    except ImportError:
        ijson = None

    if ijson is not None:
        with json_path.open("rb") as f:
            for obj in ijson.items(f, "item"):
                c = str(obj.get("company", ""))
                if c in cset:
                    p = str(obj.get("temporal_pattern") or "unknown").strip().lower()
                    out[c] = p if p in PATTERN_COLORS else "unknown"
        return {c: out.get(c, "unknown") for c in companies}

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for r in data:
        c = str(r.get("company", ""))
        if c in cset:
            p = str(r.get("temporal_pattern") or "unknown").strip().lower()
            out[c] = p if p in PATTERN_COLORS else "unknown"
    return {c: out.get(c, "unknown") for c in companies}


def _kmeans_labels(X: np.ndarray, k: int, rng: np.random.RandomState, max_iter: int = 60) -> np.ndarray:
    """Lloyd k-means；k 限制在 [2, n]。"""
    n = X.shape[0]
    k = int(max(2, min(k, n)))
    idx0 = rng.choice(n, size=k, replace=False)
    centroids = X[idx0].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = new_labels == j
            if np.any(mask):
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centroids = new_centroids
    return labels


def _communities_from_adjacency(W: np.ndarray, k_comm: int, rng: np.random.RandomState) -> np.ndarray:
    """归一化拉普拉斯前几维谱嵌入 + k-means，得到每个节点的小区 id。"""
    n = W.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=np.int32)
    k_comm = int(max(8, min(k_comm, max(8, n // 4), n - 1)))

    W = np.maximum(W, W.T)
    np.fill_diagonal(W, W.diagonal() + 1e-9)
    d = W.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
    Ln = np.eye(n) - (inv_sqrt[:, np.newaxis] * W) * inv_sqrt[np.newaxis, :]
    try:
        vals, vecs = np.linalg.eigh(Ln)
        ndim = max(2, min(k_comm, n - 1, 14))
        emb = vecs[:, 1 : ndim + 1]
        if emb.shape[1] < 2:
            emb = np.hstack([emb, rng.normal(size=(n, 2 - emb.shape[1])) * 1e-4])
    except np.linalg.LinAlgError:
        emb = rng.normal(size=(n, min(8, n)))

    labels = _kmeans_labels(emb.astype(float), k_comm, rng)
    return labels.astype(np.int32)


def _fan_bulge_series(
    node_ids: list[int],
    pos_arr: np.ndarray,
    hub: np.ndarray,
    toward: np.ndarray,
    spread: float,
) -> list[float]:
    """按围绕「hub→toward」轴的方位角排序叶子，生成一串对称展开的 bulge，扇形展开弧线。"""
    if not node_ids:
        return []
    axis = toward.astype(float) - hub.astype(float)
    la = float(np.linalg.norm(axis))
    if la < 1e-9:
        axis = np.array([0.0, 1.0, 0.0], dtype=float)
        la = 1.0
    axis = axis / la
    tmp = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(tmp, axis))) > 0.92:
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(axis, tmp)
    nu = float(np.linalg.norm(u))
    if nu < 1e-9:
        u = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        u = u / nu
    v = np.cross(axis, u)

    angs: list[float] = []
    for nid in node_ids:
        d = pos_arr[nid].astype(float) - hub.astype(float)
        d = d - axis * float(np.dot(d, axis))
        angs.append(math.atan2(float(np.dot(d, v)), float(np.dot(d, u))))
    order = sorted(range(len(node_ids)), key=lambda j: angs[j])
    n = len(node_ids)
    bulges = [0.10] * n
    for rank, j in enumerate(order):
        t = (rank / max(n - 1, 1)) - 0.5
        bulges[j] = 0.10 + spread * (2.0 * t)
    return bulges


def _aggregate_cross_community_edges(
    pairs: list[tuple[int, int, float]],
    comm: np.ndarray,
    pos_arr: np.ndarray,
    *,
    color: str,
    opacity: float,
    filter_key: str,
    tier: str,
    trunk_base_r: float,
    trunk_scale_r: float,
    trunk_cap: float,
    fan_base_r: float,
    fan_scale_r: float,
    fan_cap: float,
    max_fan_per_side: int = 26,
    fan_spread: float = 0.17,
    opacity_mult: float = 1.0,
    tube_mult: float = 1.0,
) -> tuple[list[dict], float]:
    """跨小区：质心之间细干线 + 区内扇形细枝（节点↔本侧质心）。小区内原始边仍不单独画。"""
    buckets: dict[tuple[int, int], list[tuple[int, int, float]]] = defaultdict(list)
    for ia, ib, w in pairs:
        ca_i, cb_i = int(comm[ia]), int(comm[ib])
        if ca_i == cb_i:
            continue
        ca, cb = (ca_i, cb_i) if ca_i < cb_i else (cb_i, ca_i)
        buckets[(ca, cb)].append((ia, ib, float(w)))

    max_w = 0.0
    for plist in buckets.values():
        max_w = max(max_w, sum(w for _, _, w in plist))
    max_w = max(max_w, 1.0)

    edges_out: list[dict] = []
    for (ca, cb), plist in buckets.items():
        wt = sum(w for _, _, w in plist)
        mask_a = comm == ca
        mask_b = comm == cb
        pa = pos_arr[mask_a].mean(axis=0)
        pb = pos_arr[mask_b].mean(axis=0)

        trunk_r = tube_mult * (trunk_base_r + trunk_scale_r * math.sqrt(wt / max_w))
        trunk_r = min(tube_mult * trunk_cap, trunk_r)
        trunk_seed = (ca * 13 + cb * 7) % 5
        trunk_bulge = 0.11 + 0.035 * trunk_seed / 5.0

        edges_out.append(
            {
                "agg": True,
                "trunk": True,
                "fan": False,
                "x0": float(pa[0]),
                "y0": float(pa[1]),
                "z0": float(pa[2]),
                "x1": float(pb[0]),
                "y1": float(pb[1]),
                "z1": float(pb[2]),
                "c0": ca,
                "c1": cb,
                "tubeR": trunk_r,
                "bulge": float(trunk_bulge),
                "color": color,
                "opacity": min(1.0, opacity * opacity_mult),
                "filterKey": filter_key,
                "tier": tier,
                "wt": float(wt),
                "ia": -1,
                "ib": -1,
            }
        )

        wa: dict[int, float] = defaultdict(float)
        wb: dict[int, float] = defaultdict(float)
        for ia, ib, w in plist:
            if comm[ia] == ca and comm[ib] == cb:
                wa[ia] += w
                wb[ib] += w
            elif comm[ia] == cb and comm[ib] == ca:
                wa[ib] += w
                wb[ia] += w

        def _top_nodes(wmap: dict[int, float]) -> list[int]:
            items = sorted(wmap.items(), key=lambda kv: -kv[1])[:max_fan_per_side]
            return [k for k, _ in items]

        ids_a = _top_nodes(wa)
        ids_b = _top_nodes(wb)

        max_wa = max(wa.values(), default=1.0)
        max_wb = max(wb.values(), default=1.0)

        bulges_a = _fan_bulge_series(ids_a, pos_arr, pa, pb, fan_spread)
        bulges_b = _fan_bulge_series(ids_b, pos_arr, pb, pa, fan_spread)

        for k, nid in enumerate(ids_a):
            fw = wa[nid]
            fan_r = tube_mult * (fan_base_r + fan_scale_r * math.sqrt(fw / max(max_wa, 1e-9)))
            fan_r = min(tube_mult * fan_cap, fan_r)
            p_leaf = pos_arr[nid]
            edges_out.append(
                {
                    "agg": True,
                    "trunk": False,
                    "fan": True,
                    "x0": float(p_leaf[0]),
                    "y0": float(p_leaf[1]),
                    "z0": float(p_leaf[2]),
                    "x1": float(pa[0]),
                    "y1": float(pa[1]),
                    "z1": float(pa[2]),
                    "c0": ca,
                    "c1": cb,
                    "tubeR": fan_r,
                    "bulge": float(bulges_a[k]),
                    "color": color,
                    "opacity": min(1.0, opacity * opacity_mult),
                    "filterKey": filter_key,
                    "tier": tier,
                    "wt": float(fw),
                    "ia": int(nid),
                    "ib": -1,
                }
            )

        for k, nid in enumerate(ids_b):
            fw = wb[nid]
            fan_r = tube_mult * (fan_base_r + fan_scale_r * math.sqrt(fw / max(max_wb, 1e-9)))
            fan_r = min(tube_mult * fan_cap, fan_r)
            p_leaf = pos_arr[nid]
            edges_out.append(
                {
                    "agg": True,
                    "trunk": False,
                    "fan": True,
                    "x0": float(pb[0]),
                    "y0": float(pb[1]),
                    "z0": float(pb[2]),
                    "x1": float(p_leaf[0]),
                    "y1": float(p_leaf[1]),
                    "z1": float(p_leaf[2]),
                    "c0": ca,
                    "c1": cb,
                    "tubeR": fan_r,
                    "bulge": float(bulges_b[k]),
                    "color": color,
                    "opacity": min(1.0, opacity * opacity_mult),
                    "filterKey": filter_key,
                    "tier": tier,
                    "wt": float(fw),
                    "ia": -1,
                    "ib": int(nid),
                }
            )

    return edges_out, max_w


def _select_nodes(
    pred_nodes: set[str],
    raw_preds: list[tuple[str, str, str]],
    q1_pairs: list[tuple[str, str, float, str]],
    cap: int,
) -> list[str]:
    score: defaultdict[str, float] = defaultdict(float)
    for u, v, _ in raw_preds:
        score[u] += 25
        score[v] += 25
    for a, b, conf, _ in q1_pairs:
        w = 1.0 + conf
        if a in pred_nodes:
            score[b] += w
        if b in pred_nodes:
            score[a] += w
        if a in pred_nodes and b in pred_nodes:
            score[a] += w * 0.5
            score[b] += w * 0.5

    ordered = sorted(pred_nodes, key=lambda x: -score.get(x, 0))
    if len(ordered) <= cap:
        rest = [x for x in sorted(score.keys(), key=lambda x: -score[x]) if x not in pred_nodes]
        for x in rest:
            if len(ordered) >= cap:
                break
            ordered.append(x)
        return sorted(set(ordered))

    return sorted(set(ordered[:cap]))


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MC2 · Q1+Q2 3D network</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; overflow: hidden; background: radial-gradient(ellipse 120% 90% at 70% 15%, rgba(40, 90, 140, 0.35), transparent 55%), #061525;
      font-family: "Segoe UI", Roboto, system-ui, sans-serif; color: #e0f2fe; }
    #cv { display: block; width: 100vw; height: 100vh; }
    #panel {
      position: fixed; top: 48px; left: 10px; width: 206px; max-height: calc(100vh - 120px);
      overflow-y: auto; padding: 10px 10px 12px;
      background: rgba(6, 18, 38, 0.92); border: 1px solid rgba(125, 211, 252, 0.35);
      border-radius: 12px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35); z-index: 20;
    }
    #panel h2 { margin: 0 0 8px; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #7dd3fc; font-weight: 700; }
    #panel .fil-btn {
      display: block; width: 100%; margin: 3px 0; padding: 7px 9px; border-radius: 8px;
      border: 1px solid rgba(125, 211, 252, 0.22); background: rgba(15, 40, 72, 0.65);
      color: #e0f2fe; font-size: 11px; text-align: left; cursor: pointer;
      transition: background 0.15s, border-color 0.15s;
    }
    #panel .fil-btn:hover { background: rgba(56, 130, 180, 0.35); border-color: rgba(125, 211, 252, 0.45); }
    #panel .fil-btn.active {
      background: rgba(56, 189, 248, 0.28); border-color: rgba(125, 211, 252, 0.65);
      box-shadow: inset 0 0 0 1px rgba(125, 211, 252, 0.2);
    }
    #btn-auto {
      margin-top: 10px; width: 100%; padding: 8px 10px; border-radius: 8px; cursor: pointer;
      border: 1px solid rgba(125, 211, 252, 0.45); background: rgba(14, 52, 92, 0.85);
      color: #bae6fd; font-size: 12px; font-weight: 650;
    }
    #btn-auto.on { background: rgba(34, 150, 110, 0.45); border-color: rgba(74, 222, 128, 0.55); color: #ecfdf5; }
    #hint {
      position: fixed; left: 226px; bottom: 12px; max-width: min(380px, calc(100vw - 240px));
      padding: 10px 12px; border-radius: 10px; font-size: 11px; line-height: 1.45;
      background: rgba(8, 22, 42, 0.82); border: 1px solid rgba(125, 211, 252, 0.28);
      color: #bae6fd; pointer-events: none;
    }
    #tip {
      position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
      padding: 8px 14px; border-radius: 10px; font-size: 12px;
      background: rgba(8, 22, 42, 0.78); border: 1px solid rgba(125, 211, 252, 0.25);
      color: #e0f2fe; pointer-events: none; display: none; z-index: 15;
    }
    h1.top {
      position: fixed; top: 12px; left: 228px; margin: 0; font-size: 15px; font-weight: 650;
      text-shadow: 0 1px 8px rgba(0,0,0,0.45); z-index: 12; pointer-events: none;
    }
    body.embed-dashboard #panel { display: none !important; }
    body.embed-dashboard h1.top { left: 12px; font-size: 12px; max-width: calc(100vw - 24px); }
    body.embed-dashboard #hint { display: none !important; }
  </style>
  <script src="../../vendor/three.min.js"></script>
</head>
<body>
  <aside id="panel">
    <h2>Layer filter</h2>
    <div id="filter-list"></div>
    <button id="btn-auto" type="button" class="on">Auto rotate · On</button>
  </aside>
  <h1 class="top">Q1+Q2 dandelion · community highways & fan spokes</h1>
  <canvas id="cv"></canvas>
  <div id="tip"></div>
  <div id="hint"><strong>Purpose.</strong> Nodes use Q1 temporal pattern hues, scaled by connectivity: more linked endpoints read larger, brighter, and richer color; quiet periphery is smaller and more transparent. ML predicted links (<code>reliable_links.json</code>) are tubes on a light-blue → purple spectrum per bundle; pale cyan = Q1 relationship confidence. Communities collapse clutter into inter-hub links plus fan spokes.<br>
    <strong>Interaction.</strong> Pick a layer to <strong>show only that signal</strong> (other edges hidden; unrelated nodes hidden). Layout morphs so hubs move inward for that layer. Wheel zoom (fog lightens when close); drag to orbit; hover nodes for labels; auto-rotate is on by default.</div>
  <script>
(function() {
  function decodeB64Utf8(b64) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  var DATA = decodeB64Utf8("__B64_PAYLOAD__");

  var THREERef = window.THREE;
  if (!THREERef) { document.body.innerHTML = "<p>Three.js failed to load. Check that ../../vendor/three.min.js exists.</p>"; return; }

  var scene = new THREERef.Scene();
  scene.background = new THREERef.Color(0x0d2440);
  scene.fog = new THREERef.FogExp2(0x07182e, 0.045);

  var camera = new THREERef.PerspectiveCamera(50, innerWidth / innerHeight, 0.012, 220);
  camera.position.set(5.2, 3.8, 6.4);

  var canvas = document.getElementById("cv");
  var renderer = new THREERef.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setSize(innerWidth, innerHeight);
  if (renderer.outputColorSpace !== undefined && THREERef.SRGBColorSpace !== undefined)
    renderer.outputColorSpace = THREERef.SRGBColorSpace;

  var amb = new THREERef.HemisphereLight(0x8ec5ff, 0x1a2842, 0.55);
  scene.add(amb);
  var dir = new THREERef.DirectionalLight(0xdbeafe, 0.95);
  dir.position.set(4, 10, 6);
  scene.add(dir);
  var fill = new THREERef.PointLight(0x60a5fa, 0.35, 40);
  fill.position.set(-3, 2, -4);
  scene.add(fill);

  function hexInt(hex) {
    return parseInt(hex.replace("#", ""), 16);
  }

  function v3(a) {
    return new THREERef.Vector3(a[0], a[1], a[2]);
  }

  /** 二次贝塞尔弧：控制点在中垂方向抬起 */
  function arcControl(p0, p1, bulge) {
    var a = v3(p0);
    var b = v3(p1);
    var mid = new THREERef.Vector3().addVectors(a, b).multiplyScalar(0.5);
    var chord = new THREERef.Vector3().subVectors(b, a);
    var len = chord.length();
    chord.normalize();
    var up = new THREERef.Vector3(0, 1, 0);
    var orth = new THREERef.Vector3().crossVectors(chord, up);
    if (orth.lengthSq() < 1e-6) orth = new THREERef.Vector3().crossVectors(chord, new THREERef.Vector3(1, 0, 0));
    orth.normalize();
    mid.addScaledVector(orth, bulge * len);
    return mid;
  }

  var nodeMeshes = [];
  var nodeGroup = new THREERef.Group();
  scene.add(nodeGroup);

  DATA.nodes.forEach(function (nd, ni) {
    var color = hexInt(nd.color);
    var colThree = new THREERef.Color(color);
    var baseOp = typeof nd.opacity === "number" ? nd.opacity : 0.96;
    var wgt = typeof nd.weight === "number" ? nd.weight : 0.5;
    var emMult = 0.045 + 0.11 * wgt;
    var mat = new THREERef.MeshPhysicalMaterial({
      color: color,
      metalness: 0.38,
      roughness: 0.3,
      clearcoat: 0.52,
      clearcoatRoughness: 0.24,
      emissive: colThree.clone().multiplyScalar(emMult),
      emissiveIntensity: 1,
      transparent: true,
      opacity: baseOp,
      depthWrite: baseOp > 0.34,
    });
    var geo = new THREERef.SphereGeometry(nd.r, 28, 28);
    var mesh = new THREERef.Mesh(geo, mat);
    mesh.position.set(nd.x, nd.y, nd.z);
    var glowLayers = [];
    var scales = [1.12, 1.24];
    var bases = [0.04 + 0.26 * wgt, 0.012 + 0.078 * wgt];
    for (var gl = 0; gl < scales.length; gl++) {
      var gR = nd.r * scales[gl];
      var glowGeo = new THREERef.SphereGeometry(gR, 18, 18);
      var glowMat = new THREERef.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: bases[gl],
        depthWrite: false,
        depthTest: true,
        blending: THREERef.AdditiveBlending,
      });
      var glow = new THREERef.Mesh(glowGeo, glowMat);
      glow.renderOrder = 8 + gl;
      mesh.add(glow);
      glowLayers.push({ mesh: glow, base: bases[gl] });
    }
    mesh.renderOrder = 10;
    mesh.userData = {
      name: nd.name,
      pattern: nd.pattern,
      nodeIndex: ni,
      baseOpacity: baseOp,
      weight: wgt,
      glowLayers: glowLayers,
    };
    nodeGroup.add(mesh);
    nodeMeshes.push(mesh);
  });

  var edgeGroup = new THREERef.Group();
  scene.add(edgeGroup);

  var nodeComm = DATA.nodeCommunity || [];

  function edgeMatchesFilter(ud, fid) {
    if (fid === "all") return true;
    if (fid.indexOf("tier:") === 0) return ud.tier === fid.slice(5);
    return ud.filterKey === fid;
  }

  function commCentroid(cid) {
    var sx = 0, sy = 0, sz = 0, n = 0;
    for (var i = 0; i < nodeMeshes.length; i++) {
      if (nodeComm[i] !== cid) continue;
      var p = nodeMeshes[i].position;
      sx += p.x; sy += p.y; sz += p.z;
      n++;
    }
    if (!n) return [0, 0, 0];
    return [sx / n, sy / n, sz / n];
  }

  function endpointsForEdge(e) {
    if (e.trunk) {
      return [commCentroid(e.c0), commCentroid(e.c1)];
    }
    if (e.fan && e.ia >= 0) {
      var hubA = commCentroid(e.c0);
      var lf = nodeMeshes[e.ia].position;
      return [[lf.x, lf.y, lf.z], hubA];
    }
    if (e.fan && e.ib >= 0) {
      var hubB = commCentroid(e.c1);
      var lf2 = nodeMeshes[e.ib].position;
      return [hubB, [lf2.x, lf2.y, lf2.z]];
    }
    if (e.agg) return [[e.x0, e.y0, e.z0], [e.x1, e.y1, e.z1]];
    return [
      [DATA.nodes[e.a].x, DATA.nodes[e.a].y, DATA.nodes[e.a].z],
      [DATA.nodes[e.b].x, DATA.nodes[e.b].y, DATA.nodes[e.b].z],
    ];
  }

  function getEdgeCurveBundle(e, coarse) {
    var ends = endpointsForEdge(e);
    var pa = ends[0];
    var pb = ends[1];
    var seed = e.trunk ? ((e.c0 + e.c1 * 7) % 5) : (((e.ia || 0) + (e.ib || 0) * 7 + (e.fan ? 19 : 0)) % 5);
    var bulge = (typeof e.bulge === "number") ? e.bulge : (0.18 + 0.06 * seed / 5);
    var pc = arcControl(pa, pb, bulge);
    var curve = new THREERef.QuadraticBezierCurve3(v3(pa), pc, v3(pb));
    var tubular = coarse
      ? Math.max(6, Math.min(24, Math.floor(10 + curve.getLength() * 6)))
      : Math.max(8, Math.min(48, Math.floor(14 + curve.getLength() * 9)));
    var radSeg = coarse ? 5 : ((e.fan && e.tubeR && e.tubeR < 0.0035) ? 6 : 7);
    return { curve: curve, tubular: tubular, radSeg: radSeg };
  }

  function makeEdgeGeometry(e, coarse) {
    var b = getEdgeCurveBundle(e, coarse);
    return new THREERef.TubeGeometry(b.curve, b.tubular, e.tubeR, b.radSeg, false);
  }

  function makeHaloTubeGeometry(e, coarse, radiusMult, minExtra) {
    var b = getEdgeCurveBundle(e, coarse);
    var glowR = Math.max(e.tubeR * radiusMult, e.tubeR + minExtra);
    var tubularG = Math.max(6, Math.floor(b.tubular * 0.8));
    var radG = Math.max(5, b.radSeg - 1);
    return new THREERef.TubeGeometry(b.curve, tubularG, glowR, radG, false);
  }

  function filterUdBase(e) {
    return {
      agg: !!e.agg,
      trunk: !!e.trunk,
      fan: !!e.fan,
      filterKey: e.filterKey || "q1",
      tier: e.tier || "q1",
      c0: e.c0 != null ? e.c0 : -1,
      c1: e.c1 != null ? e.c1 : -1,
      ia: e.a != null ? e.a : (e.ia != null ? e.ia : -1),
      ib: e.b != null ? e.b : (e.ib != null ? e.ib : -1),
    };
  }

  function udCoreEdge(e) {
    var u = filterUdBase(e);
    u.edgeLayer = "core";
    u.baseOpacityMain = e.opacity;
    return u;
  }

  function udHaloEdge(e, slot, haloOp) {
    var u = filterUdBase(e);
    u.edgeLayer = "halo";
    u.haloSlot = slot;
    u.baseOpacityHalo = haloOp;
    return u;
  }

  var edgeMeshes = [];
  var edgeHalos = [];

  DATA.edges.forEach(function (e) {
    var col = hexInt(e.color);
    var opSolid = Math.min(1, e.opacity);

    var geoIn = makeHaloTubeGeometry(e, false, 1.38, 0.0015);
    var geoOut = makeHaloTubeGeometry(e, false, 1.62, 0.0022);
    var opIn = Math.min(0.38, 0.14 + opSolid * 0.12);
    var opOut = Math.min(0.14, 0.045 + opSolid * 0.045);

    var matOut = new THREERef.MeshBasicMaterial({
      color: col,
      transparent: true,
      opacity: opOut,
      depthWrite: false,
      depthTest: true,
      blending: THREERef.AdditiveBlending,
      polygonOffset: true,
      polygonOffsetFactor: 0,
      polygonOffsetUnits: 1,
    });
    var matIn = new THREERef.MeshBasicMaterial({
      color: col,
      transparent: true,
      opacity: opIn,
      depthWrite: false,
      depthTest: true,
      blending: THREERef.AdditiveBlending,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });

    var haloOut = new THREERef.Mesh(geoOut, matOut);
    haloOut.renderOrder = -1;
    haloOut.userData = udHaloEdge(e, "outer", opOut);

    var haloIn = new THREERef.Mesh(geoIn, matIn);
    haloIn.renderOrder = 0;
    haloIn.userData = udHaloEdge(e, "inner", opIn);

    var coreGeo = makeEdgeGeometry(e, false);
    var matCore = new THREERef.MeshBasicMaterial({
      color: col,
      transparent: opSolid < 0.999,
      opacity: opSolid,
      depthWrite: opSolid >= 0.98,
      depthTest: true,
      polygonOffset: true,
      polygonOffsetFactor: 2,
      polygonOffsetUnits: 1,
    });
    var coreMesh = new THREERef.Mesh(coreGeo, matCore);
    coreMesh.renderOrder = 1;
    coreMesh.userData = udCoreEdge(e);

    edgeGroup.add(haloOut);
    edgeGroup.add(haloIn);
    edgeGroup.add(coreMesh);
    edgeHalos.push({ outer: haloOut, inner: haloIn });
    edgeMeshes.push(coreMesh);
  });

  function rebuildAllEdges(coarse) {
    for (var ei = 0; ei < edgeMeshes.length; ei++) {
      var coreMesh = edgeMeshes[ei];
      var H = edgeHalos[ei];
      var ed = DATA.edges[ei];
      coreMesh.geometry.dispose();
      H.inner.geometry.dispose();
      H.outer.geometry.dispose();
      coreMesh.geometry = makeEdgeGeometry(ed, coarse);
      H.inner.geometry = makeHaloTubeGeometry(ed, coarse, 1.38, 0.0015);
      H.outer.geometry = makeHaloTubeGeometry(ed, coarse, 1.62, 0.0022);

      var opSolid = Math.min(1, ed.opacity);
      var opIn = Math.min(0.38, 0.14 + opSolid * 0.12);
      var opOut = Math.min(0.14, 0.045 + opSolid * 0.045);

      coreMesh.userData.baseOpacityMain = ed.opacity;
      H.inner.userData.baseOpacityHalo = opIn;
      H.outer.userData.baseOpacityHalo = opOut;

      coreMesh.material.opacity = opSolid;
      coreMesh.material.transparent = opSolid < 0.999;
      coreMesh.material.depthWrite = opSolid >= 0.98;

      H.inner.material.opacity = opIn;
      H.outer.material.opacity = opOut;
    }
  }

  var layoutMorph = { active: false, from: null, startMs: 0, dur: 940, targetKey: "all" };

  function layoutTargetArr(fid) {
    var L = DATA.layouts && DATA.layouts[fid];
    if (!L) L = DATA.layouts && DATA.layouts["all"];
    return L;
  }

  function captureFlatPositions() {
    var out = new Float32Array(nodeMeshes.length * 3);
    for (var i = 0; i < nodeMeshes.length; i++) {
      out[i * 3] = nodeMeshes[i].position.x;
      out[i * 3 + 1] = nodeMeshes[i].position.y;
      out[i * 3 + 2] = nodeMeshes[i].position.z;
    }
    return out;
  }

  function beginLayoutMorph(fid) {
    if (!DATA.layouts) return;
    var tgt = layoutTargetArr(fid);
    if (!tgt || tgt.length !== nodeMeshes.length) return;
    layoutMorph.from = captureFlatPositions();
    layoutMorph.targetKey = fid;
    layoutMorph.startMs = performance.now();
    layoutMorph.active = true;
  }

  function snapNodesToLayout(fid) {
    var to = layoutTargetArr(fid);
    if (!to || to.length !== nodeMeshes.length) return;
    for (var i = 0; i < nodeMeshes.length; i++) {
      var ti = to[i];
      nodeMeshes[i].position.set(ti[0], ti[1], ti[2]);
    }
  }

  /* 轨道：可钻进内部（近距离裁剪 + 雾变淡） */
  var target = new THREERef.Vector3(0, 0, 0);
  var spherical = {
    radius: camera.position.length(),
    theta: Math.atan2(camera.position.z, camera.position.x),
    phi: Math.acos(THREERef.MathUtils.clamp(camera.position.y / camera.position.length(), -1, 1)),
  };
  var dragging = false, lx = 0, ly = 0;
  var autoRotate = true;
  var __embedDash = /(?:^|[?&])dashboard=1(?:&|$)/.test(location.search);
  if (__embedDash) {
    document.body.classList.add("embed-dashboard");
    autoRotate = true;
  }
  var activeFilterId = "all";

  function syncFog() {
    if (!scene.fog) return;
    var r = spherical.radius;
    var d = r < 0.85 ? 0.011 : (r < 1.6 ? 0.018 : (r < 3.2 ? 0.028 : (r < 6 ? 0.038 : 0.052)));
    scene.fog.density = d;
  }

  function camFromSpherical() {
    var r = spherical.radius;
    camera.position.set(
      r * Math.sin(spherical.phi) * Math.cos(spherical.theta),
      r * Math.cos(spherical.phi),
      r * Math.sin(spherical.phi) * Math.sin(spherical.theta)
    );
    camera.lookAt(target);
    syncFog();
  }

  function applyFilter(fid) {
    activeFilterId = fid;
    var activeComms = null;
    if (fid !== "all") {
      activeComms = new Set();
      edgeGroup.children.forEach(function (m) {
        var ud = m.userData;
        if (!ud.edgeLayer) return;
        if (!edgeMatchesFilter(ud, fid)) return;
        if (ud.c0 >= 0 && ud.c1 >= 0) {
          activeComms.add(ud.c0);
          activeComms.add(ud.c1);
        }
        if (ud.ia >= 0) activeComms.add(nodeComm[ud.ia]);
        if (ud.ib >= 0) activeComms.add(nodeComm[ud.ib]);
      });
    }

    var HI_CORE = 1.02;
    var HI_HALO = 1.03;

    edgeGroup.children.forEach(function (m) {
      var ud = m.userData;
      if (!ud.edgeLayer) return;
      var match = edgeMatchesFilter(ud, fid);
      var show = fid === "all" || match;
      m.visible = show;
      if (!show) return;
      var bo = ud.edgeLayer === "halo" ? ud.baseOpacityHalo : ud.baseOpacityMain;
      var hi = ud.edgeLayer === "halo" ? HI_HALO : HI_CORE;
      m.material.opacity = Math.min(1, bo * hi);
      m.scale.set(1, 1, 1);
    });

    nodeMeshes.forEach(function (nm) {
      var bo = nm.userData.baseOpacity;
      var ni = nm.userData.nodeIndex;
      var cid = nodeComm[ni];
      var gls = nm.userData.glowLayers;
      if (fid === "all") {
        nm.visible = true;
        nm.material.opacity = bo;
        if (gls) {
          for (var gx = 0; gx < gls.length; gx++) gls[gx].mesh.material.opacity = gls[gx].base;
        }
        return;
      }
      var hasFocus = activeComms && activeComms.size > 0;
      if (!hasFocus) {
        nm.visible = true;
        nm.material.opacity = 0.11;
        if (gls) {
          for (var g0 = 0; g0 < gls.length; g0++) gls[g0].mesh.material.opacity = gls[g0].base * 0.15;
        }
        return;
      }
      var keep = activeComms.has(cid);
      nm.visible = keep;
      if (!keep) return;
      nm.material.opacity = Math.min(1, bo + 0.04);
      if (gls) {
        for (var gy = 0; gy < gls.length; gy++) {
          var L = gls[gy];
          L.mesh.material.opacity = Math.min(0.42, L.base * 1.15);
        }
      }
    });

    var list = document.querySelectorAll("#filter-list .fil-btn");
    list.forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-fid") === fid);
    });
  }

  var flRoot = document.getElementById("filter-list");
  var FILTERS = DATA.filters || [{ id: "all", label: "All layers" }];
  FILTERS.forEach(function (f) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "fil-btn" + (f.id === "all" ? " active" : "");
    b.setAttribute("data-fid", f.id);
    b.textContent = f.label;
    b.addEventListener("click", function () {
      applyFilter(f.id);
      beginLayoutMorph(f.id);
    });
    flRoot.appendChild(b);
  });

  applyFilter("all");

  var btnAuto = document.getElementById("btn-auto");
  btnAuto.addEventListener("click", function () {
    autoRotate = !autoRotate;
    btnAuto.classList.toggle("on", autoRotate);
    btnAuto.textContent = autoRotate ? "Auto rotate · On" : "Auto rotate · Off";
  });

  canvas.addEventListener("mousedown", function (ev) { dragging = true; lx = ev.clientX; ly = ev.clientY; });
  window.addEventListener("mouseup", function () { dragging = false; });
  window.addEventListener("mousemove", function (ev) {
    if (!dragging) return;
    var dx = ev.clientX - lx, dy = ev.clientY - ly;
    lx = ev.clientX; ly = ev.clientY;
    spherical.theta -= dx * 0.006;
    spherical.phi -= dy * 0.005;
    spherical.phi = Math.max(0.08, Math.min(Math.PI - 0.08, spherical.phi));
    camFromSpherical();
  });
  canvas.addEventListener("wheel", function (ev) {
    ev.preventDefault();
    var step = 0.09 * (1 + Math.min(2.8, 4.2 / Math.max(spherical.radius, 0.35)));
    spherical.radius *= 1 + Math.sign(ev.deltaY) * step;
    spherical.radius = Math.max(0.52, Math.min(56, spherical.radius));
    camFromSpherical();
  }, { passive: false });

  camFromSpherical();

  var raycaster = new THREERef.Raycaster();
  var mouse = new THREERef.Vector2();
  var tip = document.getElementById("tip");

  var PATTERN_EN = {
    stable: "Stable",
    bursty: "Bursty",
    periodic: "Periodic",
    short_term: "Short-term",
    general: "General",
    unknown: "Unknown",
  };
  function patternEn(p) {
    var k = String(p || "").trim().toLowerCase();
    return PATTERN_EN[k] || (p || "?");
  }

  canvas.addEventListener("mousemove", function (ev) {
    var rect = canvas.getBoundingClientRect();
    mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    var hits = raycaster.intersectObjects(nodeMeshes, false);
    if (hits.length) {
      var u = hits[0].object.userData;
      tip.style.display = "block";
      var pct = u.weight != null ? Math.round(Math.max(0, Math.min(1, u.weight)) * 100) : null;
      tip.textContent = u.name + " · " + patternEn(u.pattern) + (pct != null ? " · connectivity " + pct + "%" : "");
    } else {
      tip.style.display = "none";
    }
  });

  function tick() {
    requestAnimationFrame(tick);
    var now = performance.now();
    if (layoutMorph.active && layoutMorph.from) {
      var tm = Math.min(1, (now - layoutMorph.startMs) / layoutMorph.dur);
      var ease = 1 - Math.pow(1 - tm, 3);
      var fr = layoutMorph.from;
      var to = layoutTargetArr(layoutMorph.targetKey);
      if (to && to.length === nodeMeshes.length) {
        for (var i = 0; i < nodeMeshes.length; i++) {
          var bx = i * 3;
          var ti = to[i];
          nodeMeshes[i].position.set(
            fr[bx] + (ti[0] - fr[bx]) * ease,
            fr[bx + 1] + (ti[1] - fr[bx + 1]) * ease,
            fr[bx + 2] + (ti[2] - fr[bx + 2]) * ease
          );
        }
      }
      rebuildAllEdges(tm < 1);
      if (tm >= 1) {
        snapNodesToLayout(layoutMorph.targetKey);
        rebuildAllEdges(false);
        layoutMorph.active = false;
        layoutMorph.from = null;
        applyFilter(activeFilterId);
      }
    }
    if (autoRotate && !dragging) {
      spherical.theta += __embedDash ? 0.004 : 0.0022;
      camFromSpherical();
    }
    renderer.render(scene, camera);
  }
  tick();

  window.addEventListener("resize", function () {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
})();
  </script>
</body>
</html>
"""


def main() -> None:
    FIG3_Q2.mkdir(parents=True, exist_ok=True)

    rel_json = OUTPUTS_DIR_Q2 / "reliable_links.json"
    if not rel_json.exists():
        raise FileNotFoundError(rel_json)
    if not Q1_REL_CSV.exists():
        OUT_HTML.write_text(
            "<!DOCTYPE html><meta charset=utf-8><title>Missing Q1</title>"
            "<body style='padding:2rem;font-family:system-ui'>Missing q1_relationship_patterns.csv</body>",
            encoding="utf-8",
        )
        print(f"Stub -> {OUT_HTML}")
        return

    reliable_links: list[dict] = json.loads(rel_json.read_text(encoding="utf-8"))
    pred_nodes: set[str] = set()
    raw_preds: list[tuple[str, str, str]] = []
    for lk in reliable_links:
        s, t = lk.get("source"), lk.get("target")
        if not s or not t or s == t:
            continue
        s, t = str(s), str(t)
        pred_nodes.add(s)
        pred_nodes.add(t)
        raw_preds.append((s, t, str(lk.get("generated_by") or "unknown")))

    df = pd.read_csv(Q1_REL_CSV)
    ca = df["company_a"].astype(str)
    cb = df["company_b"].astype(str)
    mask = ca.isin(pred_nodes) & cb.isin(pred_nodes) & (ca != cb)
    best: dict[tuple[str, str], tuple[float, str]] = {}
    for row in df.loc[mask].itertuples(index=False):
        a = str(row.company_a)
        b = str(row.company_b)
        key = (a, b)
        conf = float(row.confidence) if pd.notna(row.confidence) else 0.0
        pat = str(row.relationship_pattern)
        if key not in best or conf > best[key][0]:
            best[key] = (conf, pat)

    q1_pairs = [(a, b, conf, pat) for (a, b), (conf, pat) in best.items()]

    companies = _select_nodes(pred_nodes, raw_preds, q1_pairs, NODE_CAP_3D)
    cset = set(companies)
    idx = {c: i for i, c in enumerate(companies)}

    q1_lines: list[tuple[str, str, float]] = []
    for a, b, conf, _ in q1_pairs:
        if a in cset and b in cset:
            q1_lines.append((a, b, max(0.15, conf)))

    pred_lines_by_bundle: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    pred_uv_count: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for u, v, b in raw_preds:
        if u not in cset or v not in cset:
            continue
        pred_uv_count[(u, v, b)] += 1
    for (u, v, b), cnt in pred_uv_count.items():
        pred_lines_by_bundle[b].append((u, v, cnt))

    deg = np.zeros(len(companies), dtype=float)
    for a, b, w in q1_lines:
        deg[idx[a]] += math.sqrt(w)
        deg[idx[b]] += math.sqrt(w)
    for _, lst in pred_lines_by_bundle.items():
        for u, v, c in lst:
            deg[idx[u]] += 2 + math.sqrt(c)
            deg[idx[v]] += 2 + math.sqrt(c)

    rng = np.random.RandomState(42)
    pos_arr = _dandelion_xyz(deg, rng)

    temporal = _temporal_map(companies, Q1_TEMPORAL_JSON)

    dmax = float(deg.max()) if deg.max() > 0 else 1.0
    nodes_js: list[dict] = []
    for i, c in enumerate(companies):
        strength = float(deg[i]) / dmax if dmax > 0 else 0.0
        strength = max(0.0, min(1.0, strength))
        pat = temporal.get(c, "unknown")
        pat_hex = PATTERN_COLORS.get(pat, PATTERN_COLORS["unknown"])
        r = 0.009 + 0.062 * math.pow(strength, 0.62)
        op = 0.16 + 0.82 * math.pow(strength, 0.92)
        nodes_js.append(
            {
                "name": _short(c, 48),
                "pattern": temporal.get(c, "?"),
                "x": float(pos_arr[i, 0]),
                "y": float(pos_arr[i, 1]),
                "z": float(pos_arr[i, 2]),
                "r": r,
                "opacity": op,
                "weight": strength,
                "color": _blend_pattern_by_strength(pat_hex, strength),
            }
        )

    n_n = len(companies)
    W = np.zeros((n_n, n_n), dtype=float)
    for a, b, w in q1_lines:
        ia, ib = idx[a], idx[b]
        W[ia, ib] += float(w)
        W[ib, ia] += float(w)
    for _, lst in pred_lines_by_bundle.items():
        for u, v, cnt in lst:
            ia, ib = idx[u], idx[v]
            fw = float(cnt)
            W[ia, ib] += fw
            W[ib, ia] += fw

    k_target = max(10, min(24, n_n // 5))
    comm = _communities_from_adjacency(W, k_target, rng)

    q1_ix = [(idx[a], idx[b], float(w)) for a, b, w in q1_lines]

    edges_js: list[dict] = []
    agg_q1, _ = _aggregate_cross_community_edges(
        q1_ix,
        comm,
        pos_arr,
        color=Q1_EDGE_HEX,
        opacity=OP_Q1,
        filter_key="q1",
        tier="q1",
        trunk_base_r=0.0026,
        trunk_scale_r=0.014,
        trunk_cap=0.026,
        fan_base_r=0.0010,
        fan_scale_r=0.0022,
        fan_cap=0.0052,
        max_fan_per_side=26,
        fan_spread=0.18,
        opacity_mult=1.0,
        tube_mult=1.0,
    )
    edges_js.extend(agg_q1)

    bundles_sorted = sorted(pred_lines_by_bundle.keys())
    n_b = len(bundles_sorted)
    for bi, bundle in enumerate(bundles_sorted):
        lst = pred_lines_by_bundle[bundle]
        ech = _spectrum_ml_bundle_hex(bi, n_b if n_b > 1 else 1)
        pairs_ix = [(idx[u], idx[v], float(cnt)) for u, v, cnt in lst]
        agg_b, _ = _aggregate_cross_community_edges(
            pairs_ix,
            comm,
            pos_arr,
            color=ech,
            opacity=OP_ML_BUNDLE,
            filter_key=bundle,
            tier="ml",
            trunk_base_r=0.0029,
            trunk_scale_r=0.0155,
            trunk_cap=0.029,
            fan_base_r=0.0011,
            fan_scale_r=0.0024,
            fan_cap=0.0058,
            max_fan_per_side=26,
            fan_spread=0.18,
            opacity_mult=1.0,
            tube_mult=1.0,
        )
        edges_js.extend(agg_b)

    filters_js: list[dict[str, str]] = [
        {"id": "all", "label": "All layers"},
        {"id": "q1", "label": "Q1 relationships"},
        {"id": "tier:ml", "label": "All ML bundles"},
    ]
    for bundle in bundles_sorted:
        filters_js.append({"id": bundle, "label": bundle})

    layout_keys: list[str] = [f["id"] for f in filters_js]
    deg_layout: dict[str, np.ndarray] = {k: np.zeros(n_n, dtype=float) for k in layout_keys}
    deg_layout["all"] = deg.copy()

    for a, b, w in q1_lines:
        ia, ib = idx[a], idx[b]
        sq = math.sqrt(w)
        deg_layout["q1"][ia] += sq
        deg_layout["q1"][ib] += sq

    for bundle, lst in pred_lines_by_bundle.items():
        for u, v, c in lst:
            ia, ib = idx[u], idx[v]
            inc = 2.0 + math.sqrt(float(c))
            deg_layout[bundle][ia] += inc
            deg_layout[bundle][ib] += inc
            deg_layout["tier:ml"][ia] += inc
            deg_layout["tier:ml"][ib] += inc

    layouts_js: dict[str, list[list[float]]] = {}
    for k in layout_keys:
        if k == "all":
            layouts_js[k] = pos_arr.tolist()
        else:
            layouts_js[k] = _dandelion_xyz(deg_layout[k], np.random.RandomState(42)).tolist()

    n_comm = int(comm.max()) + 1 if len(comm) else 0
    payload = {
        "nodes": nodes_js,
        "edges": edges_js,
        "filters": filters_js,
        "layouts": layouts_js,
        "nodeCommunity": [int(x) for x in comm.tolist()],
    }
    raw_json = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    b64 = base64.b64encode(raw_json).decode("ascii")

    html = HTML_TEMPLATE.replace("__B64_PAYLOAD__", b64)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(
        f"Wrote {OUT_HTML} (Three.js, {len(nodes_js)} spheres, {len(edges_js)} edge segments "
        f"(trunk+fan), {n_comm} communities)"
    )


if __name__ == "__main__":
    main()
