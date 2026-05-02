#!/usr/bin/env python3
"""Q1：Top50 公司 × 月度 × HS band 的 3D 分层堆叠「河流」视图。

纵向排序：**数组 index 0 为交易量第 1 名**，3D 中 **第 50 名在近侧（y 小）、第 1 名在远侧（y 大）**。
纵深仅用透明度区分远近（RGB 与图例一致）。

**交互 HTML**：All 时仅图例蓝色系，不向 RGB 掺灰；聚焦时选中公司保持同色，其余统一浅灰且 opacity=0.05；
右侧图例为 legend-only 固定色，不随下拉改变。
"""

from __future__ import annotations

import colorsys
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_vis_root = Path(__file__).resolve().parents[1]
if str(_vis_root) not in sys.path:
    sys.path.insert(0, str(_vis_root))

from figures2d_common import FIG_DIR, ensure_dirs, load_q1_patterns

_analysis_shared = Path(__file__).resolve().parents[2] / "analysis" / "shared"
if str(_analysis_shared) not in sys.path:
    sys.path.insert(0, str(_analysis_shared))

from build_index import month_key, normalize_hscode
from config import BASE_GRAPH_PATH, FISH_HSCODE_PREFIXES

TOP_N = 50

# 极端月度总交易数：全局分位数封顶（超过 cap 的月份各 band 同比缩放）。
TOTAL_CAP_PERCENTILE = 99.0

# -------------------------- 关键修改1：增大平滑参数，曲线更顺滑 --------------------------
# 原参数 2.5 → 改为 4.0，Gaussian滤波窗口更大，曲线更平滑（和参考图的柔滑感匹配）
SMOOTH_SIGMA_MONTHS = 4.0

# 堆叠顺序：底层 other，其上按海产品前缀序。
FISH_PREFIXES_ORDER: tuple[str, ...] = tuple(FISH_HSCODE_PREFIXES)
BAND_ORDER: tuple[str, ...] = ("other",) + FISH_PREFIXES_ORDER

# 聚焦：未选中层固定中性灰 + 极低不透明（不跟随纵深变色相）
FOCUS_UNSELECTED_OPACITY = 0.05
FOCUS_UNSELECTED_GRAY = 0.72


def _band_label(code: str) -> str:
    if code == "other":
        return "Other (non-fish HS)"
    return f"Fish HS {code}*"


def classify_band(hscode_raw) -> str:
    h = normalize_hscode(hscode_raw)
    if not h:
        return "other"
    for prefix in FISH_PREFIXES_ORDER:
        if h.startswith(prefix):
            return prefix
    return "other"


def top_companies_from_q1(patterns: list[dict], n: int) -> list[str]:
    ranked = sorted(patterns, key=lambda r: int(r.get("total_links") or 0), reverse=True)
    out = []
    seen = set()
    for row in ranked:
        c = row.get("company")
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(str(c))
        if len(out) >= n:
            break
    return out


def aggregate_trade_counts_for_companies(
    links: list[dict],
    company_set: set[str],
) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
    """company -> month -> band -> trade count（每条边在端点各 +1）。"""
    raw: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    months_set: set[str] = set()

    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        mk = month_key(link.get("arrivaldate"))
        if not mk or len(mk) != 7:
            continue
        band = classify_band(link.get("hscode"))

        for node in (src, tgt):
            if not node or node not in company_set:
                continue
            raw[node][mk][band] += 1.0
            months_set.add(mk)

    months_sorted = sorted(months_set)
    nested = {c: {m: dict(raw[c][m]) for m in raw[c]} for c in raw}
    return nested, months_sorted


def build_tensor(
    companies: list[str],
    months: list[str],
    agg: dict[str, dict[str, dict[str, float]]],
    bands: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """values[c,t,b], totals[c,t]."""
    c_n, t_n, b_n = len(companies), len(months), len(bands)
    band_index = {b: i for i, b in enumerate(bands)}
    values = np.zeros((c_n, t_n, b_n), dtype=np.float64)
    month_index = {m: i for i, m in enumerate(months)}
    for ci, comp in enumerate(companies):
        per_m = agg.get(comp, {})
        for month, band_map in per_m.items():
            ti = month_index.get(month)
            if ti is None:
                continue
            for band, v in band_map.items():
                bi = band_index.get(band)
                if bi is None:
                    continue
                values[ci, ti, bi] += v
    totals = values.sum(axis=2)
    return values, totals


def truncate_monthly_totals(values: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    """按月总交易数封顶：超过全局分位数的月份，各 band 同比缩放。"""
    totals = values.sum(axis=2)
    positive = totals[totals > 0]
    if positive.size == 0:
        return values.copy(), float("nan")
    cap = float(np.percentile(positive, percentile))
    if not np.isfinite(cap) or cap <= 0:
        return values.copy(), cap
    eps = 1e-18
    scale_2d = np.where(totals > cap, cap / np.maximum(totals, eps), 1.0)
    out = values * scale_2d[:, :, np.newaxis]
    return out, cap


def smooth_company_total_curves_preserving_band_shares(values: np.ndarray, sigma: float) -> np.ndarray:
    """对每个公司沿时间平滑月度总交易数；各月各 band 按比例缩放以保持 band 占比不变。"""
    if sigma <= 0 or values.shape[1] < 2:
        return values.copy()

    totals = values.sum(axis=2)
    eps = 1e-12

    try:
        from scipy import ndimage
        totals_s = ndimage.gaussian_filter1d(totals, sigma=sigma, axis=1, mode="nearest")
    except ImportError:
        window = max(3, int(round(sigma * 2)) | 1)
        pad = window // 2
        kernel = np.ones(window, dtype=np.float64) / window
        totals_s = np.zeros_like(totals)
        for c in range(values.shape[0]):
            pad_row = np.pad(totals[c, :], (pad, pad), mode="edge")
            totals_s[c, :] = np.convolve(pad_row, kernel, mode="valid")

    totals_s = np.maximum(totals_s, 0.0)

    denom = np.maximum(totals[:, :, np.newaxis], eps)
    props = np.divide(values, denom, out=np.zeros_like(values), where=totals[:, :, np.newaxis] > eps)
    out = props * totals_s[:, :, np.newaxis]
    return np.maximum(out, 0.0)


def normalize_values_global_max(values: np.ndarray) -> tuple[np.ndarray, float]:
    """全体 (公司×月) 总交易数除以全局最大值，使峰值高度为 1。"""
    totals = values.sum(axis=2)
    mx = float(np.nanmax(totals)) if totals.size else 0.0
    if not np.isfinite(mx) or mx <= 0:
        return np.maximum(values, 0.0), mx
    return np.maximum(values / mx, 0.0), mx


def compute_layer_style(
    ci: int,
    c_n: int,
    band_rgb: tuple[float, float, float, float],
    focus_idx: int | None,
) -> tuple[int, int, int, float]:
    """All：RGB 与图例 band 一致（不乘纵深阴影，避免发灰）；不透明度和纵深仅用 alpha。
    聚焦：选中公司与 All 同色；其余固定浅灰 + FOCUS_UNSELECTED_OPACITY。"""
    n1 = max(c_n - 1, 1)
    depth_t = (c_n - 1 - ci) / n1
    depth_op = 0.34 + 0.46 * (1.0 - depth_t)

    if focus_idx is not None and 0 <= focus_idx < c_n:
        if ci == focus_idx:
            r, g, b = float(band_rgb[0]), float(band_rgb[1]), float(band_rgb[2])
            op = float(band_rgb[3]) * depth_op
        else:
            gry = float(FOCUS_UNSELECTED_GRAY)
            r = g = b = gry
            op = float(FOCUS_UNSELECTED_OPACITY)
    else:
        r, g, b = float(band_rgb[0]), float(band_rgb[1]), float(band_rgb[2])
        op = float(band_rgb[3]) * depth_op

    ri, gi, bi = int(min(255, max(0, round(r * 255)))), int(min(255, max(0, round(g * 255)))), int(
        min(255, max(0, round(b * 255)))
    )
    return ri, gi, bi, float(np.clip(op, 0.03, 0.98))


_MESH_LIGHTING = dict(
    ambient=0.38,
    diffuse=0.95,
    specular=0.42,
    roughness=0.42,
    fresnel=0.14,
)


def ocean_band_colors(bands: tuple[str, ...]) -> dict[str, tuple[float, float, float, float]]:
    """RGBA per band：高饱和蓝青～靛紫渐变，相邻色相跨度大，易于区分。"""
    alpha = 0.82
    colors: dict[str, tuple[float, float, float, float]] = {}
    fish_keys = [b for b in bands if b != "other"]
    nfish = len(fish_keys)

    # other：亮青蓝锚点，与海产色系拉开
    r0, g0, b0 = colorsys.hsv_to_rgb(0.528, 0.94, 0.99)
    colors["other"] = (r0, g0, b0, alpha)

    if nfish == 0:
        return colors

    for fi, b in enumerate(fish_keys):
        t = fi / max(nfish - 1, 1)
        # 色相约 0.46→0.74：青绿蓝 → 标准蓝 → 靛/紫蓝，间隔明显
        h = 0.458 + 0.282 * t
        s = min(1.0, 0.935 + 0.045 * math.sin(math.pi * t))
        v = min(
            1.0,
            0.855 + 0.13 * math.sin(math.pi * (0.35 + 0.65 * t)) + (0.035 if fi % 2 == 0 else 0.0),
        )
        r_ch, g_ch, b_ch = colorsys.hsv_to_rgb(h, s, v)
        colors[str(b)] = (r_ch, g_ch, b_ch, alpha)
    return colors


def plot_matplotlib(
    companies: list[str],
    months: list[str],
    values: np.ndarray,
    totals: np.ndarray,
    bands: tuple[str, ...],
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ensure_dirs()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    c_n, t_n, _ = values.shape
    if t_n < 2:
        raise RuntimeError("Too few months for 3D ribbons.")

    x = np.arange(t_n, dtype=np.float64)
    # -------------------------- 关键修改2：调整公司层间距，分层更清晰 --------------------------
    # 原 9.0 → 改为 7.5，层与层之间更紧凑，和参考图的堆叠感匹配
    depth_step = 7.5
    colors_rgba = ocean_band_colors(bands)

    fig = plt.figure(figsize=(16, 10), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    # -------------------------- 关键修改3：添加参考图风格的淡色网格线 --------------------------
    # 不改动数据配色，仅调整背景网格为淡红色，贴近参考图
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#f0c0c0')
    ax.yaxis.pane.set_edgecolor('#f0c0c0')
    ax.zaxis.pane.set_edgecolor('#f0c0c0')
    ax.grid(color='#f0c0c0', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_facecolor('white')

    cum = np.zeros((c_n, t_n), dtype=np.float64)
    for bi, band in enumerate(bands):
        layer = values[:, :, bi]
        z0 = cum.copy()
        z1 = cum + layer
        color = colors_rgba.get(band, (0.3, 0.5, 0.7, 0.75))
        polys = []
        for ci in range(c_n):
            y = (c_n - 1 - ci) * depth_step
            for t in range(t_n - 1):
                polys.append(
                    [
                        (x[t], y, z0[ci, t]),
                        (x[t + 1], y, z0[ci, t + 1]),
                        (x[t + 1], y, z1[ci, t + 1]),
                        (x[t], y, z1[ci, t]),
                    ]
                )
        poly3d = Poly3DCollection(polys, facecolors=color[:3], alpha=color[3], linewidths=0)
        ax.add_collection3d(poly3d)
        cum = z1

    zmax = float(np.nanmax(totals)) if totals.size else 1.0
    if zmax <= 0:
        zmax = 1.0
    ax.set_zlim(0.0, min(max(zmax * 1.05, 0.05), 1.05))

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-depth_step * 0.5, depth_step * (c_n - 0.5))
    # -------------------------- 关键修改4：移除月份轴（X轴）的"Month"标题 --------------------------
    ax.set_xlabel("")  # 原 ax.set_xlabel("Month")，改为空字符串
    ax.set_ylabel(f"Top {c_n} (#{c_n} near → #1 far)")
    ax.set_zlabel("Normalized transaction count")
    tick_step = max(1, t_n // 12)
    ax.set_xticks(x[::tick_step])
    # 可选：如果想完全隐藏月份文字，改为 ax.set_xticklabels([])
    ax.set_xticklabels([months[i] for i in range(0, t_n, tick_step)], rotation=35, ha="right", fontsize=7)
    ax.set_yticks([])

    # -------------------------- 关键修改5：调整视角，贴近参考图的斜俯视效果 --------------------------
    # 原 elev=22, azim=-58 → 改为 elev=28, azim=-65，更接近参考图的视角
    ax.view_init(elev=28, azim=-65)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors_rgba[b][:3], alpha=colors_rgba[b][3], label=_band_label(b))
        for b in bands
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.set_title(
        f"Top {c_n} companies — normalized monthly transaction count (stacked HS bands)\n"
        f"P{TOTAL_CAP_PERCENTILE:.0f} monthly count cap; per-company total smoothed σ={SMOOTH_SIGMA_MONTHS} mo; "
        "global max normalization",
        fontsize=11,
        pad=12,
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close(fig)


def export_plotly_html(
    companies: list[str],
    months: list[str],
    values: np.ndarray,
    totals: np.ndarray,
    bands: tuple[str, ...],
    out_html: Path,
) -> None:
    import html as html_module

    import plotly.graph_objects as go
    import plotly.io as pio

    c_n, t_n, _ = values.shape
    if t_n < 2:
        return

    x_idx = np.arange(t_n, dtype=np.float64)
    depth_step = 7.5
    colors_rgba = ocean_band_colors(bands)

    mesh_traces: list = []
    cum = np.zeros((c_n, t_n), dtype=np.float64)
    for bi, band in enumerate(bands):
        layer = values[:, :, bi]
        z0 = cum.copy()
        z1 = cum + layer
        band_tuple = colors_rgba.get(band, (0.3, 0.5, 0.7, 0.78))

        for ci in range(c_n):
            ri, gi, bi_i, op = compute_layer_style(ci, c_n, band_tuple, None)
            rgb = f"rgb({ri},{gi},{bi_i})"
            y0 = (c_n - 1 - ci) * depth_step
            xv = np.concatenate([x_idx, x_idx])
            yv = np.full(2 * t_n, y0, dtype=np.float64)
            zv = np.concatenate([z0[ci], z1[ci]])
            i_list: list[int] = []
            j_list: list[int] = []
            k_list: list[int] = []
            for t in range(t_n - 1):
                a, b_idx, c_idx, d_idx = t, t + 1, t + t_n + 1, t + t_n
                i_list.extend([a, a])
                j_list.extend([b_idx, c_idx])
                k_list.extend([c_idx, d_idx])
            mesh_traces.append(
                go.Mesh3d(
                    x=xv,
                    y=yv,
                    z=zv,
                    i=i_list,
                    j=j_list,
                    k=k_list,
                    color=rgb,
                    opacity=op,
                    flatshading=False,
                    lighting=_MESH_LIGHTING,
                    name=_band_label(band),
                    legendgroup=str(band),
                    showlegend=False,
                    uid=f"ridge_ci{ci}_b{bi}",
                    meta=dict(company_index=ci, band_key=str(band)),
                )
            )
        cum = z1

    n_legend = len(bands)
    legend_traces: list = []
    for band in bands:
        bt = colors_rgba.get(band, (0.3, 0.5, 0.7, 0.78))
        lr, lg, lb = (
            int(min(255, max(0, round(bt[0] * 255)))),
            int(min(255, max(0, round(bt[1] * 255)))),
            int(min(255, max(0, round(bt[2] * 255)))),
        )
        legend_traces.append(
            go.Scatter3d(
                x=[0.0],
                y=[0.0],
                z=[0.0],
                mode="markers",
                marker=dict(size=9, color=f"rgb({lr},{lg},{lb})", opacity=1),
                name=_band_label(band),
                legendgroup=str(band),
                showlegend=True,
                visible="legendonly",
            )
        )

    fig = go.Figure(data=legend_traces + mesh_traces)
    fig.update_layout(
        title=(
            f"Top {c_n} companies — normalized monthly transaction count (stacked HS bands)<br>"
            f"<sup>P{TOTAL_CAP_PERCENTILE:.0f} monthly count cap; per-company total σ={SMOOTH_SIGMA_MONTHS} mo; "
            "global max norm · use dropdown to focus one company</sup>"
        ),
        scene=dict(
            xaxis=dict(
                title="",
                tickmode="array",
                tickvals=list(range(0, t_n, max(1, t_n // 12))),
                ticktext=[months[i] for i in range(0, t_n, max(1, t_n // 12))],
            ),
            yaxis=dict(
                title=f"Top {c_n} (#{c_n} near → #1 far)",
                showticklabels=False,
                tickmode="linear",
                showspikes=False,
            ),
            zaxis=dict(title="Normalized transaction count", range=[0.0, 1.05]),
            aspectmode="manual",
            aspectratio=dict(x=1.8, y=1.0, z=0.4),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.52)),
            bgcolor="rgb(248,252,255)",
        ),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=0, r=140, t=52, b=0),
        paper_bgcolor="#eef6fb",
    )

    fig_json = pio.to_json(fig)
    safe_fig_json = fig_json.replace("</script>", "<\\/script>")
    band_bases_json = json.dumps({str(b): list(colors_rgba[b][:4]) for b in bands}, separators=(",", ":"))

    opt_lines = ['<option value="">— All (depth shading only) —</option>']
    for i, name in enumerate(companies):
        lab = name if len(name) <= 44 else name[:41] + "…"
        opt_lines.append(
            f'<option value="{i}" title="{html_module.escape(name)}">'
            f"{i + 1}. {html_module.escape(lab)}</option>"
        )
    options_html = "\n".join(opt_lines)

    js_logic = """
function computeLayerStyle(ci, cN, bandRgba, focusIdx) {
  const n1 = Math.max(cN - 1, 1);
  const depthT = (cN - 1 - ci) / n1;
  const depthOp = 0.34 + 0.46 * (1 - depthT);
  let r;
  let g;
  let b;
  let op;
  if (focusIdx !== null && focusIdx >= 0 && focusIdx < cN) {
    if (Number(ci) === Number(focusIdx)) {
      r = bandRgba[0];
      g = bandRgba[1];
      b = bandRgba[2];
      op = bandRgba[3] * depthOp;
    } else {
      const gy = window.RIDGE_FOCUS_UNSELECTED_GRAY;
      r = g = b = gy;
      op = window.RIDGE_FOCUS_UNSELECTED_OPACITY;
    }
  } else {
    r = bandRgba[0];
    g = bandRgba[1];
    b = bandRgba[2];
    op = bandRgba[3] * depthOp;
  }
  const ri = Math.min(255, Math.max(0, Math.round(r * 255)));
  const gi = Math.min(255, Math.max(0, Math.round(g * 255)));
  const bi = Math.min(255, Math.max(0, Math.round(b * 255)));
  op = Math.min(0.98, Math.max(0.03, op));
  return [ri, gi, bi, op];
}

function safeRestyleColors(gd, colors, opacities, traceIndices) {
  try {
    if (traceIndices && traceIndices.length > 0) {
      Plotly.restyle(gd, { color: colors, opacity: opacities }, traceIndices);
    } else {
      Plotly.restyle(gd, { color: colors, opacity: opacities });
    }
  } catch (e) {
    console.warn("Plotly.restyle skipped:", e);
  }
}

function traceCiBandKey(traceIdx, cN, bandOrder) {
  const nB = bandOrder.length;
  const bi = Math.floor(traceIdx / cN);
  const ci = traceIdx % cN;
  if (bi < 0 || bi >= nB || ci < 0 || ci >= cN) return { ci: null, bk: null };
  const bk = bandOrder[bi];
  return { ci, bk };
}

function applyCompanyFocus(gd) {
  const sel = document.getElementById("ridge-focus-select").value;
  let focusIdx = null;
  if (sel !== "") {
    const v = parseInt(sel, 10);
    if (!Number.isNaN(v)) focusIdx = v;
  }
  const cN = window.RIDGE_C_N;
  const bandOrder = window.RIDGE_BAND_ORDER;
  const nLegend = window.RIDGE_N_LEGEND || 0;
  const colors = [];
  const opacities = [];
  const meshIndices = [];
  const nTr = gd.data.length;
  const expectedMeshes = bandOrder.length * cN;
  if (nTr !== expectedMeshes + nLegend) {
    console.warn("Ridge river: trace count mismatch", nTr, "expected", expectedMeshes + nLegend);
  }
  for (let i = nLegend; i < nTr; i++) {
    meshIndices.push(i);
    const tr = gd.data[i];
    const idx = i - nLegend;
    const { ci, bk } = traceCiBandKey(idx, cN, bandOrder);
    const base = bk ? window.RIDGE_BAND_BASE[bk] : null;
    if (ci === null || !base) {
      colors.push(typeof tr.color === "string" ? tr.color : "rgb(120,150,190)");
      opacities.push(typeof tr.opacity === "number" ? tr.opacity : 0.5);
      continue;
    }
    const st = computeLayerStyle(ci, cN, base, focusIdx);
    colors.push("rgb(" + st[0] + "," + st[1] + "," + st[2] + ")");
    opacities.push(st[3]);
  }
  safeRestyleColors(gd, colors, opacities, meshIndices);
}

Plotly.newPlot("ridge-plot-div", FIG.data, FIG.layout, { responsive: true }).then(function (gd) {
  document.getElementById("ridge-focus-select").addEventListener("change", function () {
    applyCompanyFocus(gd);
  });
}).catch(function (err) {
  console.error(err);
  document.getElementById("ridge-plot-div").innerHTML =
    '<p style="padding:24px;color:#a00;font-family:system-ui">Plot failed: ' + String(err) + '</p>';
});
"""

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Q1 Ridge river — focus company</title>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.30.1/plotly.min.js"></script>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #eef6fb; }}
    .ridge-toolbar {{
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px;
      padding: 12px 16px; background: #e2eef6; border-bottom: 1px solid #bcd;
    }}
    .ridge-toolbar label {{ font-weight: 600; color: #134; }}
    #ridge-focus-select {{
      min-width: min(520px, 92vw); max-width: 720px; font-size: 14px; padding: 6px 8px;
    }}
    .ridge-hint {{ font-size: 12px; color: #456; max-width: 900px; }}
    #ridge-plot-div {{ width: 100%; height: min(900px, 96vh); }}
  </style>
</head>
<body>
  <div class="ridge-toolbar">
    <label for="ridge-focus-select">Focus company</label>
    <select id="ridge-focus-select">
{options_html}
    </select>
    <span class="ridge-hint">All: legend blues only (depth via opacity). Focus: selected keeps blues; others gray α=0.05. Legend fixed.</span>
  </div>
  <div id="ridge-plot-div"></div>
  <script type="application/json" id="ridge-fig-json">{safe_fig_json}</script>
  <script>
    window.RIDGE_C_N = {c_n};
    window.RIDGE_BAND_ORDER = {json.dumps(list(bands))};
    window.RIDGE_BAND_BASE = {band_bases_json};
    window.RIDGE_N_LEGEND = {n_legend};
    window.RIDGE_FOCUS_UNSELECTED_OPACITY = {FOCUS_UNSELECTED_OPACITY};
    window.RIDGE_FOCUS_UNSELECTED_GRAY = {FOCUS_UNSELECTED_GRAY};
    let FIG;
    try {{
      FIG = JSON.parse(document.getElementById("ridge-fig-json").textContent);
    }} catch (err) {{
      document.getElementById("ridge-plot-div").innerHTML =
        '<p style="padding:24px;color:#b00;font-family:system-ui,max-width:640px">Chart data parse failed. File may be truncated (very large). Re-run:<br><code>python visualization/q1/build_q1_ridge_river.py</code></p>';
      console.error(err);
    }}
    if (!FIG) {{ }}
    else if (typeof Plotly === "undefined") {{
      document.getElementById("ridge-plot-div").innerHTML =
        '<p style="padding:24px;color:#b00;font-family:system-ui">Plotly.js did not load (CDN blocked?). Check network or try another browser.</p>';
    }} else {{
    {js_logic}
    }}
  </script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_out, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    q1_dir = FIG_DIR / "q1"
    q1_dir.mkdir(parents=True, exist_ok=True)

    patterns = load_q1_patterns()
    companies = top_companies_from_q1(patterns, TOP_N)
    if len(companies) < TOP_N:
        print(f"警告：q1_temporal_patterns 仅能提供 {len(companies)} 家公司。", file=sys.stderr)

    if not BASE_GRAPH_PATH.is_file():
        print(f"未找到主图文件：{BASE_GRAPH_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"读取主图（按交易数聚合）：{BASE_GRAPH_PATH}")
    base_graph = json.loads(BASE_GRAPH_PATH.read_text(encoding="utf-8"))
    links = base_graph.get("links") or []
    company_set = set(companies)
    agg, months = aggregate_trade_counts_for_companies(links, company_set)
    if not months:
        print("所选公司在主图中无有效月份交易记录。", file=sys.stderr)
        sys.exit(1)

    values_raw, _ = build_tensor(companies, months, agg, BAND_ORDER)
    values_capped, cap_count = truncate_monthly_totals(values_raw, TOTAL_CAP_PERCENTILE)
    values_smoothed = smooth_company_total_curves_preserving_band_shares(
        values_capped, SMOOTH_SIGMA_MONTHS
    )
    values, norm_divisor_count = normalize_values_global_max(values_smoothed)
    totals = values.sum(axis=2)
    print(
        f"P{TOTAL_CAP_PERCENTILE:.0f} monthly total count cap ≈ {cap_count:,.1f}; "
        f"per-company Gaussian σ={SMOOTH_SIGMA_MONTHS} mo; "
        f"normalize divisor (max monthly total after cap+smooth) ≈ {norm_divisor_count:,.1f}"
    )
    meta = {
        "description": "Top companies from q1_temporal_patterns.json by total_links; MC2 edges; "
        "metric is transaction count per company-month-band (each edge counted at source and target); "
        "display normalized to [0,1] after P99 cap and per-company total smoothing.",
        "metric": "transaction_count",
        "top_n": len(companies),
        "monthly_total_cap_percentile": TOTAL_CAP_PERCENTILE,
        "monthly_total_cap_count": cap_count,
        "smooth_sigma_months": SMOOTH_SIGMA_MONTHS,
        "normalize_divisor_count": norm_divisor_count,
        "companies": companies,
        "months": months,
        "bands": [{"code": b, "label": _band_label(b)} for b in BAND_ORDER],
        "band_order": list(BAND_ORDER),
        "values_normalized": values.tolist(),
        "monthly_totals_normalized": totals.tolist(),
    }
    json_path = q1_dir / "q1_ridge_river_top50_data.json"
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入：{json_path}")

    png_path = q1_dir / "q1_ridge_river_top50.png"
    plot_matplotlib(companies, months, values, totals, BAND_ORDER, png_path)
    print(f"已写入：{png_path}")

    html_path = q1_dir / "q1_ridge_river_top50.html"
    export_plotly_html(companies, months, values, totals, BAND_ORDER, html_path)
    print(f"已写入：{html_path}")


if __name__ == "__main__":
    main()