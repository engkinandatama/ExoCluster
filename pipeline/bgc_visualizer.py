"""
phase3_visualizer.py — Phase 3 Visualization & HTML Report

Produces publication-ready figures and an HTML report for BGC prediction results:

  1. plot_bgc_class_distribution()
       Stacked bar chart: BGC class counts per strain, with confidence tiers
       shaded.

  2. plot_bgc_heatmap()
       Phylogenomic BGC heatmap: strains (rows) × BGC classes (columns),
       colour intensity = number of hits; hierarchical clustering applied.

  3. plot_bgc_confidence_landscape()
       Scatter plot: kmer_deviation vs GC-deviation, colour = BGC class,
       size = confidence, marker = confidence tier.

  4. render_phase3_html_report()
       Jinja2 HTML report embedding all plots, BGC stats table, and a
       per-gene predictions table.

  5. plot_phase3_decision_funnel()
       Funnel-style summary from accessory genes to high-confidence BGC hits.

  6. plot_bgc_neighborhood_map()
       Contig-level map of scored alien genes with BGC hits highlighted.

All functions write outputs to an output_dir and return the file path(s).
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator

# Optional seaborn / scipy for heatmap
try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    _SEABORN_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from jinja2 import Environment, BaseLoader
    _JINJA_AVAILABLE = True
except ImportError:
    Environment = None
    BaseLoader = None
    _JINJA_AVAILABLE = False

from pipeline.bgc_predictor import BGCResult, BGCGeneRecord, BGC_CLASSES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette for BGC classes
# ---------------------------------------------------------------------------
BGC_PALETTE: Dict[str, str] = {
    "NonBGC":     "#AAAAAA",
    "Alkaloid":   "#9B5DE5",
    "Terpene":    "#2A9D8F",
    "NRP":        "#E63946",
    "Polyketide": "#F4A261",
    "RiPP":       "#457B9D",
    "Saccharide": "#E9C46A",
    "Other":      "#264653",
}

CONF_MARKERS = {"High": "^", "Medium": "o", "Low": "s"}


# ===========================================================================
# Helper utilities
# ===========================================================================

def _fig_to_base64(fig: Figure) -> str:
    """Encode matplotlib figure as base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _save_and_close(fig: Figure, path: str) -> str:
    """Save figure to disk and close it; return path."""
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("  Saved: %s", path)
    return path


def _empty_plot(output_dir: str, filename: str, title: str, message: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, transform=ax.transAxes, color="#6c757d")
    ax.set_title(title, fontweight="bold")
    ax.set_axis_off()
    return _save_and_close(fig, os.path.join(output_dir, filename))


# ===========================================================================
# Plot 1 — BGC Class Distribution (stacked bar per strain)
# ===========================================================================

def plot_bgc_class_distribution(
    bgc_result: BGCResult,
    output_dir: str,
    filename: str = "bgc_class_distribution.png",
) -> str:
    """
    Stacked bar chart showing BGC class counts per strain.

    Parameters
    ----------
    bgc_result : BGCResult
    output_dir : str
    filename   : str

    Returns
    -------
    str — full path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build pivot table: strains × BGC classes (excluding Non-BGC)
    active_classes = [c for c in BGC_CLASSES if c != "NonBGC"]
    strains = sorted({r.strain_id for r in bgc_result.bgc_hits}) if bgc_result.bgc_hits else []

    if not strains:
        logger.warning("plot_bgc_class_distribution: no BGC hits — generating empty chart")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No BGC hits detected", ha="center", va="center",
                fontsize=14, transform=ax.transAxes, color="#666666")
        ax.set_title("BGC Class Distribution per Strain", fontweight="bold")
        return _save_and_close(fig, os.path.join(output_dir, filename))

    counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in active_classes} for s in strains}
    for r in bgc_result.bgc_hits:
        if r.bgc_class in active_classes:
            counts[r.strain_id][r.bgc_class] += 1

    df = pd.DataFrame(counts).T.reindex(columns=active_classes, fill_value=0)
    # Friendly strain labels (strip long prefixes)
    df.index = [s.replace("Streptomyces_abikoensis_", "S.ab. ") for s in df.index]

    fig, ax = plt.subplots(figsize=(max(8, len(strains) * 1.8), 5))
    bottom = np.zeros(len(df))
    for cls in active_classes:
        vals = df[cls].astype(float).to_numpy()
        bars = ax.bar(df.index, vals.tolist(), bottom=bottom.tolist(),
                      color=BGC_PALETTE[cls], label=cls, edgecolor="white", linewidth=0.5)
        # Label non-zero bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
        bottom = bottom + vals

    ax.set_xlabel("Strain", fontsize=11)
    ax.set_ylabel("BGC Gene Count", fontsize=11)
    ax.set_title("BGC Class Distribution per Strain", fontsize=13, fontweight="bold")
    ax.legend(title="BGC Class", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    xticks = list(ax.get_xticks())
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.set_xticklabels(df.index, rotation=20, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    return _save_and_close(fig, os.path.join(output_dir, filename))


# ===========================================================================
# Plot 2 — Phylogenomic BGC Heatmap
# ===========================================================================

def plot_bgc_heatmap(
    bgc_result: BGCResult,
    output_dir: str,
    filename: str = "bgc_heatmap.png",
) -> str:
    """
    Seaborn clustermap: strains (rows) × BGC classes (columns).
    Colour = number of BGC gene hits; hierarchical clustering on both axes.

    Returns
    -------
    str — full path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, filename)

    active_classes = [c for c in BGC_CLASSES if c != "NonBGC"]
    strains = sorted({r.strain_id for r in bgc_result.bgc_records})

    if not strains:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No records to plot", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        return _save_and_close(fig, outpath)

    # Count matrix
    counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in active_classes} for s in strains}
    for r in bgc_result.bgc_hits:
        if r.bgc_class in active_classes:
            counts[r.strain_id][r.bgc_class] += 1

    matrix = pd.DataFrame(counts).T.reindex(columns=active_classes, fill_value=0)
    matrix.index = [s.replace("Streptomyces_abikoensis_", "S.ab. ") for s in matrix.index]

    if _SEABORN_AVAILABLE and sns is not None:
        sns_mod = sns
        # Use custom palette: white → class-specific dark colour
        cmap = sns_mod.light_palette("#E63946", as_cmap=True)
        g = sns_mod.clustermap(
            matrix,
            cmap=cmap,
            annot=True,
            fmt="d",
            linewidths=0.5,
            linecolor="#dddddd",
            figsize=(max(10, len(active_classes) * 1.4), max(5, len(strains) * 1.1)),
            dendrogram_ratio=(0.15, 0.15),
            cbar_kws={"label": "BGC Gene Count"},
            row_cluster=(len(strains) > 2),
            col_cluster=(len(active_classes) > 2),
        )
        g.fig.suptitle("Phylogenomic BGC Heatmap", fontsize=14, fontweight="bold", y=1.02)
        g.fig.savefig(outpath, bbox_inches="tight", dpi=150)
        plt.close(g.fig)
        logger.info("  Saved: %s", outpath)
        return outpath
    else:
        # Fallback: plain matplotlib heatmap
        fig, ax = plt.subplots(
            figsize=(max(10, len(active_classes) * 1.4), max(5, len(strains) * 1.1))
        )
        im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(active_classes)))
        ax.xaxis.set_major_locator(FixedLocator(range(len(active_classes))))
        ax.set_xticklabels(active_classes, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(matrix)))
        ax.yaxis.set_major_locator(FixedLocator(range(len(matrix))))
        ax.set_yticklabels(matrix.index, fontsize=9)
        for i in range(len(matrix)):
            for j in range(len(active_classes)):
                ax.text(j, i, str(matrix.values[i, j]), ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="BGC Gene Count")
        ax.set_title("Phylogenomic BGC Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return _save_and_close(fig, outpath)


def plot_phase3_decision_funnel(
    bgc_result: BGCResult,
    total_accessory_genes: int,
    output_dir: str,
    filename: str = "bgc_decision_funnel.png",
) -> str:
    if total_accessory_genes <= 0:
        return _empty_plot(output_dir, filename, "Phase 3 Decision Funnel", "No accessory genes available")

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, filename)
    n_scored = int(bgc_result.stats.get("n_alien_scored", len(bgc_result.bgc_records)))
    n_hits = int(bgc_result.stats.get("n_bgc_hits", len(bgc_result.bgc_hits)))
    n_high = int(bgc_result.stats.get("n_high_confidence", 0))
    stages = [
        ("Accessory genes", int(total_accessory_genes), "Phase 2 input universe"),
        ("Alien HGT genes", n_scored, "Genes passed into Phase 3"),
        ("BGC hits", n_hits, "Predicted BGC-associated genes"),
        ("High-confidence hits", n_high, "Most actionable candidates"),
    ]
    max_count = max(count for _, count, _ in stages) or 1
    widths = [float(count) / float(max_count) for _, count, _ in stages]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ypos = np.arange(len(stages))[::-1]
    colors = ["#a8dadc", "#457b9d", "#e76f51", "#2a9d8f"]
    ax.barh(ypos, widths, color=colors, height=0.72, edgecolor="white")

    for y, width, (label, count, note) in zip(ypos, widths, stages):
        ax.text(0.02, y, label, va="center", ha="left", fontsize=11, fontweight="bold", color="#1d3557")
        ax.text(min(width + 0.02, 0.98), y, f"{count:,}", va="center", ha="left", fontsize=11, fontweight="bold")
        ax.text(0.02, y - 0.26, note, va="center", ha="left", fontsize=8.5, color="#6c757d")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.8, len(stages) - 0.2)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.5, 0.75, 1.0]))
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fraction of accessory genes retained", fontsize=10)
    ax.set_title("Phase 2 → Phase 3 Candidate Funnel", fontsize=13, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    return _save_and_close(fig, outpath)


def plot_bgc_neighborhood_map(
    bgc_result: BGCResult,
    output_dir: str,
    filename: str = "bgc_neighborhood_map.png",
    max_regions: int = 6,
) -> str:
    if not bgc_result.bgc_records:
        return _empty_plot(output_dir, filename, "BGC Neighborhood Map", "No alien genes were scored")

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, filename)
    grouped: Dict[tuple[str, str], List[BGCGeneRecord]] = {}
    for rec in bgc_result.bgc_records:
        grouped.setdefault((rec.strain_id, rec.contig_id), []).append(rec)

    ranked_regions = sorted(
        grouped.items(),
        key=lambda item: (
            sum(1 for rec in item[1] if rec.is_bgc),
            max((rec.confidence for rec in item[1]), default=0.0),
            len(item[1]),
        ),
        reverse=True,
    )
    ranked_regions = [item for item in ranked_regions if any(rec.is_bgc for rec in item[1])][:max_regions]
    if not ranked_regions:
        return _empty_plot(output_dir, filename, "BGC Neighborhood Map", "No BGC hits available for neighborhood mapping")

    fig, axes = plt.subplots(len(ranked_regions), 1, figsize=(15, max(4.5, len(ranked_regions) * 2.4)))
    if len(ranked_regions) == 1:
        axes = [axes]

    for ax, ((strain_id, contig_id), region_records) in zip(axes, ranked_regions):
        records = sorted(region_records, key=lambda rec: rec.hgt_record.gene_record.start)
        starts = [rec.hgt_record.gene_record.start for rec in records]
        ends = [rec.hgt_record.gene_record.end for rec in records]
        centers = [0.5 * (s + e) for s, e in zip(starts, ends)]
        confidences = [rec.confidence for rec in records]
        sizes = [40 + conf * 240 for conf in confidences]

        ax.hlines(0, starts[0], ends[-1], color="#ced4da", linewidth=2.0, zorder=1)
        for rec, center, size in zip(records, centers, sizes):
            color = BGC_PALETTE.get(rec.bgc_class, "#999999") if rec.is_bgc else "#adb5bd"
            marker = CONF_MARKERS.get(rec.confidence_tier, "o")
            ax.scatter(center, 0, s=size, color=color, marker=marker, edgecolors="white", linewidths=0.6, alpha=0.95, zorder=3)
            ax.vlines(
                [rec.hgt_record.gene_record.start, rec.hgt_record.gene_record.end],
                -0.08,
                0.08,
                color="#adb5bd",
                linewidth=0.8,
                zorder=2,
            )

        top_hits = [rec for rec in records if rec.is_bgc]
        top_hits = sorted(top_hits, key=lambda rec: rec.confidence, reverse=True)[:3]
        for rec in top_hits:
            center = 0.5 * (rec.hgt_record.gene_record.start + rec.hgt_record.gene_record.end)
            label = rec.hgt_record.gene_record.product or rec.gene_id
            ax.text(center, 0.2, label[:28], ha="center", va="bottom", fontsize=7.5, color="#343a40")

        n_hits = sum(1 for rec in records if rec.is_bgc)
        ax.set_title(
            f"{strain_id} | {contig_id} | {n_hits} BGC hits across {len(records)} scored alien genes",
            fontsize=9.5,
            loc="left",
        )
        ax.set_yticks([])
        ax.set_xlabel("Genomic position (bp)", fontsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    legend_handles = [
        mpatches.Patch(color="#adb5bd", label="Scored alien gene (NonBGC)"),
        mpatches.Patch(color="#e76f51", label="BGC hit (class-coloured)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True, fontsize=9)
    fig.suptitle("Contig-Level BGC Hit Neighborhoods", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save_and_close(fig, outpath)


# ===========================================================================
# Plot 3 — BGC Confidence Landscape (scatter)
# ===========================================================================

def plot_bgc_confidence_landscape(
    bgc_result: BGCResult,
    output_dir: str,
    filename: str = "bgc_confidence_landscape.png",
) -> str:
    """
    Scatter plot: kmer_deviation (x) vs gc_deviation (y).
    Colour = BGC class, size = confidence, marker shape = tier.

    Returns
    -------
    str — full path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, filename)

    if not bgc_result.bgc_records:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No records", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
        return _save_and_close(fig, outpath)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Group by confidence tier for different markers
    for tier, marker in CONF_MARKERS.items():
        tier_recs = [r for r in bgc_result.bgc_records if r.confidence_tier == tier]
        if not tier_recs:
            continue

        xs     = [r.hgt_record.kmer_deviation for r in tier_recs]
        ys     = [r.hgt_record.gc_deviation   for r in tier_recs]
        colors = [BGC_PALETTE.get(r.bgc_class, "#999999") for r in tier_recs]
        sizes  = [max(20, r.confidence * 180)              for r in tier_recs]

        ax.scatter(
            xs, ys, c=colors, s=sizes, alpha=0.75,
            marker=marker, edgecolors="white", linewidths=0.4,
            label=f"{tier} confidence",
        )

    # Legend entries for BGC classes
    class_patches = [
        mpatches.Patch(color=BGC_PALETTE[c], label=c)
        for c in BGC_CLASSES
        if any(r.bgc_class == c for r in bgc_result.bgc_records)
    ]
    leg1 = ax.legend(
        handles=class_patches, title="BGC Class",
        bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9,
    )
    ax.legend(title="Confidence Tier", loc="lower right", fontsize=9)
    ax.add_artist(leg1)

    ax.set_xlabel("K-mer Deviation (tetranucleotide)", fontsize=11)
    ax.set_ylabel("GC Deviation from Host", fontsize=11)
    ax.set_title(
        "BGC Confidence Landscape\n(Alien HGT genes coloured by predicted BGC class)",
        fontsize=12, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return _save_and_close(fig, outpath)


# ===========================================================================
# HTML Report
# ===========================================================================

_PHASE3_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PanAdapt-BGC Miner | Phase 3: BGC Prediction</title>
<style>
  :root { --accent:#E63946; --bg:#f8f9fa; --card:#fff; --border:#dee2e6; }
  body { font-family:'Segoe UI',Arial,sans-serif; background:var(--bg);
         color:#212529; margin:0; padding:0; }
  header { background:var(--accent); color:#fff; padding:1.4rem 2rem; }
  header h1 { margin:0; font-size:1.8rem; }
  header p  { margin:.3rem 0 0; opacity:.85; font-size:.95rem; }
  .container { max-width:1200px; margin:0 auto; padding:1.5rem 2rem; }
  .stats-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                gap:1rem; margin-bottom:1.5rem; }
  .stat-card  { background:var(--card); border:1px solid var(--border);
                border-radius:.5rem; padding:1rem; text-align:center; }
  .stat-card .val { font-size:2rem; font-weight:700; color:var(--accent); }
  .stat-card .lbl { font-size:.8rem; color:#6c757d; margin-top:.2rem; }
  .plot-card  { background:var(--card); border:1px solid var(--border);
                border-radius:.5rem; padding:1.2rem; margin-bottom:1.5rem; }
  .plot-card h3 { margin:0 0 1rem; font-size:1.1rem; color:#343a40; }
  .plot-card img { max-width:100%; border-radius:.3rem; }
  table { width:100%; border-collapse:collapse; font-size:.82rem; }
  th    { background:#343a40; color:#fff; padding:.6rem .8rem; text-align:left; }
  td    { padding:.5rem .8rem; border-bottom:1px solid var(--border); }
  tr:hover td { background:#f1f3f5; }
  .badge { display:inline-block; padding:.2rem .55rem; border-radius:.9rem;
           font-size:.75rem; font-weight:600; color:#fff; }
  .badge-nonbgc     { background:#AAAAAA; }
  .badge-alkaloid   { background:#9B5DE5; }
  .badge-terpene    { background:#2A9D8F; }
  .badge-nrp        { background:#E63946; }
  .badge-polyketide { background:#F4A261; }
  .badge-ripp       { background:#457B9D; }
  .badge-saccharide { background:#E9C46A; color:#333; }
  .badge-other      { background:#264653; }
  .badge-high   { background:#198754; }
  .badge-med    { background:#FFC107; color:#333; }
  .badge-low    { background:#6c757d; }
  footer { text-align:center; padding:1rem; color:#adb5bd; font-size:.8rem;
           border-top:1px solid var(--border); margin-top:2rem; }
</style>
</head>
<body>
<header>
  <h1>PanAdapt-BGC Miner &mdash; Phase 3: BGC Prediction</h1>
  <p>AI-assisted Biosynthetic Gene Cluster scoring of Alien HGT genes</p>
</header>
<div class="container">

<!-- Stats cards -->
<div class="stats-grid">
  <div class="stat-card"><div class="val">{{ stats.n_alien_scored }}</div>
    <div class="lbl">Alien Genes Scored</div></div>
  <div class="stat-card"><div class="val">{{ stats.n_bgc_hits }}</div>
    <div class="lbl">BGC Hits</div></div>
  <div class="stat-card"><div class="val">{{ "%.1f"|format(stats.bgc_hit_rate*100) }}%</div>
    <div class="lbl">BGC Hit Rate</div></div>
  <div class="stat-card"><div class="val">{{ stats.n_high_confidence }}</div>
    <div class="lbl">High Confidence</div></div>
  <div class="stat-card"><div class="val">{{ stats.top_class }}</div>
    <div class="lbl">Top BGC Class</div></div>
  <div class="stat-card"><div class="val">{{ "BGC-Prophet" if stats.prophet_used else ("PyTorch" if stats.torch_used else "NumPy") }}</div>
    <div class="lbl">Inference Engine</div></div>
</div>

<!-- BGC class breakdown table -->
<div class="plot-card">
  <h3>BGC Class Summary</h3>
  <table>
    <thead><tr><th>BGC Class</th><th>Hits</th><th>% of BGC Hits</th></tr></thead>
    <tbody>
    {% for cls, cnt in class_distribution.items() %}
    <tr>
      <td>{{ cls }}</td>
      <td>{{ cnt }}</td>
      <td>{{ "%.1f"|format(cnt / ([stats.n_bgc_hits, 1]|max) * 100) }}%</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<!-- Plots -->
{% if plot_funnel %}
<div class="plot-card">
  <h3>Phase 2 → Phase 3 Candidate Funnel</h3>
  <img src="data:image/png;base64,{{ plot_funnel }}" alt="Phase 3 funnel">
</div>
{% endif %}

{% if plot_neighborhood %}
<div class="plot-card">
  <h3>Contig-Level BGC Hit Neighborhoods</h3>
  <img src="data:image/png;base64,{{ plot_neighborhood }}" alt="BGC neighborhood map">
</div>
{% endif %}

{% if plot_distribution %}
<div class="plot-card">
  <h3>BGC Class Distribution per Strain</h3>
  <img src="data:image/png;base64,{{ plot_distribution }}" alt="BGC distribution">
</div>
{% endif %}

{% if plot_heatmap %}
<div class="plot-card">
  <h3>Phylogenomic BGC Heatmap</h3>
  <img src="data:image/png;base64,{{ plot_heatmap }}" alt="BGC heatmap">
</div>
{% endif %}

{% if plot_landscape %}
<div class="plot-card">
  <h3>BGC Confidence Landscape</h3>
  <img src="data:image/png;base64,{{ plot_landscape }}" alt="Confidence landscape">
</div>
{% endif %}

<!-- Per-gene predictions table -->
<div class="plot-card">
  <h3>Top BGC Predictions (high confidence first)</h3>
  <table>
    <thead><tr>
      <th>Gene ID</th><th>Strain</th><th>BGC Class</th>
      <th>Confidence</th><th>Tier</th><th>Keywords</th>
    </tr></thead>
    <tbody>
    {% for r in top_predictions %}
    <tr>
      <td style="font-family:monospace;font-size:.78rem">{{ r.gene_id }}</td>
      <td>{{ r.strain_id | replace("Streptomyces_abikoensis_","S.ab. ") }}</td>
      <td><span class="badge badge-{{ r.bgc_class|lower|replace('-','') }}">{{ r.bgc_class }}</span></td>
      <td>{{ "%.3f"|format(r.confidence) }}</td>
      <td><span class="badge badge-{{ r.confidence_tier|lower }}">{{ r.confidence_tier }}</span></td>
      <td style="font-size:.75rem">{{ r.keyword_hits | join(", ") if r.keyword_hits else "—" }}</td>
    </tr>
    {% endfor %}
    {% if total_bgc > max_rows %}
    <tr><td colspan="6" style="color:#6c757d;font-style:italic">
      ... and {{ total_bgc - max_rows }} more BGC hits (see prediction_matrix.csv)
    </td></tr>
    {% endif %}
    </tbody>
  </table>
</div>

</div><!-- /container -->
<footer>PanAdapt-BGC Miner &bull; Phase 3 Report &bull; Generated by BGCPredictor</footer>
</body>
</html>
"""


def render_phase3_html_report(
    bgc_result: BGCResult,
    output_dir: str,
    plot_funnel_path: Optional[str] = None,
    plot_neighborhood_path: Optional[str] = None,
    plot_distribution_path: Optional[str] = None,
    plot_heatmap_path: Optional[str] = None,
    plot_landscape_path: Optional[str] = None,
    max_table_rows: int = 100,
    filename: str = "phase3_report.html",
) -> str:
    """
    Render Jinja2 HTML report for Phase 3 results.

    Embeds all provided plot images as base64 PNG data URIs.

    Parameters
    ----------
    bgc_result            : BGCResult
    output_dir            : str
    plot_*_path           : Optional paths to pre-generated PNG files
    max_table_rows        : int  — cap predictions table
    filename              : str

    Returns
    -------
    str — full path to written HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not _JINJA_AVAILABLE or Environment is None or BaseLoader is None:
        logger.error("Jinja2 not installed — skipping HTML report")
        return ""

    def _embed(path: Optional[str]) -> str:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Sort BGC hits by confidence descending
    top_preds = sorted(bgc_result.bgc_hits, key=lambda r: r.confidence, reverse=True)

    env = Environment(loader=BaseLoader())
    template = env.from_string(_PHASE3_TEMPLATE)
    html = template.render(
        stats               = bgc_result.stats,
        class_distribution  = bgc_result.class_distribution,
        top_predictions     = top_preds[:max_table_rows],
        total_bgc           = len(bgc_result.bgc_hits),
        max_rows            = max_table_rows,
        plot_funnel         = _embed(plot_funnel_path),
        plot_neighborhood   = _embed(plot_neighborhood_path),
        plot_distribution   = _embed(plot_distribution_path),
        plot_heatmap        = _embed(plot_heatmap_path),
        plot_landscape      = _embed(plot_landscape_path),
    )

    outpath = os.path.join(output_dir, filename)
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info("Phase 3 HTML report written: %s", outpath)
    return outpath
