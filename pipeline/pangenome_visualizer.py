"""
PanAdapt-BGC Miner — Phase 1 Visualizer
=========================================
Module  : pipeline/phase1_visualizer.py
Purpose : Generate all Phase 1 output artefacts:
            1. Phylogenomic BGC Heatmap (seaborn clustermap with dendrogram)
            2. Pangenome Pie/Bar summary chart
            3. Phase 1 HTML report (Jinja2)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Phylogenomic Presence/Absence Heatmap
# ---------------------------------------------------------------------------


def plot_presence_absence_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str = "Pangenome Gene Presence/Absence",
    figsize: tuple = (16, 10),
) -> Path:
    """
    Clustered heatmap (strains × gene clusters) with hierarchical dendrogram.

    Columns (strains) are clustered by Jaccard distance on binary presence vectors.
    Rows (gene clusters) subsampled to max 500 for readability.

    Parameters
    ----------
    matrix : pd.DataFrame
        Boolean DataFrame, index=cluster_ids, columns=strain_ids.
    output_path : Path
        Where to save the PNG.

    Returns
    -------
    Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Work with transpose so strains are rows for clustering
    mat = matrix.astype(int)

    # Subsample rows for large pangenomes (keep variety: core + accessory mix)
    max_rows = 500
    if len(mat) > max_rows:
        # Keep equal proportions: top N by presence fraction variability (std)
        row_std = mat.std(axis=1)
        mat = mat.loc[row_std.nlargest(max_rows).index]
        logger.info("Heatmap: subsampled to %d most variable clusters", max_rows)

    # Compute strain-level Jaccard distance for column clustering
    mat_T = mat.T  # strains × clusters
    try:
        col_linkage = hierarchy.linkage(
            pdist(mat_T.values, metric="jaccard"),
            method="average",
        )
    except Exception:
        col_linkage = None

    # Color palette: white = absent, dark teal = present
    cmap = sns.color_palette(["#f0f0f0", "#1a7a5e"], as_cmap=True)

    # Fraction present per cluster → row colors (blue=core, red=accessory)
    n_strains = mat.shape[1]
    fraction = mat.sum(axis=1) / n_strains
    row_colors = fraction.map(
        lambda f: (
            "#c0392b"
            if f <= (1.0 / n_strains + 0.01)
            else ("#27ae60" if f >= 0.95 else "#f39c12")
        )
    )
    row_colors.name = "Category"

    g = sns.clustermap(
        mat,
        col_linkage=col_linkage,
        row_cluster=True,
        cmap=cmap,
        figsize=figsize,
        linewidths=0,
        row_colors=row_colors,
        xticklabels=True,
        yticklabels=False,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        dendrogram_ratio=(0.15, 0.12),
    )

    g.ax_heatmap.set_xlabel("Strains", fontsize=12, labelpad=8)
    g.ax_heatmap.set_ylabel(f"Gene Clusters (n={len(mat):,})", fontsize=12, labelpad=8)
    g.figure.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#1a7a5e", label="Present"),
        mpatches.Patch(color="#f0f0f0", label="Absent"),
        mpatches.Patch(color="#c0392b", label="Accessory (strain-specific)"),
        mpatches.Patch(color="#f39c12", label="Shell genome"),
        mpatches.Patch(color="#27ae60", label="Core genome (≥95%)"),
    ]
    g.figure.legend(
        handles=legend_patches,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        fontsize=9,
        frameon=True,
    )

    g.figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Heatmap saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2.1 Rarefaction Curve
# ---------------------------------------------------------------------------


def plot_rarefaction_curve(
    rarefaction_data: List[float],
    output_path: Path,
    title: str = "Pangenome Rarefaction Curve",
    figsize: tuple = (8, 6),
) -> Path:
    """
    Plots the rarefaction curve showing unique gene clusters discovered
    as number of genomes increases.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    x = np.arange(1, len(rarefaction_data) + 1)
    y = np.array(rarefaction_data)

    plt.plot(x, y, marker="o", linestyle="-", color="#1a7a5e", linewidth=2)
    plt.fill_between(x, y, alpha=0.2, color="#1a7a5e")

    plt.xlabel("Number of Genomes", fontsize=12)
    plt.ylabel("Unique Gene Clusters", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close("all")
    logger.info("Rarefaction curve saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. Pangenome Summary Bar Chart
# ---------------------------------------------------------------------------


def plot_pangenome_summary(
    stats: Dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Horizontal stacked bar chart showing Core / Shell / Accessory proportions,
    plus a separate bar for total clusters per strain.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "PanAdapt-BGC Miner — Phase 1 Pangenome Summary", fontsize=13, fontweight="bold"
    )

    # --- Left: stacked bar of pangenome composition ---
    ax = axes[0]
    categories = ["Core\n(≥95%)", "Shell", "Accessory\n(strain-specific)"]
    counts = [stats["n_core"], stats["n_shell"], stats["n_accessory"]]
    colors = ["#27ae60", "#f39c12", "#c0392b"]
    total = sum(counts)

    bars = ax.barh(["Pangenome"], [counts[0]], color=colors[0], label=categories[0])
    ax.barh(
        ["Pangenome"],
        [counts[1]],
        left=[counts[0]],
        color=colors[1],
        label=categories[1],
    )
    ax.barh(
        ["Pangenome"],
        [counts[2]],
        left=[counts[0] + counts[1]],
        color=colors[2],
        label=categories[2],
    )

    # Annotate each segment
    lefts = [0, counts[0], counts[0] + counts[1]]
    for c, l, cat in zip(counts, lefts, categories):
        if c > total * 0.03:
            pct = c / total * 100
            ax.text(
                l + c / 2,
                0,
                f"{c:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

    ax.set_xlabel("Number of Ortholog Clusters")
    ax.set_title(
        f"Pangenome Composition\n(n={stats['n_strains']} strains, {total:,} total clusters)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, total * 1.05)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])

    # --- Right: accessory genes per strain breakdown ---
    ax2 = axes[1]
    ax2.text(
        0.5,
        0.5,
        f"Total Pangenome Clusters: {total:,}\n\n"
        f"  Core genome       : {stats['n_core']:>6,} ({stats['n_core']/total*100:.1f}%)\n"
        f"  Shell genome      : {stats['n_shell']:>6,} ({stats['n_shell']/total*100:.1f}%)\n"
        f"  Accessory genome  : {stats['n_accessory']:>6,} ({stats['n_accessory']/total*100:.1f}%)\n\n"
        f"Accessory gene records: {stats['n_accessory_records']:,}\n"
        f"  → Passed to Phase 2 (HGT Detection)",
        transform=ax2.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#eaf4fb",
            edgecolor="#2980b9",
            linewidth=1.5,
        ),
    )
    ax2.axis("off")
    ax2.set_title("Phase 1 Statistics")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Summary chart saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. Phase 1 HTML Report (Jinja2)
# ---------------------------------------------------------------------------

PHASE1_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PanAdapt-BGC Miner — Phase 1 Report</title>
  <style>
    :root { --accent: #1a7a5e; --warn: #c0392b; --shell: #f39c12; --core: #27ae60; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f7fa; color: #2c3e50; }
    header { background: var(--accent); color: white; padding: 2rem 3rem; }
    header h1 { font-size: 1.8rem; letter-spacing: 0.5px; }
    header p  { opacity: 0.85; margin-top: 0.4rem; font-size: 0.95rem; }
    .container { max-width: 1100px; margin: 2rem auto; padding: 0 2rem; }
    .card { background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 1.8rem; margin-bottom: 1.8rem; }
    .card h2 { font-size: 1.2rem; color: var(--accent); border-bottom: 2px solid var(--accent);
               padding-bottom: 0.5rem; margin-bottom: 1.2rem; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }
    .stat-box { text-align: center; padding: 1.2rem; border-radius: 8px; }
    .stat-box .value { font-size: 2rem; font-weight: 700; }
    .stat-box .label { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.3rem; }
    .core-box    { background: #eafaf1; color: var(--core); }
    .shell-box   { background: #fef9e7; color: var(--shell); }
    .access-box  { background: #fdedec; color: var(--warn); }
    .total-box   { background: #eaf4fb; color: #2980b9; }
    .strain-box  { background: #f4ecf7; color: #8e44ad; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th { background: #ecf0f1; text-align: left; padding: 0.6rem 0.8rem; font-size: 0.82rem;
         text-transform: uppercase; letter-spacing: 0.4px; }
    td { padding: 0.55rem 0.8rem; border-bottom: 1px solid #ecf0f1; }
    tr:hover td { background: #f9fbfc; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px;
             font-size: 0.78rem; font-weight: 600; }
    .badge-core   { background: #eafaf1; color: var(--core); }
    .badge-access { background: #fdedec; color: var(--warn); }
    .badge-shell  { background: #fef9e7; color: var(--shell); }
    .plot-img { width: 100%; border-radius: 6px; border: 1px solid #dde; }
    .alert { background: #fdf2e9; border-left: 4px solid var(--warn); padding: 1rem 1.2rem;
             border-radius: 4px; margin-bottom: 1rem; font-size: 0.92rem; }
    .alert strong { color: var(--warn); }
    footer { text-align: center; padding: 2rem; color: #95a5a6; font-size: 0.82rem; }
  </style>
</head>
<body>
<header>
  <h1>🧬 PanAdapt-BGC Miner</h1>
  <p>Phase 1 Report — The Comparatist Engine (Pangenome Analysis)</p>
  <p style="margin-top:0.6rem; font-size:0.85rem; opacity:0.7;">Generated: {{ generated_at }}</p>
</header>

<div class="container">

  <!-- Stats overview -->
  <div class="card">
    <h2>📊 Pangenome Statistics</h2>
    <div class="stats-grid">
      <div class="stat-box strain-box">
        <div class="value">{{ stats.n_strains }}</div>
        <div class="label">Strains Analysed</div>
      </div>
      <div class="stat-box total-box">
        <div class="value">{{ "{:,}".format(stats.n_total_clusters) }}</div>
        <div class="label">Total Clusters</div>
      </div>
      <div class="stat-box core-box">
        <div class="value">{{ "{:,}".format(stats.n_core) }}</div>
        <div class="label">Core Genome</div>
      </div>
      <div class="stat-box shell-box">
        <div class="value">{{ "{:,}".format(stats.n_shell) }}</div>
        <div class="label">Shell Genome</div>
      </div>
      <div class="stat-box access-box">
        <div class="value">{{ "{:,}".format(stats.n_accessory) }}</div>
        <div class="label">Accessory Clusters</div>
      </div>
    </div>
  </div>

  <!-- Alert: accessory gene payload -->
  <div class="alert">
    <strong>⚠ Phase 2 Input:</strong>
    {{ "{:,}".format(stats.n_accessory_records) }} individual accessory gene records
    (from {{ "{:,}".format(stats.n_accessory) }} unique clusters) have been flagged as
    strain-specific and will be passed to the <strong>HGT Detective</strong> for anomaly analysis.
  </div>

  <!-- Strains table -->
  <div class="card">
    <h2>🦠 Input Strains</h2>
    <table>
      <thead><tr><th>#</th><th>Strain ID</th><th>Status</th></tr></thead>
      <tbody>
        {% for sid in strain_ids %}
        <tr>
          <td>{{ loop.index }}</td>
          <td><code>{{ sid }}</code></td>
          <td><span class="badge badge-core">loaded</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Top accessory genes table -->
  <div class="card">
    <h2>🔬 Top Accessory Gene Records (Phase 2 candidates)</h2>
    <p style="font-size:0.85rem; color:#7f8c8d; margin-bottom:1rem;">
      Showing first {{ top_accessory|length }} of {{ "{:,}".format(stats.n_accessory_records) }} accessory gene records.
      These are strain-specific genes not found in the core pangenome.
    </p>
    <table>
      <thead>
        <tr>
          <th>Gene ID</th><th>Strain</th><th>Contig</th>
          <th>Start</th><th>End</th><th>Strand</th><th>Product</th>
        </tr>
      </thead>
      <tbody>
        {% for rec in top_accessory %}
        <tr>
          <td><code>{{ rec.gene_id[:35] }}</code></td>
          <td><code>{{ rec.strain_id }}</code></td>
          <td><code>{{ rec.contig[:20] }}</code></td>
          <td>{{ "{:,}".format(rec.start) }}</td>
          <td>{{ "{:,}".format(rec.end) }}</td>
          <td>{{ rec.strand }}</td>
          <td style="color:#555; font-size:0.85rem;">{{ rec.product[:60] if rec.product else "—" }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Plots -->
  <div class="card">
    <h2>📈 Pangenome Summary Chart</h2>
    <img class="plot-img" src="{{ summary_chart_rel }}" alt="Pangenome summary chart">
  </div>

  {% if rarefaction_curve_rel %}
  <div class="card">
    <h2>📉 Rarefaction Curve</h2>
    <img class="plot-img" src="{{ rarefaction_curve_rel }}" alt="Rarefaction curve">
  </div>
  {% endif %}

  <div class="card">
    <h2>🗺 Presence/Absence Heatmap</h2>
    <p style="font-size:0.85rem; color:#7f8c8d; margin-bottom:0.8rem;">
      Rows = gene clusters (most variable shown). Columns = strains.
      Red = accessory | Orange = shell | Green = core.
      Column order reflects hierarchical clustering by Jaccard distance.
    </p>
    <img class="plot-img" src="{{ heatmap_rel }}" alt="Presence/absence heatmap">
  </div>

</div>

<footer>
  PanAdapt-BGC Miner v0.1 &mdash; Phase 1 complete.
  Proceed to Phase 2 (HGT Detection) with {{ "{:,}".format(stats.n_accessory_records) }} accessory gene candidates.
</footer>
</body>
</html>
"""


def render_phase1_html_report(
    stats: Dict[str, Any],
    strain_ids: List[str],
    accessory_records: list,
    heatmap_path: Path,
    summary_chart_path: Path,
    rarefaction_path: Optional[Path] = None,
    output_path: Path = Path("report.html"),
) -> Path:
    """Render the Phase 1 HTML report using Jinja2."""
    try:
        from jinja2 import Environment, BaseLoader
    except ImportError:
        logger.warning("jinja2 not installed — skipping HTML report")
        return output_path

    from datetime import datetime

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=BaseLoader())
    template = env.from_string(PHASE1_HTML_TEMPLATE)

    # Make paths relative to report location for portability
    report_dir = output_path.parent

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(report_dir))
        except ValueError:
            return str(p)

    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stats=stats,
        strain_ids=strain_ids,
        top_accessory=accessory_records[:100],
        heatmap_rel=_rel(heatmap_path),
        summary_chart_rel=_rel(summary_chart_path),
        rarefaction_curve_rel=_rel(rarefaction_path) if rarefaction_path else None,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("Phase 1 HTML report saved → %s", output_path)
    return output_path
