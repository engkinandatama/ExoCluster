"""
PanAdapt-BGC Miner — Main Pipeline Entry Point
===============================================
Module  : main.py
Purpose : CLI entry point for the full PanAdapt-BGC Miner pipeline.
          Currently wires Phase 1 (PangenomeMiner) and is scaffolded
          for Phase 2 (HGT Detective) and Phase 3 (AI BGC Predictor).

Usage
-----
  # Run with real data:
  python main.py --genomes genomes/ --annotations annotations/ --output output/

  # Run with auto-generated mock data (for testing):
  python main.py --mock --output output/
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from pipeline.pangenome_miner import PangenomeMiner, PangenomeResult
from pipeline.pangenome_visualizer import (
    plot_presence_absence_heatmap,
    plot_pangenome_summary,
    render_phase1_html_report,
)
from pipeline.hgt_detective import HGTDetective, HGTResult
from pipeline.hgt_visualizer import (
    plot_genomic_island_architecture,
    plot_hgt_feature_distributions,
    render_phase2_html_report,
)
from pipeline.bgc_predictor import BGCPredictor, BGCResult
from pipeline.bgc_visualizer import (
    plot_phase3_decision_funnel,
    plot_bgc_neighborhood_map,
    plot_bgc_class_distribution,
    plot_bgc_heatmap,
    plot_bgc_confidence_landscape,
    render_phase3_html_report,
)

logger = logging.getLogger("panadapt_bgc_miner")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panadapt-bgc-miner",
        description=(
            "PanAdapt-BGC Miner — Pangenome Adaptive & Alien Biosynthetic Gene "
            "Cluster Miner. Integrates pangenome analysis, HGT detection, and "
            "AI-driven BGC prediction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--genomes", type=Path, default=Path("genomes"),
        help="Directory containing FASTA genome files (.fasta / .fna / .fa).",
    )
    parser.add_argument(
        "--annotations", type=Path, default=Path("annotations"),
        help="Directory containing GFF3 annotation files (.gff / .gff3).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output"),
        help="Root output directory for all pipeline artefacts.",
    )
    parser.add_argument(
        "--core-threshold", type=float, default=0.95,
        help="Fraction of strains a gene must appear in to be Core (default: 0.95).",
    )
    parser.add_argument(
        "--accessory-threshold", type=float, default=0.10,
        help="Max fraction of strains for a gene to be Accessory (default: 0.10).",
    )
    parser.add_argument(
        "--identity", type=float, default=0.80,
        help="Ortholog clustering sequence-identity threshold (default: 0.80).",
    )
    parser.add_argument(
        "--cluster-method", type=str, default="mmseqs2",
        choices=["kmer", "mmseqs2"],
        help="Ortholog clustering method (default: mmseqs2).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Generate and use synthetic mock data for testing.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Directory with BGC-Prophet model weights (annotator.pt + classifier.pt)."
    )
    parser.add_argument(
        "--esm-model", type=str, default="esm2_t6_8M_UR50D",
        help=(
            "ESM2 protein language model variant for Phase 3 embeddings. "
            "Options: esm2_t6_8M_UR50D (default, ~30 MB), esm2_t12_35M_UR50D (~140 MB), "
            "esm2_t30_150M_UR50D (~600 MB), esm2_t33_650M_UR50D (~2.5 GB), "
            "esm2_t36_3B_UR50D (~11 GB), esm2_t48_15B_UR50D (~60 GB)."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help=(
            "Compute device for Phase 3 ESM2 / BGC-Prophet inference. "
            "Options: auto (default, picks CUDA > MPS > CPU), cuda, mps, cpu."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Phase 1 runner
# ---------------------------------------------------------------------------
def run_phase1(args: argparse.Namespace) -> tuple[PangenomeResult, PangenomeMiner]:
    logger.info("=" * 60)
    logger.info("PHASE 1 — The Comparatist Engine (Pangenome Analysis)")
    logger.info("=" * 60)

    miner = PangenomeMiner(
        core_threshold=args.core_threshold,
        accessory_threshold=args.accessory_threshold,
        identity_threshold=args.identity,
        cluster_method=args.cluster_method,
    )

    result: PangenomeResult = miner.run(args.genomes, args.annotations)

    # Persist matrix
    matrix_out = args.output / "pangenome_presence_absence_matrix.csv"
    miner.save_matrix(matrix_out)

    # Phase 1 Visualizations
    logger.info("Generating Pangenome visualizations …")
    p1_dir = args.output / "pangenome"
    p1_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = plot_presence_absence_heatmap(
        result.presence_absence_matrix,
        output_path=p1_dir / "presence_absence_heatmap.png",
    )
    summary_chart_path = plot_pangenome_summary(
        result.stats,
        output_path=p1_dir / "pangenome_summary.png",
    )
    html_path = render_phase1_html_report(
        stats=result.stats,
        strain_ids=result.strain_ids,
        accessory_records=result.accessory_records,
        heatmap_path=heatmap_path,
        summary_chart_path=summary_chart_path,
        output_path=p1_dir / "pangenome_report.html",
    )

    stats = result.stats
    print(
        f"\n{'─' * 55}\n"
        f"  Phase 1 Summary\n"
        f"{'─' * 55}\n"
        f"  Strains analysed   : {stats['n_strains']}\n"
        f"  Total clusters     : {stats['n_total_clusters']}\n"
        f"  Core genome        : {stats['n_core']} clusters\n"
        f"  Shell genome       : {stats['n_shell']} clusters\n"
        f"  Accessory genome   : {stats['n_accessory']} clusters "
        f"({stats['n_accessory_records']} individual genes)\n"
        f"{'─' * 55}\n"
        f"  Outputs → {p1_dir}/\n"
    )
    return result, miner


# ---------------------------------------------------------------------------
# Phase 2 stub
# ---------------------------------------------------------------------------
def run_phase2(phase1_result: PangenomeResult, miner: PangenomeMiner, args: argparse.Namespace) -> HGTResult:
    logger.info("=" * 60)
    logger.info("PHASE 2 — The HGT Detective")
    logger.info("=" * 60)

    detective = HGTDetective(contamination=0.30, n_estimators=200)
    hgt_result = detective.run(phase1_result=phase1_result, fasta_store=miner._fasta_store)

    # Phase 2 Visualizations
    logger.info("Generating HGT visualizations …")
    p2_dir = args.output / "hgt"
    p2_dir.mkdir(parents=True, exist_ok=True)

    genomic_plot_paths = {}
    for strain_id in phase1_result.strain_ids:
        safe_name = strain_id.replace('/', '_').replace(' ', '_')
        plot_path = plot_genomic_island_architecture(
            hgt_result=hgt_result,
            strain_id=strain_id,
            output_path=p2_dir / f"genomic_island_{safe_name}.png",
        )
        if plot_path:
            genomic_plot_paths[strain_id] = plot_path

    feat_dist_path = plot_hgt_feature_distributions(
        hgt_result=hgt_result,
        output_path=p2_dir / "hgt_feature_distributions.png",
    )

    p2_html = render_phase2_html_report(
        hgt_result=hgt_result,
        genomic_plot_paths=genomic_plot_paths,
        feature_dist_path=feat_dist_path,
        output_path=p2_dir / "hgt_report.html",
    )

    stats = hgt_result.stats
    print(
        f"\n{'─' * 55}\n"
        f"  Phase 2 Summary\n"
        f"{'─' * 55}\n"
        f"  Accessory genes analysed  : {stats['n_accessory_input']}\n"
        f"  Alien HGT regions flagged : {stats['n_alien_hgt']} ({stats['hgt_fraction']:.1%})\n"
        f"  MGE-proximal genes        : {stats['n_mge_proximal']}\n"
        f"{'─' * 55}\n"
        f"  Outputs → {p2_dir}/\n"
    )
    return hgt_result


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------
def run_phase3(
    hgt_result: HGTResult,
    all_gene_records,
    args: argparse.Namespace,
) -> BGCResult:
    logger.info("=" * 60)
    logger.info("PHASE 3 — The AI Discoverer (BGC Prediction)")
    logger.info("=" * 60)

    predictor = BGCPredictor(seed=42, min_confidence=0.25, use_keyword_boost=True, model_dir=args.model_dir, esm_model_name=args.esm_model, device=args.device)
    bgc_result = predictor.run(hgt_result, all_gene_records=all_gene_records)

    # Phase 3 Visualizations
    logger.info("Generating BGC visualizations ...")
    p3_dir = args.output / "bgc"
    p3_dir.mkdir(parents=True, exist_ok=True)
    funnel_path = plot_phase3_decision_funnel(
        bgc_result=bgc_result,
        total_accessory_genes=len(hgt_result.hgt_records),
        output_dir=str(p3_dir),
    )
    neighborhood_path = plot_bgc_neighborhood_map(
        bgc_result=bgc_result,
        output_dir=str(p3_dir),
    )

    dist_path = plot_bgc_class_distribution(
        bgc_result=bgc_result,
        output_dir=str(p3_dir),
    )
    heatmap_path = plot_bgc_heatmap(
        bgc_result=bgc_result,
        output_dir=str(p3_dir),
    )
    landscape_path = plot_bgc_confidence_landscape(
        bgc_result=bgc_result,
        output_dir=str(p3_dir),
    )
    render_phase3_html_report(
        bgc_result=bgc_result,
        output_dir=str(p3_dir),
        plot_funnel_path=funnel_path,
        plot_neighborhood_path=neighborhood_path,
        plot_distribution_path=dist_path,
        plot_heatmap_path=heatmap_path,
        plot_landscape_path=landscape_path,
    )

    # Export prediction matrix
    if not bgc_result.prediction_matrix.empty:
        pred_csv = p3_dir / "prediction_matrix.csv"
        bgc_result.prediction_matrix.to_csv(pred_csv, index=False)
        logger.info("Prediction matrix saved: %s", pred_csv)

    stats = bgc_result.stats
    print(
        f"\n{'─' * 55}\n"
        f"  Phase 3 Summary\n"
        f"{'─' * 55}\n"
        f"  Alien genes scored    : {stats.get('n_alien_scored', 0)}\n"
        f"  BGC hits              : {stats.get('n_bgc_hits', 0)} ({stats.get('bgc_hit_rate', 0):.1%})\n"
        f"  High-confidence hits  : {stats.get('n_high_confidence', 0)}\n"
        f"  Top BGC class         : {stats.get('top_class', 'N/A')}\n"
        f"  Inference engine      : {'BGC-Prophet' if stats.get('prophet_used') else ('PyTorch' if stats.get('torch_used') else 'NumPy mock')}\n"
        f"  ESM2 model            : {stats.get('esm_model', 'N/A')}\n"
        f"  Device                : {stats.get('device', 'N/A')}\n"
        f"  Outputs → {p3_dir}/\n"
    )
    return bgc_result

# ---------------------------------------------------------------------------
# Mock data helper
# ---------------------------------------------------------------------------
def prepare_mock_data(args: argparse.Namespace) -> None:
    mock_dir = Path(__file__).parent / "mock_data"
    gen_script = mock_dir / "generate_mock_data.py"

    logger.info("Generating mock data …")
    result = subprocess.run(
        [sys.executable, str(gen_script)],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Mock data generation failed.")

    # Override paths to point at generated data
    args.genomes = mock_dir / "genomes"
    args.annotations = mock_dir / "annotations"
    logger.info("Mock data ready: %s / %s", args.genomes, args.annotations)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging level
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Optional: generate mock data
    if args.mock:
        prepare_mock_data(args)

    phase1_result, miner = run_phase1(args)
    hgt_result = run_phase2(phase1_result, miner, args)
    run_phase3(hgt_result, miner._all_records, args)

    logger.info("Pipeline complete. All outputs in: %s", args.output)


if __name__ == "__main__":
    main()
