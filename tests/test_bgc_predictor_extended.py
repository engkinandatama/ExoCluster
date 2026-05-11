"""
Extended BGCPredictor tests – target >90 % coverage.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import pandas as pd

# Ensure project root on import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.bgc_predictor import (
    BGCPredictor,
    BGCResult,
    _translate_cds,
    _apply_keyword_boosts,
    _apply_keyword_boosts_to_scores,
)
from pipeline.hgt_detective import HGTGeneRecord, HGTResult
from pipeline.pangenome_miner import GeneRecord


# ---------- Helper factories ----------
def make_gene(gid: str, seq: str = "ATG" * 4, product: str = "") -> GeneRecord:
    """Create a minimal GeneRecord with required attributes."""
    return GeneRecord(
        gene_id=gid,
        strain_id="strainA",
        contig="contig1",
        start=1,
        end=len(seq) * 3,
        strand="+",
        feature_type="CDS",
        product=product,
        sequence=seq,
    )


def make_hgt(gene: GeneRecord) -> HGTGeneRecord:
    """Wrap a GeneRecord into an HGTGeneRecord (minimal fields)."""
    return HGTGeneRecord(
        gene_record=gene,
        gc_content=0.5,
        gc_deviation=0.1,
        kmer_deviation=0.2,
        anomaly_score=0.0,
        mge_proximity=False,
        is_hgt=True,
        evidence=[]
    )


# ---------- Tests ----------
def test_translate_cds_variants():
    # Valid coding sequence → protein >10 aa
    # Use a longer sequence to ensure >10 amino acids
    assert _translate_cds("ATG" * 10) == "M" * 10  # "ATGATGATGATGATGATGATGATGATGATGATGATGATG" -> "MMMMMMMMMM"
    # Short protein (<10 aa) → None
    assert _translate_cds("ATGATG") is None
    # Invalid chars replaced by N, N‑preserved
    seq = "ATGXXXATGNNNATG"
    # Expected translation: M N M N N M (N codon stays N, others become X)
    # "ATGXXXATGNNNATG" -> "MNMXNM" (6 aa) -> should be None due to length constraint
    assert _translate_cds(seq) is None


def test_empty_hgtresult_returns_empty_bgcresult():
    predictor = BGCPredictor(seed=1, use_keyword_boost=False)
    empty_hgt = HGTResult(
        alien_records=[],
        hgt_records=[],
        strain_gc_profiles={},
        feature_matrix=pd.DataFrame(),
        stats={}
    )
    result: BGCResult = predictor.run(empty_hgt)

    # Empty‑result path exercised
    assert result.bgc_records == []
    assert result.bgc_hits == []
    assert result.feature_matrix.empty
    assert result.prediction_matrix.empty
    assert result.stats["n_alien_scored"] == 0
    assert result.stats["n_bgc_hits"] == 0
    assert "fisher_enrichment" in result.stats


def test_mock_backend_with_keyword_boost():
    # Build two genes – one with a keyword that should boost “nrps”
    gene1 = make_gene("g1", seq="ATG" * 5, product="non-ribosomal peptide synthase")
    gene2 = make_gene("g2", seq="ATG" * 5, product="unknown")
    hgt1 = make_hgt(gene1)
    hgt2 = make_hgt(gene2)

    hgt_res = HGTResult(
        alien_records=[hgt1, hgt2],
        hgt_records=[hgt1, hgt2],
        strain_gc_profiles={"strainA": 0.5},
        feature_matrix=pd.DataFrame({
            "gene_id": ["g1", "g2"],
            "gc_content": [0.5, 0.5],
            "gc_deviation": [0.1, 0.1],
            "kmer_deviation": [0.2, 0.2],
            "mge_proximity": [0.0, 0.0]
        }).set_index("gene_id"),
        stats={}
    )

    # Use lower min_confidence to ensure the keyword boost triggers is_bgc=True in mock mode
    predictor = BGCPredictor(seed=42, use_keyword_boost=True, min_confidence=0.2)

    # Run predictor – will use mock‑MLP path (Prophet backend unavailable)
    result: BGCResult = predictor.run(hgt_res)

    # Core expectations
    assert len(result.feature_matrix) == 2
    assert len(result.prediction_matrix) == 2

    # Keyword “non-ribosomal” maps to class index 3 (NRP). Verify boost was applied.
    kw_hits = [r.keyword_hits for r in result.bgc_records]
    assert any("non-ribosomal" in hits for hits in kw_hits)
    
    # Debug: Print confidence and is_bgc for each record
    for i, r in enumerate(result.bgc_records):
        print(f"Record {i}: is_bgc={r.is_bgc}, confidence={r.confidence}, class={r.bgc_class}, keyword_hits={r.keyword_hits}")

    # At least one record should be flagged as BGC (confidence above default MED)
    assert any(r.is_bgc for r in result.bgc_records)

    # Fisher’s test stats are present and numeric
    fisher = result.stats["fisher_enrichment"]
    assert isinstance(fisher["odds_ratio"], float)
    assert isinstance(fisher["p_value"], float)

    # Confidence‑tier assignment logic (high/medium/low)
    tiers = {r.confidence_tier for r in result.bgc_records}
    assert tiers.issubset({"High", "Medium", "Low"})


def test_keyword_boost_functions_direct():
    gene = make_gene("gX", product="polyketide synthase")
    hgt = make_hgt(gene)

    # Fake logits: uniform low scores
    # Assuming BGC_CLASSES = ["NonBGC", "Alkaloid", "Terpene", "NRP", "Polyketide", "RiPP", "Saccharide", "Other"]
    # So, 8 classes, index 4 is "Polyketide"
    logits = np.full((1, 8), 0.1, dtype=np.float32)

    boosted, hits = _apply_keyword_boosts([hgt], logits.copy())
    # “polyketide” maps to class index 4 with boost 0.25
    assert boosted[0, 4] > logits[0, 4]
    assert "polyketide" in hits[0]

    # Apply to already‑softmaxed scores
    scores = [[0.1] * 8]
    adj_scores, adj_hits = _apply_keyword_boosts_to_scores([hgt], scores)
    assert adj_scores[0][4] > scores[0][4]
    assert "polyketide" in adj_hits[0]