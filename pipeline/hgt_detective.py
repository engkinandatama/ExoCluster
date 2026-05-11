"""
PanAdapt-BGC Miner — Phase 2: The HGT Detective
=================================================
Module  : pipeline/hgt_detective.py
Purpose : Detect Horizontal Gene Transfer (HGT) signatures in accessory/unique
          genes from Phase 1. Uses three complementary anomaly signals:

            1. Tetranucleotide (k-mer) frequency deviation from host genome
            2. GC content skew relative to host genome average
            3. Proximity to Mobile Genetic Elements (MGEs) in GFF annotation

          Genes passing an Isolation Forest anomaly threshold are flagged as
          Alien_HGT_Regions and returned as HGTResult for Phase 3.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from pipeline.pangenome_miner import GeneRecord, PangenomeResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MGE keyword list — product strings that indicate mobile elements
# ---------------------------------------------------------------------------
MGE_KEYWORDS = frozenset(
    [
        "transposase",
        "integrase",
        "recombinase",
        "resolvase",
        "phage",
        "prophage",
        "insertion sequence",
        "is element",
        "conjugal",
        "conjugative",
        "plasmid",
        "relaxase",
        "mob protein",
        "mobilization",
        "mobilase",
        "terminase",
        "capsid",
        "tail fiber",
        "portal protein",
        "site-specific recombinase",
        "serine recombinase",
        "tyrosine recombinase",
        "hin recombinase",
        "xre",
        "toxin",
        "antitoxin",
        "crispr",
        "crispr-associated",
        "cas protein",
    ]
)

# Distance threshold (bp) for MGE proximity flag
MGE_PROXIMITY_BP = 10_000


# ===========================================================================
# Data containers
# ===========================================================================


@dataclass
class HGTGeneRecord:
    """Phase 2 output: an accessory gene annotated with HGT evidence scores."""

    gene_record: GeneRecord
    gc_content: float  # GC fraction of this gene
    gc_deviation: float  # |gene_gc - host_gc| normalised by host_gc
    kmer_deviation: float  # Mahalanobis-like distance from host k-mer profile
    mge_proximity: bool  # True if within MGE_PROXIMITY_BP of an MGE
    anomaly_score: float  # IsolationForest decision score (higher = more anomalous)
    is_hgt: bool  # True if flagged as Alien_HGT_Region
    evidence: List[str] = field(default_factory=list)  # human-readable evidence list


@dataclass
class HGTResult:
    """Full Phase 2 output — passed to Phase 3."""

    hgt_records: List[HGTGeneRecord]  # all accessory genes, scored
    alien_records: List[HGTGeneRecord]  # subset flagged is_hgt=True
    strain_gc_profiles: Dict[str, float]  # strain → mean GC content
    feature_matrix: pd.DataFrame  # genes × features (for inspection)
    stats: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Helper functions
# ===========================================================================


def _gc_content(seq: str) -> float:
    """Return GC fraction of a nucleotide sequence. Returns 0.0 if empty."""
    if not seq:
        return 0.0
    seq_upper = seq.upper()
    gc = seq_upper.count("G") + seq_upper.count("C")
    return gc / len(seq_upper)


def _tetranucleotide_freq(seq: str) -> np.ndarray:
    """
    Compute normalised tetranucleotide frequency vector (4^4 = 256 dims).
    Returns a zero vector if sequence is too short.
    """
    k = 4
    if len(seq) < k:
        return np.zeros(256, dtype=float)

    counts = np.zeros(256, dtype=float)
    bases = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_upper = seq.upper()

    for i in range(len(seq_upper) - k + 1):
        mer = seq_upper[i : i + k]
        if all(b in bases for b in mer):
            idx = 0
            for b in mer:
                idx = idx * 4 + bases[b]
            counts[idx] += 1

    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _build_host_kmer_profile(
    fasta_store: Dict[str, Any],
    strain_id: str,
) -> Tuple[np.ndarray, float]:
    """
    Build genome-wide tetranucleotide frequency profile and mean GC for a strain.
    Processes entire sequence in one pass for efficiency.

    Returns
    -------
    (kmer_profile_256d, mean_gc)
    """
    records = fasta_store.get(strain_id, {})
    full_seq = "".join(str(r.seq) for r in records.values()).upper()

    if not full_seq:
        return np.zeros(256), 0.5

    # GC content
    gc = _gc_content(full_seq)

    # Tetranucleotide frequency on whole genome
    kmer_profile = _tetranucleotide_freq(full_seq[:500_000])  # cap at 500 kb for speed

    return kmer_profile, gc


def _find_mge_positions(
    all_records: List[GeneRecord],
    strain_id: str,
) -> List[Tuple[str, int, int]]:
    """
    Return list of (contig, start, end) for all MGE-related features in a strain's
    annotation, identified by product keyword matching.
    """
    mge_positions: List[Tuple[str, int, int]] = []
    for rec in all_records:
        if rec.strain_id != strain_id:
            continue
        product_lower = rec.product.lower()
        if any(kw in product_lower for kw in MGE_KEYWORDS):
            mge_positions.append((rec.contig, rec.start, rec.end))
    return mge_positions


def _is_near_mge(
    gene: GeneRecord,
    mge_positions: List[Tuple[str, int, int]],
    proximity_bp: int = MGE_PROXIMITY_BP,
) -> bool:
    """Check whether *gene* falls within *proximity_bp* of any known MGE."""
    for contig, mge_start, mge_end in mge_positions:
        if contig != gene.contig:
            continue
        dist = max(0, max(mge_start, gene.start) - min(mge_end, gene.end))
        if dist <= proximity_bp:
            return True
    return False


# ===========================================================================
# Main Class
# ===========================================================================


class HGTDetective:
    """
    Phase 2 — The HGT Detective Engine.

    Algorithm
    ---------
    For each strain's accessory genes (from Phase 1):
      1. Compute tetranucleotide k-mer frequency deviation from host genome
      2. Compute GC content deviation from host genome mean
      3. Flag proximity to Mobile Genetic Elements (MGEs)
      4. Run Isolation Forest on the feature matrix to score anomalies
      5. Apply threshold to produce binary HGT labels

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalous (HGT) genes. Default: 0.30.
        Higher → more genes flagged; lower → stricter.
    n_estimators : int
        Number of trees in Isolation Forest. Default: 200.
    random_state : int
        Reproducibility seed. Default: 42.
    min_seq_length : int
        Minimum nucleotide length to perform k-mer analysis. Default: 90.
    """

    def __init__(
        self,
        contamination: float = 0.30,
        n_estimators: int = 200,
        random_state: int = 42,
        min_seq_length: int = 90,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_seq_length = min_seq_length

        logger.info(
            "HGTDetective initialised | contamination=%.2f | trees=%d",
            contamination,
            n_estimators,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        phase1_result: PangenomeResult,
        fasta_store: Dict[str, Any],
    ) -> HGTResult:
        """
        Full Phase 2 pipeline.

        Parameters
        ----------
        phase1_result : PangenomeResult
            Output from Phase 1 (PangenomeMiner).
        fasta_store   : dict
            {strain_id → {contig_id → BioPython SeqRecord}} — same store
            used/built during Phase 1. Passed in from PangenomeMiner._fasta_store.

        Returns
        -------
        HGTResult
        """
        if not phase1_result.accessory_records:
            raise ValueError("No accessory records from Phase 1. Cannot run Phase 2.")

        logger.info(
            "Phase 2 starting: %d accessory gene records across %d strains",
            len(phase1_result.accessory_records),
            len(phase1_result.strain_ids),
        )

        # 1. Build host genomic profiles per strain
        host_kmer_profiles: Dict[str, np.ndarray] = {}
        host_gc: Dict[str, float] = {}
        mge_map: Dict[str, List[Tuple[str, int, int]]] = {}

        all_annotation_records = (
            phase1_result.accessory_records
        )  # we reuse product info
        # But we need ALL records for MGE search — accessory_records only has accessory
        # We'll search within accessory for MGE labels (proxy for nearby MGEs)

        for strain_id in phase1_result.strain_ids:
            kmer_prof, gc = _build_host_kmer_profile(fasta_store, strain_id)
            host_kmer_profiles[strain_id] = kmer_prof
            host_gc[strain_id] = gc
            mge_positions = _find_mge_positions(
                phase1_result.accessory_records, strain_id
            )
            mge_map[strain_id] = mge_positions
            logger.debug(
                "  %s: host GC=%.2f%%, %d MGE loci found",
                strain_id,
                gc * 100,
                len(mge_positions),
            )

        # 2. Compute per-gene features
        hgt_records_all: List[HGTGeneRecord] = []
        feature_rows: List[Dict] = []

        for rec in phase1_result.accessory_records:
            sid = rec.strain_id
            seq = rec.sequence

            # GC deviation
            gene_gc = _gc_content(seq) if seq else host_gc.get(sid, 0.5)
            host_gc_val = host_gc.get(sid, 0.5)
            gc_dev = abs(gene_gc - host_gc_val) / (host_gc_val + 1e-9)

            # K-mer deviation (cosine distance from host profile)
            if seq and len(seq) >= self.min_seq_length:
                gene_kmer = _tetranucleotide_freq(seq)
                host_kmer = host_kmer_profiles.get(sid, np.zeros(256))
                # Use top-10 most discriminative k-mers for speed
                denom = (np.linalg.norm(gene_kmer) * np.linalg.norm(host_kmer)) + 1e-12
                cosine_sim = np.dot(gene_kmer, host_kmer) / denom
                kmer_dev = float(1.0 - cosine_sim)
            else:
                kmer_dev = 0.0

            # MGE proximity
            near_mge = _is_near_mge(rec, mge_map.get(sid, []))
            # Also check the gene's own product
            self_is_mge = any(kw in rec.product.lower() for kw in MGE_KEYWORDS)

            hgt_rec = HGTGeneRecord(
                gene_record=rec,
                gc_content=gene_gc,
                gc_deviation=gc_dev,
                kmer_deviation=kmer_dev,
                mge_proximity=near_mge or self_is_mge,
                anomaly_score=0.0,  # filled in after IsolationForest
                is_hgt=False,
            )
            hgt_records_all.append(hgt_rec)

            feature_rows.append(
                {
                    "gene_id": rec.gene_id,
                    "strain_id": sid,
                    "gc_content": gene_gc,
                    "gc_deviation": gc_dev,
                    "kmer_deviation": kmer_dev,
                    "mge_proximity": float(near_mge or self_is_mge),
                    "gene_length": rec.length,
                }
            )

        # 3. Isolation Forest anomaly detection
        feature_df = pd.DataFrame(feature_rows).set_index("gene_id")
        feature_cols = ["gc_deviation", "kmer_deviation", "mge_proximity", "gc_content"]
        X = feature_df[feature_cols].values.astype(float)

        # Handle degenerate case (all zeros)
        if X.std(axis=0).sum() < 1e-9:
            logger.warning(
                "Feature matrix has near-zero variance — skipping Isolation Forest; "
                "falling back to GC/k-mer threshold rules."
            )
            scores = np.zeros(len(X))
            labels = np.where(
                (feature_df["gc_deviation"] > 0.05)
                | (feature_df["kmer_deviation"] > 0.05),
                -1,
                1,
            )
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(X_scaled)
            scores = clf.decision_function(X_scaled)  # higher = more normal
            labels = clf.predict(X_scaled)  # -1 = anomaly, 1 = normal

        # 4. Annotate HGT records
        gene_id_to_score = dict(zip(feature_df.index, scores))
        gene_id_to_label = dict(zip(feature_df.index, labels))

        alien_records: List[HGTGeneRecord] = []

        for hgt_rec in hgt_records_all:
            gid = hgt_rec.gene_record.gene_id
            score = float(gene_id_to_score.get(gid, 0.0))
            label = int(gene_id_to_label.get(gid, 1))

            hgt_rec.anomaly_score = -score  # negate: high = anomalous
            hgt_rec.is_hgt = label == -1

            # Build evidence list
            evidence: List[str] = []
            if hgt_rec.gc_deviation > 0.05:
                evidence.append(
                    f"GC deviation {hgt_rec.gc_deviation:.3f} "
                    f"(gene {hgt_rec.gc_content:.1%} vs host {host_gc.get(hgt_rec.gene_record.strain_id, 0):.1%})"
                )
            if hgt_rec.kmer_deviation > 0.03:
                evidence.append(f"k-mer cosine deviation {hgt_rec.kmer_deviation:.4f}")
            if hgt_rec.mge_proximity:
                evidence.append(
                    "proximal to Mobile Genetic Element (transposase/integrase/phage)"
                )
            if not evidence and hgt_rec.is_hgt:
                evidence.append(
                    "IsolationForest multi-feature anomaly (combined signal)"
                )
            hgt_rec.evidence = evidence

            if hgt_rec.is_hgt:
                alien_records.append(hgt_rec)

        n_hgt = len(alien_records)
        n_total = len(hgt_records_all)
        logger.info(
            "Phase 2 complete: %d / %d accessory genes flagged as Alien_HGT_Regions (%.1f%%)",
            n_hgt,
            n_total,
            n_hgt / n_total * 100 if n_total else 0,
        )

        stats = {
            "n_accessory_input": n_total,
            "n_alien_hgt": n_hgt,
            "n_normal": n_total - n_hgt,
            "hgt_fraction": n_hgt / n_total if n_total else 0.0,
            "n_mge_proximal": sum(1 for r in hgt_records_all if r.mge_proximity),
            "strain_gc_profiles": {k: round(v, 4) for k, v in host_gc.items()},
        }

        return HGTResult(
            hgt_records=hgt_records_all,
            alien_records=alien_records,
            strain_gc_profiles=host_gc,
            feature_matrix=feature_df,
            stats=stats,
        )
