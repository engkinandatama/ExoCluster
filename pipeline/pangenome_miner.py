"""
PanAdapt-BGC Miner — Phase 1: The Comparatist Engine
=====================================================
Module  : pipeline/pangenome_miner.py
Purpose : Ingest multi-genome GFF/FASTA data, cluster orthologs, build a
          gene presence/absence matrix, and partition the genome into
          Core (<= 5% absent) and Accessory/Unique (>= 90% absent) gene sets.
          Only the Accessory/Unique gene coordinates are passed downstream.

Author  : PanAdapt-BGC Miner Project
Python  : >=3.9
"""

from __future__ import annotations

import logging
import os
import re
import hashlib
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

# ---------------------------------------------------------------------------
# Logger — module-level, uses Python stdlib logging so callers can configure
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ===========================================================================
# Data-container dataclasses (lightweight, no external deps)
# ===========================================================================
from dataclasses import dataclass, field


@dataclass
class GeneRecord:
    """Lightweight representation of a single annotated gene locus."""

    gene_id: str  # Unique identifier (from GFF attributes or constructed)
    strain_id: str  # Which genome/strain this gene belongs to
    contig: str  # Contig / chromosome name
    start: int  # 1-based genomic start coordinate
    end: int  # 1-based genomic end coordinate
    strand: str  # '+', '-', or '.'
    feature_type: str  # CDS, gene, mRNA, …
    product: str = ""  # Functional annotation (if present)
    sequence: str = ""  # Nucleotide sequence (populated later if FASTA given)

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class PangenomeResult:
    """Container for all Phase-1 outputs handed to Phase-2."""

    presence_absence_matrix: pd.DataFrame  # genes × strains (bool)
    core_genes: pd.Index  # gene_ids present in > core_threshold
    accessory_genes: pd.Index  # gene_ids present in < accessory_threshold
    accessory_records: List[GeneRecord]  # Full GeneRecord objects for accessory genes
    strain_ids: List[str]
    stats: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Helper utilities (private)
# ===========================================================================


def _parse_gff_attributes(attr_str: str) -> Dict[str, str]:
    """
    Parse GFF3-style attribute column (key=value;key=value) into a dict.
    Handles both GFF3 (key=val) and GTF-style (key "val") loosely.
    """
    attrs: Dict[str, str] = {}
    for part in attr_str.strip().rstrip(";").split(";"):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            attrs[k.strip()] = v.strip().strip('"')
        elif " " in part:
            k, _, v = part.partition(" ")
            attrs[k.strip()] = v.strip().strip('"')
    return attrs


def _extract_gene_id(attrs: Dict[str, str]) -> str:
    """
    Attempt to extract a meaningful gene identifier from attribute dict.
    Priority: ID > locus_tag > gene > protein_id > Name > generated hash.
    """
    for key in ("ID", "locus_tag", "gene", "protein_id", "Name"):
        if key in attrs and attrs[key]:
            return attrs[key]
    # Fallback: deterministic hash of the entire attribute string
    return "gene_" + hashlib.md5(str(attrs).encode()).hexdigest()[:8]


def _sequence_for_region(
    fasta_records: Dict[str, SeqRecord],
    contig: str,
    start: int,
    end: int,
    strand: str,
) -> str:
    """
    Extract and optionally reverse-complement a subsequence from a FASTA dict.
    Coordinates are 1-based inclusive (GFF convention) and converted internally.
    """
    if contig not in fasta_records:
        return ""
    contig_len = len(fasta_records[contig].seq)
    # Clamp coordinates to valid range (1-based inclusive → 0-based slice)
    s = max(start - 1, 0)
    e = min(end, contig_len)
    if s >= e:
        return ""
    seq = fasta_records[contig].seq[s:e]
    if strand == "-":
        seq = seq.reverse_complement()
    return str(seq)


def _cluster_genes_by_sequence_identity(
    all_records: List[GeneRecord],
    identity_threshold: float = 0.80,
) -> Dict[str, str]:
    """
    Lightweight ortholog clustering via k-mer (tetranucleotide) Jaccard
    similarity — used as a fast *mock* when MMseqs2 is unavailable.

    Returns
    -------
    Dict[gene_id → cluster_id]
        Cluster IDs are the gene_id of the first member encountered (centroid).

    Notes
    -----
    For production use, replace this function body with a subprocess call to
    ``mmseqs easy-cluster`` and parse the resulting TSV.  The interface
    (input: list of GeneRecord, output: dict gene_id→cluster_id) stays the same.
    """
    logger.info(
        "Running lightweight k-mer ortholog clustering on %d sequences …",
        len(all_records),
    )

    def _kmer_set(seq: str, k: int = 4) -> set:
        return (
            {seq[i : i + k] for i in range(len(seq) - k + 1)}
            if len(seq) >= k
            else set()
        )

    # Build kmer sets once
    kmer_sets: Dict[str, set] = {r.gene_id: _kmer_set(r.sequence) for r in all_records}

    cluster_map: Dict[str, str] = {}  # gene_id → centroid gene_id
    centroids: List[str] = []  # ordered list of cluster centroids

    for record in all_records:
        gid = record.gene_id
        if not record.sequence:
            # If sequence unavailable, each gene is its own cluster
            cluster_map[gid] = gid
            centroids.append(gid)
            continue

        assigned = False
        q_set = kmer_sets[gid]

        for centroid in centroids:
            c_set = kmer_sets[centroid]
            union = q_set | c_set
            if not union:
                continue
            jaccard = len(q_set & c_set) / len(union)
            if jaccard >= identity_threshold:
                cluster_map[gid] = centroid
                assigned = True
                break

        if not assigned:
            cluster_map[gid] = gid  # becomes a new centroid
            centroids.append(gid)

    n_clusters = len(set(cluster_map.values()))
    logger.info(
        "Clustering complete: %d genes → %d ortholog clusters (identity ≥ %.0f%%)",
        len(all_records),
        n_clusters,
        identity_threshold * 100,
    )
    return cluster_map


def _cluster_genes_with_mmseqs2(
    all_records: List[GeneRecord],
    identity_threshold: float = 0.80,
    temp_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Ortholog clustering using MMseqs2 (faster and more accurate than k-mer).

    Requires mmseqs2 to be installed and available in PATH.

    Parameters
    ----------
    all_records : List[GeneRecord]
        List of gene records to cluster.
    identity_threshold : float
        Minimum sequence identity for clustering (passed to mmseqs2 as --min-seq-id).
    temp_dir : str, optional
        Directory to store temporary FASTA and TSV files. If None, uses system temp.

    Returns
    -------
    Dict[gene_id → cluster_id]
        Cluster IDs are MMseqs-generated cluster numbers.

    Raises
    ------
    RuntimeError
        If mmseqs2 is not found or fails.
    """

    logger.info(
        "Running MMseqs2 ortholog clustering on %d sequences (identity ≥ %.0f%%) …",
        len(all_records),
        identity_threshold * 100,
    )

    try:
        import subprocess
    except ImportError:
        raise RuntimeError("subprocess module is not available, cannot run MMseqs2")

    if not all_records:
        return {}

    # Use a separate variable for Path operations to avoid type confusion
    work_dir: Path = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write input FASTA
    input_fasta = work_dir / "input.fasta"
    with open(input_fasta, "w") as f:
        for record in all_records:
            # Use gene_id as header, store original gene_id in description
            f.write(f">{record.gene_id}\n{record.sequence}\n")

    # Run MMseqs2 clustering
    cluster_db = work_dir / "cluster_db"
    cluster_tsv = work_dir / "cluster.tsv"
    cluster_pref = work_dir / "cluster_pref"

    try:
        # Step 1: Create database
        subprocess.run(
            ["mmseqs", "createdb", str(input_fasta), str(cluster_db)],
            check=True,
            capture_output=True,
            text=True,
        )
        # Step 2: Cluster
        subprocess.run(
            [
                "mmseqs",
                "cluster",
                str(cluster_db),
                str(cluster_pref),
                str(cluster_tsv),
                "--min-seq-id",
                str(identity_threshold),
                "-c",
                "0.8",
                "--threads",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        # Step 3: Create TSV file
        subprocess.run(
            ["mmseqs", "createtsv", str(cluster_db), str(cluster_db), str(cluster_tsv)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Parse results
        cluster_map: Dict[str, str] = {}
        with open(cluster_tsv, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        cluster_map[parts[0]] = parts[1]

    finally:
        # Clean up temp files
        shutil.rmtree(work_dir, ignore_errors=True)

    return cluster_map


# ===========================================================================
# Main Class
# ===========================================================================


class PangenomeMiner:
    """
    Phase 1 — The Comparatist Engine.

    Workflow
    --------
    1. ``load_genomes(genomes_dir, annotations_dir)``   — parse FASTA + GFF files
    2. ``build_presence_absence_matrix()``              — cluster orthologs; build matrix
    3. ``partition_pangenome()``                        — split into Core / Accessory
    4. ``extract_accessory_coordinates()``              — return coordinates for Phase 2

    Parameters
    ----------
    core_threshold : float
        Fraction of strains a gene must be present in to be labelled Core.
        Default: 0.95  (present in ≥ 95 % of strains).
    accessory_threshold : float
        Upper fraction of strains for a gene to qualify as Accessory/Unique.
        Default: 0.10  (present in ≤ 10 % of strains).
    identity_threshold : float
        Sequence identity (Jaccard k-mer proxy) for clustering orthologs.
        Default: 0.80.
    feature_types : list[str]
        GFF feature types to extract (default: CDS only).
    min_gene_length : int
        Minimum gene length in bp; shorter features are skipped.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        core_threshold: float = 0.95,
        accessory_threshold: float = 0.10,
        identity_threshold: float = 0.80,
        feature_types: Optional[List[str]] = None,
        min_gene_length: int = 100,
        cluster_method: str = "kmer",
    ) -> None:
        if not (0.0 < core_threshold <= 1.0):
            raise ValueError(f"core_threshold must be in (0, 1], got {core_threshold}")
        if not (0.0 <= accessory_threshold < 1.0):
            raise ValueError(
                f"accessory_threshold must be in [0, 1), got {accessory_threshold}"
            )
        if accessory_threshold >= core_threshold:
            raise ValueError(
                f"accessory_threshold ({accessory_threshold}) must be < "
                f"core_threshold ({core_threshold})"
            )

        self.core_threshold = core_threshold
        self.accessory_threshold = accessory_threshold
        if not (0.0 < identity_threshold <= 1.0):
            raise ValueError(
                f"identity_threshold must be in (0, 1], got {identity_threshold}"
            )
        self.identity_threshold = identity_threshold
        self.feature_types = set(feature_types or ["CDS"])
        self.min_gene_length = min_gene_length

        valid_methods = ["kmer", "mmseqs2"]
        if cluster_method not in valid_methods:
            raise ValueError(
                f"cluster_method must be one of {valid_methods}, got {cluster_method}"
            )
        self.cluster_method = cluster_method

        # Internal state (populated during pipeline run)
        self._strain_ids: List[str] = []
        self._all_records: List[GeneRecord] = []
        self._fasta_store: Dict[str, Dict[str, SeqRecord]] = (
            {}
        )  # strain → contig → SeqRecord
        self._matrix: Optional[pd.DataFrame] = None
        self._cluster_map: Optional[Dict[str, str]] = None
        self.stats: Dict[str, Any] = {}  # Container for runtime statistics

        logger.info(
            "PangenomeMiner initialised | core_threshold=%.2f | "
            "accessory_threshold=%.2f | identity=%.2f",
            self.core_threshold,
            self.accessory_threshold,
            self.identity_threshold,
        )

    # ------------------------------------------------------------------
    # Step 1 — Data ingestion
    # ------------------------------------------------------------------
    def load_genomes(
        self,
        genomes_dir: str | Path,
        annotations_dir: str | Path,
    ) -> "PangenomeMiner":
        """
        Scan *genomes_dir* for FASTA files (.fasta / .fna / .fa) and
        *annotations_dir* for GFF3 files (.gff / .gff3).

        Matching is done by stem (e.g. ``strain_A.fasta`` ↔ ``strain_A.gff``).
        Unmatched FASTA files are skipped with a warning.

        Returns
        -------
        self   (enables method chaining)
        """
        genomes_dir = Path(genomes_dir)
        annotations_dir = Path(annotations_dir)

        if not genomes_dir.is_dir():
            raise FileNotFoundError(f"Genomes directory not found: {genomes_dir}")
        if not annotations_dir.is_dir():
            raise FileNotFoundError(
                f"Annotations directory not found: {annotations_dir}"
            )

        # Build index of available GFF files by stem
        gff_index: Dict[str, Path] = {}
        for ext in ("*.gff", "*.gff3"):
            for p in annotations_dir.glob(ext):
                gff_index[p.stem] = p

        if not gff_index:
            raise ValueError(f"No GFF files found in: {annotations_dir}")

        # Iterate FASTA files and pair with GFF
        loaded = 0
        for ext in ("*.fasta", "*.fna", "*.fa"):
            for fasta_path in sorted(genomes_dir.glob(ext)):
                stem = fasta_path.stem
                if stem not in gff_index:
                    logger.warning(
                        "FASTA file '%s' has no matching GFF annotation — skipping.",
                        stem,
                    )
                    continue

                strain_id = stem
                logger.info("Loading strain: %s", strain_id)

                # Parse FASTA
                fasta_records = {
                    rec.id: rec for rec in SeqIO.parse(fasta_path, "fasta")
                }
                self._fasta_store[strain_id] = fasta_records

                # Parse GFF → GeneRecord list
                records = self._parse_gff(
                    gff_path=gff_index[stem],
                    strain_id=strain_id,
                    fasta_records=fasta_records,
                )
                self._all_records.extend(records)
                self._strain_ids.append(strain_id)
                loaded += 1
                logger.info(
                    "  └─ %d features loaded from %s", len(records), fasta_path.name
                )

        if loaded == 0:
            raise RuntimeError(
                "No strains were loaded. Check that genomes and annotations "
                "directories contain properly paired FASTA/GFF files."
            )

        logger.info(
            "Data ingestion complete: %d strains, %d total gene features.",
            len(self._strain_ids),
            len(self._all_records),
        )
        return self

    def _parse_gff(
        self,
        gff_path: Path,
        strain_id: str,
        fasta_records: Dict[str, SeqRecord],
    ) -> List[GeneRecord]:
        """
        Parse a single GFF3 file and return a list of GeneRecord objects.

        Rules
        -----
        - Lines starting with '#' are comments/directives — skipped.
        - Only features whose *type* column is in ``self.feature_types`` are kept.
        - Features shorter than ``self.min_gene_length`` are discarded.
        - Sequences are extracted from *fasta_records* if available.
        """
        records: List[GeneRecord] = []
        seen_ids: Dict[str, int] = defaultdict(int)  # handle duplicate IDs

        try:
            with gff_path.open("r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    cols = line.split("\t")
                    if len(cols) < 9:
                        logger.debug(
                            "%s:%d — expected 9 columns, got %d; skipping.",
                            gff_path.name,
                            lineno,
                            len(cols),
                        )
                        continue

                    (
                        seqid,
                        source,
                        feat_type,
                        start_s,
                        end_s,
                        score,
                        strand,
                        phase,
                        attr_s,
                    ) = cols

                    # Filter by feature type
                    if feat_type not in self.feature_types:
                        continue

                    # Coordinate parsing
                    try:
                        start = int(start_s)
                        end = int(end_s)
                    except ValueError:
                        logger.warning(
                            "%s:%d — non-integer coordinates (%s, %s); skipping.",
                            gff_path.name,
                            lineno,
                            start_s,
                            end_s,
                        )
                        continue

                    if end < start:
                        start, end = end, start  # tolerate inverted coords

                    if (end - start + 1) < self.min_gene_length:
                        continue

                    # Attribute parsing
                    attrs = _parse_gff_attributes(attr_s)
                    gene_id = _extract_gene_id(attrs)

                    # Ensure uniqueness within strain
                    seen_ids[gene_id] += 1
                    if seen_ids[gene_id] > 1:
                        gene_id = f"{gene_id}_{seen_ids[gene_id]}"

                    product = attrs.get("product", attrs.get("Note", ""))

                    # Optional: attach nucleotide sequence
                    seq = _sequence_for_region(fasta_records, seqid, start, end, strand)

                    records.append(
                        GeneRecord(
                            gene_id=gene_id,
                            strain_id=strain_id,
                            contig=seqid,
                            start=start,
                            end=end,
                            strand=strand if strand in ("+", "-") else ".",
                            feature_type=feat_type,
                            product=product,
                            sequence=seq,
                        )
                    )

        except OSError as exc:
            raise RuntimeError(f"Could not read GFF file {gff_path}: {exc}") from exc

        return records

    # ------------------------------------------------------------------
    # Step 2 — Presence/absence matrix
    # ------------------------------------------------------------------
    def build_presence_absence_matrix(self) -> "PangenomeMiner":
        """
        Cluster all genes into orthogroups and build a boolean
        (gene_cluster × strain) presence/absence matrix.

        After this call, ``self._matrix`` is a DataFrame with:
        - Index   : ortholog cluster IDs
        - Columns : strain IDs
        - Values  : True / False

        Returns
        -------
        self
        """
        if not self._all_records:
            raise RuntimeError("No gene records loaded. Call load_genomes() first.")

        # Ortholog clustering
        if self.cluster_method == "mmseqs2":
            self._cluster_map = _cluster_genes_with_mmseqs2(
                self._all_records,
                identity_threshold=self.identity_threshold,
            )
        else:  # kmer
            self._cluster_map = _cluster_genes_by_sequence_identity(
                self._all_records,
                identity_threshold=self.identity_threshold,
            )

        logger.info("Building presence/absence matrix …")

        # cluster_id → set of strains that have ≥1 representative
        cluster_strain_map: Dict[str, set] = defaultdict(set)
        for record in self._all_records:
            centroid = self._cluster_map[record.gene_id]
            cluster_strain_map[centroid].add(record.strain_id)

        # Build matrix
        all_clusters = sorted(cluster_strain_map.keys())
        matrix_data = {
            strain: [strain in cluster_strain_map[c] for c in all_clusters]
            for strain in self._strain_ids
        }
        self._matrix = pd.DataFrame(
            matrix_data,
            index=pd.Index(all_clusters, name="cluster_id"),
        )

        logger.info(
            "Presence/absence matrix: %d clusters × %d strains",
            self._matrix.shape[0],
            self._matrix.shape[1],
        )
        return self

    # ------------------------------------------------------------------
    # Step 3 — Pangenome partitioning
    # ------------------------------------------------------------------
    def calculate_rarefaction(self, iterations: int = 10) -> np.ndarray:
        """Calculate pangenome rarefaction curve (average unique clusters per k genomes)."""
        if self._matrix is None:
            raise RuntimeError("Matrix not built.")
        n_strains = self._matrix.shape[1]
        rarefaction_means = np.zeros(n_strains)
        vals = self._matrix.values
        for k in range(1, n_strains + 1):
            counts = []
            for _ in range(iterations):
                indices = np.random.choice(n_strains, k, replace=False)
                count = np.any(vals[:, indices], axis=1).sum()
                counts.append(count)
            rarefaction_means[k - 1] = np.mean(counts)
        return rarefaction_means

    def partition_pangenome(self) -> Tuple[pd.Index, pd.Index, pd.Index]:
        """
        Partition the pangenome into three tiers based on strain-fraction.
        Also calculates Heaps' Law and Rarefaction curve statistics.
        """
        if self._matrix is None:
            raise RuntimeError(
                "Matrix not built. Call build_presence_absence_matrix() first."
            )

        n_strains = len(self._strain_ids)
        fraction_present = self._matrix.sum(axis=1) / n_strains

        effective_acc = max(self.accessory_threshold, 1.0 / n_strains)
        effective_acc = min(effective_acc, self.core_threshold - 1e-6)

        core_genes = self._matrix.index[fraction_present >= self.core_threshold]
        accessory_genes = self._matrix.index[fraction_present <= effective_acc]
        shell_genes = self._matrix.index[
            (fraction_present > effective_acc)
            & (fraction_present < self.core_threshold)
        ]

        # Heaps' Law
        n_genomes = np.arange(1, n_strains + 1)
        unique_genes = []
        for i in n_genomes:
            subset = self._matrix.sample(n=i, axis=1)
            unique_genes.append(subset.sum(axis=1).gt(0).sum())

        from scipy.optimize import curve_fit

        def heaps_law(n, k, beta):
            return k * np.power(n, beta)

        try:
            # Ensure we have enough data points for fitting
            if len(n_genomes) < 3:
                raise ValueError("Insufficient data points for Heaps' Law fitting")

            # Use bounds to prevent unphysical values (k > 0, 0 < beta <= 1)
            # k: intercept/scaling factor, beta: exponent
            popt, _ = curve_fit(
                heaps_law,
                n_genomes,
                unique_genes,
                p0=[max(unique_genes) / (len(n_genomes) ** 0.5), 0.5],
                bounds=([1e-3, 0.001], [np.inf, 1.0]),
            )
            k, beta = popt
            heaps_stats = {"k": float(k), "beta": float(beta), "open": bool(beta > 0.5)}
        except Exception as e:
            logger.warning(
                "Heaps' Law calculation failed (likely insufficient variance or data): %s",
                e,
            )
            heaps_stats = {"k": 0.0, "beta": 0.0, "open": None}

        rarefaction = self.calculate_rarefaction()

        logger.info(
            "Pangenome partitioning (%d strains):\n"
            "  Core       (≥%.0f%%): %5d clusters\n"
            "  Shell                : %5d clusters\n"
            "  Accessory  (≤%.0f%%): %5d clusters\n"
            "  Heaps' Law           : k=%.2f, beta=%.2f (Open=%s)",
            n_strains,
            self.core_threshold * 100,
            len(core_genes),
            len(shell_genes),
            effective_acc * 100,
            len(accessory_genes),
            heaps_stats["k"],
            heaps_stats["beta"],
            heaps_stats["open"],
        )

        self.stats.update(
            {"heaps_law": heaps_stats, "rarefaction_curve": rarefaction.tolist()}
        )
        return core_genes, accessory_genes, shell_genes

    # ------------------------------------------------------------------
    # Step 4 — Extract accessory coordinates (output to Phase 2)
    # ------------------------------------------------------------------
    def extract_accessory_coordinates(self) -> PangenomeResult:
        """
        Full Phase-1 pipeline run in one call:
        build matrix → partition → collect GeneRecord objects for
        Accessory/Unique clusters → return PangenomeResult.

        This is the primary output consumed by Phase 2 (HGT Detective).

        Returns
        -------
        PangenomeResult
            Dataclass carrying the matrix, partitioned indices, accessory
            GeneRecord list, and summary statistics.
        """
        # Ensure all steps have run
        if self._matrix is None:
            self.build_presence_absence_matrix()

        core_genes, accessory_genes, shell_genes = self.partition_pangenome()

        # Collect all GeneRecord objects belonging to accessory clusters
        accessory_set = set(accessory_genes)
        accessory_records: List[GeneRecord] = [
            rec
            for rec in self._all_records
            if self._cluster_map.get(rec.gene_id) in accessory_set
        ]

        stats = {
            "n_strains": len(self._strain_ids),
            "n_total_clusters": len(self._matrix),
            "n_core": len(core_genes),
            "n_shell": len(shell_genes),
            "n_accessory": len(accessory_genes),
            "n_accessory_records": len(accessory_records),
        }

        logger.info(
            "Phase 1 complete. Passing %d accessory gene coordinates to Phase 2.",
            len(accessory_records),
        )

        return PangenomeResult(
            presence_absence_matrix=self._matrix,
            core_genes=core_genes,
            accessory_genes=accessory_genes,
            accessory_records=accessory_records,
            strain_ids=self._strain_ids,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Convenience: run full Phase 1 from directory paths
    # ------------------------------------------------------------------
    def run(
        self,
        genomes_dir: str | Path,
        annotations_dir: str | Path,
    ) -> PangenomeResult:
        """
        Convenience method: ingest → cluster → partition → extract in one call.

        Parameters
        ----------
        genomes_dir : path to directory with .fasta / .fna / .fa files
        annotations_dir : path to directory with .gff / .gff3 files

        Returns
        -------
        PangenomeResult
        """
        return (
            self.load_genomes(genomes_dir, annotations_dir)
            .build_presence_absence_matrix()
            .extract_accessory_coordinates()
        )

    # ------------------------------------------------------------------
    # Matrix persistence helpers
    # ------------------------------------------------------------------
    def save_matrix(self, output_path: str | Path) -> None:
        """
        Persist the presence/absence matrix to a CSV file.
        Useful for downstream inspection in R (vegan, ape).
        """
        if self._matrix is None:
            raise RuntimeError(
                "No matrix to save. Run build_presence_absence_matrix() first."
            )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._matrix.to_csv(out)
        logger.info("Presence/absence matrix saved → %s", out)

    def load_matrix(self, input_path: str | Path) -> "PangenomeMiner":
        """Load a pre-computed presence/absence matrix from CSV (skip re-clustering)."""
        inp = Path(input_path)
        if not inp.exists():
            raise FileNotFoundError(f"Matrix file not found: {inp}")
        self._matrix = pd.read_csv(inp, index_col=0)
        # Restore strain list from columns
        self._strain_ids = list(self._matrix.columns)
        logger.info(
            "Pre-computed matrix loaded: %d clusters × %d strains",
            self._matrix.shape[0],
            self._matrix.shape[1],
        )
        return self
