"""
PanAdapt-BGC Miner — Phase 1 Tests
====================================
Module  : tests/test_pangenome_miner.py
Purpose : Unit tests for the PangenomeMiner class using synthetic mock data.

Run with:
    pytest tests/test_pangenome_miner.py -v
"""

from __future__ import annotations


import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import random
import pytest

# Make sure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pangenome_miner import (
    GeneRecord,
    PangenomeMiner,
    PangenomeResult,
    _extract_gene_id,
    _parse_gff_attributes,
    _sequence_for_region,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def tmp_dirs():
    """Create temporary genome + annotation directories, return (genomes_dir, annot_dir)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "genomes").mkdir()
        (root / "annotations").mkdir()
        yield root / "genomes", root / "annotations"


def _write_fasta(path: Path, contig_id: str, sequence: str) -> None:
    with path.open("w") as fh:
        fh.write(f">{contig_id}\n")
        for i in range(0, len(sequence), 60):
            fh.write(sequence[i: i + 60] + "\n")


def _write_gff(path: Path, records: list[tuple]) -> None:
    """records: list of (seqid, feat_type, start, end, strand, gene_id, product)"""
    with path.open("w") as fh:
        fh.write("##gff-version 3\n")
        for seqid, feat_type, start, end, strand, gene_id, product in records:
            fh.write(
                f"{seqid}\tmock\t{feat_type}\t{start}\t{end}\t.\t{strand}\t0\t"
                f"ID={gene_id};product={product}\n"
            )


# ===========================================================================
# Helper / util tests
# ===========================================================================

class TestParseGffAttributes:
    def test_standard_gff3(self):
        attrs = _parse_gff_attributes("ID=gene_001;product=hypothetical protein;Name=geneA")
        assert attrs["ID"] == "gene_001"
        assert attrs["product"] == "hypothetical protein"
        assert attrs["Name"] == "geneA"

    def test_trailing_semicolon(self):
        attrs = _parse_gff_attributes("ID=x;locus_tag=LT_01;")
        assert attrs["locus_tag"] == "LT_01"

    def test_empty_string(self):
        attrs = _parse_gff_attributes("")
        assert attrs == {}


class TestExtractGeneId:
    def test_prefers_ID(self):
        assert _extract_gene_id({"ID": "abc", "locus_tag": "lt"}) == "abc"

    def test_falls_back_to_locus_tag(self):
        assert _extract_gene_id({"locus_tag": "LT_001"}) == "LT_001"

    def test_generates_hash_when_no_known_key(self):
        gid = _extract_gene_id({"random_key": "val"})
        assert gid.startswith("gene_")


# ===========================================================================
# PangenomeMiner: initialisation tests
# ===========================================================================

class TestPangenomeMinerInit:
    def test_valid_thresholds(self):
        miner = PangenomeMiner(core_threshold=0.9, accessory_threshold=0.1)
        assert miner.core_threshold == 0.9

    def test_invalid_core_threshold(self):
        with pytest.raises(ValueError, match="core_threshold"):
            PangenomeMiner(core_threshold=1.5)

    def test_invalid_accessory_threshold(self):
        with pytest.raises(ValueError, match="accessory_threshold"):
            PangenomeMiner(core_threshold=0.9, accessory_threshold=0.95)

    def test_accessory_must_be_less_than_core(self):
        with pytest.raises(ValueError):
            PangenomeMiner(core_threshold=0.5, accessory_threshold=0.6)


# ===========================================================================
# Integration test with synthetic data
# ===========================================================================

class TestPangenomeMinerIntegration:
    """End-to-end test using three synthetic strains."""

    CONTIG_ID = "ctg1"
    CONTIG_ID = "ctg1"

    # Gene definitions: (start, end, strand, gene_id, product, shared_in)
    # shared_in = list of strain names that carry this gene
    GENE_DEFS = [
        (100, 399, "+", "shared_gene_1", "hypothetical protein", ["A", "B", "C"]),
        (500, 799, "-", "shared_gene_2", "methyltransferase",    ["A", "B", "C"]),
        (900, 1199, "+", "shared_gene_3", "permease",            ["A", "B", "C"]),
        (1300, 1599, "+", "unique_A_1",   "transposase",         ["A"]),
        (1700, 1999, "+", "unique_A_2",   "putative BGC gene",   ["A"]),
        (2100, 2399, "-", "shared_AB",    "integrase",           ["A", "B"]),
    ]

    def _build_data(self, tmp_dirs):
        genomes_dir, annot_dir = tmp_dirs
        strains = ["A", "B", "C"]

        # Build a genome where each gene slot holds a DISTINCT random sequence
        # so k-mer clustering treats them as separate orthogroups.
        rng = random.Random(0)
        def rand_seq(length: int) -> str:
            return "".join(rng.choices("ACGT", k=length))

        # Assign unique nucleotide templates per gene_id
        # Shared genes get the *same* template across strains (minor mutations ok
        # because identity_threshold=0.70, but we skip mutations here for clarity).
        gene_seqs = {
            gene_id: rand_seq(end - start + 1)
            for start, end, strand, gene_id, product, shared_in in self.GENE_DEFS
        }

        # Build a 20 kbp background genome (low-complexity repetition replaced
        # with random sequence to avoid spurious k-mer matches).
        GENOME_LEN = 20_000

        for strain in strains:
            genome = list(rand_seq(GENOME_LEN))
            gff_records = []
            for start, end, strand, gene_id, product, shared_in in self.GENE_DEFS:
                if strain not in shared_in:
                    continue
                # Embed gene-specific sequence into the contig
                seq = gene_seqs[gene_id]
                genome[start - 1: end] = list(seq)
                gff_records.append(
                    (self.CONTIG_ID, "CDS", start, end, strand, gene_id, product)
                )
            _write_fasta(genomes_dir / f"strain_{strain}.fasta", self.CONTIG_ID, "".join(genome))
            _write_gff(annot_dir / f"strain_{strain}.gff", gff_records)

        return genomes_dir, annot_dir

    def test_run_returns_pangenome_result(self, tmp_dirs):
        genomes_dir, annot_dir = self._build_data(tmp_dirs)
        miner = PangenomeMiner(
            core_threshold=0.95,
            accessory_threshold=0.40,
            identity_threshold=0.70,
        )
        result = miner.run(genomes_dir, annot_dir)
        assert isinstance(result, PangenomeResult)

    def test_matrix_has_correct_dimensions(self, tmp_dirs):
        genomes_dir, annot_dir = self._build_data(tmp_dirs)
        miner = PangenomeMiner(core_threshold=0.95, accessory_threshold=0.40, identity_threshold=0.70)
        result = miner.run(genomes_dir, annot_dir)
        # 3 strains in columns
        assert set(result.presence_absence_matrix.columns) == {"strain_A", "strain_B", "strain_C"}
        assert len(result.presence_absence_matrix) > 0

    def test_core_genes_identified(self, tmp_dirs):
        genomes_dir, annot_dir = self._build_data(tmp_dirs)
        miner = PangenomeMiner(core_threshold=0.95, accessory_threshold=0.40, identity_threshold=0.70)
        result = miner.run(genomes_dir, annot_dir)
        # shared_gene_1/2/3 are present in 100% strains → must be core
        assert len(result.core_genes) >= 3

    def test_accessory_genes_are_strain_specific(self, tmp_dirs):
        genomes_dir, annot_dir = self._build_data(tmp_dirs)
        miner = PangenomeMiner(core_threshold=0.95, accessory_threshold=0.40, identity_threshold=0.70)
        result = miner.run(genomes_dir, annot_dir)
        # unique_A_1 and unique_A_2 are only in strain_A (33%) — accessory
        assert len(result.accessory_genes) >= 2

    def test_accessory_records_are_gene_record_objects(self, tmp_dirs):
        genomes_dir, annot_dir = self._build_data(tmp_dirs)
        miner = PangenomeMiner(core_threshold=0.95, accessory_threshold=0.40, identity_threshold=0.70)
        result = miner.run(genomes_dir, annot_dir)
        for rec in result.accessory_records:
            assert isinstance(rec, GeneRecord)
            assert rec.start < rec.end
            assert rec.strand in ("+", "-", ".")

    def test_stats_keys_present(self, tmp_dirs):
        genomes_dir, annot_dir = self._build_data(tmp_dirs)
        miner = PangenomeMiner(core_threshold=0.95, accessory_threshold=0.40, identity_threshold=0.70)
        result = miner.run(genomes_dir, annot_dir)
        for key in ("n_strains", "n_total_clusters", "n_core", "n_shell", "n_accessory"):
            assert key in result.stats

    def test_save_and_reload_matrix(self, tmp_dirs):
        genomes_dir, annot_dir = self._build_data(tmp_dirs)
        miner = PangenomeMiner(core_threshold=0.95, accessory_threshold=0.40, identity_threshold=0.70)
        result = miner.run(genomes_dir, annot_dir)

        out_csv = genomes_dir.parent / "matrix.csv"
        miner.save_matrix(out_csv)
        assert out_csv.exists()

        miner2 = PangenomeMiner()
        miner2.load_matrix(out_csv)
        pd.testing.assert_frame_equal(result.presence_absence_matrix, miner2._matrix)

    def test_missing_gff_raises_error(self, tmp_dirs):
        genomes_dir, annot_dir = tmp_dirs
        _write_fasta(genomes_dir / "strain_X.fasta", "ctg1", "ACGT" * 100)
        # No corresponding GFF → no strains loaded → RuntimeError
        miner = PangenomeMiner()
        with pytest.raises((RuntimeError, ValueError)):
            miner.run(genomes_dir, annot_dir)


# ===========================================================================
# Test rarefaction and Heaps' Law calculations
# ===========================================================================

class TestPangenomeAnalytics:
    """Test advanced analytics: rarefaction curves and Heaps' Law."""

    def test_calculate_rarefaction(self, tmp_dirs):
        """Test rarefaction curve calculation."""
        genomes_dir, annot_dir = tmp_dirs
        strains = ["A", "B", "C"]
        
        # Build simple data
        CONTIG_ID = "ctg1"
        GENE_DEFS = [
            (100, 399, "+", "gene1", "protein1", strains),  # Present in all
            (500, 799, "-", "gene2", "protein2", ["A", "B"]),  # Present in A, B
            (900, 1199, "+", "gene3", "protein3", ["C"]),  # Present in C only
        ]
        
        from pathlib import Path
        import random
        rng = random.Random(42)
        rand_seq = lambda l: "".join(rng.choices("ACGT", k=l))
        
        for strain in strains:
            genome = list(rand_seq(2000))
            gff_records = []
            for start, end, strand, gene_id, product, shared_in in GENE_DEFS:
                if strain not in shared_in:
                    continue
                seq = rand_seq(end - start + 1)
                genome[start - 1:end] = list(seq)
                gff_records.append((CONTIG_ID, "CDS", start, end, strand, gene_id, product))
            
            _write_fasta(genomes_dir / f"strain_{strain}.fasta", CONTIG_ID, "".join(genome))
            _write_gff(annot_dir / f"strain_{strain}.gff", gff_records)
        
        miner = PangenomeMiner(core_threshold=0.9, accessory_threshold=0.1)
        miner.run(genomes_dir, annot_dir)
        
        # Test rarefaction calculation
        rarefaction = miner.calculate_rarefaction(iterations=5)
        assert len(rarefaction) == 3  # 3 strains
        assert all(rarefaction > 0)
        # Rarefaction should be non-decreasing
        assert all(rarefaction[i] <= rarefaction[i+1] for i in range(len(rarefaction)-1))

    def test_heaps_law_calculation(self, tmp_dirs):
        """Test Heaps' Law calculation."""
        genomes_dir, annot_dir = tmp_dirs
        strains = ["A", "B", "C"]
        
        CONTIG_ID = "ctg1"
        # Define genes to ensure pangenome is "open"
        GENE_DEFS = [
            (100, 399, "+", "gene1", "protein1", strains),
            (500, 799, "-", "gene2", "protein2", ["A", "B"]),
            (900, 1199, "+", "gene3", "protein3", ["A", "C"]),
            (1300, 1599, "+", "gene4", "protein4", ["B"]),
            (1700, 1999, "+", "gene5", "protein5", ["C"]),
        ]
        
        from pathlib import Path
        import random
        rng = random.Random(42)
        rand_seq = lambda l: "".join(rng.choices("ACGT", k=l))
        
        for strain in strains:
            genome = list(rand_seq(3000))
            gff_records = []
            for start, end, strand, gene_id, product, shared_in in GENE_DEFS:
                if strain not in shared_in:
                    continue
                seq = rand_seq(end - start + 1)
                genome[start - 1:end] = list(seq)
                gff_records.append((CONTIG_ID, "CDS", start, end, strand, gene_id, product))
            
            _write_fasta(genomes_dir / f"strain_{strain}.fasta", CONTIG_ID, "".join(genome))
            _write_gff(annot_dir / f"strain_{strain}.gff", gff_records)
        
        miner = PangenomeMiner(core_threshold=0.9, accessory_threshold=0.1)
        miner.run(genomes_dir, annot_dir)
        
        # Check Heaps' Law stats
        assert "heaps_law" in miner.stats
        heaps = miner.stats["heaps_law"]
        assert "k" in heaps
        assert "beta" in heaps
        assert "open" in heaps
        assert isinstance(heaps["k"], (int, float))
        assert isinstance(heaps["beta"], (int, float))
        assert isinstance(heaps["open"], (bool, np.bool_))

    def test_heaps_law_failure_handling(self, tmp_dirs, monkeypatch):
        """Test that Heaps' Law failures are handled gracefully."""
        genomes_dir, annot_dir = tmp_dirs
        strains = ["A", "B"]
        
        CONTIG_ID = "ctg1"
        GENE_DEFS = [
            (100, 399, "+", "gene1", "protein1", strains),
        ]
        
        from pathlib import Path
        import random
        rng = random.Random(42)
        rand_seq = lambda l: "".join(rng.choices("ACGT", k=l))
        
        for strain in strains:
            genome = list(rand_seq(1000))
            gff_records = []
            for start, end, strand, gene_id, product, shared_in in GENE_DEFS:
                if strain not in shared_in:
                    continue
                seq = rand_seq(end - start + 1)
                genome[start - 1:end] = list(seq)
                gff_records.append((CONTIG_ID, "CDS", start, end, strand, gene_id, product))
            
            _write_fasta(genomes_dir / f"strain_{strain}.fasta", CONTIG_ID, "".join(genome))
            _write_gff(annot_dir / f"strain_{strain}.gff", gff_records)
        
        # Force curve_fit to raise an exception to test error handling
        def mock_curve_fit(*args, **kwargs):
            raise RuntimeError("Simulated failure")
        
        monkeypatch.setattr("scipy.optimize.curve_fit", mock_curve_fit)
        
        miner = PangenomeMiner(core_threshold=0.9, accessory_threshold=0.1)
        miner.run(genomes_dir, annot_dir)
        
        # Even if curve_fit fails, we should still have stats with defaults
        assert "heaps_law" in miner.stats
        heaps = miner.stats["heaps_law"]
        assert heaps["k"] == 0.0
        assert heaps["beta"] == 0.0
        assert heaps["open"] is None
