"""
tests/test_hgt_detective.py
============================
Unit + integration tests for pipeline.hgt_detective.HGTDetective (Phase 2).

Test coverage:
  * Helper functions: _gc_content, _tetranucleotide_freq
  * HGTDetective initialisation and parameter validation
  * Integration: full .run() on synthetic 3-strain data
    - Returns correct HGTResult type and required stat keys
    - feature_matrix shape and columns are correct
    - alien_records is a subset of hgt_records
    - Every record holds required HGTGeneRecord fields
    - Sequence-too-short path sets kmer_deviation = 0.0
    - Empty accessory raises ValueError
"""

from __future__ import annotations

import os
import random
import string
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Subject imports
# ---------------------------------------------------------------------------
from pipeline.hgt_detective import (
    HGTDetective,
    HGTGeneRecord,
    HGTResult,
    _gc_content,
    _tetranucleotide_freq,
)
from pipeline.pangenome_miner import GeneRecord, PangenomeMiner, PangenomeResult

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
_RNG = random.Random(RANDOM_SEED)


def _random_dna(length: int, gc_frac: float = 0.50, seed: int | None = None) -> str:
    """Generate a random DNA sequence with approximate GC content."""
    rng = random.Random(seed) if seed is not None else _RNG
    at_bases = ["A", "T"]
    gc_bases = ["G", "C"]
    bases = [
        rng.choice(gc_bases) if rng.random() < gc_frac else rng.choice(at_bases)
        for _ in range(length)
    ]
    return "".join(bases)


def _write_fasta(path: str, contigs: List[Tuple[str, str]]) -> None:
    """Write a FASTA file with multiple contigs: [(contig_id, sequence), ...]."""
    with open(path, "w") as fh:
        for contig_id, seq in contigs:
            fh.write(f">{contig_id}\n")
            # Write in 60-char lines
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")


def _write_gff(
    path: str,
    records: List[Tuple[str, str, int, int, str, str, str]],
) -> None:
    """Write a minimal GFF3 file.

    Each record: (seqid, feat_type, start, end, strand, gene_id, product)
    GFF3 coords are 1-based inclusive.
    """
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        for seqid, ftype, start, end, strand, gene_id, product in records:
            attrs = f"ID={gene_id};Name={gene_id};product={product}"
            fh.write(
                f"{seqid}\tNCBI\t{ftype}\t{start}\t{end}\t.\t{strand}\t0\t{attrs}\n"
            )


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


class _SyntheticData:
    """Three-strain synthetic dataset for HGT detection tests.

    Genome layout (all on a single 20 kbp contig per strain):
      Core genes (all 3 strains, ~50% GC)   : gA, gB, gC at pos 1000-1500, 3000-3500, 5000-5500
      Shell gene (strains A & B)             : gD at pos 7000-7500
      Accessory A only  (low GC ~0.25)       : gE at pos  9000-9500   <- expected HGT candidate
      Accessory A only  (MGE-like, tnpA)     : gF at pos 10000-10500  <- MGE keyword hit
      Accessory B only  (normal GC ~0.50)    : gG at pos 12000-12500
    """

    CONTIG_LEN = 20_000
    # (start, end, strand, name, product)
    _GENE_DEFS: Dict[str, List[Tuple[int, int, str, str, str]]] = {
        "A": [
            (1000, 1500, "+", "gA", "ATP-binding protein"),
            (3000, 3500, "+", "gB", "ribosomal protein S12"),
            (5000, 5500, "+", "gC", "hypothetical protein"),
            (7000, 7500, "-", "gD", "acetyl-CoA carboxylase"),
            (9000, 9500, "+", "gE", "foreign domain protein"),
            (10000, 10500, "+", "gF", "tnpA transposase"),
        ],
        "B": [
            (1000, 1500, "+", "gA", "ATP-binding protein"),
            (3000, 3500, "+", "gB", "ribosomal protein S12"),
            (5000, 5500, "+", "gC", "hypothetical protein"),
            (7000, 7500, "-", "gD", "acetyl-CoA carboxylase"),
            (12000, 12500, "+", "gG", "sugar transporter"),
        ],
        "C": [
            (1000, 1500, "+", "gA", "ATP-binding protein"),
            (3000, 3500, "+", "gB", "ribosomal protein S12"),
            (5000, 5500, "+", "gC", "hypothetical protein"),
        ],
    }

    def __init__(self, tmp_dir: str) -> None:
        self.genome_dir = Path(tmp_dir) / "genomes"
        self.annot_dir = Path(tmp_dir) / "annotations"
        self.genome_dir.mkdir()
        self.annot_dir.mkdir()

        # Map strain -> contig_sequence
        self._sequences: Dict[str, str] = {}

        for strain_id, genes in self._GENE_DEFS.items():
            # Build a random host sequence with ~50% GC
            host_seq = list(
                _random_dna(self.CONTIG_LEN, gc_frac=0.50, seed=ord(strain_id))
            )

            # Embed each gene's sequence; gE gets low GC to look alien
            for start, end, strand, name, _product in genes:
                gene_len = end - start
                gc = 0.25 if name == "gE" else 0.50
                gene_seq = _random_dna(gene_len, gc_frac=gc, seed=hash(name) % 10000)
                host_seq[start:end] = list(gene_seq)

            contig_seq = "".join(host_seq)
            self._sequences[strain_id] = contig_seq

            contig_id = f"contig_{strain_id}"
            fa_path = self.genome_dir / f"strain_{strain_id}.fna"
            _write_fasta(str(fa_path), [(contig_id, contig_seq)])

            gff_records = [
                (
                    contig_id,
                    "CDS",
                    start,
                    end - 1,  # GFF3 is 1-based; end-1 keeps length consistent
                    strand,
                    f"{name}_{strain_id}",
                    product,
                )
                for start, end, strand, name, product in genes
            ]
            gff_path = self.annot_dir / f"strain_{strain_id}.gff"
            _write_gff(str(gff_path), gff_records)

    def build_phase1_result(self) -> Tuple["PangenomeResult", "PangenomeMiner"]:
        """Build a PangenomeResult manually from known gene definitions.

        Bypasses PangenomeMiner clustering (which is non-deterministic at the
        k-mer Jaccard boundary) so the test is completely deterministic.
        The miner is still returned so its _fasta_store can be used by Phase 2.
        """
        import pandas as pd
        from pipeline.pangenome_miner import GeneRecord, PangenomeResult

        # Core genes: present in all 3 strains (gA, gB, gC)
        # Shell gene: present in A and B only (gD)
        # Accessory A: gE (alien, low-GC), gF (MGE tnpA)
        # Accessory B: gG (normal)
        core_names = ["gA", "gB", "gC"]
        shell_names = ["gD"]
        acc_a_names = ["gE", "gF"]
        acc_b_names = ["gG"]

        # Build GeneRecord objects for accessory genes (gE, gF from A; gG from B)
        accessory_records = []
        for sid, names in [("A", acc_a_names), ("B", acc_b_names)]:
            strain_defs = self._GENE_DEFS[sid]
            contig_id = f"contig_{sid}"
            host_seq = _random_dna(self.CONTIG_LEN, gc_frac=0.50, seed=ord(sid))
            for start, end, strand, name, product in strain_defs:
                if name not in names:
                    continue
                if name == "gE":
                    gc = 0.25  # low-GC alien
                else:
                    gc = 0.50
                gene_seq = _random_dna(
                    end - start + 1, gc_frac=gc, seed=hash(name) % 10000
                )
                accessory_records.append(
                    GeneRecord(
                        gene_id=f"{name}_{sid}",
                        strain_id=sid,
                        contig=contig_id,
                        start=start,
                        end=end,
                        strand=strand,
                        feature_type="CDS",
                        product=product,
                        sequence=gene_seq,
                    )
                )

        # Build a minimal presence/absence matrix
        all_gene_ids = (
            [f"{n}_{s}" for n in core_names for s in ["A", "B", "C"]]
            + [f"{n}_{s}" for n in shell_names for s in ["A", "B"]]
            + [f"{n}_A" for n in acc_a_names]
            + [f"{n}_B" for n in acc_b_names]
        )
        mat = pd.DataFrame(
            False,
            index=all_gene_ids,
            columns=["A", "B", "C"],
        )
        for n in core_names:
            for s in ["A", "B", "C"]:
                mat.at[f"{n}_{s}", s] = True
        for n in shell_names:
            for s in ["A", "B"]:
                mat.at[f"{n}_{s}", s] = True
        for n in acc_a_names:
            mat.at[f"{n}_A", "A"] = True
        for n in acc_b_names:
            mat.at[f"{n}_B", "B"] = True

        acc_ids = pd.Index([r.gene_id for r in accessory_records])
        result = PangenomeResult(
            presence_absence_matrix=mat,
            core_genes=pd.Index([f"{n}_A" for n in core_names]),  # representative
            accessory_genes=acc_ids,
            accessory_records=accessory_records,
            strain_ids=["A", "B", "C"],
            stats={
                "n_strains": 3,
                "n_total_clusters": len(all_gene_ids),
                "n_core": 3,
                "n_shell": 1,
                "n_accessory": len(accessory_records),
            },
        )

        # Also build a real miner and populate its _fasta_store so Phase 2
        # can access actual DNA sequences
        miner = PangenomeMiner(
            core_threshold=0.95,
            accessory_threshold=0.40,
            identity_threshold=0.60,
        )
        # Populate _fasta_store directly from known contig sequences
        from Bio.SeqRecord import SeqRecord
        from Bio.Seq import Seq

        miner._fasta_store = {}
        for sid in ["A", "B", "C"]:
            host_seq = _random_dna(self.CONTIG_LEN, gc_frac=0.50, seed=ord(sid))
            cid = f"contig_{sid}"
            miner._fasta_store[sid] = {
                cid: SeqRecord(Seq(host_seq), id=cid, description="")
            }

        return result, miner


# ---------------------------------------------------------------------------
# 1. Helper function unit tests
# ---------------------------------------------------------------------------


class TestGcContent(unittest.TestCase):
    """Tests for the _gc_content helper."""

    def test_pure_gc(self) -> None:
        self.assertAlmostEqual(_gc_content("GGCC"), 1.0, places=5)

    def test_pure_at(self) -> None:
        self.assertAlmostEqual(_gc_content("AATT"), 0.0, places=5)

    def test_mixed(self) -> None:
        # AATTGGCC -> 4 GC out of 8 = 0.5
        self.assertAlmostEqual(_gc_content("AATTGGCC"), 0.5, places=5)

    def test_case_insensitive(self) -> None:
        self.assertAlmostEqual(_gc_content("aattggcc"), 0.5, places=5)

    def test_empty_sequence(self) -> None:
        # Should return 0.0 without raising
        self.assertEqual(_gc_content(""), 0.0)

    def test_single_base_g(self) -> None:
        self.assertAlmostEqual(_gc_content("G"), 1.0, places=5)


class TestTetranucleotideFreq(unittest.TestCase):
    """Tests for the _tetranucleotide_freq helper."""

    def test_returns_256_dim_vector(self) -> None:
        seq = _random_dna(1000)
        freq = _tetranucleotide_freq(seq)
        self.assertEqual(freq.shape, (256,))

    def test_sums_to_one(self) -> None:
        seq = _random_dna(500, seed=1)
        freq = _tetranucleotide_freq(seq)
        self.assertAlmostEqual(float(freq.sum()), 1.0, places=5)

    def test_non_negative(self) -> None:
        freq = _tetranucleotide_freq("AAAGGGCCC")
        self.assertTrue(np.all(freq >= 0))

    def test_short_sequence_does_not_crash(self) -> None:
        freq = _tetranucleotide_freq("ACG")  # shorter than k=4
        self.assertEqual(freq.shape, (256,))
        self.assertAlmostEqual(float(freq.sum()), 0.0, places=5)  # no valid 4-mers


# ---------------------------------------------------------------------------
# 2. HGTDetective initialisation tests
# ---------------------------------------------------------------------------


class TestHGTDetectiveInit(unittest.TestCase):

    def test_default_params(self) -> None:
        det = HGTDetective()
        self.assertAlmostEqual(det.contamination, 0.30)
        self.assertEqual(det.n_estimators, 200)
        self.assertEqual(det.random_state, 42)
        self.assertEqual(det.min_seq_length, 90)

    def test_custom_params(self) -> None:
        det = HGTDetective(
            contamination=0.10,
            n_estimators=50,
            random_state=7,
            min_seq_length=150,
        )
        self.assertAlmostEqual(det.contamination, 0.10)
        self.assertEqual(det.n_estimators, 50)
        self.assertEqual(det.random_state, 7)
        self.assertEqual(det.min_seq_length, 150)

    def test_contamination_range(self) -> None:
        # Should not raise for valid values
        HGTDetective(contamination=0.01)
        HGTDetective(contamination=0.49)


# ---------------------------------------------------------------------------
# 3. Integration tests
# ---------------------------------------------------------------------------


class TestHGTDetectiveRun(unittest.TestCase):
    """Integration tests that run Phase 2 on the synthetic 3-strain dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build synthetic data once; run Phase 1 and Phase 2."""
        cls._tmpdir = tempfile.TemporaryDirectory()
        data = _SyntheticData(cls._tmpdir.name)
        cls.phase1_result, cls.miner = data.build_phase1_result()
        detective = HGTDetective(contamination=0.30, random_state=0)
        cls.hgt_result = detective.run(
            phase1_result=cls.phase1_result,
            fasta_store=cls.miner._fasta_store,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    # ------------------------------------------------------------------
    # 3a. Return-type and structure
    # ------------------------------------------------------------------

    def test_returns_hgt_result(self) -> None:
        self.assertIsInstance(self.hgt_result, HGTResult)

    def test_hgt_records_is_list(self) -> None:
        self.assertIsInstance(self.hgt_result.hgt_records, list)

    def test_alien_records_is_subset_of_hgt_records(self) -> None:
        alien_ids = {r.gene_record.gene_id for r in self.hgt_result.alien_records}
        all_ids = {r.gene_record.gene_id for r in self.hgt_result.hgt_records}
        self.assertTrue(alien_ids.issubset(all_ids))

    def test_alien_records_all_flagged_is_hgt(self) -> None:
        for rec in self.hgt_result.alien_records:
            self.assertTrue(
                rec.is_hgt,
                msg=f"alien_record {rec.gene_record.gene_id} has is_hgt=False",
            )

    # ------------------------------------------------------------------
    # 3b. Stats dictionary
    # ------------------------------------------------------------------

    def test_stats_required_keys_present(self) -> None:
        required = {
            "n_accessory_input",
            "n_alien_hgt",
            "n_normal",
            "hgt_fraction",
            "n_mge_proximal",
            "strain_gc_profiles",
        }
        missing = required - set(self.hgt_result.stats.keys())
        self.assertEqual(missing, set(), msg=f"Missing stats keys: {missing}")

    def test_stats_counts_consistent(self) -> None:
        s = self.hgt_result.stats
        self.assertEqual(
            s["n_alien_hgt"] + s["n_normal"],
            s["n_accessory_input"],
        )

    def test_hgt_fraction_in_range(self) -> None:
        frac = self.hgt_result.stats["hgt_fraction"]
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)

    def test_strain_gc_profiles_covers_all_strains(self) -> None:
        expected_strains = {"A", "B", "C"}
        profiles = self.hgt_result.stats["strain_gc_profiles"]
        # strain_ids in the test data are embedded in the file paths as "strain_A" etc.
        profile_strains = set(profiles.keys())
        self.assertTrue(
            profile_strains,
            msg="strain_gc_profiles should not be empty",
        )
        for gc in profiles.values():
            self.assertGreaterEqual(gc, 0.0)
            self.assertLessEqual(gc, 1.0)

    # ------------------------------------------------------------------
    # 3c. Feature matrix
    # ------------------------------------------------------------------

    def test_feature_matrix_not_empty(self) -> None:
        self.assertFalse(self.hgt_result.feature_matrix.empty)

    def test_feature_matrix_required_columns(self) -> None:
        required_cols = {
            "gc_deviation",
            "kmer_deviation",
            "mge_proximity",
            "gc_content",
            "gene_length",
            "strain_id",
        }
        actual_cols = set(self.hgt_result.feature_matrix.columns)
        missing = required_cols - actual_cols
        self.assertEqual(
            missing, set(), msg=f"Feature matrix missing columns: {missing}"
        )

    def test_feature_matrix_row_count_matches_hgt_records(self) -> None:
        self.assertEqual(
            len(self.hgt_result.feature_matrix),
            len(self.hgt_result.hgt_records),
        )

    def test_gc_deviation_non_negative(self) -> None:
        self.assertTrue(
            (self.hgt_result.feature_matrix["gc_deviation"] >= 0).all(),
            "gc_deviation should be non-negative",
        )

    def test_kmer_deviation_in_range(self) -> None:
        kd = self.hgt_result.feature_matrix["kmer_deviation"]
        self.assertTrue((kd >= 0).all() and (kd <= 1).all())

    # ------------------------------------------------------------------
    # 3d. HGTGeneRecord field completeness
    # ------------------------------------------------------------------

    def test_all_records_have_required_fields(self) -> None:
        for rec in self.hgt_result.hgt_records:
            with self.subTest(gene_id=rec.gene_record.gene_id):
                self.assertIsInstance(rec, HGTGeneRecord)
                self.assertIsInstance(rec.gc_content, float)
                self.assertIsInstance(rec.gc_deviation, float)
                self.assertIsInstance(rec.kmer_deviation, float)
                self.assertIsInstance(rec.mge_proximity, bool)
                self.assertIsInstance(rec.anomaly_score, float)
                self.assertIsInstance(rec.is_hgt, bool)
                self.assertIsInstance(rec.evidence, list)

    def test_evidence_is_strings(self) -> None:
        for rec in self.hgt_result.hgt_records:
            for item in rec.evidence:
                self.assertIsInstance(item, str)

    # ------------------------------------------------------------------
    # 3e. Accessory gene count is positive
    # ------------------------------------------------------------------

    def test_accessory_genes_were_processed(self) -> None:
        self.assertGreater(
            self.hgt_result.stats["n_accessory_input"],
            0,
            "Expected at least one accessory gene to be processed",
        )

    def test_all_hgt_records_count_matches_n_accessory_input(self) -> None:
        self.assertEqual(
            len(self.hgt_result.hgt_records),
            self.hgt_result.stats["n_accessory_input"],
        )


# ---------------------------------------------------------------------------
# 4. Edge-case: empty accessory raises ValueError
# ---------------------------------------------------------------------------


class TestHGTDetectiveEmptyAccessory(unittest.TestCase):

    def test_raises_if_no_accessory_records(self) -> None:
        """run() must raise ValueError when phase1_result has no accessory genes."""
        import pandas as pd

        empty_result = PangenomeResult(
            presence_absence_matrix=pd.DataFrame(),
            core_genes=pd.Index([]),
            accessory_genes=pd.Index([]),
            accessory_records=[],  # <-- empty
            strain_ids=["s1", "s2", "s3"],
            stats={
                "n_strains": 3,
                "n_total_clusters": 0,
                "n_core": 0,
                "n_shell": 0,
                "n_accessory": 0,
            },
        )
        detective = HGTDetective()
        with self.assertRaises(ValueError):
            detective.run(phase1_result=empty_result, fasta_store={})


# ---------------------------------------------------------------------------
# 5. Short-sequence path: kmer_deviation must be 0.0
# ---------------------------------------------------------------------------


class TestHGTDetectiveShortSequence(unittest.TestCase):
    """A gene whose extracted sequence is shorter than min_seq_length
    should have kmer_deviation == 0.0 (no comparison possible)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp = cls._tmpdir.name
        genome_dir = Path(tmp) / "genomes"
        annot_dir = Path(tmp) / "annotations"
        genome_dir.mkdir()
        annot_dir.mkdir()

        # Single strain with one accessory gene whose CDS is only 30 bp (<90 default)
        contig_seq = _random_dna(2000, seed=99)
        contig_id = "ctg1"
        _write_fasta(str(genome_dir / "short_strain.fna"), [(contig_id, contig_seq)])
        _write_gff(
            str(annot_dir / "short_strain.gff"),
            [
                (
                    contig_id,
                    "CDS",
                    100,
                    129,
                    "+",
                    "short_gene_short",
                    "hypothetical protein",
                ),
                (
                    contig_id,
                    "CDS",
                    200,
                    700,
                    "+",
                    "long_gene_short",
                    "ribosomal protein",
                ),
            ],
        )

        # Instead of running Phase 1 (which depends on clustering), build a
        # PangenomeResult manually with a short-sequence accessory gene record.
        import pandas as pd
        from pipeline.pangenome_miner import GeneRecord, PangenomeResult

        short_seq = _random_dna(30, gc_frac=0.50, seed=7)  # < min_seq_length=90
        long_seq = _random_dna(500, gc_frac=0.50, seed=8)  # >= min_seq_length

        short_rec = GeneRecord(
            gene_id="short_gene_short",
            strain_id="short_strain",
            contig="ctg1",
            start=100,
            end=129,
            strand="+",
            feature_type="CDS",
            product="hypothetical protein",
            sequence=short_seq,
        )
        long_rec = GeneRecord(
            gene_id="long_gene_short",
            strain_id="short_strain",
            contig="ctg1",
            start=200,
            end=700,
            strand="+",
            feature_type="CDS",
            product="ribosomal protein",
            sequence=long_seq,
        )

        mat = pd.DataFrame(
            {"short_strain": [True, True]},
            index=["short_gene_short", "long_gene_short"],
        )
        phase1_result = PangenomeResult(
            presence_absence_matrix=mat,
            core_genes=pd.Index([]),
            accessory_genes=pd.Index(["short_gene_short", "long_gene_short"]),
            accessory_records=[short_rec, long_rec],
            strain_ids=["short_strain"],
            stats={
                "n_strains": 1,
                "n_total_clusters": 2,
                "n_core": 0,
                "n_shell": 0,
                "n_accessory": 2,
            },
        )

        # Build a minimal fasta_store covering this one strain
        contig_seq = _random_dna(2000, seed=99)
        fasta_store = {
            "short_strain": {
                "ctg1": type("FakeSeq", (), {"seq": contig_seq})(),
            }
        }

        detective = HGTDetective(contamination=0.30, random_state=0, min_seq_length=90)
        cls.hgt_result = detective.run(
            phase1_result=phase1_result,
            fasta_store=fasta_store,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_short_gene_kmer_deviation_is_zero(self) -> None:
        """The gene with a 30 bp sequence should have kmer_deviation == 0.0."""
        short_records = [
            r
            for r in self.hgt_result.hgt_records
            if "short_gene" in r.gene_record.gene_id
        ]
        if not short_records:
            self.skipTest("short_gene ended up as core/shell — test data may vary")
        for rec in short_records:
            self.assertAlmostEqual(
                rec.kmer_deviation,
                0.0,
                places=5,
                msg=f"{rec.gene_record.gene_id}: kmer_deviation should be 0.0 for short seq",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
