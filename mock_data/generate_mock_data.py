"""
PanAdapt-BGC Miner — Mock Data Generator
=========================================
Module  : mock_data/generate_mock_data.py
Purpose : Generate synthetic FASTA + GFF3 files for 5 mock bacterial strains
          to allow immediate testing of Phase 1 without real genome data.

Usage   : python mock_data/generate_mock_data.py
"""

from __future__ import annotations

import random
import string
from pathlib import Path

STRAINS = ["strain_A", "strain_B", "strain_C", "strain_D", "strain_E"]
CONTIG_LEN = 50_000      # bp per mock contig
N_GENES = 80             # genes per strain (some shared, some unique)
SEED = 42


def random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choices("ACGT", k=length))


def mutate_seq(seq: str, mutation_rate: float, rng: random.Random) -> str:
    """Introduce point mutations to simulate homologs."""
    bases = list(seq)
    for i in range(len(bases)):
        if rng.random() < mutation_rate:
            bases[i] = rng.choice("ACGT")
    return "".join(bases)


def generate_mock_data(output_dir: Path) -> None:
    rng = random.Random(SEED)

    genomes_dir = output_dir / "genomes"
    annotations_dir = output_dir / "annotations"
    genomes_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Generate a shared "core" gene pool (present in ALL strains)
    # ---------------------------------------------------------------
    n_core = 50        # shared across all 5 strains → fraction = 1.0 (core)
    n_unique = 10      # unique to each strain         → fraction = 0.2 (accessory)
    n_shell = 20       # present in 2–4 strains         → shell

    # Each gene is a (start, end, strand, sequence, product) tuple on a 50 kbp contig
    # We lay them out sequentially with ~200 bp gaps
    GENE_LEN = 300

    print("Generating mock genome data …")

    # Core gene sequences — shared template, slight per-strain mutation
    core_templates = {
        f"core_gene_{i:03d}": random_dna(GENE_LEN, rng) for i in range(n_core)
    }
    # Shell gene sequences — shared among subset
    shell_templates = {
        f"shell_gene_{i:03d}": random_dna(GENE_LEN, rng) for i in range(n_shell)
    }
    # Unique gene pool — one unique gene block per strain
    unique_templates: dict[str, dict[str, str]] = {
        strain: {
            f"unique_{strain}_gene_{i:02d}": random_dna(GENE_LEN, rng)
            for i in range(n_unique)
        }
        for strain in STRAINS
    }

    for strain in STRAINS:
        contig_seq = list(random_dna(CONTIG_LEN, rng))  # background genome
        gff_lines: list[str] = [
            "##gff-version 3",
            f"##sequence-region mock_contig_1 1 {CONTIG_LEN}",
        ]

        cursor = 500   # start placing genes at position 500

        # --- core genes (slight per-strain mutation) ---
        for gene_name, tmpl in core_templates.items():
            seq = mutate_seq(tmpl, 0.02, rng)   # 2% mutation → still clusters together
            start = cursor
            end = start + GENE_LEN - 1
            if end >= CONTIG_LEN:
                break
            contig_seq[start - 1: end] = list(seq)
            strand = rng.choice(["+", "-"])
            gff_lines.append(
                f"mock_contig_1\tmock\tCDS\t{start}\t{end}\t.\t{strand}\t0\t"
                f"ID={gene_name}_{strain};product=core hypothetical protein"
            )
            cursor = end + rng.randint(100, 250)

        # --- shell genes (present in subset) ---
        strain_idx = STRAINS.index(strain)
        for i, (gene_name, tmpl) in enumerate(shell_templates.items()):
            # include shell gene if strain_idx < 3 (60% strains) → shell range
            if (strain_idx + i) % 3 == 0:
                seq = mutate_seq(tmpl, 0.05, rng)
                start = cursor
                end = start + GENE_LEN - 1
                if end >= CONTIG_LEN:
                    break
                contig_seq[start - 1: end] = list(seq)
                strand = rng.choice(["+", "-"])
                gff_lines.append(
                    f"mock_contig_1\tmock\tCDS\t{start}\t{end}\t.\t{strand}\t0\t"
                    f"ID={gene_name}_{strain};product=shell gene"
                )
                cursor = end + rng.randint(100, 250)

        # --- unique genes (strain-specific → accessory) ---
        for gene_name, seq in unique_templates[strain].items():
            start = cursor
            end = start + GENE_LEN - 1
            if end >= CONTIG_LEN:
                break
            contig_seq[start - 1: end] = list(seq)
            strand = rng.choice(["+", "-"])
            gff_lines.append(
                f"mock_contig_1\tmock\tCDS\t{start}\t{end}\t.\t{strand}\t0\t"
                f"ID={gene_name};product=putative HGT gene transposase-related"
            )
            cursor = end + rng.randint(100, 250)

        # --- Write FASTA ---
        fasta_path = genomes_dir / f"{strain}.fasta"
        with fasta_path.open("w") as fh:
            fh.write(f">mock_contig_1 {strain} mock genome\n")
            seq_str = "".join(contig_seq)
            for chunk_start in range(0, len(seq_str), 60):
                fh.write(seq_str[chunk_start: chunk_start + 60] + "\n")

        # --- Write GFF3 ---
        gff_path = annotations_dir / f"{strain}.gff"
        with gff_path.open("w") as fh:
            fh.write("\n".join(gff_lines) + "\n")

        print(f"  [OK] {strain}: {len(gff_lines) - 2} CDS features written")

    print(f"\nMock data written to:\n  {genomes_dir}\n  {annotations_dir}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    generate_mock_data(script_dir)
