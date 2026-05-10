# Usage & Examples

## Quick start — bundled test data (5 Streptomyces abikoensis MAGs)

```bash
python main.py \
  --genomes    tests/real_data/genomes \
  --annotations tests/real_data/annotations \
  --output     tests/output \
  --verbose
```

Expected output:
```
═══ Phase 1: Pangenome Analysis ═══
  Strains analyzed       : 5
  Total CDS genes        : ~32,000
  Ortholog clusters      : ~5,050
  Accessory genes        : ~2,021

═══ Phase 2: HGT Detection ═══
  Genes screened         : ~2,021
  Alien HGT genes        : ~606 (~30%)

═══ Phase 3: BGC Prediction ═══
  Alien genes scored     : ~606
  BGC hits               : ~43 (~7%)
  High-confidence hits   : ~27
  Top BGC class          : RiPP
  Inference engine       : BGC-Prophet
  ESM2 model             : esm2_t6_8M_UR50D
```

Expected runtime: ~4–6 minutes (Phase 1 k-mer clustering dominates; Phase 3 ESM2 embedding ~30 seconds).

## Minimal run

```bash
python main.py \
  --genomes    data/genomes \
  --annotations data/annotations \
  --output     output/
```

## Full run — all parameters specified

```bash
python main.py \
  --genomes              tests/real_data/genomes \
  --annotations          tests/real_data/annotations \
  --output               output/ \
  --core-threshold       0.95 \
  --accessory-threshold  0.10 \
  --identity             0.80 \
  --model-dir            models/model \
  --esm-model            esm2_t6_8M_UR50D \
  --verbose
```

## Using a larger ESM2 model

```bash
# 35M parameter model — richer embeddings, ~4x slower
python main.py \
  --genomes    tests/real_data/genomes \
  --annotations tests/real_data/annotations \
  --output     output_35M/ \
  --esm-model  esm2_t12_35M_UR50D \
  --verbose

# 150M parameter model — needs ~2 GB RAM
python main.py \
  --genomes    tests/real_data/genomes \
  --annotations tests/real_data/annotations \
  --output     output_150M/ \
  --esm-model  esm2_t30_150M_UR50D \
  --verbose
```

## Quick smoke test — synthetic mock data

```bash
python main.py --mock --output output/mock_run/
```

## Running on your own MAGs

Expected input layout:
```
genomes/
  strain_1.fna
  strain_2.fna
  strain_3.fna
annotations/
  strain_1.gff     ← GFF3 from NCBI PGAP or Prokka
  strain_2.gff
  strain_3.gff
```

```bash
python main.py \
  --genomes     /path/to/my_mags/genomes \
  --annotations /path/to/my_mags/annotations \
  --output      /path/to/results \
  --identity    0.75 \
  --verbose
```

## CLI Reference

```
usage: exocluster [-h]
        [--genomes GENOMES] [--annotations ANNOTATIONS]
        [--output OUTPUT]
        [--core-threshold CORE_THRESHOLD]
        [--accessory-threshold ACCESSORY_THRESHOLD]
        [--identity IDENTITY]
        [--model-dir MODEL_DIR]
        [--esm-model ESM_MODEL]
        [--mock] [--verbose]

options:
  --genomes              Directory of FASTA genome files (.fna/.fasta/.fa)
  --annotations          Directory of GFF3 annotation files (.gff/.gff3)
  --output               Root output directory  [default: output/]
  --core-threshold       Fraction of strains for Core genes  [default: 0.95]
  --accessory-threshold  Max fraction of strains for Accessory  [default: 0.10]
  --identity             Ortholog clustering Jaccard threshold  [default: 0.80]
  --model-dir            Directory with BGC-Prophet model weights  [default: models/model/]
  --esm-model            ESM2 model variant for protein embeddings  [default: esm2_t6_8M_UR50D]
                          Options: esm2_t6_8M_UR50D, esm2_t12_35M_UR50D,
                          esm2_t30_150M_UR50D, esm2_t33_650M_UR50D,
                          esm2_t36_3B_UR50D, esm2_t48_15B_UR50D
  --mock                 Use auto-generated synthetic data (ignores --genomes/--annotations)
  --verbose, -v          Enable DEBUG logging