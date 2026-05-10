# System Architecture

ExoCluster uses a modular, sequential pipeline architecture designed for bioinformatics scalability.

## High-Level Pipeline
The data flows linearly through three core modules:

1. **Pangenome Analysis**
   - Input: FASTA genomes + GFF annotations.
   - Output: Presence/absence matrix of ortholog clusters.
   - Key Class: `PangenomeMiner`

2. **HGT Detection**
   - Input: Accessory gene list from Pangenome Analysis.
   - Output: List of "Alien" gene candidates flagged for potential HGT.
   - Key Class: `HGTDetective`

3. **BGC Prediction**
   - Input: Alien gene candidates.
   - Output: Predicted BGC class (e.g., RiPPs, PKS).
   - Key Class: `BGCPredictor`

## Component Responsibilities

| Module | Responsibility | Core Tech |
|--------|----------------|-----------|
| `pipeline/pangenome_miner.py` | Ortholog clustering & matrix gen | NumPy, BioPython |
| `pipeline/hgt_detective.py` | Anomaly detection | Scikit-learn (Isolation Forest) |
| `pipeline/bgc_predictor.py` | ESM2 embeddings & BGC classification | PyTorch, BGC-Prophet |

## Data Flow
- **Initialization:** `main.py` parses CLI arguments and configures the `ModelRegistry`.
- **Phase Execution:** Data is passed via temporary intermediate files between phases to ensure checkpointing capability.
- **Reporting:** Each module has a dedicated visualizer (`phaseX_visualizer.py`) that exports HTML/plot reports.