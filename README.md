# ExoCluster
**ExoCluster: Pangenome Adaptive & Alien Biosynthetic Gene Cluster Miner**

A modular, end-to-end Python pipeline for discovering Biosynthetic Gene Clusters (BGCs) in microbial genomes by chaining three analytical phases: **Pangenome Analysis → HGT Detection → AI BGC Prediction**.

## 📖 Documentation

For detailed information, please see the documentation in the `docs/` directory:

- [Installation Guide](docs/installation.md)
- [Usage & Examples](docs/usage.md)
- [Methodology Details](docs/methodology/)
- [Development Guide](docs/development/)

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd ExoCluster

# 2. Create a conda environment (recommended)
conda create -n exocluster python=3.10
conda activate exocluster

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download BGC-Prophet model weights
mkdir -p models/model
wget -q "https://github.com/HUST-NingKang-Lab/BGC-Prophet/files/12733164/model.tar.gz" \
  -O models/model.tar.gz
tar -xzf models/model.tar.gz -C models/
rm models/model.tar.gz

# 5. Run on test data
python main.py \
  --genomes    tests/real_data/genomes \
  --annotations tests/real_data/annotations \
  --output     tests/output \
  --verbose
```

## 🔬 Project Structure

```
ExoCluster/
├── main.py                        # Pipeline entry point & CLI
├── requirements.txt               # Python dependencies
├── pipeline/
│   ├── pangenome_miner.py         # Phase 1: Pangenome Analysis
│   ├── hgt_detective.py           # Phase 2: HGT Detection
│   ├── bgc_predictor.py           # Phase 3: BGC Prediction
│   └── *_visualizer.py            # HTML report generators
├── models/
│   └── model/                     # BGC-Prophet weights
├── tests/                         # Unit tests
├── mock_data/                     # Synthetic data generator
└── docs/                          # Comprehensive documentation
```

## 🛠 Project Evolution & Roadmap

ExoCluster is actively being developed as a research and learning tool. The project is transitioning from a sequential script to a robust, modular bioinformatics framework.

**Internal Development Plans:**
- [Contribution & Optimization Plan](.dev/contribution-plan.md) — Technical strategy for scalability and science depth.
- [Reproducibility Roadmap](.dev/REPRODUCIBILITY_ROADMAP.md) — Path toward HPC-readiness and containerization.

## 👥 Acknowledgments

- **BGC-Prophet** — [HUST-NingKang-Lab/BGC-Prophet](https://github.com/HUST-NingKang-Lab/BGC-Prophet) (MIT License) for the trained BGC annotator and classifier models
- **ESM2** — [facebookresearch/esm](https://github.com/facebookresearch/esm) for protein language models
- Test data: 5 *Streptomyces abikoensis* soil MAG assemblies from NCBI