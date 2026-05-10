# Installation Guide

## Prerequisites

- Python 3.10+
- conda (recommended) or pip
- ~200 MB disk for model weights (BGC-Prophet + ESM2-8M)

## Steps

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

# 5. (Optional) Install MMseqs2 for faster ortholog clustering at scale
conda install -c bioconda mmseqs2
```

## Verify Installation

```bash
# Check all dependencies
python -c "
from pipeline.bgc_predictor import BGCPredictor, ESM2_REGISTRY, _PROPHET_AVAILABLE
print(f'BGC-Prophet available: {_PROPHET_AVAILABLE}')
print(f'Supported ESM2 models: {list(ESM2_REGISTRY.keys())}')
"

# Run tests
python -m pytest tests/ -v
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `fair-esm` | ≥2.0.0 | ESM2 protein language models |
| `bgc-prophet` | ≥0.1.2 | BGC-Prophet annotator/classifier models |
| `lmdb` | ≥1.4.0 | Required by bgc-prophet internals |
| `biopython` | ≥1.80 | GFF/FASTA parsing, DNA→protein translation |
| `scikit-learn` | ≥1.2 | Isolation Forest (HGT Detection) |
| `pandas` | ≥1.5 | Data manipulation |
| `numpy` | ≥1.23 | Numerical operations |
| `matplotlib` | ≥3.6 | Plotting |
| `seaborn` | ≥0.12 | Statistical plots |
| `plotly` | ≥5.0 | Interactive plots |
| `jinja2` | ≥3.1 | HTML report templating |
| `loguru` | ≥0.7 | Structured logging |
| `tqdm` | ≥4.60 | Progress bars |