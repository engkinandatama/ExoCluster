# BGC Prediction

The BGC Predictor module uses deep learning to predict the class of Biosynthetic Gene Clusters (BGCs) from a pool of candidate genes (typically from HGT Detection).

## Methodology

### 1. Input Processing
- Takes a set of candidate genes (typically from HGT detection).
- Translates DNA sequences to amino acid sequences.
- Generates protein embeddings using ESM2 language models.

### 2. Model Architecture
- Uses BGC-Prophet (ensemble of two models):
  1. **Annotator:** 1D CNN for gene cluster annotation.
  2. **Classifier:** 1D CNN for BGC class prediction.
- Accepts ESM2 embeddings as input instead of raw AA sequences.

### 3. ESM2 Integration
- Supports multiple ESM2 model sizes (8M to 15B parameters).
- Larger models capture more complex protein patterns but require more memory/compute.

## Parameters
- `--model-dir`: Directory containing BGC-Prophet model weights.
- `--esm-model`: ESM2 variant for protein embeddings.

## Output
- Predicted BGC class (e.g., RiPP, PKS, NRPS, etc.) with confidence scores.
- Detailed HTML report with visualizations.