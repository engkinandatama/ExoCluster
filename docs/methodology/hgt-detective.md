# HGT Detection

The HGT Detective module identifies potential Horizontally Transferred Genes (HGTs) by analyzing accessory genes for signals of rapid evolutionary change or abnormal k-mer composition.

## Methodology

### 1. Feature Extraction
- Scans genes identified as "Accessory" in the Pangenome Analysis.
- Computes genomic signatures and evolutionary metrics.

### 2. Detection Algorithm
- **Isolation Forest:** Uses unsupervised anomaly detection to identify genes with deviating patterns compared to the core genomic background.
- Genes flagged as "Alien" are prioritized for downstream BGC prediction.

## Output
- List of high-confidence HGT candidates passed to BGC Predictor.