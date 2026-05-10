# Pangenome Analysis

The Pangenome Analysis module identifies the core and accessory gene sets across a group of microbial genomes.

## Methodology

### 1. Ortholog Clustering
- **Method:** Jaccard Similarity based on k-mer frequency (default).
- **Process:** 
    - Translates genomic sequences to protein-coding regions (CDS).
    - Extracts k-mers from protein sequences.
    - Computes similarity matrix between all gene pairs.
    - Clusters genes into orthologous groups (OGs).

### 2. Pangenome Categorization
- **Core Genome:** OGs present in $\ge$ `core-threshold` (default 0.95) of strains.
- **Accessory Genome:** OGs present in $\le$ `accessory-threshold` (default 0.10) of strains.

## Parameters
- `--core-threshold`: Fraction of strains required to classify a gene as 'core'.
- `--accessory-threshold`: Max fraction of strains to classify a gene as 'accessory'.
- `--identity`: Jaccard similarity threshold for clustering.