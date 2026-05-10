# Testing Guide

ExoCluster employs a comprehensive testing strategy to ensure the reliability of the bioinformatics pipeline.

## Test Suite Overview

The project uses `pytest` for all levels of testing, located in the `tests/` directory.

### 1. Unit Tests
- **`tests/test_pangenome_miner.py`**: Verifies k-mer clustering, core/accessory categorization, and presence-absence matrix generation.
- **`tests/test_hgt_detective.py`**: Validates the Isolation Forest implementation and anomaly detection logic.
- **`tests/test_bgc_predictor.py`**: Tests the ESM2 embedding pipeline and BGC-Prophet model integration.

### 2. Integration Tests
- Tests the full flow from genome input $\rightarrow$ BGC prediction using small, curated datasets in `tests/real_data/`.

### 3. Mock Testing
- A synthetic data generator (`mock_data/generate_mock_data.py`) allows for high-throughput testing without requiring real genomic files.

## How to Run Tests

### Basic Execution
```bash
python -m pytest tests/
```

### Verbose Output
```bash
python -m pytest tests/ -v
```

### Running Specific Modules
```bash
python -m pytest tests/test_pangenome_miner.py
```

## Adding New Tests

1. Create a new test file in `tests/` following the `test_*.py` naming convention.
2. Use `pytest` fixtures for common setup (e.g., initializing a `PangenomeMiner` instance).
3. Follow the **Arrange-Act-Assert** pattern.
4. Ensure tests are independent and do not rely on external file paths (use `tmp_path` fixture).

## Test Data Management
- **Real Data:** Stored in `tests/real_data/` for gold-standard verification.
- **Mock Data:** Generated on-the-fly using the `--mock` flag in `main.py`.