# Contributing to ExoCluster

## How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/ExoCluster.git
cd ExoCluster

# Create development environment
conda create -n exocluster-dev python=3.10
conda activate exocluster-dev

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

## Code Style

- Follow [PEP 8](https://pep8.org/) with 4-space indentation
- Maximum line length: 88 characters
- Use [Black](https://black.readthedocs.io/) for auto-formatting
- Use [ruff](https://docs.astral.sh/ruff/) for linting
- Type hints required for all public functions
- Docstrings in [Google style](https://sphinx.google.github.io/pydoc-templates.html#google-style)

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=pipeline --cov-report=html

# Run specific test
python -m pytest tests/test_pangenome_miner.py -v
```

## Documentation

- Docstrings should follow Google style
- Update documentation when changing functionality
- Run `make docs` to build documentation locally