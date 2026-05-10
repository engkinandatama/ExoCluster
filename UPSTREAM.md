# ExoCluster Fork Divergence Tracking

**Original Repository:** https://github.com/nauvalrajwaa/pangenome_miner  
**Fork Repository:** https://github.com/engkinandatama/ExoCluster  
**Last Sync:** 2026-05-10  
**Status:** Active Development

---

## Divergence Summary

| Aspect | Original (PanAdapt-BGC Miner) | ExoCluster (Fork) | Status |
|--------|-------------------------------|-------------------|--------|
| **Core Pipeline** | 3-phase sequential | 3-phase sequential | ✅ Compatible |
| **Workflow Management** | None | Snakemake (planned) | 🆕 New |
| **Containerization** | None | Docker/Singularity (planned) | 🆕 New |
| **Scalability** | < 20 genomes | 100+ genomes (target) | 🆕 Improved |
| **Reproducibility** | Manual setup | Containerized (target) | 🆕 Improved |
| **Testing** | 50/50 passing | 50/50 passing | ✅ Maintained |
| **Documentation** | README.md | README + docs (planned) | 🆕 Extended |

---

## Key Changes in ExoCluster

### 1. Workflow Management (Planned)
- **Add:** Snakemake workflow orchestration
- **Add:** Conda environment per phase
- **Add:** HPC support (Slurm/SGE)
- **Add:** Checkpointing untuk resume

### 2. Containerization (Planned)
- **Add:** Dockerfile untuk reproducibility
- **Add:** Singularity.def untuk HPC
- **Add:** docker-compose.yml untuk local testing

### 3. Performance Optimization (Planned)
- **Add:** MMseqs2 integration untuk clustering
- **Add:** Hybrid k-mer/MMseqs2 backend
- **Add:** Memory optimization untuk large datasets

### 4. Configuration Management (Planned)
- **Add:** YAML-based configuration
- **Add:** CLI override support
- **Add:** Environment-specific configs

### 5. Data Versioning (Planned)
- **Add:** DVC untuk model weights
- **Add:** DVC untuk test data
- **Add:** Version tracking untuk large files

### 6. CI/CD (Planned)
- **Add:** GitHub Actions untuk testing
- **Add:** Automated coverage reporting
- **Add:** Container build verification

---

## New Features (Not in Original)

| Feature | Description | Priority |
|---------|-------------|----------|
| Snakemake Workflow | Automatic dependency management | P0 |
| Docker Container | Reproducible execution | P0 |
| MMseqs2 Integration | Scalable clustering | P1 |
| HPC Profile | Slurm/SGE support | P1 |
| Config Management | YAML configuration | P1 |
| DVC Tracking | Large file versioning | P1 |
| CI/CD Pipeline | Automated testing | P1 |

---

## Backward Compatibility

### Maintained
- ✅ All existing 3-phase pipeline logic
- ✅ Same input/output formats
- ✅ Same CLI arguments (with extensions)
- ✅ All existing tests (50/50 passing)

### Extended
- 🆕 New `--cluster-method` flag (kmer/mmseqs2)
- 🆕 New `--config` flag for YAML config
- 🆕 New `--profile` flag for Snakemake profiles

---

## Migration Path

### For Users of Original Repository

**Current (Original):**
```bash
python main.py --genomes data/ --annotations data/ --output output/
```

**Future (ExoCluster):**
```bash
# Option 1: Direct execution (backward compatible)
python main.py --genomes data/ --annotations data/ --output output/

# Option 2: Snakemake workflow
snakemake --cores 8 --configfile config/default.yaml

# Option 3: Containerized
docker run -v $(pwd):/app exocluster:latest --genomes /app/data/ --output /app/output/
```

### For Developers

**Current:**
```bash
python -m pytest tests/ -v
```

**Future:**
```bash
# Option 1: Direct testing
python -m pytest tests/ -v

# Option 2: CI/CD testing (automatic)
# Push to GitHub → Actions run tests automatically
```

---

## Future Enhancements (Beyond Original)

| Feature | Description | Estimated Effort |
|---------|-------------|------------------|
| Advanced Statistics | Heaps' Law, Fisher's Exact Test | 3-5 days |
| Interactive Dashboard | Web-based visualization | 5-7 days |
| Cloud Deployment | AWS/GCP integration | 7-10 days |
| API Server | REST API untuk pipeline | 10-14 days |
| GUI Application | Desktop app dengan Electron | 14-21 days |

---

## Versioning Strategy

### Semantic Versioning
- **MAJOR:** Breaking changes (e.g., API changes)
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: Feature development
- `hotfix/*`: Critical bug fixes

---

## Contribution Guidelines

### Before Submitting PR
1. [ ] Run tests: `python -m pytest tests/ -v`
2. [ ] Check linting: `flake8 pipeline/`
3. [ ] Type checking: `mypy pipeline/`
4. [ ] Update documentation
5. [ ] Add tests for new features

### PR Template
```markdown
## Summary
- [Feature/bug fix] description

## Changes
- Change 1
- Change 2

## Testing
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Backward compatible

## Related Issues
Fixes #issue-number
```

---

## Contact & Support

**Maintainer:** @engkinandatama  
**Original Author:** @nauvalrajwaa  
**Repository:** https://github.com/engkinandatama/ExoCluster  
**Issues:** https://github.com/engkinandatama/ExoCluster/issues

---

## Acknowledgments

- Original work by [Nauval Rajwaa](https://github.com/nauvalrajwaa)
- BGC-Prophet by [HUST-NingKang-Lab](https://github.com/HUST-NingKang-Lab/BGC-Prophet)
- ESM2 by [Facebook Research](https://github.com/facebookresearch/esm)

---

**Last Updated:** 2026-05-10  
**Next Review:** 2026-06-10
