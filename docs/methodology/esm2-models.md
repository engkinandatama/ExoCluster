# ESM2 Model Selection Guide

## Supported Models

| Model | Parameters | Speed | Memory | Quality | Best Use Case |
|-------|------------|-------|--------|---------|--------------|
| `esm2_t6_8M_UR50D` | 8M | Fast | 1GB | Good | Default, quick screening |
| `esm2_t12_35M_UR50D` | 35M | Medium | 2GB | Better | Higher accuracy, moderate cost |
| `esm2_t30_150M_UR50D` | 150M | Slow | 4GB | Good | Best accuracy if resources allow |
| `esm2_t33_650M_UR50D` | 650M | Very slow | 12GB | Excellent | High-precision research |
| `esm2_t36_3B_UR50D` | 3B | Very slow | 24GB | Excellent | State-of-the-art, requires GPU |
| `esm2_t48_15B_UR50D` | 15B | Extremely slow | 48GB | Best available | Enterprise-scale projects |

## Recommendations

### Quick Testing / Small Projects
- Start with `esm2_t6_8M_UR50D`
- Runs in ~30 seconds on CPU
- Good for initial screening

### Standard Research
- `esm2_t12_35M_UR50D` offers best balance
- ~2 minutes on CPU, 30s on GPU

### High-Precision Analysis
- Use `esm2_t30_150M_UR50D` for better accuracy
- Requires ~4GB RAM

### Large-Scale Projects
- Consider `esm2_t33_650M_UR50D` or larger
- Use GPU acceleration for feasible runtime