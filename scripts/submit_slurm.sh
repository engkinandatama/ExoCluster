#!/bin/bash
# ExoCluster Slurm submission script
# Usage: ./scripts/submit_slurm.sh [snakemake options]

# Default snakemake options
SNAKEMAKE_OPTS="--profile workflow/profiles/slurm --use-conda"

# Allow user to pass additional options
if [ $# -gt 0 ]; then
    SNAKEMAKE_OPTS="$SNAKEMAKE_OPTS $*"
fi

echo "Submitting ExoCluster workflow to Slurm cluster..."
echo "Using profile: workflow/profiles/slurm"
echo "With options: $SNAKEMAKE_OPTS"

# Run snakemake
snakemake $SNAKEMAKE_OPTS

echo "Slurm submission completed."