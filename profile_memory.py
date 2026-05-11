"""
Memory profiling script for the kmer clustering method.
This version imports the pipeline directly to allow tracemalloc to capture allocations.
"""

import argparse
import tracemalloc
import os
import shutil
import sys
import time
from pathlib import Path

# Add current directory to sys.path to allow importing pipeline
sys.path.append(os.getcwd())

from pipeline.pangenome_miner import PangenomeMiner


def generate_mock_data(num_genomes=10):
    """Generate mock data for profiling."""
    mock_dir = Path(__file__).parent / "mock_data"
    gen_script = mock_dir / "generate_mock_data.py"

    # Clean old mock data if exists
    genomes_mock = mock_dir / "genomes"
    annotations_mock = mock_dir / "annotations"
    if genomes_mock.exists():
        shutil.rmtree(genomes_mock)
    if annotations_mock.exists():
        shutil.rmtree(annotations_mock)

    # Generate new data
    import subprocess

    cmd = [sys.executable, str(gen_script), "--num-strains", str(num_genomes)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Mock data generation failed: {result.stderr}")

    print(f"Generated mock data: {num_genomes} genomes")
    return genomes_mock, annotations_mock


def profile_pangenome_miner(genomes_dir, annotations_dir, num_genomes):
    """Profile the PangenomeMiner memory usage."""
    # Setup Miner
    miner = PangenomeMiner(
        core_threshold=0.95,
        accessory_threshold=0.10,
        identity_threshold=0.80,
        cluster_method="kmer",
    )

    print(f"Profiling PangenomeMiner with {num_genomes} genomes...")

    # Start tracing
    tracemalloc.start()
    start_time = time.time()

    try:
        # Run the core logic
        result = miner.run(genomes_dir, annotations_dir)
        elapsed_time = time.time() - start_time

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return elapsed_time, current, peak
    except Exception as e:
        tracemalloc.stop()
        print(f"Error during profiling: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Profile memory usage for kmer method")
    parser.add_argument(
        "--num-genomes", type=int, default=10, help="Number of mock genomes to generate"
    )
    args = parser.parse_args()

    # Generate mock data
    genomes_mock, annotations_mock = generate_mock_data(args.num_genomes)

    # Profile
    elapsed, current, peak = profile_pangenome_miner(
        genomes_mock, annotations_mock, args.num_genomes
    )

    # Print results
    print("\nMemory Profiling Results (Internal Process):")
    print("-------------------------------------------")
    print(f"Number of genomes: {args.num_genomes}")
    if elapsed is not None:
        print(f"Execution time: {elapsed:.2f} seconds")
        print(f"Current memory usage: {current / 10**6:.2f} MB")
        print(f"Peak memory usage: {peak / 10**6:.2f} MB")
    else:
        print("Profiling failed")


if __name__ == "__main__":
    main()
