"""
Benchmark script for cluster_method comparison.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def generate_mock_data(num_genomes=10):
    """Generate mock data for benchmarking."""
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
    cmd = [sys.executable, str(gen_script), "--num-strains", str(num_genomes)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Mock data generation failed: {result.stderr}")

    print(f"Generated mock data: {num_genomes} genomes")
    return genomes_mock, annotations_mock


def run_pipeline(genomes_dir, annotations_dir, output_dir, cluster_method):
    """Run the pipeline with a specific cluster_method."""
    cmd = [
        sys.executable,
        "main.py",
        "--genomes",
        str(genomes_dir),
        "--annotations",
        str(annotations_dir),
        "--output",
        str(output_dir),
        "--cluster-method",
        cluster_method,
        "--verbose",
    ]
    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    end_time = time.time()

    if result.returncode != 0:
        print(f"Error running pipeline with {cluster_method}:")
        print(result.stderr)
        return None

    elapsed_time = end_time - start_time
    return elapsed_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark cluster methods")
    parser.add_argument(
        "--num-genomes", type=int, default=10, help="Number of mock genomes to generate"
    )
    args = parser.parse_args()

    # Generate mock data
    genomes_mock, annotations_mock = generate_mock_data(args.num_genomes)

    # Benchmark kmer
    print("\nBenchmarking kmer method...")
    kmer_output = Path("output/benchmark_kmer")
    if kmer_output.exists():
        shutil.rmtree(kmer_output)
    kmer_time = run_pipeline(genomes_mock, annotations_mock, kmer_output, "kmer")

    # Benchmark mmseqs2
    print("\nBenchmarking mmseqs2 method...")
    mmseqs2_output = Path("output/benchmark_mmseqs2")
    if mmseqs2_output.exists():
        shutil.rmtree(mmseqs2_output)
    mmseqs2_time = run_pipeline(
        genomes_mock, annotations_mock, mmseqs2_output, "mmseqs2"
    )

    # Print results
    print("\nBenchmark Results:")
    print("------------------")
    print(f"Number of genomes: {args.num_genomes}")
    print(
        f"kmer method time: {kmer_time:.2f} seconds"
        if kmer_time
        else "kmer method failed"
    )
    print(
        f"mmseqs2 method time: {mmseqs2_time:.2f} seconds"
        if mmseqs2_time
        else "mmseqs2 method failed"
    )

    if kmer_time and mmseqs2_time:
        print(
            f"\nmmseqs2 was {kmer_time / mmseqs2_time:.2f}x {'faster' if mmseqs2_time < kmer_time else 'slower'} than kmer"
        )


if __name__ == "__main__":
    main()
