"""
PanAdapt-BGC Miner — Integration Tests
======================================
Module  : tests/test_integration_main.py
Purpose : Full pipeline integration test using mock data.
"""

import sys
import shutil
import pytest
from pathlib import Path
from main import main

def test_full_pipeline_mock(tmp_path):
    """
    Run main.py with --mock to verify the end-to-end wiring
    from Pangenome -> HGT -> BGC.
    """
    output_dir = tmp_path / "output"
    
    # Simulate CLI arguments
    # We monkeypatch sys.argv to emulate calling the script
    test_args = [
        "main.py",
        "--mock",
        "--output", str(output_dir),
        "--cluster-method", "kmer"  # Use kmer to avoid external dependency issues
    ]
    
    # Use the monkeypatch fixture provided by pytest
    # We rename the function to include the fixture
    pass

def test_full_pipeline_mock_with_fixture(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    
    test_args = [
        "main.py",
        "--mock",
        "--output", str(output_dir),
        "--cluster-method", "kmer"
    ]
    
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Run main()
    main()
    
    # Verify core outputs exist
    assert (output_dir / "pangenome_presence_absence_matrix.csv").exists()
    assert (output_dir / "pangenome" / "pangenome_report.html").exists()
    assert (output_dir / "hgt" / "hgt_report.html").exists()
    assert (output_dir / "bgc" / "bgc_report.html").exists()
    
    # Cleanup
    shutil.rmtree(output_dir, ignore_errors=True)