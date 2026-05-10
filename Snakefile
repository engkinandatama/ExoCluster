# ExoCluster: Snakemake Workflow
# Author: Engki Nandatama
# Version: 0.1.0

import pandas as pd
from pathlib import Path

# Load configuration
configfile: "config/default.yaml"

# Global variables
PIPELINE_NAME = config["pipeline"]["name"]
PIPELINE_VERSION = config["pipeline"]["version"]
OUTPUT_DIR = Path(config["paths"]["output"])
GENOMES_DIR = Path(config["paths"]["genomes"])
ANNOTATIONS_DIR = Path(config["paths"]["annotations"])
MODEL_DIR = Path(config["paths"]["model_dir"])

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define wildcard constraints
wildcard_constraints:
    phase="pangenome|hgt|bgc",
    method="kmer|mmseqs2",
    model="esm2_t[0-9]+_[0-9]+M_UR50D|esm2_t[0-9]+_[0-9]+B_UR50D"

# Define rules
rule all:
    input:
        # Pangenome outputs
        f"{OUTPUT_DIR}/pangenome/pangenome_matrix.csv",
        f"{OUTPUT_DIR}/pangenome/pangenome_stats.json",
        f"{OUTPUT_DIR}/pangenome/visualization/pangenome_plot.html",
        
        # HGT outputs
        f"{OUTPUT_DIR}/hgt/hgt_predictions.csv",
        f"{OUTPUT_DIR}/hgt/visualization/hgt_plot.html",
        
        # BGC outputs
        f"{OUTPUT_DIR}/bgc/bgc_predictions.csv",
        f"{OUTPUT_DIR}/bgc/visualization/bgc_plot.html",
        f"{OUTPUT_DIR}/bgc/reports/final_report.html",

    message:
        "ExoCluster pipeline completed successfully!"

rule pangenome:
    input:
        genomes = f"{GENOMES_DIR}/{{genome_id}}.fasta",
        annotations = f"{ANNOTATIONS_DIR}/{{genome_id}}.gff"
    output:
        matrix = f"{OUTPUT_DIR}/pangenome/pangenome_matrix.csv",
        stats = f"{OUTPUT_DIR}/pangenome/pangenome_stats.json"
    params:
        core_threshold = config["pangenome"]["core_threshold"],
        accessory_threshold = config["pangenome"]["accessory_threshold"],
        identity = config["pangenome"]["identity"],
        cluster_method = config["pangenome"]["cluster_method"],
        min_seq_length = config["pangenome"]["min_seq_length"]
    conda:
        "workflow/envs/pangenome.yaml"
    threads: config["resources"]["pangenome"]["threads"]
    resources:
        mem_mb = config["resources"]["pangenome"]["mem_mb"],
        time = config["resources"]["pangenome"]["time"]
    log:
        f"{OUTPUT_DIR}/pangenome/logs/pangenome_{{wildcards.genome_id}}.log"
    shell:
        """
        python workflow/scripts/run_pangenome.py \
            --genomes {input.genomes} \
            --annotations {input.annotations} \
            --output {output.matrix} \
            --stats {output.stats} \
            --core-threshold {params.core_threshold} \
            --accessory-threshold {params.accessory_threshold} \
            --identity {params.identity} \
            --cluster-method {params.cluster_method} \
            --min-seq-length {params.min_seq_length} \
            > {log} 2>&1
        """

rule hgt:
    input:
        genome = f"{GENOMES_DIR}/{{genome_id}}.fasta",
        annotations = f"{ANNOTATIONS_DIR}/{{genome_id}}.gff",
        pangenome_stats = f"{OUTPUT_DIR}/pangenome/pangenome_stats.json"
    output:
        predictions = f"{OUTPUT_DIR}/hgt/hgt_predictions_{{genome_id}}.csv"
    params:
        contamination = config["hgt"]["contamination"],
        n_estimators = config["hgt"]["n_estimators"],
        random_state = config["hgt"]["random_state"],
        mge_proximity_bp = config["hgt"]["mge_proximity_bp"],
        min_seq_length = config["hgt"]["min_seq_length"]
    conda:
        "workflow/envs/hgt.yaml"
    threads: config["resources"]["hgt"]["threads"]
    resources:
        mem_mb = config["resources"]["hgt"]["mem_mb"],
        time = config["resources"]["hgt"]["time"]
    log:
        f"{OUTPUT_DIR}/hgt/logs/hgt_{{wildcards.genome_id}}.log"
    shell:
        """
        python workflow/scripts/run_hgt.py \
            --genome {input.genome} \
            --annotations {input.annotations} \
            --pangenome-stats {input.pangenome_stats} \
            --output {output.predictions} \
            --contamination {params.contamination} \
            --n-estimators {params.n_estimators} \
            --random-state {params.random_state} \
            --mge-proximity-bp {params.mge_proximity_bp} \
            --min-seq-length {params.min_seq_length} \
            > {log} 2>&1
        """

rule bgc:
    input:
        genome = f"{GENOMES_DIR}/{{genome_id}}.fasta",
        annotations = f"{ANNOTATIONS_DIR}/{{genome_id}}.gff",
        hgt_predictions = f"{OUTPUT_DIR}/hgt/hgt_predictions_{{genome_id}}.csv"
    output:
        predictions = f"{OUTPUT_DIR}/bgc/bgc_predictions_{{genome_id}}.csv",
        model_weights = f"{MODEL_DIR}/annotator_{{genome_id}}.pt"
    params:
        seed = config["bgc"]["seed"],
        min_confidence = config["bgc"]["min_confidence"],
        use_keyword_boost = config["bgc"]["use_keyword_boost"],
        esm_model = config["bgc"]["esm_model"],
        device = config["bgc"]["device"],
        model_dir = config["paths"]["model_dir"]
    conda:
        "workflow/envs/bgc.yaml"
    threads: config["resources"]["bgc"]["threads"]
    resources:
        mem_mb = config["resources"]["bgc"]["mem_mb"],
        time = config["resources"]["bgc"]["time"],
        gpu = config["resources"]["bgc"]["gpu"]
    log:
        f"{OUTPUT_DIR}/bgc/logs/bgc_{{wildcards.genome_id}}.log"
    shell:
        """
        python workflow/scripts/run_bgc.py \
            --genome {input.genome} \
            --annotations {input.annotations} \
            --hgt-predictions {input.hgt_predictions} \
            --output {output.predictions} \
            --model-weights {output.model_weights} \
            --seed {params.seed} \
            --min-confidence {params.min_confidence} \
            --use-keyword-boost {params.use_keyword_boost} \
            --esm-model {params.esm_model} \
            --device {params.device} \
            --model-dir {params.model_dir} \
            > {log} 2>&1
        """

rule pangenome_visualize:
    input:
        matrix = f"{OUTPUT_DIR}/pangenome/pangenome_matrix.csv",
        stats = f"{OUTPUT_DIR}/pangenome/pangenome_stats.json"
    output:
        plot = f"{OUTPUT_DIR}/pangenome/visualization/pangenome_plot.html"
    params:
        output_dir = f"{OUTPUT_DIR}/pangenome/visualization"
    conda:
        "workflow/envs/pangenome.yaml"
    shell:
        """
        python pipeline/phase1_visualizer.py \
            --matrix {input.matrix} \
            --stats {input.stats} \
            --output {output.plot} \
            --output-dir {params.output_dir}
        """

rule hgt_visualize:
    input:
        hgt_predictions = expand(f"{OUTPUT_DIR}/hgt/hgt_predictions_{{genome_id}}.csv", genome_id=config["samples"]["sample_genomes"])
    output:
        plot = f"{OUTPUT_DIR}/hgt/visualization/hgt_plot.html"
    params:
        output_dir = f"{OUTPUT_DIR}/hgt/visualization"
    conda:
        "workflow/envs/hgt.yaml"
    shell:
        """
        python pipeline/phase2_visualizer.py \
            --hgt-predictions {input.hgt_predictions} \
            --output {output.plot} \
            --output-dir {params.output_dir}
        """

rule bgc_visualize:
    input:
        bgc_predictions = expand(f"{OUTPUT_DIR}/bgc/bgc_predictions_{{genome_id}}.csv", genome_id=config["samples"]["sample_genomes"])
    output:
        plot = f"{OUTPUT_DIR}/bgc/visualization/bgc_plot.html"
    params:
        output_dir = f"{OUTPUT_DIR}/bgc/visualization"
    conda:
        "workflow/envs/bgc.yaml"
    shell:
        """
        python pipeline/phase3_visualizer.py \
            --bgc-predictions {input.bgc_predictions} \
            --output {output.plot} \
            --output-dir {params.output_dir}
        """

rule generate_final_report:
    input:
        pangenome_plot = f"{OUTPUT_DIR}/pangenome/visualization/pangenome_plot.html",
        hgt_plot = f"{OUTPUT_DIR}/hgt/visualization/hgt_plot.html",
        bgc_plot = f"{OUTPUT_DIR}/bgc/visualization/bgc_plot.html",
        pangenome_stats = f"{OUTPUT_DIR}/pangenome/pangenome_stats.json",
        hgt_predictions = expand(f"{OUTPUT_DIR}/hgt/hgt_predictions_{{genome_id}}.csv", genome_id=config["samples"]["sample_genomes"]),
        bgc_predictions = expand(f"{OUTPUT_DIR}/bgc/bgc_predictions_{{genome_id}}.csv", genome_id=config["samples"]["sample_genomes"])
    output:
        report = f"{OUTPUT_DIR}/bgc/reports/final_report.html"
    params:
        pipeline_name = PIPELINE_NAME,
        pipeline_version = PIPELINE_VERSION,
        output_dir = f"{OUTPUT_DIR}/bgc/reports"
    conda:
        "workflow/envs/bgc.yaml"
    shell:
        """
        python workflow/scripts/generate_report.py \
            --pangenome-plot {input.pangenome_plot} \
            --hgt-plot {input.hgt_plot} \
            --bgc-plot {input.bgc_plot} \
            --pangenome-stats {input.pangenome_stats} \
            --hgt-predictions {input.hgt_predictions} \
            --bgc-predictions {input.bgc_predictions} \
            --output {output.report} \
            --pipeline-name {params.pipeline_name} \
            --pipeline-version {params.pipeline_version} \
            --output-dir {params.output_dir}
        """
