FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Snakemake
RUN conda install -c conda-forge -c bioconda snakemake mamba

# Copy environment files
COPY workflow/envs/ /app/workflow/envs/

# Create environments
RUN mamba env create -f /app/workflow/envs/pangenome.yaml && \
    mamba env create -f /app/workflow/envs/hgt.yaml && \
    mamba env create -f /app/workflow/envs/bgc.yaml

# Copy project files
COPY . /app/

# Set environment
ENV PATH /opt/conda/envs/pangenome/bin:$PATH

ENTRYPOINT ["snakemake"]