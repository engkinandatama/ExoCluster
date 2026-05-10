import argparse
import sys
import json
from pathlib import Path
from loguru import logger
from pipeline.pangenome_miner import PangenomeMiner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genomes", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--core-threshold", type=float, required=True)
    parser.add_argument("--accessory-threshold", type=float, required=True)
    parser.add_argument("--identity", type=float, required=True)
    parser.add_argument("--cluster-method", required=True)
    parser.add_argument("--min-seq-length", type=int, required=True)
    args = parser.parse_args()

    logger.info(f"Running Pangenome Analysis for {args.genomes}")
    
    try:
        miner = PangenomeMiner(
            core_threshold=args.core_threshold,
            accessory_threshold=args.accessory_threshold,
            identity=args.identity,
            cluster_method=args.cluster_method,
            min_seq_length=args.min_seq_length
        )
        
        matrix, stats = miner.run(args.genomes, args.annotations)
        
        matrix.to_csv(args.output)
        with open(args.stats, 'w') as f:
            json.dump(stats, f, indent=4)
            
        logger.info("Pangenome Analysis completed successfully.")
    except Exception as e:
        logger.error(f"Pangenome Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()