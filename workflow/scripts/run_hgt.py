import argparse
import sys
import json
from pathlib import Path
from loguru import logger
from pipeline.hgt_detective import HGTDetective

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--pangenome-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contamination", type=float, required=True)
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument("--random-state", type=int, required=True)
    parser.add_argument("--mge-proximity-bp", type=int, required=True)
    parser.add_argument("--min-seq-length", type=int, required=True)
    args = parser.parse_args()

    logger.info(f"Running HGT Detection for {args.genome}")
    
    try:
        with open(args.pangenome_stats, 'r') as f:
            stats = json.load(f)

        detective = HGTDetective(
            contamination=args.contamination,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            mge_proximity_bp=args.mge_proximity_bp,
            min_seq_length=args.min_seq_length
        )
        
        predictions = detective.run(args.genome, args.annotations, stats)
        
        predictions.to_csv(args.output, index=False)
            
        logger.info("HGT Detection completed successfully.")
    except Exception as e:
        logger.error(f"HGT Detection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()