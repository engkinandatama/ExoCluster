import argparse
import sys
from pathlib import Path
from loguru import logger
from pipeline.bgc_predictor import BGCPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--hgt-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-confidence", type=float, required=True)
    parser.add_argument("--use-keyword-boost", type=bool, required=True)
    parser.add_argument("--esm-model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    logger.info(f"Running BGC Prediction for {args.genome}")

    try:
        predictor = BGCPredictor(
            seed=args.seed,
            min_confidence=args.min_confidence,
            use_keyword_boost=args.use_keyword_boost,
            esm_model=args.esm_model,
            device=args.device,
            model_dir=args.model_dir,
        )

        predictions = predictor.run(args.genome, args.annotations, args.hgt_predictions)

        predictions.to_csv(args.output, index=False)
        # Note: model_weights output logic depends on existing BGC predictor implementation
        # Placeholder for weight saving if not handled by predictor

        logger.info("BGC Prediction completed successfully.")
    except Exception as e:
        logger.error(f"BGC Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
