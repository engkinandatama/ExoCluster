import argparse
import sys
from pathlib import Path
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pangenome-plot", required=True)
    parser.add_argument("--hgt-plot", required=True)
    parser.add_argument("--bgc-plot", required=True)
    parser.add_argument("--pangenome-stats", required=True)
    parser.add_argument("--hgt-predictions", nargs='+', required=True)
    parser.add_argument("--bgc-predictions", nargs='+', required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--pipeline-version", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    logger.info("Generating final unified report...")
    
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simple HTML report generation
        html_content = f"""
        <html>
        <head>
            <title>{args.pipeline_name} Report v{args.pipeline_version}</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; }}
                .section {{ margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .section h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{args.pipeline_name} Analysis Report</h1>
                <p style="text-align: center;">Version: {args.pipeline_version}</p>
                
                <div class="section">
                <h2>Pangenome Analysis</h2>
                    <iframe src="{args.pangenome_plot}"></iframe>
                </div>
                
                <div class="section">
                <h2>HGT Detection</h2>
                    <iframe src="{args.hgt_plot}"></iframe>
                </div>
                
                <div class="section">
                <h2>BGC Prediction</h2>
                    <iframe src="{args.bgc_plot}"></iframe>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Final report generated at {args.output}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()