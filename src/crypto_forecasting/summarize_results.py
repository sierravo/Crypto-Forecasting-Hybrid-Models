"""
Summarize evaluation outputs into a single comparison table.

Run from the project root after evaluating models/baselines:

    python -m crypto_forecasting.summarize_results

This reads results/*_metrics.json and optional results/*_eval_config.json files,
then writes results/summary_metrics.csv.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_results(results_dir=RESULTS_DIR, output_name="summary_metrics.csv"):
    """
    Build a compact CSV summary from metrics/config JSON files.

    Args:
        results_dir: directory containing *_metrics.json outputs.
        output_name: filename for the summary CSV.

    Returns:
        pandas.DataFrame with one row per evaluated model/baseline.
    """
    results_dir = Path(results_dir)
    rows = []

    for metrics_path in sorted(results_dir.glob("*_metrics.json")):
        stem = metrics_path.name.replace("_metrics.json", "")
        metrics = _load_json(metrics_path)

        config_path = results_dir / f"{stem}_eval_config.json"
        config = _load_json(config_path) if config_path.exists() else {}

        row = {
            "model_name": stem,
            "eval_model": config.get("eval_model"),
            "adjacency_mode": config.get("adjacency_mode"),
            "rolling_window": config.get("rolling_window"),
            "seq_len": config.get("seq_len"),
            "seed": config.get("seed"),
            "use_mock_data": config.get("use_mock_data"),
            "mse": metrics.get("mse"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "directional_accuracy": metrics.get("directional_accuracy"),
            "n_observations": metrics.get("n_observations"),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(["rmse", "mae"], na_position="last")

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / output_name
    summary.to_csv(output_path, index=False)
    return summary


def main(results_dir=RESULTS_DIR, output_name="summary_metrics.csv"):
    summary = summarize_results(results_dir=results_dir, output_name=output_name)
    output_path = Path(results_dir) / output_name
    print(f"Wrote {len(summary)} rows to {output_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(RESULTS_DIR),
                        help="Directory containing *_metrics.json files")
    parser.add_argument("--output_name", default="summary_metrics.csv",
                        help="Output CSV filename")
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_name=args.output_name)
