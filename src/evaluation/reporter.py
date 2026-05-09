"""Generate evaluation reports in JSON/HTML format."""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate evaluation reports in various formats."""

    def __init__(self, output_dir: str = "artifacts/reports"):
        """
        Initialize reporter.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(
        self,
        metrics: Dict[str, float],
        model_name: str,
        dataset_name: str,
        extra_data: Optional[Dict] = None,
    ) -> str:
        """
        Generate JSON evaluation report.

        Args:
            metrics: Evaluation metrics
            model_name: Name of the model
            dataset_name: Name of the dataset
            extra_data: Additional data to include

        Returns:
            Path to saved report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metrics": metrics,
        }

        if extra_data:
            report["extra"] = extra_data

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"JSON report saved to {filepath}")
        return str(filepath)

    def generate_summary_report(
        self,
        all_results: List[Dict[str, Any]],
        output_filename: str = "evaluation_summary.json",
    ) -> str:
        """
        Generate summary report from multiple evaluation runs.

        Args:
            all_results: List of evaluation result dictionaries
            output_filename: Name of output file

        Returns:
            Path to saved report
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(all_results),
            "results": all_results,
        }

        # Calculate aggregate statistics
        if all_results:
            metric_names = set()
            for result in all_results:
                metric_names.update(result.get("metrics", {}).keys())

            aggregates = {}
            for metric in metric_names:
                values = [
                    r["metrics"].get(metric, 0)
                    for r in all_results
                    if "metrics" in r
                ]
                if values:
                    aggregates[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }

            summary["aggregates"] = aggregates

        filepath = self.output_dir / output_filename

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary report saved to {filepath}")
        return str(filepath)

    def print_report(self, metrics: Dict[str, float], model_name: str) -> None:
        """
        Print evaluation report to console.

        Args:
            metrics: Evaluation metrics
            model_name: Name of the model
        """
        print(f"\n{'=' * 50}")
        print(f"Evaluation Report: {model_name}")
        print(f"{'=' * 50}")

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print(f"{'=' * 50}\n")

