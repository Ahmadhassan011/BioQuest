#!/usr/bin/env python3
"""BioQuest CLI — python -m cli <command> [args]

Commands:
    prepare DATASET         prepare a dataset: dti, toxicity, vae, property, all
    train [MODEL ...]       train models: dti, toxicity, vae, property, all (default)
    cache                   show cached dataset info
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(
        prog="bioquest",
        description="BioQuest — AI Drug Discovery",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prepare ---
    prep = sub.add_parser("prepare", help="Prepare a dataset (dti, toxicity, vae, property, all)")
    prep.add_argument("dataset", choices=["dti", "toxicity", "vae", "property", "all"],
                       help="Dataset to prepare")
    prep.add_argument("--scaffold", action="store_true", help="Use scaffold split")
    prep.add_argument("--assay", default="NR-AR", help="Tox21 assay (default: NR-AR)")
    prep.add_argument("--chembl-frac", type=float, default=0.052, help="ChEMBL fraction (default: 0.052)")
    prep.add_argument("--prop", default="Lipophilicity_AstraZeneca", help="Property dataset name")
    prep.add_argument("--save-config", help="Save pipeline config to path")

    # --- train ---
    tr = sub.add_parser("train", help="Train models: dti, toxicity, vae, property, all")
    tr.add_argument("models", nargs="*", choices=["dti", "toxicity", "vae", "property", "all"],
                    default=["all"], help="Models to train (default: all)")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate")
    tr.add_argument("--gpu", action="store_true", dest="use_gpu", help="Use GPU")
    tr.add_argument("--checkpoint-dir", default="artifacts/models")
    tr.add_argument("--assay", default="NR-AR", help="Tox21 assay")
    tr.add_argument("--chembl-frac", type=float, default=0.052)
    tr.add_argument("--config", help="Load PipelineConfig JSON")
    tr.add_argument("--save-config", help="Save pipeline config to path")

    # --- cache ---
    sub.add_parser("cache", help="Show cached dataset info")

    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if args.command == "prepare":
        from . import data as _data
        _data.run_prepare(args)
    elif args.command == "train":
        from . import train as _train
        _train.run(args)
    elif args.command == "cache":
        from . import data as _data
        _data.run_info(args)


if __name__ == "__main__":
    main()
