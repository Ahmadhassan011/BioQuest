"""bioquest prepare / cache — dataset preparation."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logger = logging.getLogger(__name__)

DATASETS = {"dti", "toxicity", "vae", "property"}


def run_prepare(args) -> None:
    from src.data.preparation.dti import DTIDatasetPreparer
    from src.data.preparation.toxicity import Tox21DatasetPreparer
    from src.data.preparation.vae import VAEDatasetPreparer
    from src.data.preparation.property import PropertyDatasetPreparer

    targets = DATASETS if args.dataset == "all" else {args.dataset}

    for ds in targets:
        logger.info(f"Preparing {ds}...")

        if ds == "dti":
            prep = DTIDatasetPreparer()
            _, splits, meta = prep.prepare_dti_dataset(
                dataset_name="DAVIS",
                use_scaffold_split=args.scaffold,
            )
            logger.info(f"  DTI: {meta['total_samples']} graphs — "
                        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

        elif ds == "toxicity":
            prep = Tox21DatasetPreparer()
            X, y, splits, meta = prep.prepare_tox21_dataset(
                assay=args.assay,
                use_scaffold_split=args.scaffold,
            )
            logger.info(f"  Tox21 {args.assay}: X{X.shape} — "
                        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

        elif ds == "vae":
            prep = VAEDatasetPreparer()
            X, splits, meta = prep.prepare_vae_dataset(sample_frac=args.chembl_frac)
            logger.info(f"  ChEMBL ({args.chembl_frac}): {meta['total_samples']} sequences — "
                        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

        elif ds == "property":
            prep = PropertyDatasetPreparer()
            features, targets, splits, meta = prep.prepare_property_dataset(
                dataset_name=args.prop,
                use_scaffold_split=args.scaffold,
            )
            logger.info(f"  {args.prop}: {meta['total_samples']} compounds — "
                        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    if args.save_config:
        from src.utils.pipeline import PipelineConfig
        cfg = PipelineConfig(
            models=list(targets),
            tox_assay=args.assay,
            chembl_frac=args.chembl_frac,
            prop_dataset=args.prop,
        )
        cfg.save(args.save_config)


def run_info(args) -> None:
    from src.data.storage import DataCache
    DataCache.print_cache_summary()
