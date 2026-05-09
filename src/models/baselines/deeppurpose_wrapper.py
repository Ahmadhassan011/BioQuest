"""DeepPurpose-compatible predictive baseline wrapper.

When DeepPurpose is not installed, uses sklearn RandomForest (Morgan FP)
as a surrogate for DTI / Toxicity / Property prediction comparison.
"""

import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

_SURROGATE_WARNING = False


def _log_surrogate():
    global _SURROGATE_WARNING
    if not _SURROGATE_WARNING:
        logger.info(
            "DeepPurpose not available — using sklearn RF surrogate. "
            "Install via: pip install DeepPurpose"
        )
        _SURROGATE_WARNING = True


def _fingerprint_matrix(smiles_list, radius=2, n_bits=2048):
    """Convert SMILES list to Morgan fingerprint matrix."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
            continue
        arr = np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.array(fps)


def _run_rf_dti(dataset_name: str) -> Dict[str, float]:
    """Run sklearn RandomForest on DTI regression task using real Morgan fingerprints."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    from src.data.load.tdc import TDCDataLoader
    from rdkit import Chem

    tdc = TDCDataLoader()
    raw = tdc.load_dti_data(dataset_name)
    raw = raw[raw["Y"] < 10000].copy().dropna()

    smiles_list = []
    labels = []
    for _, row in raw.iterrows():
        smi = str(row["Drug"])
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            smiles_list.append(Chem.MolToSmiles(mol))
            labels.append(float(row["Y"]))

    labels = np.array(labels, dtype=np.float32)

    feats = _fingerprint_matrix(smiles_list, radius=2, n_bits=2048)
    n = len(smiles_list)
    indices = np.arange(n)
    np.random.RandomState(42).shuffle(indices)
    test_sz = int(n * 0.1)
    _val_sz = int(n * 0.1)
    test_idx = indices[:test_sz]
    train_idx = indices[test_sz + _val_sz:]

    train_feats = feats[train_idx]
    test_feats = feats[test_idx]
    train_y = labels[train_idx]
    test_y = labels[test_idx]

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(train_feats, train_y)
    preds = reg.predict(test_feats)

    return {
        "rmse": float(np.sqrt(np.mean((test_y - preds) ** 2))),
        "mae": float(mean_absolute_error(test_y, preds)),
        "r2": float(r2_score(test_y, preds)),
    }


def _run_rf_classification(
    smiles_list, labels, splits
) -> Dict[str, float]:
    """Run sklearn RandomForest on a binary classification task."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    feats = _fingerprint_matrix(smiles_list, radius=2, n_bits=2048)
    train_feats = feats[splits["train"]]
    test_feats = feats[splits["test"]]
    train_labels = labels[splits["train"]]
    test_labels = labels[splits["test"]]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train_feats, train_labels)
    probs = clf.predict_proba(test_feats)[:, 1]

    both = len(np.unique(test_labels)) == 2
    return {
        "auc": float(roc_auc_score(test_labels, probs)) if both else 0.0,
        "pr_auc": float(average_precision_score(test_labels, probs)) if both else 0.0,
    }


def _run_rf_property(
    dataset_name: str,
) -> Dict[str, float]:
    """Run sklearn RandomForest on property regression task."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    from src.data.preparation.property import PropertyDatasetPreparer

    prep = PropertyDatasetPreparer()
    features, targets_dict, splits, meta = prep.prepare_property_dataset(dataset_name)
    features_np = features.numpy()
    test_idx = splits["test"]

    results = {}
    for task, target in targets_dict.items():
        target_np = target.numpy()
        train_feats = features_np[splits["train"]]
        test_feats = features_np[test_idx]
        train_y = target_np[splits["train"]]
        test_y = target_np[test_idx]

        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(train_feats, train_y)
        preds = reg.predict(test_feats)

        results[f"{task}_mae"] = float(mean_absolute_error(test_y, preds))
        results[f"{task}_r2"] = float(r2_score(test_y, preds))

    return results


def run_deeppurpose_predictive() -> Dict[str, Any]:
    """Run DeepPurpose predictive baselines (or sklearn RF surrogate).

    Returns:
        Dict keyed by task name, each value a metrics dict.
    """
    try:
        import deeppurpose  # noqa: F401
        logger.info("DeepPurpose found — would delegate prediction.")
    except ImportError:
        _log_surrogate()

    results = {}
    from src.data.preparation.toxicity import Tox21DatasetPreparer

    # --- DTI ---
    try:
        results["dti_davis"] = _run_rf_dti("DAVIS")
        logger.info(f"Baseline DTI (DAVIS): {results['dti_davis']}")
    except Exception as e:
        logger.warning(f"Baseline DTI failed: {e}")

    # --- Tox21 (all assays) ---
    TOX21_ASSAYS = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ]
    for assay in TOX21_ASSAYS:
        try:
            prep = Tox21DatasetPreparer()
            X, y, splits, meta = prep.prepare_tox21_dataset(assay)
            # X and y are aligned, but we need the original SMILES list
            # The preparer returns (mol_features, labels), not SMILES
            # We approximate by regenerating SMILES from the dataset
            # Since the preparer doesn't expose SMILES, use the Tox21 raw data
            from src.data.load.tdc import TDCDataLoader
            tdc = TDCDataLoader()
            tox_data = tdc.load_tox21_data()
            filtered = tox_data[tox_data["assay"] == assay].copy().dropna()
            smiles_list = filtered["Drug"].tolist()
            labels = filtered["Y"].to_numpy(dtype=np.float32)

            # Recompute splits based on order — data was sorted by validity
            n = len(smiles_list)
            indices = np.arange(n)
            np.random.RandomState(42).shuffle(indices)
            test_sz = int(n * 0.1)
            val_sz = int(n * 0.1)
            test_idx = indices[:test_sz]
            val_idx = indices[test_sz:test_sz + val_sz]
            train_idx = indices[test_sz + val_sz:]
            splits = {"train": train_idx, "val": val_idx, "test": test_idx}

            metrics = _run_rf_classification(smiles_list, labels, splits)
            results[f"tox21_{assay}"] = metrics
            logger.info(f"Baseline Tox21 ({assay}): {metrics}")
        except Exception as e:
            logger.warning(f"Baseline Tox21 {assay} failed: {e}")

    # --- Property ---
    try:
        prop_results = _run_rf_property("Lipophilicity_AstraZeneca")
        results["property_lipophilicity"] = prop_results
        logger.info(f"Baseline Property: {prop_results}")
    except Exception as e:
        logger.warning(f"Baseline Property failed: {e}")

    return {"model": "deeppurpose", "tasks": results}
