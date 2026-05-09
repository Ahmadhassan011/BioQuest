"""Generation metrics for molecule quality evaluation."""

import logging
from typing import List, Dict, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)


def compute_validity(molecules: List[str]) -> float:
    """Fraction of chemically valid SMILES."""
    if not molecules:
        return 0.0
    valid = sum(1 for smi in molecules if Chem.MolFromSmiles(smi) is not None)
    return valid / len(molecules)


def compute_uniqueness(molecules: List[str]) -> float:
    """Fraction of non-duplicate molecules among valid ones."""
    valid = [Chem.MolToSmiles(Chem.MolFromSmiles(smi))
             for smi in molecules if Chem.MolFromSmiles(smi) is not None]
    if not valid:
        return 0.0
    return len(set(valid)) / len(valid)


def compute_novelty(generated: List[str], reference: List[str]) -> float:
    """Fraction of generated molecules not present in reference set."""
    ref_set = set()
    for smi in reference:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ref_set.add(Chem.MolToSmiles(mol))

    if not generated:
        return 0.0

    novel = 0
    total_valid = 0
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            total_valid += 1
            canon = Chem.MolToSmiles(mol)
            if canon not in ref_set:
                novel += 1

    return novel / total_valid if total_valid > 0 else 0.0


def compute_internal_diversity(molecules: List[str]) -> float:
    """Average pairwise Tanimoto dissimilarity among valid molecules."""
    from rdkit.Chem import AllChem, DataStructs

    fps = []
    for smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)

    if len(fps) < 2:
        return 0.0

    scores = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            scores.append(1.0 - sim)

    return float(np.mean(scores)) if scores else 0.0


def compute_qed_sa_distribution(molecules: List[str]) -> Dict[str, float]:
    """Compute mean QED and SA scores for a set of molecules."""
    qeds, sas = [], []
    for smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            qeds.append(Descriptors.qed(mol))
            sas.append(_compute_sa_score(mol))
    return {
        "qed_mean": float(np.mean(qeds)) if qeds else 0.0,
        "qed_std": float(np.std(qeds)) if qeds else 0.0,
        "sa_mean": float(np.mean(sas)) if sas else 0.0,
        "sa_std": float(np.std(sas)) if sas else 0.0,
    }


def compute_kl_divergence(
    generated_props: Dict[str, List[float]],
    reference_props: Dict[str, List[float]],
    bins: int = 20,
) -> Dict[str, float]:
    """Compute KL divergence between property distributions."""
    kl_results = {}
    for prop_name in generated_props:
        if prop_name not in reference_props:
            continue
        gen = np.array(generated_props[prop_name])
        ref = np.array(reference_props[prop_name])
        if len(gen) == 0 or len(ref) == 0:
            kl_results[prop_name] = float("inf")
            continue
        all_vals = np.concatenate([gen, ref])
        if np.all(all_vals == all_vals[0]):
            kl_results[prop_name] = 0.0
            continue
        bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
        gen_hist, _ = np.histogram(gen, bins=bin_edges, density=True)
        ref_hist, _ = np.histogram(ref, bins=bin_edges, density=True)
        gen_hist = gen_hist + 1e-10
        ref_hist = ref_hist + 1e-10
        gen_hist /= gen_hist.sum()
        ref_hist /= ref_hist.sum()
        kl = np.sum(gen_hist * np.log(gen_hist / ref_hist))
        kl_results[prop_name] = float(kl)
    return kl_results


def compute_fcd_score(generated: List[str], reference: List[str]) -> float:
    """Compute Fréchet ChemNet Distance (FCD) proxy using Morgan fingerprints.

    Uses ECFP6 (radius=3, 2048 bits) as the feature space instead of
    ChemNet activations.  This is a common approximation when the full
    ChemNet model is not available.

    Args:
        generated: List of generated SMILES.
        reference: List of reference (training set) SMILES.

    Returns:
        FCD score (lower = more similar distributions).
        Returns inf if either set has <2 valid molecules.
    """
    from rdkit.Chem import AllChem, DataStructs
    from scipy.linalg import sqrtm

    def _fingerprint_matrix(smiles_list):
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            arr = np.zeros(2048, dtype=np.float32)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        return np.array(fps) if fps else None

    mu_gen = _fingerprint_matrix(generated)
    mu_ref = _fingerprint_matrix(reference)

    if mu_gen is None or mu_ref is None or len(mu_gen) < 2 or len(mu_ref) < 2:
        return float("inf")

    mean_gen = np.mean(mu_gen, axis=0)
    mean_ref = np.mean(mu_ref, axis=0)
    cov_gen = np.cov(mu_gen, rowvar=False)
    cov_ref = np.cov(mu_ref, rowvar=False)

    diff = mean_gen - mean_ref
    cov_mean = sqrtm(cov_gen @ cov_ref, disp_maxiter=1000)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    return float(np.real(diff @ diff + np.trace(cov_gen + cov_ref - 2 * cov_mean)))


def _compute_sa_score(mol: Chem.Mol) -> float:
    """Estimate synthetic accessibility (0 = hard, 1 = easy)."""
    num_atoms = mol.GetNumAtoms()
    return 1.0 - min(np.log10(num_atoms + 1) / 2.0, 1.0)


def compute_all_generation_metrics(
    generated: List[str],
    reference_smiles: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute all generation metrics at once."""
    metrics = {
        "validity": compute_validity(generated),
        "uniqueness": compute_uniqueness(generated),
        "internal_diversity": compute_internal_diversity(generated),
    }
    dist = compute_qed_sa_distribution(generated)
    metrics.update({f"gen_{k}": v for k, v in dist.items()})

    if reference_smiles is not None:
        metrics["novelty"] = compute_novelty(generated, reference_smiles)
        metrics["fcd"] = compute_fcd_score(generated, reference_smiles)
        gen_qeds = []
        gen_sas = []
        for smi in generated:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                gen_qeds.append(Descriptors.qed(mol))
                gen_sas.append(_compute_sa_score(mol))
        ref_qeds = []
        ref_sas = []
        for smi in reference_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                ref_qeds.append(Descriptors.qed(mol))
                ref_sas.append(_compute_sa_score(mol))
        kl = compute_kl_divergence(
            {"qed": gen_qeds, "sa": gen_sas},
            {"qed": ref_qeds, "sa": ref_sas},
        )
        metrics["kl_div_qed"] = kl.get("qed", float("inf"))
        metrics["kl_div_sa"] = kl.get("sa", float("inf"))

    return metrics
