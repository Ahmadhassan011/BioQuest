"""ADMET property computation from SMILES using RDKit."""

import logging
from typing import Dict, List
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

logger = logging.getLogger(__name__)


def compute_admet_properties(smiles: str) -> Dict[str, float]:
    """Compute a comprehensive set of ADMET-related properties from SMILES.

    All properties are computed directly from RDKit (no trained models).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "num_rings": Descriptors.RingCount(mol),
        "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_h_atoms": Descriptors.NumHDonors(mol) + sum(
            1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1
        ),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "fraction_csp3": Descriptors.FractionCSP3(mol),
        "qed": Descriptors.qed(mol),
    }


def batch_compute_admet_properties(smiles_list: List[str]) -> Dict[str, np.ndarray]:
    """Compute ADMET properties for a batch of SMILES."""
    results = {}
    for smi in smiles_list:
        props = compute_admet_properties(smi)
        if not props:
            continue
        for key, val in props.items():
            results.setdefault(key, []).append(val)
    return {k: np.array(v, dtype=np.float32) for k, v in results.items()}


LIPINSKI_RULES = {
    "mw": (0, 500),
    "logp": (-float("inf"), 5),
    "hba": (0, 10),
    "hbd": (0, 5),
}


def check_lipinski_rule_of_five(props: Dict[str, float]) -> Dict[str, bool]:
    """Check Lipinski Rule of Five violations.

    Returns dict of rule_name -> passed (True/False).
    """
    violations = {}
    for rule, (lo, hi) in LIPINSKI_RULES.items():
        val = props.get(rule)
        if val is None:
            violations[rule] = False
        else:
            violations[rule] = lo <= val <= hi
    violations["total_violations"] = sum(1 for v in violations.values() if not v)
    violations["passes_lipinski"] = violations["total_violations"] <= 1
    return violations
