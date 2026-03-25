"""
Molecular Featurization Module.

Provides sophisticated featurization using Morgan fingerprints
and RDKit descriptors.
"""

import logging
from typing import List
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def _get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """
    Calculate atom features for a given atom.

    Args:
        atom: RDKit atom object.

    Returns:
        A numpy array of atom features.
    """
    # Features: one-hot encoding of atom symbol, plus other properties
    possible_atoms = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
    atom_type = [int(atom.GetSymbol() == s) for s in possible_atoms]

    # Other features
    atom_features = [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetNumRadicalElectrons(),
        atom.GetTotalNumHs(),
    ]

    return np.array(atom_type + atom_features, dtype=np.float32)


class MolecularFeaturizer:
    """
    Sophisticated molecular featurization using Morgan fingerprints and
    RDKit descriptors optimized for neural network input.
    """

    def __init__(self, radius: int = 2, n_bits: int = 256):
        """
        Initialize molecular featurizer.

        Args:
            radius: Radius for Morgan fingerprints
            n_bits: Number of bits for fingerprints
        """
        self.radius = radius
        self.n_bits = n_bits

    def featurize_molecule(self, smiles: str) -> np.ndarray:
        """
        Featurize molecule as Morgan fingerprint.

        Args:
            smiles: SMILES string

        Returns:
            Feature vector of shape (n_bits + n_descriptors,)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits + 8)

        # Morgan fingerprint (256 bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        morgan_features = np.array(fp, dtype=np.float32)

        # RDKit descriptors (8 additional features)
        from rdkit.Chem import Descriptors, Crippen

        mw = min(Descriptors.MolWt(mol) / 500.0, 1.0)  # Normalize to 0-1
        logp = (Crippen.MolLogP(mol) + 3) / 8.0  # Normalize to ~0-1
        hba = min(Descriptors.NumHAcceptors(mol) / 10.0, 1.0)
        hbd = min(Descriptors.NumHDonors(mol) / 5.0, 1.0)
        num_rings = min(Descriptors.RingCount(mol) / 4.0, 1.0)
        num_rotatable = min(Descriptors.NumRotatableBonds(mol) / 10.0, 1.0)
        tpsa = min(Descriptors.TPSA(mol) / 140.0, 1.0)
        aromatic_rings = min(Descriptors.NumAromaticRings(mol) / 4.0, 1.0)

        descriptor_features = np.array(
            [mw, logp, hba, hbd, num_rings, num_rotatable, tpsa, aromatic_rings],
            dtype=np.float32,
        )

        return np.concatenate([morgan_features, descriptor_features])

    def featurize_molecule_graph(self, smiles: str) -> Data:
        """
        Featurize a molecule into a graph representation for GNN input.

        Args:
            smiles: The SMILES string of the molecule.

        Returns:
            A torch_geometric.data.Data object.
            Returns None if the SMILES string is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get atom features
        atom_features_list = [_get_atom_features(atom) for atom in mol.GetAtoms()]
        atom_features = torch.from_numpy(np.array(atom_features_list, dtype=np.float32))

        # Get edge index and edge attributes
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_type = bond.GetBondTypeAsDouble()
            edge_indices.append((i, j))
            edge_attrs.append(bond_type)

            # Add reverse bond
            edge_indices.append((j, i))
            edge_attrs.append(bond_type)

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)

        return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr)

    def batch_featurize_molecules(self, smiles_list: List[str]) -> np.ndarray:
        """Featurize multiple molecules."""
        features = []
        for smiles in smiles_list:
            features.append(self.featurize_molecule(smiles))
        return np.array(features, dtype=np.float32)

    def featurize_protein(self, sequence: str, max_len: int = 512) -> np.ndarray:
        """
        Convert a protein amino-acid sequence into integer token indices suitable
        for embedding layers. Uses 20 standard amino acids mapped to 1..20 and
        0 reserved for padding.

        Args:
            sequence: Raw protein sequence (string of amino-acid single-letter codes)
            max_len: Maximum sequence length to pad/truncate to

        Returns:
            Numpy array of shape (max_len,) with dtype int64 containing indices
        """
        aa_list = [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ]
        aa_to_idx = {aa: i + 1 for i, aa in enumerate(aa_list)}  # 1..20, 0=padding

        seq = sequence.upper() if sequence is not None else ""
        indices = [aa_to_idx.get(ch, 0) for ch in seq]

        # Truncate or pad
        if len(indices) >= max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [0] * (max_len - len(indices))

        return np.array(indices, dtype=np.int64)
