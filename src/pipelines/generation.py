"""
Generator Module: Hybrid molecule generation using RDKit evolutionary algorithms.

This module provides:
- RDKit-based evolutionary algorithms for guided exploration
- Hybrid generation combining VAE and evolutionary approaches
- SMILES validity checking and canonicalization
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
from rdkit import Chem
import random

from src.models.vae import MoleculeVAE

logger = logging.getLogger(__name__)


class RDKitEvolutionaryGenerator:
    """RDKit-based evolutionary algorithm for molecule generation."""

    def __init__(self, mutation_rate: float = 0.3, crossover_rate: float = 0.7):
        """
        Initialize evolutionary generator.

        Args:
            mutation_rate: Probability of mutation per molecule
            crossover_rate: Probability of crossover
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_operations = [
            self._add_bond,
            self._remove_bond,
            self._change_atom,
            self._add_ring,
            self._remove_atom,
        ]

    def _add_bond(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add a random bond to molecule."""
        try:
            atoms = list(range(mol.GetNumAtoms()))
            if len(atoms) < 2:
                return mol

            atom1, atom2 = random.sample(atoms, 2)
            if mol.GetBondBetweenAtoms(atom1, atom2):
                return mol

            editable_mol = Chem.EditableMol(mol)
            editable_mol.AddBond(atom1, atom2, Chem.BondType.SINGLE)
            new_mol = editable_mol.GetMol()

            try:
                Chem.SanitizeMol(new_mol)
                return new_mol
            except ValueError:
                return mol
        except ValueError:
            return mol

    def _remove_bond(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Remove a random bond from molecule."""
        try:
            if mol.GetNumBonds() == 0:
                return mol

            bond = random.choice(list(mol.GetBonds()))
            editable_mol = Chem.EditableMol(mol)
            editable_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            new_mol = editable_mol.GetMol()

            try:
                Chem.SanitizeMol(new_mol)
                return new_mol
            except ValueError:
                return mol
        except (ValueError, IndexError):
            return mol

    def _change_atom(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Change a random atom in molecule."""
        try:
            atom = random.choice(list(mol.GetAtoms()))
            atomic_numbers = [1, 6, 7, 8, 9, 16, 17, 35]  # H, C, N, O, F, S, Cl, Br
            new_atomic_num = random.choice(atomic_numbers)

            if new_atomic_num == atom.GetAtomicNum():
                return mol

            editable_mol = Chem.EditableMol(mol)
            atom_idx = atom.GetIdx()
            editable_mol.ReplaceAtom(atom_idx, Chem.Atom(new_atomic_num))
            new_mol = editable_mol.GetMol()

            try:
                Chem.SanitizeMol(new_mol)
                return new_mol
            except ValueError:
                return mol
        except (ValueError, IndexError):
            return mol

    def _add_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add a small ring to molecule."""
        try:
            if mol.GetNumAtoms() < 2:
                return mol

            # Simple ring addition
            atoms = list(range(mol.GetNumAtoms()))
            atom1, atom2 = random.sample(atoms, 2)

            editable_mol = Chem.EditableMol(mol)
            new_atom_idx = editable_mol.AddAtom(Chem.Atom(6))

            if not mol.GetBondBetweenAtoms(atom1, atom2):
                editable_mol.AddBond(atom1, new_atom_idx, Chem.BondType.SINGLE)
                editable_mol.AddBond(new_atom_idx, atom2, Chem.BondType.SINGLE)
            else:
                return mol

            new_mol = editable_mol.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
                return new_mol
            except ValueError:
                return mol
        except ValueError:
            return mol

    def _remove_atom(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Remove a random atom from molecule."""
        try:
            if mol.GetNumAtoms() < 2:
                return mol

            atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
            editable_mol = Chem.EditableMol(mol)
            editable_mol.RemoveAtom(atom_idx)
            new_mol = editable_mol.GetMol()

            try:
                Chem.SanitizeMol(new_mol)
                return new_mol
            except ValueError:
                return mol
        except (ValueError, IndexError):
            return mol

    def mutate(self, smiles: str) -> Optional[str]:
        """
        Apply random mutation to molecule SMILES.

        Args:
            smiles: Molecule SMILES string

        Returns:
            Mutated SMILES or None if mutation fails
        """
        if random.random() > self.mutation_rate:
            return smiles

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        operation = random.choice(self.mutation_operations)
        new_mol = operation(mol)

        if new_mol is None:
            return None

        new_smiles = Chem.MolToSmiles(new_mol)
        return new_smiles if new_smiles else None

    def crossover(self, smiles1: str, smiles2: str) -> Optional[str]:
        """
        Perform crossover between two molecules.

        Args:
            smiles1: First parent SMILES
            smiles2: Second parent SMILES

        Returns:
            Offspring SMILES or None
        """
        if random.random() > self.crossover_rate:
            return random.choice([smiles1, smiles2])

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return random.choice([smiles1, smiles2])

        # Simple crossover: combine fragments
        try:
            # Get common substructures and combine
            if mol1.GetNumAtoms() > 0 and mol2.GetNumAtoms() > 0:
                # Randomly choose which parent to start from
                parent = random.choice([mol1, mol2])
                other = mol2 if parent == mol1 else mol1

                # Simple approach: return parent
                return Chem.MolToSmiles(parent)
        except ValueError:
            pass

        return random.choice([smiles1, smiles2])


class HybridMoleculeGenerator:
    """Hybrid generator combining VAE and RDKit evolutionary approaches."""

    def __init__(
        self,
        vae_model: Optional[MoleculeVAE] = None,
        vae_enabled: bool = True,
        evolutionary_enabled: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize hybrid generator.

        Args:
            vae_model: Pre-trained VAE model (optional)
            vae_enabled: Whether to use VAE
            evolutionary_enabled: Whether to use evolutionary algorithm
            device: Computation device
        """
        self.device = torch.device(device)
        self.vae = vae_model
        self.evolutionary = RDKitEvolutionaryGenerator()
        self.vae_enabled = vae_enabled and vae_model is not None
        self.evolutionary_enabled = evolutionary_enabled

        # SMILES vocabulary - expanded to include all common SMILES characters
        # Uppercase atoms, lowercase aromatic atoms, digits, special chars
        self.smiles_chars = (
            "CNOPSFClBrIBr"  # Uppercase atoms
            "nops"  # Aromatic atoms
            "0123456789"  # Ring closures and counts
            "()[]#"  # Branches and rings
            "=-+"  # Bond types
            "/\\@"  # Stereochemistry and chirality
            "%"  # Double-digit ring closures
        )
        self.char_to_idx = {c: i for i, c in enumerate(self.smiles_chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        logger.info(
            f"Hybrid generator initialized (VAE: {self.vae_enabled}, Evolutionary: {self.evolutionary_enabled})"
        )

    def generate_from_seeds(
        self,
        seeds: List[str],
        num_molecules: int = 10,
        mutations_per_seed: int = 2,
    ) -> List[str]:
        """
        Generate molecules from seed compounds.

        Args:
            seeds: List of seed SMILES
            num_molecules: Target number of molecules to generate
            mutations_per_seed: Number of mutations per seed

        Returns:
            List of generated SMILES
        """
        generated = set(seeds)

        for _ in range(mutations_per_seed):
            for seed in seeds:
                for _ in range(num_molecules // len(seeds)):
                    # Evolutionary mutation
                    mutated = self.evolutionary.mutate(seed)
                    if mutated and self._is_valid_smiles(mutated):
                        generated.add(mutated)

                    # Crossover with other seeds
                    if len(seeds) > 1:
                        other_seed = random.choice([s for s in seeds if s != seed])
                        crossed = self.evolutionary.crossover(seed, other_seed)
                        if crossed and self._is_valid_smiles(crossed):
                            generated.add(crossed)

        return list(generated)[:num_molecules]

    def generate_from_latent_space(
        self,
        num_molecules: int = 10,
        latent_dim: int = 64,
    ) -> List[str]:
        """
        Generate molecules by sampling latent space.

        Args:
            num_molecules: Number of molecules to generate
            latent_dim: Latent space dimension

        Returns:
            List of generated SMILES
        """
        if not self.vae_enabled:
            logger.warning("VAE not enabled, cannot generate from latent space")
            return []

        generated = []
        self.vae.eval()

        with torch.no_grad():
            for _ in range(num_molecules):
                # Sample from standard normal distribution
                z = torch.randn(1, latent_dim).to(self.device)

                # Decode to molecule
                output_tokens = self.vae._decode_tokens(z)
                smiles = self._decode_smiles(output_tokens[0].cpu().numpy())

                if self._is_valid_smiles(smiles):
                    generated.append(smiles)

        return generated

    def generate_hybrid(
        self,
        seeds: List[str],
        num_molecules: int = 10,
    ) -> List[str]:
        """
        Generate molecules using hybrid approach.

        Args:
            seeds: List of seed SMILES
            num_molecules: Target number of molecules

        Returns:
            List of generated SMILES
        """
        generated = set()

        # 60% from evolutionary mutations
        if self.evolutionary_enabled:
            evolutionary_mols = self.generate_from_seeds(
                seeds,
                num_molecules=int(num_molecules * 0.6),
                mutations_per_seed=3,
            )
            generated.update(evolutionary_mols)

        # 40% from VAE latent space
        if self.vae_enabled:
            vae_mols = self.generate_from_latent_space(
                num_molecules=int(num_molecules * 0.4),
            )
            generated.update(vae_mols)

        return list(generated)[:num_molecules]

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is valid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except ValueError:
            return False

    def _decode_smiles(self, indices: np.ndarray) -> str:
        """Decode integer array to SMILES string."""
        try:
            smiles = ""
            for idx in indices:
                if idx < len(self.idx_to_char):
                    smiles += self.idx_to_char[int(idx)]
            return smiles
        except (ValueError, KeyError, IndexError):
            return ""

    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Canonicalize SMILES string.

        Args:
            smiles: Input SMILES string

        Returns:
            Canonical SMILES or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except ValueError:
            return None

    def get_unique_molecules(self, smiles_list: List[str]) -> List[str]:
        """
        Get unique molecules from list (based on canonical SMILES).

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of unique canonical SMILES
        """
        unique = set()
        for smiles in smiles_list:
            canonical = self.canonicalize_smiles(smiles)
            if canonical:
                unique.add(canonical)
        return list(unique)

    def batch_canonicalize(self, smiles_list: List[str]) -> List[str]:
        """Canonicalize multiple SMILES."""
        result = []
        for smiles in smiles_list:
            canonical = self.canonicalize_smiles(smiles)
            if canonical:
                result.append(canonical)
        return result
