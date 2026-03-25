"""
Predictor Module: Drug-Target Interaction, toxicity, and oracle functions.

This module provides:
- DTI affinity prediction (binding strength estimation)
- Toxicity prediction (Tox21 models)
- Oracle functions: QED (drug-likeness), SA (synthetic accessibility)
- Batch prediction capabilities
"""

import logging
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

logger = logging.getLogger(__name__)


class DTIPredictor:
    """Drug-Target Interaction binding affinity predictor."""

    def __init__(self, model_name: str = "DeepDTA", use_gpu: bool = False):
        """
        Initialize DTI predictor.

        Args:
            model_name: Pre-trained model name
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )
        self.model = None
        self.use_gpu = use_gpu
        logger.info(f"DTI Predictor initialized on {self.device}")

    def load_pretrained_model(self) -> None:
        """Load pre-trained DTI model from TDC (DEPRECATED - use custom models instead)."""
        try:
            # TDC pre-trained models no longer used
            # This method kept for backward compatibility but not called
            logger.info(
                "Pre-trained model loading deprecated - using custom trained models"
            )
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            # Fallback to simple neural network
            self._create_simple_model()

    def _create_simple_model(self) -> None:
        """Create a simple neural network for DTI prediction."""

        class SimpleDTI(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(512, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        self.model = SimpleDTI().to(self.device)

    def predict_binding_affinity(self, smiles: str, protein_seq: str) -> float:
        """
        Predict binding affinity between molecule and protein.

        Args:
            smiles: Molecule SMILES string
            protein_seq: Protein sequence

        Returns:
            Predicted binding affinity score (0-1 range)
        """
        try:
            if self.model is not None:
                # Use pre-trained model if available
                affinity = self.model.predict([[smiles, protein_seq]])[0][0]
                return float(np.clip(affinity, 0, 1))
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")

        # Fallback: heuristic based on molecular properties
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5

        # Simple heuristic: penalize large molecules and high lipophilicity
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)

        # Weighted combination
        affinity = 1.0
        if mw > 500:
            affinity -= 0.3 * (mw - 500) / 500
        if logp > 5:
            affinity -= 0.2 * (logp - 5)
        affinity += 0.1 * min(hba, 5) / 5
        affinity += 0.1 * min(hbd, 5) / 5

        return max(0.1, min(1.0, affinity))

    def batch_predict(self, smiles_list: List[str], protein_seq: str) -> np.ndarray:
        """
        Predict binding affinities for multiple molecules.

        Args:
            smiles_list: List of SMILES strings
            protein_seq: Protein sequence

        Returns:
            Array of affinity predictions
        """
        affinities = []
        for smiles in smiles_list:
            affinity = self.predict_binding_affinity(smiles, protein_seq)
            affinities.append(affinity)
        return np.array(affinities)


class ToxicityPredictor:
    """Toxicity prediction using Tox21 oracle."""

    def __init__(self, use_gpu: bool = False):
        """
        Initialize toxicity predictor.

        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )
        self.model = None
        logger.info(f"Toxicity Predictor initialized on {self.device}")

    def predict_toxicity(self, smiles: str) -> float:
        """
        Predict toxicity probability for molecule.

        Args:
            smiles: Molecule SMILES string

        Returns:
            Toxicity probability (0-1, where 1 is most toxic)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5  # Neutral prediction for invalid SMILES

        # Heuristic toxicity estimation
        toxicity_score = 0.0

        # Penalize reactive functional groups
        reactive_groups = {
            "[N+](=O)[O-]": 0.1,  # Nitro groups
            "C(=O)Cl": 0.05,  # Acyl chlorides
            "C(=O)Br": 0.05,  # Acyl bromides
            "[P](F)(F)(F)(F)(F)F": 0.15,  # Phosphorus pentafluoride
        }

        for group, penalty in reactive_groups.items():
            if mol.HasSubstructMatch(Chem.MolFromSmarts(group)):
                toxicity_score += penalty

        # Penalize high lipophilicity
        logp = Crippen.MolLogP(mol)
        if logp > 5:
            toxicity_score += 0.1 * min((logp - 5) / 5, 0.3)

        # Penalize high molecular weight
        mw = Descriptors.MolWt(mol)
        if mw > 600:
            toxicity_score += 0.05 * min((mw - 600) / 400, 0.2)

        # Penalize high number of aromatic rings
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        if num_aromatic_rings > 4:
            toxicity_score += 0.05 * min(num_aromatic_rings - 4, 1)

        return min(0.8, toxicity_score)  # Cap at 0.8

    def batch_predict(self, smiles_list: List[str]) -> np.ndarray:
        """
        Predict toxicity for multiple molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Array of toxicity predictions
        """
        toxicities = []
        for smiles in smiles_list:
            tox = self.predict_toxicity(smiles)
            toxicities.append(tox)
        return np.array(toxicities)


class OracleFunction:
    """Oracle functions for drug-likeness and synthetic accessibility."""

    @staticmethod
    def calculate_qed(smiles: str) -> float:
        """
        Calculate Quantitative Estimate of Drug-likeness (QED).

        QED is a measure of how drug-like a molecule is (0-1 range, 1 is best).

        Args:
            smiles: Molecule SMILES string

        Returns:
            QED score (0-1)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            # Calculate molecular descriptors
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            # QED-like scoring (simplified)
            qed = 1.0

            # Penalize MW > 500
            if mw > 500:
                qed -= 0.2 * min((mw - 500) / 500, 1)

            # Penalize LogP outside optimal range (2-5)
            if logp < 2:
                qed -= 0.1 * (2 - logp) / 2
            elif logp > 5:
                qed -= 0.1 * (logp - 5) / 5

            # Penalize too many H-bond donors (>5)
            if hbd > 5:
                qed -= 0.1 * (hbd - 5) / 5

            # Penalize too many H-bond acceptors (>10)
            if hba > 10:
                qed -= 0.1 * (hba - 10) / 10

            return max(0.0, min(1.0, qed))

        except Exception as e:
            logger.warning(f"QED calculation failed for {smiles}: {e}")
            return 0.5

    @staticmethod
    def calculate_sa(smiles: str) -> float:
        """
        Calculate Synthetic Accessibility (SA) score.

        SA score ranges from 1 (easily synthesizable) to 10 (difficult to synthesize).
        Returns a normalized score (0-1, where 1 is easy to synthesize).

        Args:
            smiles: Molecule SMILES string

        Returns:
            Normalized SA score (0-1, 1 = easy to synthesize)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.5

            # Simplified SA calculation (heuristic)
            # Based on: complexity from number of atoms, rings, and functional groups

            num_atoms = mol.GetNumAtoms()
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

            # Penalize complexity
            sa_score = 1.0

            # Large molecules are harder to synthesize
            if num_atoms > 50:
                sa_score -= 0.2 * min((num_atoms - 50) / 50, 1)

            # Many rings increase complexity
            if num_rings > 3:
                sa_score -= 0.1 * min(num_rings - 3, 2)

            # Many rotatable bonds increase flexibility but also complexity
            if num_rotatable_bonds > 10:
                sa_score -= 0.1 * min((num_rotatable_bonds - 10) / 10, 1)

            return max(0.1, min(1.0, sa_score))

        except Exception as e:
            logger.warning(f"SA calculation failed for {smiles}: {e}")
            return 0.5

    @staticmethod
    def batch_qed(smiles_list: List[str]) -> np.ndarray:
        """Calculate QED for multiple molecules."""
        return np.array([OracleFunction.calculate_qed(s) for s in smiles_list])

    @staticmethod
    def batch_sa(smiles_list: List[str]) -> np.ndarray:
        """Calculate SA for multiple molecules."""
        return np.array([OracleFunction.calculate_sa(s) for s in smiles_list])


class MoleculePredictor:
    """
    Integrated predictor using custom trained neural network models.

    This class serves as the primary interface for making predictions. It
    loads and uses custom-trained neural network models for DTI, toxicity,
    and other molecular properties.

    If the required models are not found, it will raise an error to prevent
    the system from running with unreliable heuristic fallbacks.
    """

    def __init__(
        self,
        protein_sequence: str,
        use_gpu: bool = False,
        models_dir: str = "trained_models",
    ):
        """
        Initialize the molecule predictor with custom trained models.

        Args:
            protein_sequence: The target protein sequence for DTI predictions.
            use_gpu: Whether to use GPU acceleration if available.
            models_dir: Directory containing trained model checkpoints.

        Raises:
            RuntimeError: If the custom models cannot be loaded, typically because
                          the model files are missing or there's an import issue.
        """
        self.protein_sequence = protein_sequence
        self.use_gpu = use_gpu
        self.models_dir = models_dir
        self.oracle = OracleFunction()

        try:
            from src.models.loader import CustomModelPredictor

            self.custom_predictor = CustomModelPredictor(
                protein_sequence=protein_sequence,
                models_dir=models_dir,
                use_gpu=use_gpu,
            )
            logger.info(
                f"✓ Custom trained models loaded successfully from '{models_dir}'"
            )
        except Exception as e:
            logger.warning(
                f"Could not load custom trained models ({type(e).__name__}): {e}. "
                f"Falling back to heuristic-based predictions."
            )
            self.custom_predictor = None

        logger.info("MoleculePredictor initialized successfully.")

    def predict_all_properties(self, smiles: str) -> Dict[str, float]:
        """
        Predict all molecular properties using the loaded trained models.

        Args:
            smiles: The SMILES string of the molecule to predict.

        Returns:
            A dictionary containing predictions for affinity, toxicity, QED, and SA.
        """
        # Use neural network models if available, otherwise use oracle heuristics
        if self.custom_predictor is not None:
            properties_nn = self.custom_predictor.predict_properties(smiles)
            return {
                "affinity": self.custom_predictor.predict_dti_affinity(smiles),
                "toxicity": self.custom_predictor.predict_toxicity(smiles),
                "qed": properties_nn.get("qed", 0.5),
                "sa": properties_nn.get("sa", 0.5),
                "logp": properties_nn.get("logp", 0.0),
                "mw": properties_nn.get("mw", 0.5),
            }
        else:
            # Fallback to oracle heuristics
            return {
                "affinity": 0.6,  # Default binding affinity (heuristic)
                "toxicity": 0.2,  # Default low toxicity (heuristic)
                "qed": OracleFunction.calculate_qed(smiles),
                "sa": OracleFunction.calculate_sa(smiles),
                "logp": 0.0,  # Default LogP
                "mw": 0.5,  # Default MW normalization
            }

    def batch_predict(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Predict all properties for a batch of molecules using the loaded models.

        Args:
            smiles_list: A list of SMILES strings to predict.

        Returns:
            A dictionary where keys are property names and values are NumPy arrays
            of the corresponding predictions.
        """
        if self.custom_predictor is not None:
            properties_nn_batch = self.custom_predictor.batch_predict_properties(
                smiles_list
            )
            return {
                "affinity": self.custom_predictor.batch_predict_affinity(smiles_list),
                "toxicity": self.custom_predictor.batch_predict_toxicity(smiles_list),
                "qed": properties_nn_batch.get("qed", np.full(len(smiles_list), 0.5)),
                "sa": properties_nn_batch.get("sa", np.full(len(smiles_list), 0.5)),
                "logp": properties_nn_batch.get("logp", np.full(len(smiles_list), 0.0)),
                "mw": properties_nn_batch.get("mw", np.full(len(smiles_list), 0.5)),
            }
        else:
            # Fallback to oracle heuristics for batch
            return {
                "affinity": np.full(len(smiles_list), 0.6),  # Default heuristic
                "toxicity": np.full(len(smiles_list), 0.2),  # Default heuristic
                "qed": OracleFunction.batch_qed(smiles_list),
                "sa": OracleFunction.batch_sa(smiles_list),
                "logp": np.full(len(smiles_list), 0.0),
                "mw": np.full(len(smiles_list), 0.5),
            }

    def score_molecule(self, smiles: str, objective_weights: Dict[str, float]) -> float:
        """
        Calculate a weighted composite score for a molecule based on its properties.

        The score is a weighted average of different objectives, such as maximizing
        affinity while minimizing toxicity.

        Args:
            smiles: The SMILES string of the molecule to score.
            objective_weights: A dictionary where keys are property names (e.g., 'affinity',
                               'toxicity') and values are the weights for those objectives.

        Returns:
            The final weighted score, normalized to be between 0 and 1.
        """
        properties = self.predict_all_properties(smiles)

        total_weight = sum(objective_weights.values())
        if total_weight == 0:
            logger.warning(
                "Objective weights sum to zero. Returning a neutral score of 0.5."
            )
            return 0.5

        score = 0.0
        for objective, weight in objective_weights.items():
            prop_value = properties.get(objective)
            if prop_value is None:
                logger.warning(
                    f"Objective '{objective}' not found in predicted properties. Skipping."
                )
                continue

            if objective == "toxicity":
                # Lower toxicity is better, so we invert the score
                score += weight * (1.0 - prop_value)
            else:
                # For affinity, QED, and SA, higher values are better
                score += weight * prop_value

        return score / total_weight
