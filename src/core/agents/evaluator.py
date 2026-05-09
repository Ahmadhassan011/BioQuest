"""
Evaluator Agent: Scores molecules on desired properties.

This agent decides WHAT molecules to evaluate but delegates HOW to evaluate
to the MoleculePredictor and OptimizationEvaluator.
"""

import logging
from typing import Dict, List, Tuple
from datetime import datetime

from .messages import AgentMessage

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """
    Agent responsible for evaluating molecules on desired properties.

    Scores molecules on binding affinity, toxicity, drug-likeness (QED),
    and synthetic accessibility (SA).
    """

    def __init__(self, predictor, evaluator):
        """
        Initialize evaluator agent.

        Args:
            predictor: MoleculePredictor instance (from inference/)
            evaluator: OptimizationEvaluator instance (from core/optimization/)
        """
        self.predictor = predictor
        self.evaluator = evaluator
        self.name = "EvaluatorAgent"
        self.molecules_evaluated = 0
        self.evaluation_history: List[Dict] = []

    def evaluate_batch(
        self,
        molecules: List[str],
        iteration: int = 0,
    ) -> Tuple[List[Dict], AgentMessage]:
        """
        Evaluate batch of molecules.

        Args:
            molecules: List of SMILES strings
            iteration: Current iteration number

        Returns:
            Tuple of (evaluated_molecules, message_to_refiner)
        """
        evaluated = []
        evaluated_for_history = []

        try:
            # Batch predict all properties
            properties = self.predictor.batch_predict(molecules)

            # Combine with SMILES and scores
            for i, smiles in enumerate(molecules):
                mol_data = {
                    "smiles": smiles,
                    "affinity": float(properties["affinity"][i]),
                    "toxicity": float(properties["toxicity"][i]),
                    "qed": float(properties["qed"][i]),
                    "sa": float(properties["sa"][i]),
                }

                # Evaluate and get composite score
                evaluated_molecule_obj = self.evaluator.evaluate_molecule(
                    smiles,
                    {k: v for k, v in mol_data.items() if k != "smiles"},
                    iteration,
                )

                mol_data["composite_score"] = evaluated_molecule_obj.composite_score
                evaluated.append(mol_data)
                evaluated_for_history.append(evaluated_molecule_obj)

            self.molecules_evaluated += len(evaluated)

            # Update history
            self.evaluator.update_iteration(evaluated_for_history, iteration)

            # Get best molecules
            best_molecules = self.evaluator.get_top_molecules(10)
            best_dicts = [m.to_dict() for m in best_molecules]

            # Create message for refiner
            message = AgentMessage(
                sender=self.name,
                receiver="RefinerAgent",
                message_type="evaluation_results",
                content={
                    "evaluated_count": len(evaluated),
                    "best_molecules": best_dicts,
                    "pareto_front": [
                        m.to_dict() for m in self.evaluator.get_pareto_front()
                    ],
                    "convergence_metrics": self.evaluator.get_convergence_metrics(),
                    "iteration": iteration,
                },
            )

            self.evaluation_history.append(
                {
                    "iteration": iteration,
                    "evaluated_count": len(evaluated),
                    "best_score": max([m["composite_score"] for m in evaluated])
                    if evaluated
                    else 0,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info(
                f"Evaluated {len(evaluated)} molecules "
                f"(total: {self.molecules_evaluated}, "
                f"best: {best_dicts[0]['composite_score']:.4f})"
            )

            return evaluated, message

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            message = AgentMessage(
                sender=self.name,
                receiver="RefinerAgent",
                message_type="evaluation_error",
                content={"error": str(e)},
            )
            return [], message

    def get_statistics(self) -> Dict:
        """Get evaluator statistics."""
        return {
            "agent": self.name,
            "total_evaluated": self.molecules_evaluated,
            "evaluation_rounds": len(self.evaluation_history),
            "history": self.evaluation_history,
        }

    def get_best_molecule(self):
        """Get best evaluated molecule."""
        return self.evaluator.get_best_molecule()

    def get_top_molecules(self, k: int = 10):
        """Get top k molecules."""
        return self.evaluator.get_top_molecules(k)

    def get_pareto_front(self):
        """Get Pareto front molecules."""
        return self.evaluator.get_pareto_front()

    def get_convergence_metrics(self) -> Dict:
        """Get convergence metrics."""
        return self.evaluator.get_convergence_metrics()