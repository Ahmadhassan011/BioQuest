"""
Agents Module: LangGraph-based orchestration of molecule discovery agents.

This module provides:
- Generator Agent: Creates new molecules via evolutionary and VAE methods
- Evaluator Agent: Scores molecules on desired properties
- Refiner Agent: Optimizes molecules toward objectives
- Agent state management and message passing
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for agent workflow."""

    iteration: int
    protein_sequence: str
    seeds: List[str]
    objectives: Dict[str, float]

    # Population tracking
    current_population: List[Dict[str, Any]]
    best_molecules: List[Dict[str, Any]]
    pareto_front: List[Dict[str, Any]]

    # Metrics
    convergence_metrics: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]

    # Control
    should_continue: bool
    termination_reason: Optional[str]


@dataclass
class AgentMessage:
    """Message passed between agents."""

    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


class GeneratorAgent:
    """
    Agent responsible for generating new molecules.

    Uses hybrid approach combining RDKit evolutionary algorithms
    and PyTorch VAE for diverse molecule exploration.
    """

    def __init__(self, generator):
        """
        Initialize generator agent.

        Args:
            generator: HybridMoleculeGenerator instance
        """
        self.generator = generator
        self.name = "GeneratorAgent"
        self.molecules_generated = 0
        self.generation_history: List[Dict] = []

    def generate_batch(
        self,
        seeds: List[str],
        num_molecules: int = 50,
        strategy: str = "hybrid",
    ) -> Tuple[List[str], AgentMessage]:
        """
        Generate batch of molecules.

        Args:
            seeds: Seed molecules for generation
            num_molecules: Number of molecules to generate
            strategy: Generation strategy ("hybrid", "evolutionary", "vae")

        Returns:
            Tuple of (generated_smiles, message_to_evaluator)
        """
        generated = []

        try:
            if strategy == "hybrid":
                generated = self.generator.generate_hybrid(seeds, num_molecules)
            elif strategy == "evolutionary":
                generated = self.generator.generate_from_seeds(seeds, num_molecules)
            elif strategy == "vae":
                generated = self.generator.generate_from_latent_space(num_molecules)
            else:
                generated = self.generator.generate_hybrid(seeds, num_molecules)

            # Canonicalize and deduplicate
            unique = self.generator.get_unique_molecules(generated)

            self.molecules_generated += len(unique)

            # Create message for evaluator
            message = AgentMessage(
                sender=self.name,
                receiver="EvaluatorAgent",
                message_type="generated_molecules",
                content={
                    "molecules": unique,
                    "count": len(unique),
                    "strategy": strategy,
                    "generation_round": len(self.generation_history) + 1,
                },
            )

            # Record history
            self.generation_history.append(
                {
                    "round": len(self.generation_history) + 1,
                    "strategy": strategy,
                    "count": len(unique),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info(
                f"Generated {len(unique)} molecules using {strategy} "
                f"(total: {self.molecules_generated})"
            )

            return unique, message

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            message = AgentMessage(
                sender=self.name,
                receiver="EvaluatorAgent",
                message_type="generation_error",
                content={"error": str(e)},
            )
            return [], message

    def get_statistics(self) -> Dict:
        """Get generator statistics."""
        return {
            "agent": self.name,
            "total_generated": self.molecules_generated,
            "generation_rounds": len(self.generation_history),
            "history": self.generation_history,
        }


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
            predictor: MoleculePredictor instance
            evaluator: OptimizationEvaluator instance
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


class RefinerAgent:
    """
    Agent responsible for refining optimization strategy.

    Analyzes convergence metrics, adjusts generation parameters,
    and decides on termination conditions.
    """

    def __init__(self, evaluator):
        """
        Initialize refiner agent.

        Args:
            evaluator: OptimizationEvaluator instance
        """
        self.evaluator = evaluator
        self.name = "RefinerAgent"
        self.refinement_history: List[Dict] = []
        self.current_strategy = "hybrid"

    def analyze_and_refine(
        self,
        best_molecules: List[Dict],
        iteration: int,
        max_iterations: int = 100,
    ) -> Tuple[Dict[str, Any], AgentMessage]:
        """
        Analyze performance and refine strategy.

        Args:
            best_molecules: Top performing molecules
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            Tuple of (strategy_updates, message_to_generator)
        """
        try:
            # Get convergence metrics
            convergence = self.evaluator.convergence.get_convergence_metrics()

            # Determine termination
            should_terminate, reason = self.evaluator.should_terminate(max_iterations)

            # Analyze convergence and adjust strategy
            if convergence["iterations_since_improvement"] > 5:
                # Switch strategy if stuck
                if self.current_strategy == "hybrid":
                    self.current_strategy = "evolutionary"
                elif self.current_strategy == "evolutionary":
                    self.current_strategy = "vae"
                else:
                    self.current_strategy = "hybrid"

                logger.info(f"Switching to {self.current_strategy} strategy")

            # Prepare refinement instructions
            refinements = {
                "strategy": self.current_strategy,
                "num_molecules": 50,
                "mutation_rate": 0.3
                + (convergence["iterations_since_improvement"] * 0.02),
                "should_terminate": should_terminate,
                "termination_reason": reason,
            }

            # Create message for generator
            message = AgentMessage(
                sender=self.name,
                receiver="GeneratorAgent",
                message_type="strategy_update",
                content=refinements,
            )

            self.refinement_history.append(
                {
                    "iteration": iteration,
                    "strategy": self.current_strategy,
                    "should_terminate": should_terminate,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info(
                f"Refiner analysis: iterations_since_improvement="
                f"{convergence['iterations_since_improvement']}, "
                f"termination={should_terminate} ({reason})"
            )

            return refinements, message

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            message = AgentMessage(
                sender=self.name,
                receiver="GeneratorAgent",
                message_type="refinement_error",
                content={"error": str(e)},
            )
            return {"should_terminate": True, "termination_reason": str(e)}, message

    def get_statistics(self) -> Dict:
        """Get refiner statistics."""
        return {
            "agent": self.name,
            "refinement_rounds": len(self.refinement_history),
            "current_strategy": self.current_strategy,
            "history": self.refinement_history,
        }


class AgentOrchestrator:
    """
    Orchestrates agent workflow using message passing.

    Controls the interaction between Generator, Evaluator, and Refiner agents.
    """

    def __init__(
        self,
        generator_agent: GeneratorAgent,
        evaluator_agent: EvaluatorAgent,
        refiner_agent: RefinerAgent,
    ):
        """
        Initialize agent orchestrator.

        Args:
            generator_agent: GeneratorAgent instance
            evaluator_agent: EvaluatorAgent instance
            refiner_agent: RefinerAgent instance
        """
        self.generator = generator_agent
        self.evaluator = evaluator_agent
        self.refiner = refiner_agent
        self.name = "AgentOrchestrator"

        self.message_log: List[AgentMessage] = []
        self.iteration_count = 0

        logger.info("AgentOrchestrator initialized with 3 agents")

    def run_iteration(
        self,
        seeds: List[str],
        objectives: Dict[str, float],
        iteration: int,
        max_iterations: int = 100,
        batch_size: int = 50,
    ) -> Tuple[bool, str]:
        """
        Run single optimization iteration with all agents.

        Args:
            seeds: Seed molecules
            objectives: Optimization objectives and weights
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            batch_size: Number of molecules per iteration

        Returns:
            Tuple of (should_continue, reason)
        """
        self.iteration_count = iteration

        logger.info(f"\n{'=' * 60}")
        logger.info(f"ITERATION {iteration}/{max_iterations}")
        logger.info(f"{'=' * 60}\n")

        try:
            # Step 1: Generator Agent
            logger.info("Step 1: Generator Agent - Creating new molecules...")
            strategy = "hybrid" if iteration == 0 else self.refiner.current_strategy
            molecules, gen_msg = self.generator.generate_batch(
                seeds, batch_size, strategy
            )
            self.message_log.append(gen_msg)

            if not molecules:
                logger.warning("No molecules generated, attempting retry...")
                molecules, gen_msg = self.generator.generate_batch(
                    seeds, batch_size, "evolutionary"
                )
                self.message_log.append(gen_msg)

            # Step 2: Evaluator Agent
            logger.info("Step 2: Evaluator Agent - Evaluating molecules...")
            evaluated, eval_msg = self.evaluator.evaluate_batch(molecules, iteration)
            self.message_log.append(eval_msg)

            if not evaluated:
                logger.warning("No molecules evaluated")
                return False, "Evaluation failed"

            # Step 3: Refiner Agent
            logger.info("Step 3: Refiner Agent - Analyzing performance...")
            best_molecules = self.evaluator.get_top_molecules(10)
            refinements, ref_msg = self.refiner.analyze_and_refine(
                [m.to_dict() for m in best_molecules],
                iteration,
                max_iterations,
            )
            self.message_log.append(ref_msg)

            # Check termination
            should_terminate = refinements.get("should_terminate", False)
            reason = refinements.get("termination_reason", "")

            if should_terminate:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"OPTIMIZATION TERMINATING: {reason}")
                logger.info(f"{'=' * 60}\n")
                return False, reason

            return True, "Continuing optimization"

        except Exception as e:
            logger.error(f"Iteration failed: {e}")
            return False, f"Iteration error: {str(e)}"

    def get_final_results(self) -> Dict[str, Any]:
        """Get final optimization results."""

        best = self.evaluator.get_best_molecule()

        return {
            "best_molecule": best.to_dict() if best else None,
            "top_10": [m.to_dict() for m in self.evaluator.get_top_molecules(10)],
            "pareto_front": [m.to_dict() for m in self.evaluator.get_pareto_front()],
            "convergence_metrics": self.evaluator.get_convergence_metrics(),
            "agent_statistics": {
                "generator": self.generator.get_statistics(),
                "evaluator": self.evaluator.get_statistics(),
                "refiner": self.refiner.get_statistics(),
            },
            "total_iterations": self.iteration_count,
            "total_molecules_generated": self.generator.molecules_generated,
            "total_molecules_evaluated": self.evaluator.molecules_evaluated,
            "messages_exchanged": len(self.message_log),
        }
