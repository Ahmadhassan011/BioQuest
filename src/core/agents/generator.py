"""
Generator Agent: Creates new molecules via evolutionary and VAE methods.

This agent decides WHAT molecules to generate but delegates HOW to generate
to the HybridMoleculeGenerator pipeline.
"""

import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

from .messages import AgentMessage

logger = logging.getLogger(__name__)


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
            generator: HybridMoleculeGenerator instance (from inference/)
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