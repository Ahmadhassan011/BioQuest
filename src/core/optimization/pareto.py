"""
Pareto: Pareto dominance calculations.

This module contains pure business logic for Pareto front calculations.
"""

from typing import List, Dict


def is_pareto_dominated(
    candidate: Dict[str, float],
    population: List[Dict[str, float]],
    objectives: List[str],
) -> bool:
    """
    Check if candidate is Pareto dominated by any member in population.

    A candidate is dominated if there exists another solution that is
    better or equal in all objectives and strictly better in at least one.

    Args:
        candidate: Candidate solution
        population: List of solutions to check against
        objectives: List of objective names to compare

    Returns:
        True if candidate is dominated, False otherwise
    """
    for other in population:
        if other is candidate:
            continue

        # Check if other dominates candidate
        at_least_as_good = True
        strictly_better = False

        for obj in objectives:
            # Higher is better for all objectives here (toxicity should be inverted before calling)
            if other.get(obj, 0.0) < candidate.get(obj, 0.0):
                at_least_as_good = False
                break
            if other.get(obj, 0.0) > candidate.get(obj, 0.0):
                strictly_better = True

        if at_least_as_good and strictly_better:
            return True

    return False


def get_pareto_front(
    molecules: List[Dict[str, float]],
    objectives: List[str],
) -> List[Dict[str, float]]:
    """
    Calculate Pareto front from molecule population.

    Args:
        molecules: List of molecule dictionaries with properties
        objectives: List of objective names (all should be higher-is-better)

    Returns:
        List of molecules on Pareto front
    """
    if not molecules:
        return []

    pareto_front = []
    for mol in molecules:
        if not is_pareto_dominated(mol, molecules, objectives):
            pareto_front.append(mol)

    return pareto_front if pareto_front else [molecules[0]]