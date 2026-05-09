"""
Agent messages and state definitions.

This module contains pure data classes for agent communication.
No framework dependencies.
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Literal
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


AgentResponse = Dict[str, Any]
MessageType = Literal["generate", "evaluate", "refine", "result", "error"]


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