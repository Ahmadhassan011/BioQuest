"""PipelineConfig — serializable dataclass for reproducible data + training pipelines."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Captures all parameters for a data preparation + training run.

    Serializes to/from JSON for reproducibility across runs.
    """

    # --- model selection ---
    models: List[str] = field(default_factory=lambda: ["dti", "toxicity", "vae", "property"])

    # --- training hyperparams ---
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    use_gpu: bool = False
    gradient_accumulation_steps: int = 1

    # --- data params ---
    dti_dataset: str = "DAVIS"
    tox_assay: str = "NR-AR"
    chembl_frac: float = 0.052
    prop_dataset: str = "Lipophilicity_AstraZeneca"
    val_split: float = 0.1
    test_split: float = 0.1
    use_scaffold_split: bool = False
    max_prot_len: int = 1024
    vae_max_smiles_len: int = 100
    kl_anneal_epochs: int = 50

    # --- paths ---
    checkpoint_dir: str = "artifacts/models"
    log_file: Optional[str] = "training.log"

    # --- metadata ---
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def model_checkpoint_dir(self, model: str) -> str:
        mapping = {"dti": "dti", "toxicity": "toxicity", "vae": "vae", "property": "properties"}
        return str(Path(self.checkpoint_dir) / mapping.get(model, model))
