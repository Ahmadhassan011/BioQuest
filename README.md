# BioQuest - AI Drug Discovery

BioQuest is an AI system for autonomous drug discovery that combines multi-agent orchestration, neural networks, hybrid molecule generation, and multi-objective optimization.

It discovers novel drug candidates through an iterative process:

1. **Generator** creates new molecules using evolutionary algorithms and VAE
2. **Evaluator** scores molecules on binding affinity, toxicity, QED, and synthetic accessibility
3. **Refiner** optimizes and ranks candidates using Pareto front selection

This loop repeats until convergence, outputting a Pareto front of optimized drug candidates.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent System** | Generator, Evaluator, Refiner agents working in loop |
| **Neural Networks** | GNN-DTI (binding affinity), Toxicity classifier, Property predictor |
| **Hybrid Generation** | RDKit evolutionary algorithms + VAE for novel molecules |
| **Pareto Optimization** | Multi-objective optimization with weighted sum |
| **Multiple Interfaces** | CLI, Streamlit web UI, Python API |

---

## Quick Start

```bash
# Install
git clone <repo-url>
cd BioQuest
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run with config
python src/app/main.py --config configs/config_example.json

# Or use web UI
streamlit run src/app/ui.py
```

---

## Training

```bash
# Train all models (50 epochs, GPU)
python scripts/train_models.py --all --epochs 50 --use-gpu

# Train specific model
python scripts/train_models.py --model gnn_dti --epochs 100
python scripts/train_models.py --model toxicity --epochs 50
python scripts/train_models.py --model property --epochs 80
python scripts/train_models.py --model vae --epochs 50
```

---

## Configuration

Config files in `configs/`:

| File | Use Case |
|------|----------|
| `config_example.json` | Full example |
| `config_dti_training.json` | DTI-focused |
| `config_toxicity.json` | Toxicity-focused |
| `config_multiobjective.json` | Balanced objectives |
| `config_vae_pretraining.json` | VAE training |
| `config_dev.json` | Quick testing |

### Required Config

```json
{
  "protein_sequence": "MKFLK...",
  "seeds": ["CC(C)Cc1...", "CCCC..."],
  "objectives": {
    "affinity": 0.4,
    "toxicity": 0.2,
    "qed": 0.2,
    "sa": 0.2
  }
}
```

Objectives weights must sum to 1.0.

---

## Python API

```python
from src.app.core import AgentOrchestrator
from src.utils.config import Config

config = Config.from_file('configs/config_example.json')
orchestrator = AgentOrchestrator(config)
results = orchestrator.run()
```

### Load Models

```python
from src.models.gnn_dti import GNNDTIPredictor
from src.models.toxicity import ToxicityClassifier
from src.models.property import PropertyPredictor

dti = GNNDTIPredictor.load('trained_models/dti.pt')
tox = ToxicityClassifier.load('trained_models/toxicity.pt')
prop = PropertyPredictor.load('trained_models/property.pt')
```

---

## Data Sources

| Dataset | Purpose | Size |
|---------|---------|------|
| DAVIS | DTI prediction | 30,056 interactions |
| Tox21 | Toxicity classification | 7,831 compounds |
| ChEMBL | VAE pretraining | Millions of molecules |

---

## Documentation

| File | Description |
|------|-------------|
| [docs/README.md](docs/README.md) | Overview and quick start |
| [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | CLI commands, code patterns |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design with diagrams |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Training, config, caching, logging |

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- RDKit
- PyTDC (Therapeutics Data Commons)
- Streamlit

See `requirements.txt` for full list.

---

## License

MIT License - See LICENSE file.
