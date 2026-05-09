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
| **Neural Networks** | GNN-DTI (binding affinity), Toxicity classifier, Property predictor, VAE |
| **Hybrid Generation** | RDKit evolutionary algorithms + VAE for novel molecules |
| **Pareto Optimization** | Multi-objective optimization with weighted sum |
| **Multiple Interfaces** | CLI, Streamlit web UI, Python API |

---

## Quick Start

```bash
# Install
git clone https://github.com/Ahmadhassan011/BioQuest.git
cd BioQuest
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run with config
python -m cli.main --config configs/config_example.json

# Or use web UI
streamlit run ui/streamlit_app.py
```

---

## Training

```bash
# Train all models (50 epochs, GPU)
python scripts/train_models.py --all --epochs 50 --use-gpu

# Train specific models
python scripts/train_models.py --models dti --epochs 100
python scripts/train_models.py --models toxicity --epochs 50
python scripts/train_models.py --models property --epochs 80
python scripts/train_models.py --models vae --epochs 50

# Custom checkpoint directory (default: artifacts/models/)
python scripts/train_models.py --all --checkpoint-dir ./checkpoints
```

Trained model checkpoints are saved to `artifacts/models/{dti,toxicity,vae,properties}/best_model.pt`.

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

### Run Optimization

```python
from src.inference import MoleculePredictor
from src.core.optimization import OptimizationEvaluator
from src.core.agents import GeneratorAgent, EvaluatorAgent, RefinerAgent, AgentOrchestrator

predictor = MoleculePredictor(protein_sequence="MKFLK...", models_dir="artifacts/models")
vae_gen = predictor.get_vae_generator()

opt = OptimizationEvaluator(
    objective_weights={"affinity": 0.4, "toxicity": 0.3, "qed": 0.2, "sa": 0.1}
)

orchestrator = AgentOrchestrator(
    GeneratorAgent(vae_gen),
    EvaluatorAgent(predictor, opt),
    RefinerAgent(opt),
)
```

### Predict Properties

```python
from src.inference import MoleculePredictor

predictor = MoleculePredictor(protein_sequence="MKFLK...", models_dir="artifacts/models")
props = predictor.predict_all_properties("CCO")
# Returns: {"affinity": 0.87, "toxicity": 0.12, "qed": 0.78, "sa": 0.65, "logp": 0.24, "mw": 46.07}
```

### Load Trained Models

```python
from src.models.loader import ModelLoader

loader = ModelLoader(models_dir="artifacts/models", use_gpu=False)
dti_model = loader.load_dti_model()
tox_model = loader.load_toxicity_model()
prop_model = loader.load_property_model()
```

---

## Data Sources

| Dataset | Purpose | Size |
|---------|---------|------|
| DAVIS | DTI prediction | 7,429 interactions (after filtering) |
| Tox21 | Toxicity classification | 7,831 compounds |
| ChEMBL | VAE pretraining | Millions of molecules |

---

## Documentation

| File | Description |
|------|-------------|
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
