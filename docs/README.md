# BioQuest Documentation

BioQuest is an AI system for autonomous drug discovery combining multi-agent orchestration, neural networks, hybrid molecule generation, and multi-objective optimization.

---

## What is BioQuest?

- **Generator Agent** creates molecules via evolutionary algorithms and VAE
- **Evaluator Agent** scores molecules on binding affinity, toxicity, QED, synthetic accessibility
- **Refiner Agent** optimizes candidates using Pareto front selection

---

## Installation

```bash
git clone <repository-url>
cd BioQuest
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Quick Start

```bash
# Run with config
python src/app/main.py --config configs/config_example.json

# Web UI
streamlit run src/app/ui.py

# Train models
python scripts/train_models.py --all --epochs 50 --use-gpu
```

---

## Project Structure

```
BioQuest/
├── src/
│   ├── app/           # Application layer
│   │   ├── main.py    # CLI entry point
│   │   ├── ui.py      # Streamlit web interface
│   │   └── core.py    # Agent orchestration
│   ├── data/          # Data handling
│   │   ├── loaders.py
│   │   ├── preparers.py
│   │   └── storage.py
│   ├── models/        # Neural networks
│   │   ├── gnn_dti.py
│   │   ├── toxicity.py
│   │   ├── property.py
│   │   └── ...
│   ├── pipelines/      # ML workflows
│   │   ├── generation.py
│   │   ├── prediction.py
│   │   └── optimization.py
│   ├── training/       # Training infrastructure
│   │   ├── gnn_dti_trainer.py
│   │   └── ...
│   └── utils/         # Utilities
│       ├── config.py
│       └── logging_config.py
├── configs/           # Configuration files
├── scripts/           # Training scripts
├── docs/             # Documentation
└── trained_models/   # Model checkpoints (auto-created)
```

---

## Documentation

| File | Description |
|------|-------------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | CLI commands, code examples |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, diagrams |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Training, config, caching |

---

## Configuration

Config files in `configs/`: `config_example.json`, `config_dti_training.json`, `config_toxicity.json`, `config_multiobjective.json`, `config_vae_pretraining.json`.

Key sections: `protein_sequence`, `seeds`, `objectives` (must sum to 1.0), `optimization`.

---

## Help

```bash
python src/app/main.py --help
python -c "from src.app.core import AgentOrchestrator; print('OK')"
```
