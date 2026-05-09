# Architecture

System design and component overview for BioQuest.

---

## Overview

BioQuest combines:
- **Multi-agent orchestration** for drug discovery workflow
- **Neural networks** for property prediction (GNN-DTI, Toxicity, Property, VAE)
- **Hybrid generation** (evolutionary + VAE) for novel molecules
- **Pareto optimization** for multi-objective selection

---

## System Architecture

<img src="./diagrams/system.png" width="500" />

---

## Agent Loop

The orchestrator runs an iterative loop:

1. **Generator** creates new molecules
2. **Evaluator** scores molecules on properties
3. **Refiner** optimizes and ranks candidates
4. Repeat until convergence

<img src="./diagrams/agent-loop.png" width="700" />

---

## Data Flow

```
PyTDC → TDCDataLoader → DatasetPreparer → featurization → DataCache → training → artifacts/models/
```

<img src="./diagrams/data-flow.png" width="700" />

### Dataset Counts

| Dataset | Raw | Processed | Used By |
|---------|-----|-----------|---------|
| DAVIS DTI | 25,772 pairs | 7,429 graphs | GNNDTITrainer |
| Tox21 NR-AR | 77,946 rows | 7,265 fingerprints | ToxicityClassifierTrainer |
| ChEMBL (0.052) | 100,931 mols | 100,931 tokenized | MoleculeVAETrainer |
| Lipophilicity | 4,200 mols | 4,200 fingerprints | PropertyPredictorTrainer |

All datasets split 80/10/10 train/val/test (scaffold split available via `use_scaffold_split=True`).

---

## Key Components

### src/core/agents/
- **orchestrator.py**: AgentOrchestrator — main workflow coordinator
- **generator.py**: GeneratorAgent — creates molecules via VAE generator
- **evaluator.py**: EvaluatorAgent — scores molecules via MoleculePredictor + OptimizationEvaluator
- **refiner.py**: RefinerAgent — analyzes convergence, adjusts strategy
- **messages.py**: AgentMessage, AgentState dataclasses for inter-agent communication

### src/core/optimization/
- **__init__.py**: OptimizationEvaluator, MoleculeScore — combines objectives + convergence
- **objectives.py**: MultiObjectiveEvaluator — weighted-sum scoring
- **pareto.py**: Pareto front selection (higher-is-better)
- **convergence.py**: ConvergenceTracker — plateau detection, patience tracking

### src/core/types/
- **config.py**: OptimizationConfig, ModelConfig dataclasses
- **smiles.py**: SMILES validation/canonicalization
- **protein.py**: Amino acid constants, sequence validation

### src/models/
- **gnn_dti.py**: GNNDTIPredictor (860K params) — GCN + attention-LSTM
- **toxicity.py**: ToxicityClassifier (540K params) — residual MLP + feature attention
- **property.py**: PropertyPredictor (380K params) — multi-task shared encoder
- **vae.py**: MoleculeVAE (600K params) — GRU encoder/decoder with reparameterization
- **featurization.py**: Morgan fingerprints, RDKit descriptors, graph featurization
- **attention.py**: MultiHeadAttention module
- **registry.py**: ModelRegistry, checkpoint save/load
- **loader.py**: ModelLoader, CustomModelPredictor, ModelEvaluator

### src/training/
- **base.py**: Base Trainer (optimizer, scheduler, early stopping)
- **gnn_dti.py**: GNNDTITrainer (mixed precision, gradient accumulation)
- **toxicity.py**: ToxicityClassifierTrainer (weighted BCE, AUC monitoring)
- **property.py**: PropertyPredictorTrainer (multi-task losses)
- **vae.py**: MoleculeVAETrainer (KL annealing, reconstruction accuracy)
- **utils.py**: create_data_loaders, convert_numpy_types, save_training_config

### src/inference/
- **dti.py**: DTIPredictor
- **toxicity.py**: ToxicityPredictor
- **property.py**: PropertyPredictor
- **vae.py**: VAEGenerator
- **predict.py**: MoleculePredictor (orchestrates all 4 inference models)

### src/evaluation/
- **metrics.py**: Regression (RMSE, MAE, R²) and classification (AUC, F1) metrics
- **generation.py**: Molecule generation quality metrics (validity, uniqueness, novelty, diversity, KL divergence)
- **admet.py**: ADMET property computation (HBA, HBD, TPSA, Lipinski Rule of Five) via RDKit
- **reporter.py**: JSON evaluation reports with aggregate statistics (mean ± std)
- **validator.py**: Input shape/sanity checks for models and predictions

### src/data/
- **constants.py**: Amino acid constants, sequence utilities
- **storage.py**: DataCache (raw + processed with versioning)
- **load/tdc.py**: TDCDataLoader (PyTDC download with local caching)
- **load/handlers.py**: Protein/Drug/Graph data handlers
- **load/dataset.py**: BioQuestDataset, ObjectiveHandler
- **preparation/dti.py**: DTIDatasetPreparer
- **preparation/toxicity.py**: Tox21DatasetPreparer
- **preparation/property.py**: PropertyDatasetPreparer
- **preparation/vae.py**: VAEDatasetPreparer

---

## Configuration Schema

```json
{
  "protein_sequence": "required - target protein amino acids",
  "seeds": ["required - starting SMILES"],
  "objectives": {
    "affinity": 0.0-1.0,
    "toxicity": 0.0-1.0,
    "qed": 0.0-1.0,
    "sa": 0.0-1.0,
    "diversity": 0.0-1.0
  },
  "optimization": {
    "max_iterations": 100,
    "batch_size": 50,
    "patience": 20
  },
  "generation": {
    "vae_enabled": true,
    "evolutionary_enabled": true,
    "strategy": "hybrid"
  }
}
```

Objectives weights must sum to 1.0.


