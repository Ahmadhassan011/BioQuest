# Development Guide

Training, configuration, caching, and logging for BioQuest.

---

## Training Models

### Quick Start
```bash
# Train all models
python scripts/train_models.py --all --epochs 50 --use-gpu

# Train specific model
python scripts/train_models.py --models dti --epochs 100
python scripts/train_models.py --models toxicity --epochs 50
python scripts/train_models.py --models property --epochs 80
python scripts/train_models.py --models vae --epochs 50
```

### Programmatic Training

#### GNN-DTI
```python
import torch
from src.models.gnn_dti import GNNDTIPredictor
from src.training.gnn_dti import GNNDTITrainer
from src.data.preparation.dti import DTIDatasetPreparer, DTIGraphDataset
from src.training.utils import create_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preparer = DTIDatasetPreparer()
data_list, splits, metadata = preparer.prepare_dti_dataset("DAVIS")
dataset = DTIGraphDataset(data_list)
train_loader, val_loader, _ = create_data_loaders(
    dataset, splits, batch_size=32, dataset_type="dti"
)

model = GNNDTIPredictor(atom_feature_dim=metadata["atom_feature_dim"])
trainer = GNNDTITrainer(model, device)
results = trainer.fit(
    train_loader, val_loader,
    epochs=50,
    checkpoint_dir="artifacts/models/dti",
)
```

#### Toxicity
```python
import torch
from torch.utils.data import TensorDataset
from src.models.toxicity import ToxicityClassifier
from src.training.toxicity import ToxicityClassifierTrainer
from src.data.preparation.toxicity import Tox21DatasetPreparer
from src.training.utils import create_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preparer = Tox21DatasetPreparer()
X, y, splits, metadata = preparer.prepare_tox21_dataset(assay="NR-AR")
dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
train_loader, val_loader, _ = create_data_loaders(
    dataset, splits, batch_size=32, dataset_type="toxicity"
)

model = ToxicityClassifier(input_dim=metadata["mol_feature_dim"])
trainer = ToxicityClassifierTrainer(model, device)
results = trainer.fit(
    train_loader, val_loader,
    epochs=50,
    checkpoint_dir="artifacts/models/toxicity",
)
```

#### Property
```python
import torch
from src.models.property import PropertyPredictor
from src.training.property import PropertyPredictorTrainer
from src.data.preparation.property import PropertyDatasetPreparer, PropertyPredictionDataset
from src.training.utils import create_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preparer = PropertyDatasetPreparer()
features, targets_dict, splits, metadata = preparer.prepare_property_dataset("Lipophilicity_AstraZeneca")
dataset = PropertyPredictionDataset(features, targets_dict)
train_loader, val_loader, _ = create_data_loaders(
    dataset, splits, batch_size=32, dataset_type="property"
)

model = PropertyPredictor(input_dim=metadata["mol_feature_dim"])
trainer = PropertyPredictorTrainer(model, device)
results = trainer.fit(
    train_loader, val_loader,
    epochs=80,
    checkpoint_dir="artifacts/models/properties",
)
```

#### VAE
```python
import torch
from torch.utils.data import TensorDataset
from src.models.vae import MoleculeVAE
from src.training.vae import MoleculeVAETrainer
from src.data.preparation.vae import VAEDatasetPreparer
from src.training.utils import create_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preparer = VAEDatasetPreparer()
X, splits, metadata = preparer.prepare_vae_dataset(sample_frac=0.1)
dataset = TensorDataset(torch.from_numpy(X).long())
train_loader, val_loader, _ = create_data_loaders(
    dataset, splits, batch_size=64, dataset_type="vae"
)

model = MoleculeVAE(
    vocab_size=len(preparer.smiles_chars),
    latent_dim=64,
)
trainer = MoleculeVAETrainer(model, device, kl_anneal_epochs=50)
results = trainer.fit(
    train_loader, val_loader,
    epochs=100,
    checkpoint_dir="artifacts/models/vae",
)
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Number of training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 0.001 | Learning rate |
| `dropout` | 0.3 | Dropout probability |
| `early_stopping_patience` | 10 | Epochs without improvement before stopping |

---

## Configuration

### Config Files
Located in `configs/`:
- `config_example.json` - Full reference
- `config_dti_training.json` - DTI-focused
- `config_toxicity.json` - Toxicity-focused
- `config_multiobjective.json` - Balanced objectives
- `config_vae_pretraining.json` - VAE training
- `config_dev.json` - Quick testing

### Loading Config
```python
from src.utils.config import Config

config = Config.from_file('configs/config_example.json')
value = config.get("optimization.max_iterations", 100)
```

### Config Sections

#### Required
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

#### Optional
```json
{
  "optimization": {
    "max_iterations": 100,
    "batch_size": 50,
    "patience": 20,
    "mutation_rate": 0.3,
    "crossover_rate": 0.7
  },
  "generation": {
    "vae_enabled": true,
    "evolutionary_enabled": true,
    "vae_latent_dim": 64,
    "strategy": "hybrid"
  },
  "predictor": {
    "use_gpu": false,
    "models_dir": "artifacts/models"
  }
}
```

### Environment Variables
```bash
export BIOQUEST_LOG_LEVEL=DEBUG
```

---

## Caching

### Overview
Data is cached in `data/` after first download:
- `data/raw/` - Raw datasets (DAVIS, Tox21, ChEMBL) as pickle files
- `data/processed/` - Featurized data as torch tensors + JSON splits

### Cache Management
```python
from src.data.storage import DataCache

# View cache
DataCache.print_cache_summary()

# Get stats
stats = DataCache.get_cache_stats()
print(f"Size: {stats['total_size_mb']:.2f} MB")

# Clear specific dataset
DataCache.clear_cache("DAVIS")

# Clear all
DataCache.clear_cache()
```

### Cache Behavior
- First run: Downloads from PyTDC, featurizes, saves to cache
- Subsequent runs: Loads from cache (< 1 second)

---

## Logging

### Setup
```python
from src.utils.logging_config import setup_logging, get_module_logger

# Root logger
logger = setup_logging("bioquest", level="INFO")

# Module-specific logger
logger = get_module_logger("training")
```

## Ablation Studies

The `--ablation` CLI flag disables specific agents to measure their contribution.

| Mode | Description |
|------|-------------|
| `full` | All three agents (default) |
| `no_refiner` | Skip RefinerAgent — no strategy switching or termination checks |
| `no_generator` | Use seeds directly, skip molecule generation |
| `single_pass` | One generation + evaluation pass, no iterative loop |

```bash
python -m cli.main --config configs/config_example.json --ablation no_refiner
python -m cli.main --config configs/config_example.json --ablation single_pass
```

---

## ADMET Properties

In addition to the 4 trained model predictions (affinity, toxicity, QED, SA),
every `predict_all_properties()` call now returns RDKit-computed ADMET properties:

| Property | Description | Source |
|----------|-------------|--------|
| `hba` | Hydrogen bond acceptors | RDKit Lipinski |
| `hbd` | Hydrogen bond donors | RDKit Lipinski |
| `tpsa` | Topological polar surface area | RDKit Descriptors |
| `num_rings` | Ring count | RDKit Descriptors |
| `num_aromatic_rings` | Aromatic ring count | RDKit Descriptors |
| `num_rotatable_bonds` | Rotatable bond count | RDKit Descriptors |
| `num_heavy_atoms` | Heavy atom count | RDKit |
| `fraction_csp3` | Fraction of sp3 carbons | RDKit Descriptors |
| `passes_lipinski` | Lipinski Rule of Five | RDKit computation |

```python
from src.inference import MoleculePredictor

predictor = MoleculePredictor("MKFLK...", models_dir="artifacts/models")
props = predictor.predict_all_properties("CCO")
print(props["passes_lipinski"])  # True / False
```

---

### Log Levels
- `DEBUG`: Detailed diagnostic info
- `INFO`: General events
- `WARNING`: Unexpected but not an error
- `ERROR`: Function failed

### Performance Timer
```python
from src.utils.logging_config import PerformanceTimer

logger = get_module_logger("training")

with PerformanceTimer(logger, "Data loading"):
    data = load_dataset()
# Logs: "Data loading completed in 2.34s"
```

### Environment Control
```bash
export BIOQUEST_LOG_LEVEL=DEBUG
```

---

## Benchmarking

```bash
# Full benchmark suite
python scripts/benchmark.py

# Per-category benchmarks
python scripts/benchmark.py --predictive-only
python scripts/benchmark.py --generative-only
python scripts/benchmark.py --optimization-only
python scripts/benchmark.py --ablation-only
python scripts/benchmark.py --system-only

# Multiple trials for statistical significance (mean ± std)
python scripts/benchmark.py --n-trials 5
```

Output: `artifacts/benchmark_scorecard.json`

## Data Sources

Datasets are downloaded via PyTDC to `data/raw/` and featurized to `data/processed/`.

| Dataset | Model | Raw Records | Processed Records | Featurization |
|---------|-------|-------------|-------------------|---------------|
| **DAVIS** | GNN-DTI | 25,772 drug–target pairs<br>(485 drugs, 4,485 targets) | 7,429 graphs<br>(censored filter: Y ≥ 3) | Drug: atom graphs (15-dim node features)<br>Target: 1024-char AA sequence |
| **Tox21 NR-AR** | Toxicity Classifier | 77,946 rows across 12 assays | 7,265 compounds<br>(single assay: NR-AR) | Morgan fingerprints (264-bit)<br>Binary labels (active/inactive) |
| **ChEMBL** | MoleculeVAE | ~194M molecules<br>(100,931 @ default 0.052 frac) | 100,931 sequences<br>(tokenized, max 100 chars) | Character-level tokenization<br>(one-hot → indices) |
| **Lipophilicity** | Property Predictor | 4,200 compounds | 4,200 compounds | Morgan fingerprints (264-bit)<br>Targets: QED, SA, logP, MW |

### Split Ratios
All datasets use 80/10/10 train/val/test splits.

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| DAVIS | 5,945 | 742 | 742 |
| Tox21 NR-AR | 5,813 | 726 | 726 |
| ChEMBL (0.052) | 80,745 | 10,093 | 10,093 |
| Lipophilicity | 3,360 | 420 | 420 |

### Scaffold Splitting
Pass `use_scaffold_split=True` to any preparer to split by Murcko scaffold
(no scaffold overlap between train/val/test). This gives more realistic
generalization estimates than random splitting.

```python
from src.data.preparation.dti import DTIDatasetPreparer

preparer = DTIDatasetPreparer()
data_list, splits, meta = preparer.prepare_dti_dataset(
    "DAVIS", use_scaffold_split=True,
)
```

### Supplementary Datasets (Raw Only)
Downloaded via `scripts/download_all_datasets.py`; not used in model training directly.

| Dataset | Source | Description |
|---------|--------|-------------|
| **KIBA** | PyTDC (DTI) | Kinase inhibitor bioactivity (117,657 interactions) |
| **BindingDB** | PyTDC (DTI) | Binding affinity database (52,274 interactions) |

---

## Troubleshooting

### Out of Memory
```bash
python scripts/train_models.py --models dti --batch-size 16
```

### Loss Not Decreasing
```bash
python scripts/train_models.py --models dti --learning-rate 0.0005
```

### CUDA Errors
```bash
python -c "import torch; print(torch.cuda.is_available())"
python scripts/train_models.py --models dti  # CPU fallback
```
