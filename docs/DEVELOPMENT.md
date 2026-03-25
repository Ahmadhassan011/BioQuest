# Development Guide

Training, configuration, caching, and logging for BioQuest.

---

## Training Models

### Quick Start
```bash
# Train all models
python scripts/train_models.py --all --epochs 50 --use-gpu

# Train specific model
python scripts/train_models.py --model gnn_dti --epochs 100
python scripts/train_models.py --model toxicity --epochs 50
python scripts/train_models.py --model property --epochs 80
python scripts/train_models.py --model vae --epochs 50
```

### Programmatic Training

#### GNN-DTI
```python
from src.training.gnn_dti_trainer import GNNDTITrainer
from src.data.preparers import DTIDatasetPreparer

preparer = DTIDatasetPreparer()
data = preparer.prepare_dti_dataset("DAVIS")

trainer = GNNDTITrainer(config)
trainer.train(
    X_mol=data['X_mol'],
    X_prot=data['X_prot'],
    y=data['y'],
    splits=data['splits'],
    epochs=50,
    batch_size=32
)
trainer.save_checkpoint('trained_models/dti.pt')
```

#### Toxicity
```python
from src.training.toxicity_classifier_trainer import ToxicityClassifierTrainer
from src.data.loaders import TDCDataLoader

loader = TDCDataLoader()
X, y = loader.load_tox21_data()

trainer = ToxicityClassifierTrainer(config)
trainer.train(X, y, epochs=50, batch_size=32)
trainer.save_checkpoint('trained_models/toxicity.pt')
```

#### Property
```python
from src.training.property_predictor_trainer import PropertyPredictorTrainer

trainer = PropertyPredictorTrainer(config)
trainer.train(X_train, properties_train, X_val, properties_val, epochs=80)
trainer.save_checkpoint('trained_models/property.pt')
```

#### VAE
```python
from src.training.molecule_vae_trainer import MoleculeVAETrainer
from src.data.preparers import VAEDatasetPreparer

preparer = VAEDatasetPreparer()
data = preparer.prepare_vae_dataset(sample_frac=0.05)

trainer = MoleculeVAETrainer(config)
trainer.train(data['train'], data['val'], epochs=50)
trainer.save_checkpoint('trained_models/vae.pt')
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
    "models_dir": "trained_models"
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
- `data/raw/` - Raw datasets (DAVIS, Tox21, ChEMBL)
- `data/processed/` - Featurized data

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
- First run: Downloads from PyTDC, saves to cache
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

## Data Sources

### DAVIS (DTI)
- 4,485 proteins, 68 drugs, 30,056 interactions

### Tox21 (Toxicity)
- 7,831 compounds, 12 assays

### ChEMBL (VAE)
- Millions of drug-like molecules

---

## Troubleshooting

### Out of Memory
```bash
python scripts/train_models.py --model gnn_dti --batch-size 16
```

### Loss Not Decreasing
```bash
python scripts/train_models.py --model gnn_dti --learning-rate 0.0005
```

### CUDA Errors
```bash
python -c "import torch; print(torch.cuda.is_available())"
python scripts/train_models.py --model gnn_dti  # CPU fallback
```
