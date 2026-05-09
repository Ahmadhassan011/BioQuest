# Quick Reference

Essential commands and patterns for BioQuest.

---

## Running BioQuest

### CLI
```bash
python -m cli.main --config configs/config_example.json
python -m cli.main --help
```

### Data & Training CLI
```bash
python -m cli prepare all              # prepare all datasets
python -m cli prepare dti --scaffold   # prepare with scaffold split
python -m cli train                    # train all models (default)
python -m cli train dti vae           # train specific models
python -m cli train all --epochs 100 --gpu
python -m cli cache                    # show cached datasets
```

### Web UI
```bash
streamlit run ui/streamlit_app.py
```

### Python API
```python
from src.inference import MoleculePredictor
from src.core.optimization import OptimizationEvaluator
from src.core.agents import GeneratorAgent, EvaluatorAgent, RefinerAgent, AgentOrchestrator

predictor = MoleculePredictor(
    protein_sequence="MKFLK...",
    models_dir="artifacts/models",
)
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

---

## Training Models

```bash
# Train all models (default)
python -m cli train

# Train specific
python -m cli train dti
python -m cli train dti toxicity vae property
python -m cli train all --epochs 100 --gpu
```

---

## Datasets

| Dataset | Records | Split | Model |
|---------|---------|-------|-------|
| DAVIS | 7,429 graphs | 80/10/10 | GNN-DTI |
| Tox21 NR-AR | 7,265 compounds | 80/10/10 | Toxicity |
| ChEMBL (0.052) | 100,931 molecules | 80/10/10 | VAE |
| Lipophilicity | 4,200 compounds | 80/10/10 | Property |

Supplementary raw: KIBA (117,657), BindingDB (52,274).

---

## Common Tasks

### Predict Properties
```python
from src.inference import MoleculePredictor

predictor = MoleculePredictor(protein_sequence="MKFLK...", models_dir="artifacts/models")
props = predictor.predict_all_properties("CCO")
print(props["affinity"], props["toxicity"], props["qed"], props["sa"])
```

### Batch Predict
```python
results = predictor.batch_predict(["CCO", "c1ccccc1", "CC(=O)O"])
# results["affinity"], results["toxicity"], results["qed"], results["sa"]
```

### Load Data
```python
from src.data.load.tdc import TDCDataLoader
from src.data.preparation.dti import DTIDatasetPreparer

loader = TDCDataLoader()
raw = loader.load_dti_data("DAVIS")

preparer = DTIDatasetPreparer()
data_list, splits, metadata = preparer.prepare_dti_dataset("DAVIS")
```

### Train a Model
```python
import torch
from src.models.gnn_dti import GNNDTIPredictor
from src.training.gnn_dti import GNNDTITrainer
from src.data.preparation.dti import DTIDatasetPreparer, DTIGraphDataset
from src.training.utils import create_data_loaders

device = torch.device("cpu")
preparer = DTIDatasetPreparer()
data_list, splits, metadata = preparer.prepare_dti_dataset("DAVIS")
dataset = DTIGraphDataset(data_list)
train_loader, val_loader, _ = create_data_loaders(
    dataset, splits, batch_size=32, dataset_type="dti"
)

model = GNNDTIPredictor(atom_feature_dim=metadata["atom_feature_dim"])
trainer = GNNDTITrainer(model, device)
results = trainer.fit(train_loader, val_loader, epochs=50, checkpoint_dir="artifacts/models/dti")
```

---

## Configuration

### Example Config
```json
{
  "protein_sequence": "MKFLK...",
  "seeds": ["CC(C)Cc1ccc...", "CCCC..."],
  "objectives": {
    "affinity": 0.4,
    "toxicity": 0.2,
    "qed": 0.2,
    "sa": 0.2
  },
  "optimization": {
    "max_iterations": 100,
    "batch_size": 50,
    "patience": 20
  }
}
```

### Create Custom Config
```bash
cp configs/config_example.json configs/my_config.json
# Edit my_config.json
python -m cli.main --config configs/my_config.json
```

---

## Troubleshooting

### Import Errors
```bash
# Check Python path
echo $PYTHONPATH

# Add path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA/GPU Issues
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU only
python -m cli train  # defaults to CPU
```

### Memory Issues
```bash
# Reduce batch size
python -m cli train dti --batch-size 16
```

### View Logs
```bash
tail -f training.log
```

---

## Benchmarking

```bash
# Full suite
python scripts/benchmark.py

# Quick (1 trial)
python scripts/benchmark.py --quick

# Ablation modes
python -m cli.main --config configs/config_example.json --ablation no_refiner
python -m cli.main --config configs/config_example.json --ablation single_pass
```

---

## ADMET Properties

`predict_all_properties()` returns RDKit-computed ADMET:

```python
props = predictor.predict_all_properties("CCO")
props["hba"], props["hbd"], props["tpsa"], props["passes_lipinski"]
```

---

## File Locations

| Item | Path |
|------|------|
| Source | `src/` |
| Configs | `configs/` |
| Scripts | `scripts/` |
| Trained Models | `artifacts/models/` |
| Logs | `training.log` |
