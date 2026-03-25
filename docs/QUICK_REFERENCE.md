# Quick Reference

Essential commands and patterns for BioQuest.

---

## Running BioQuest

### CLI
```bash
python src/app/main.py --config configs/config_example.json
python src/app/main.py --help
```

### Web UI
```bash
streamlit run src/app/ui.py
```

### Python API
```python
from src.app.core import AgentOrchestrator
from src.utils.config import Config

config = Config.from_file('configs/config_example.json')
orchestrator = AgentOrchestrator(config)
results = orchestrator.run()
```

---

## Training Models

```bash
# Train all models
python scripts/train_models.py --all --epochs 50 --use-gpu

# Train specific model
python scripts/train_models.py --model gnn_dti --epochs 100
python scripts/train_models.py --model toxicity --epochs 50
python scripts/train_models.py --model property --epochs 80
python scripts/train_models.py --model vae --epochs 50

# Or use shell script
bash scripts/train.sh
```

---

## Common Tasks

### Load Models
```python
from src.models.gnn_dti import GNNDTIPredictor
from src.models.toxicity import ToxicityClassifier
from src.models.property import PropertyPredictor

dti = GNNDTIPredictor.load('trained_models/dti.pt')
tox = ToxicityClassifier.load('trained_models/toxicity.pt')
prop = PropertyPredictor.load('trained_models/property.pt')
```

### Generate Molecules
```python
from src.pipelines.generation import HybridMoleculeGenerator

generator = HybridMoleculeGenerator()
molecules = generator.generate(num_molecules=100)
```

### Predict Properties
```python
from src.pipelines.prediction import MoleculePredictor

predictor = MoleculePredictor()
scores = predictor.predict(molecules)
```

### Load Data
```python
from src.data.loaders import TDCDataLoader
from src.data.preparers import DTIDatasetPreparer

loader = TDCDataLoader()
davis_data = loader.load_dti_data('DAVIS')

preparer = DTIDatasetPreparer()
data_list, splits, metadata = preparer.prepare_dti_dataset("DAVIS")
```

### Train a Model
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
python src/app/main.py --config configs/my_config.json
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
python scripts/train_models.py --all --epochs 50  # no --use-gpu
```

### Memory Issues
```bash
# Reduce batch size
python scripts/train_models.py --model gnn_dti --batch-size 16
```

### View Logs
```bash
tail -f aamd.log
grep ERROR aamd.log
```

---

## File Locations

| Item | Path |
|------|------|
| Source | `src/` |
| Configs | `configs/` |
| Scripts | `scripts/` |
| Trained Models | `trained_models/` |
| Logs | `aamd.log` |
