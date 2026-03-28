# BioQuest Bug Fixes Progress

## HIGH Severity Issues

### 1. [x] Fix `self.evaluator.evaluator` double-nesting bug
**File:** `src/app/core.py` (lines 493, 520, 525, 528, 530)
**Problem:** `self.evaluator.evaluator.get_top_molecules()` should be `self.evaluator.get_top_molecules()`
**Status:** FIXED - Added delegation methods to EvaluatorAgent and updated AgentOrchestrator to use them

### 2. [x] Replace bare except clauses with specific exception handling
**File:** `src/pipelines/generation.py` (14 locations)
**Problem:** Bare `except:` catches all exceptions including KeyboardInterrupt
**Status:** FIXED - Replaced with specific exception types (ValueError, IndexError, KeyError)

### 3. [x] Make fallback predictions use RDKit oracles
**File:** `src/pipelines/prediction.py` (lines 401-408)
**Problem:** Returns hardcoded values (0.6 affinity, 0.2 toxicity) instead of computing via oracles
**Status:** FIXED - Now uses DTIPredictor and ToxicityPredictor oracle methods as fallbacks

---

## MEDIUM Severity Issues

### 4. [x] Fix ChEMBL cache name mismatch
**File:** `src/data/preparers.py` (line 522)
**Problem:** Saves as "ChEMBL_V29", loads from tdc_loader which uses "ChEMBL"
**Status:** FIXED - Changed to use consistent "ChEMBL" cache name

### 5. [x] Unify train_epoch method signatures
**Files:** `src/training/gnn_dti_trainer.py`, `toxicity_classifier_trainer.py`, `property_predictor_trainer.py`
**Status:** FIXED - Made scaler and gradient_accumulation_steps optional with defaults

### 6. [x] Fix SMILES vocabulary inconsistency
**File:** `src/pipelines/generation.py` (line 376)
**Status:** FIXED - Expanded vocabulary to include aromatic atoms (nops), digits, and more special characters

### 7. [x] Unify config access patterns
**File:** `src/utils/config.py`, `src/app/main.py`
**Status:** FIXED - Updated main.py to access max_iterations and batch_size from optimization section

---

## LOW Severity Issues

### 8. [x] Extract magic number 264 to constant
**File:** `src/models/toxicity.py` (line 36)
**Status:** FIXED - Added DEFAULT_TOXICITY_INPUT_DIM constant

### 9. [x] Update deprecated matthews_corrcoef import
**File:** `src/training/toxicity_classifier_trainer.py` (line 11)
**Status:** FIXED - Added try/except fallback to import from sklearn.metrics.cluster

### 10. [x] Complete docstrings for trainer functions
**Files:** `src/training/*.py`
**Status:** VERIFIED - All public methods in trainers already have docstrings

### 11. [ ] Increase default batch size from 4 to 32
**File:** `scripts/train_models.py` (line 280)
**Status:** TODO

### 12. [x] Remove dead code (commented import json)
**File:** `src/app/core.py` (line 15)
**Status:** FIXED - Removed commented import

### 13. [ ] Remove empty pass statement
**File:** `src/app/ui.py` (line 227)
**Status:** TODO

---

## Progress Log

| # | Issue | Status | Commit Message |
|---|-------|--------|----------------|
| 1 | Fix double-nesting bug | DONE | `fix: add delegation methods to EvaluatorAgent to avoid double-nested evaluator access` |
| 2 | Replace bare except clauses | DONE | `fix: replace bare except with specific exception handling in generation.py` |
| 3 | Use RDKit oracles for fallback | DONE | `fix: use oracle heuristics for fallback predictions instead of hardcoded values` |
| 4 | Fix ChEMBL cache name | DONE | `fix: use consistent ChEMBL cache name in VAE dataset preparer` |
| 5 | Unify train_epoch signatures | DONE | `fix: make scaler and gradient_accumulation optional with defaults` |
| 6 | Fix SMILES vocabulary | DONE | `fix: expand VAE SMILES vocabulary to include aromatic atoms and digits` |
| 7 | Unify config access | DONE | `fix: access max_iterations and batch_size from optimization section` |
| 8 | Extract magic number | DONE | `chore: extract 264 to DEFAULT_TOXICITY_INPUT_DIM constant` |
| 9 | Fix deprecated MCC import | DONE | `fix: add fallback import for matthews_corrcoef from sklearn.metrics.cluster` |
| 10 | Complete docstrings | DONE | `chore: verified all trainer methods have docstrings` |
| 11 | | | |
| 12 | Remove dead code | DONE | `chore: remove unused commented import in core.py` |
| 13 | | | |
