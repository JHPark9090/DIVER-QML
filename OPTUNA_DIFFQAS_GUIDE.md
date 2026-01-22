# Optuna + DiffQAS Hyperparameter and Architecture Search Guide

This document describes how to run the hierarchical optimization for finding optimal quantum circuit architectures for the QTSTransformer with DIVER-1 backbone.

## Overview

The optimization uses a **hierarchical/nested** approach:

```
┌─────────────────────────────────────────────────────────────┐
│  OUTER LOOP: Optuna Hyperparameter Search                   │
│  Searches: n_qubits, degree, lr, weight_decay, dropout,     │
│            warmup_epochs                                    │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  INNER LOOP: DiffQAS Architecture Search              │  │
│  │  Searches: 24 timestep circuits × 24 QFF circuits     │  │
│  │  (softmax-weighted ensemble during training)          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Current Configuration

### Training Mode
- **frozen**: False (end-to-end fine-tuning)
- **width**: 512 (DIVER backbone)
- **use_amp**: True (mixed precision for memory efficiency)
- **precompute_features**: False (required for unfrozen training)

### Optuna Hyperparameter Search Space

| Parameter | Search Range | Type | Notes |
|-----------|-------------|------|-------|
| n_qubits | {6, 8, 10} | Categorical | Quantum expressivity |
| degree | {2, 3} | Categorical | QSVT polynomial degree |
| lr | {2e-5, 5e-5, 1e-4, 2e-4} | Categorical | Includes successful 5e-5 |
| weight_decay | {0.005, 0.01, 0.02, 0.05, 0.1} | Categorical | Includes successful 0.01 |
| dropout | {0.05, 0.1, 0.15, 0.2} | Categorical | Includes successful 0.1 |
| warmup_epochs | {3, 5} | Categorical | DiffQAS warmup |

### Fixed Parameters

| Parameter | Value |
|-----------|-------|
| n_ansatz_layers | 2 |
| n_qff_layers | 1 |
| epochs | 50 |
| batch_size | 32 |
| early_stop_patience | 15 |
| depth (DIVER) | 12 |

### DiffQAS Circuit Search Space (24 candidates each)

| Dimension | Options |
|-----------|---------|
| H-layer | True, False |
| Entangling | linear_cnot, cyclic_cnot, crx_forward, crx_backward |
| Variational | RX, RY, RZ |

**Total**: 2 × 4 × 3 = 24 circuits for timestep + 24 for QFF = 48 circuit evaluations per forward pass

## Datasets

| Dataset | Task | Classes | Channels |
|---------|------|---------|----------|
| PhysioNet-MI | Motor Imagery | 4 | 64 EEG |
| FACED | Emotion Recognition | 9 | 32 EEG |

## Running Experiments

### Prerequisites

```bash
conda activate DIVER_QML
cd /home/connectome/justin/DIVER_QML/DiffQAS_QLSTM_Deploy/DIVER_DiffQAS/scripts
```

### PhysioNet-MI

```bash
# Direct run with live output
python -u optuna_diffqas_search.py \
    --dataset PhysioNet-MI \
    --n_trials 50 \
    --epochs 50 \
    --cuda 0 \
    --seed 42

# Background run with logging
nohup python -u optuna_diffqas_search.py \
    --dataset PhysioNet-MI \
    --n_trials 50 \
    --epochs 50 \
    --cuda 0 \
    --seed 42 \
    > /home/connectome/justin/DIVER_QML/DIVER_data/optuna_physionet_log.txt 2>&1 &
```

### FACED

```bash
# Direct run with live output
python -u optuna_diffqas_search.py \
    --dataset FACED \
    --n_trials 50 \
    --epochs 50 \
    --cuda 0 \
    --seed 42

# Background run with logging
nohup python -u optuna_diffqas_search.py \
    --dataset FACED \
    --n_trials 50 \
    --epochs 50 \
    --cuda 0 \
    --seed 42 \
    > /home/connectome/justin/DIVER_QML/DIVER_data/optuna_faced_log.txt 2>&1 &
```

## Monitoring Progress

### Live Log Monitoring

```bash
# PhysioNet-MI
tail -f /home/connectome/justin/DIVER_QML/DIVER_data/optuna_physionet_log.txt

# FACED
tail -f /home/connectome/justin/DIVER_QML/DIVER_data/optuna_faced_log.txt
```

### Check Optuna Study Status

```bash
conda run -n DIVER_QML python -c "
import optuna
import glob
import os

# Find latest study
studies = sorted(glob.glob('/home/connectome/justin/DIVER_QML/DIVER_data/optuna_studies/diffqas_*/optuna_study.db'))
if studies:
    latest = studies[-1]
    study_name = os.path.basename(os.path.dirname(latest))
    study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{latest}')
    print(f'Study: {study_name}')
    print(f'Total trials: {len(study.trials)}')
    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    print(f'Completed: {len(completed)}')
    if study.best_trial:
        print(f'Best val_f1: {study.best_value:.4f}')
        print(f'Best params: {study.best_params}')
"
```

### Check GPU Usage

```bash
watch -n 5 nvidia-smi
```

### Check Running Processes

```bash
ps aux | grep optuna_diffqas | grep -v grep
```

## Output Files

Results are saved to: `/home/connectome/justin/DIVER_QML/DIVER_data/optuna_studies/`

Each study creates:
```
diffqas_{dataset}_{timestamp}/
├── optuna_study.db          # Optuna SQLite database
├── best_results.json        # Best trial summary
├── all_trials.json          # All completed trials
├── param_importance.json    # Hyperparameter importance
├── {dataset}/
│   └── architecture_*.json  # Best circuit architectures
└── trial_N/
    ├── trial_results.json   # Individual trial results
    └── best_state.pth       # Model checkpoint
```

## Analyzing Results

### After Completion

```bash
# View best results
cat /home/connectome/justin/DIVER_QML/DIVER_data/optuna_studies/diffqas_PhysioNet-MI_*/best_results.json

# View architecture selection
cat /home/connectome/justin/DIVER_QML/DIVER_data/optuna_studies/diffqas_PhysioNet-MI_*/PhysioNet-MI/architecture_*.json
```

### Hyperparameter Importance

After the study completes, parameter importance is automatically calculated and saved to `param_importance.json`.

## Key Files

| File | Purpose |
|------|---------|
| `optuna_diffqas_search.py` | Main Optuna + DiffQAS script |
| `diffqas_qts_transformer.py` | DiffQAS-QTSTransformer model |
| `diffqas_module.py` | DiffQAS weight mechanism |
| `circuit_generation.py` | 24 circuit generator |
| `finetune_model_diffqas.py` | DIVER integration |

## Expected Runtime

- **Per trial**: 30-90 minutes (depending on early stopping)
- **50 trials**: 1-3 days per dataset
- **Total for both datasets**: 2-6 days

## Reference

Based on DiffQAS paper: arXiv:2508.14955v1 (Differentiable Quantum Architecture Search)

---

*Last updated: January 22, 2026*
*Configuration: frozen=False, width=512, unfrozen end-to-end fine-tuning*
