# QTSTransformer Fine-tuning Experiment Results

This document summarizes the fine-tuning experiments using the Quantum Time Series Transformer (QTSTransformer) as a classification head for the DIVER-1 foundation model on EEG classification tasks.

## Table of Contents
1. [Experimental Setup](#experimental-setup)
2. [Results: QTSTransformer Classifier](#results-qtstransformer-classifier)
3. [Comparison: QTSTransformer vs MLP Classifier](#comparison-qtstransformer-vs-mlp-classifier)
4. [Analysis and Key Findings](#analysis-and-key-findings)
5. [Recommendations](#recommendations)

---

## Experimental Setup

### Model Architecture
- **Backbone**: DIVER-1 (Deep Integration of Various EEG Representations)
- **Classifier**: QTSTransformer (Quantum Time Series Transformer)
- **Pretrained Weights**: `dmodel256_mp_rank_00_model_states.pt` and `dmodel512_mp_rank_00_model_states.pt`

### QTSTransformer Configuration
| Parameter | Value |
|-----------|-------|
| Number of Qubits | 8 |
| QSVT Polynomial Degree | 2 |
| Ansatz Layers | 2 |
| Dropout | 0.1 |

### Datasets
| Dataset | Task | Classes | Channels | Duration |
|---------|------|---------|----------|----------|
| FACED | Emotion Recognition | 9 | 32 | 3 seconds |
| PhysioNet-MI | Motor Imagery | 4 | 64 | 4 seconds |

### Training Configuration
| Setting | Frozen | Unfrozen |
|---------|--------|----------|
| DIVER Backbone | Fixed | Trainable |
| Precompute Features | Yes | No |
| AMP (Mixed Precision) | No | Yes |
| Learning Rate (256) | 1e-3 | 1e-4 |
| Learning Rate (512) | 5e-4 | 5e-5 |
| Weight Decay | 1e-2 | 1e-2 |
| Early Stop Patience | 20 epochs | 20 epochs |

---

## Results: QTSTransformer Classifier

### Summary Table

| Dataset | Width | Mode | Epochs | Test Acc | Test F1 | Test Kappa |
|---------|-------|------|--------|----------|---------|------------|
| FACED | 256 | Frozen | 51 | 15.70% | 15.35% | 4.98% |
| FACED | 256 | Unfrozen | 58 | 22.36% | 21.83% | 12.52% |
| FACED | 512 | Frozen | 69 | 20.73% | 20.68% | 10.95% |
| FACED | 512 | Unfrozen | 50 | **27.56%** | **27.48%** | **18.53%** |
| PhysioNet-MI | 256 | Frozen | 27 | 46.53% | 46.35% | 28.70% |
| PhysioNet-MI | 256 | Unfrozen | 50 | 63.50% | 63.73% | 51.32% |
| PhysioNet-MI | 512 | Frozen | 33 | 54.51% | 54.60% | 39.35% |
| PhysioNet-MI | 512 | Unfrozen | 33 | **63.78%** | **63.94%** | **51.69%** |

### Frozen vs Unfrozen Comparison

#### FACED Dataset (9-class Emotion Recognition)
| Width | Mode | Test F1 | Improvement |
|-------|------|---------|-------------|
| 256 | Frozen | 15.35% | - |
| 256 | Unfrozen | 21.83% | +6.48% |
| 512 | Frozen | 20.68% | - |
| 512 | Unfrozen | 27.48% | +6.80% |

#### PhysioNet-MI Dataset (4-class Motor Imagery)
| Width | Mode | Test F1 | Improvement |
|-------|------|---------|-------------|
| 256 | Frozen | 46.35% | - |
| 256 | Unfrozen | 63.73% | +17.38% |
| 512 | Frozen | 54.60% | - |
| 512 | Unfrozen | 63.94% | +9.34% |

### Best Confusion Matrices

#### FACED 512 Unfrozen (Best: 27.48% F1)
```
Predicted:  0   1   2   3   4   5   6   7   8
Class 0   [32  26  18  14  39   7  38  25   8]
Class 1   [...]
...
```

#### PhysioNet-MI 512 Unfrozen (Best: 63.94% F1)
```
Predicted:    0    1    2    3
Class 0   [ 318   18   83   36]
Class 1   [  30  313   46   68]
Class 2   [  72   51  261   72]
Class 3   [  45   75   60  279]
```

---

## Comparison: QTSTransformer vs MLP Classifier

### Original DIVER Results (from Paper, Table 14)
Configuration: `width=512`, `frozen=False`, `ft_config=flatten_mlp`, `num_mlp_layers=3`, `mup_weights=True`

| Dataset | Accuracy | F1 Score | Kappa |
|---------|----------|----------|-------|
| FACED | 60.1% ± 0.8% | 60.7% ± 0.9% | 55.0% ± 0.9% |
| PhysioNet-MI | 67.6% ± 0.3% | 67.8% ± 0.4% | 56.7% ± 0.4% |

### Performance Comparison

#### FACED (9-class Emotion Recognition)
| Model | Accuracy | F1 Score | Kappa | Gap vs MLP |
|-------|----------|----------|-------|------------|
| DIVER + MLP (Paper) | **60.1%** | **60.7%** | **55.0%** | - |
| DIVER + QTS (Frozen 512) | 20.7% | 20.7% | 11.0% | -40.0% |
| DIVER + QTS (Unfrozen 512) | 27.6% | 27.5% | 18.5% | -33.2% |

#### PhysioNet-MI (4-class Motor Imagery)
| Model | Accuracy | F1 Score | Kappa | Gap vs MLP |
|-------|----------|----------|-------|------------|
| DIVER + MLP (Paper) | **67.6%** | **67.8%** | **56.7%** | - |
| DIVER + QTS (Frozen 512) | 54.5% | 54.6% | 39.3% | -13.2% |
| DIVER + QTS (Unfrozen 512) | 63.8% | 63.9% | 51.7% | -3.9% |

### Parameter Efficiency

| Dataset | MLP Params | QTS Params | Reduction Factor |
|---------|------------|------------|------------------|
| FACED | 29.6M | 1.05M | **28.2x fewer** |
| PhysioNet-MI | 105M | 2.1M | **50.1x fewer** |

### Performance per Parameter

| Dataset | Model | F1 Score | Params | F1 / Million Params |
|---------|-------|----------|--------|---------------------|
| FACED | MLP | 60.7% | 29.6M | 2.05 |
| FACED | QTS | 27.5% | 1.05M | **26.2** |
| PhysioNet-MI | MLP | 67.8% | 105M | 0.65 |
| PhysioNet-MI | QTS | 63.9% | 2.1M | **30.4** |

**QTS achieves 13-47x better performance-per-parameter ratio.**

### Architectural Differences

| Aspect | MLP Classifier | QTSTransformer |
|--------|----------------|----------------|
| Structure | Fully connected layers | Quantum circuit + classical layers |
| Core Operation | Matrix multiplication | Quantum state evolution (unitary) |
| Non-linearity | ELU activation | Quantum interference + measurement |
| Time Handling | Flattens all timesteps | Processes timesteps via LCU/QSVT |
| Expressiveness | Polynomial in width | Exponential in qubits (2^n states) |
| Training Speed | Fast (GPU) | Slow (CPU quantum simulation) |

### QTSTransformer Parameter Breakdown

```
Total Parameters: ~1-2M (depending on input dimensions)

├── feature_projection: ~99.97% - Classical linear layer
├── output_ff: ~0.02% - Classical output layer
└── Quantum Parameters: ~0.01%
    ├── poly_coeffs: 3 (QSVT polynomial)
    ├── mix_coeffs: 3-4 (LCU mixing)
    └── qff_params: 32 (Quantum ansatz rotations)
```

**Note**: Only ~38 parameters are truly "quantum" trainable parameters.

---

## Analysis and Key Findings

### 1. Parameter Efficiency
- QTSTransformer uses **28-50x fewer parameters** than MLP
- Achieves competitive results on PhysioNet-MI (94% of MLP performance)
- Significant gap remains on FACED (45% of MLP performance)

### 2. Task Complexity Matters
- **PhysioNet-MI (4-class)**: QTS achieves 94% of MLP performance
- **FACED (9-class)**: QTS achieves only 45% of MLP performance
- QTS performs better on simpler classification tasks

### 3. Unfrozen Training is Critical
- Unfrozen consistently outperforms frozen (6-17% F1 improvement)
- End-to-end fine-tuning allows DIVER features to adapt to QTS

### 4. Model Width Impact
- 512-width generally outperforms 256-width
- Larger embeddings provide richer features for QTS

### 5. Bottleneck Analysis
- `feature_projection` layer dominates QTS parameters (99.97%)
- True quantum parameters are minimal (~38 params)
- Potential for further compression

---

## Recommendations

### For Improving QTS Performance

1. **Match Paper Hyperparameters**
   - Use `i_eeg_pretrained_weights.pt` with `--mup_weights True`
   - Increase learning rate to 2e-4
   - Increase weight decay to 0.3

2. **Increase Quantum Capacity**
   - More qubits (10-12 instead of 8)
   - Higher QSVT degree (3-4 instead of 2)
   - More ansatz layers (3-4 instead of 2)

3. **Architectural Improvements**
   - Add intermediate dimensionality reduction before QTS
   - Hybrid QTS + small MLP head
   - Learnable feature pooling across channels

4. **Training Strategies**
   - Curriculum learning (start with simpler tasks)
   - Layer-wise learning rate decay
   - Longer training with lower learning rate

### For Future Research

1. **Quantum Hardware**: Test on real quantum devices when available
2. **Scalability**: Investigate tensor network methods for larger qubit counts
3. **Interpretability**: Analyze quantum state representations
4. **Other Tasks**: Evaluate on regression tasks (e.g., SEED-VIG)

---

## Appendix: Experimental Commands

### Frozen Training
```bash
conda activate DIVER_QML
cd /home/connectome/justin/DIVER_QML/DIVER-1
./scripts/finetune_faced_qts.sh
./scripts/finetune_physionet_qts.sh
```

### Unfrozen Training
```bash
./scripts/finetune_faced_qts_unfrozen.sh
./scripts/finetune_physionet_qts_unfrozen.sh
```

### Monitor Progress
```bash
tail -f /home/connectome/justin/DIVER_QML/DIVER_data/qts_checkpoints/*/train.log
```

---

*Generated: January 2025*
*Environment: DIVER_QML (Python 3.10, PyTorch 2.9.1+cu130, PennyLane 0.42.3)*
