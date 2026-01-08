#!/bin/bash
# Fine-tuning DIVER-1 with Quantum Time Series Transformer (QTS) on FACED dataset
# UNFROZEN: Both DIVER backbone and QTS head are trained end-to-end
# FACED: 9-class emotion classification, 32 EEG channels, 3 seconds

# Activate conda environment
# conda activate DIVER_QML

python finetune_main.py \
    --seed 42 \
    --cuda 0 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1.00e-04 \
    --weight_decay 1.00e-02 \
    --downstream_dataset FACED \
    --datasets_dir /home/connectome/justin/DIVER_QML/DIVER_data/FACED \
    --model_dir /home/connectome/justin/DIVER_QML/DIVER_data/qts_checkpoints/faced_unfrozen \
    --foundation_dir /home/connectome/justin/DIVER_QML/DIVER_data/dmodel256_mp_rank_00_model_states.pt \
    --width 256 \
    --depth 12 \
    --patch_size 500 \
    --ft_config flatten_qts \
    --early_stop_criteria val_f1 \
    --early_stop_patience 20 \
    --mup_weights False \
    --ft_mup False \
    --use_amp True \
    --deepspeed_pth_format True \
    --frozen False \
    --precompute_features False \
    --qts_n_qubits 8 \
    --qts_degree 2 \
    --qts_n_ansatz_layers 2 \
    --qts_dropout 0.1
