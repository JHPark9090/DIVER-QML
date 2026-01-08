#!/bin/bash
# Fine-tuning DIVER-1 with Quantum Time Series Transformer (QTS) on PhysioNet-MI dataset
# PhysioNet-MI: 4-class motor imagery classification, 64 EEG channels

# Activate conda environment
# conda activate DIVER_QML

python finetune_main.py \
    --seed 42 \
    --cuda 0 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1.00e-03 \
    --weight_decay 1.00e-02 \
    --downstream_dataset PhysioNet-MI \
    --datasets_dir /home/connectome/justin/DIVER_QML/DIVER_data/PhysioNet_MI \
    --model_dir /home/connectome/justin/DIVER_QML/DIVER_data/qts_checkpoints/physionet \
    --foundation_dir /home/connectome/justin/DIVER_QML/DIVER_data/dmodel256_mp_rank_00_model_states.pt \
    --width 256 \
    --depth 12 \
    --patch_size 500 \
    --ft_config flatten_qts \
    --early_stop_criteria val_f1 \
    --early_stop_patience 20 \
    --mup_weights False \
    --ft_mup False \
    --use_amp False \
    --deepspeed_pth_format True \
    --frozen True \
    --precompute_features True \
    --qts_n_qubits 8 \
    --qts_degree 2 \
    --qts_n_ansatz_layers 2 \
    --qts_dropout 0.1
