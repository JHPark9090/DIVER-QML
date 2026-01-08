#!/bin/bash
# Hyperparameter sweep for QTS on FACED dataset
# Tests different quantum circuit configurations

# Activate conda environment
# conda activate DIVER_QML

# Hyperparameter configurations
SEEDS=(42 43 44)
N_QUBITS=(6 8 10)
DEGREES=(1 2 3)
N_ANSATZ_LAYERS=(1 2 3)

# Base paths
DATASETS_DIR="/home/connectome/justin/DIVER_QML/DIVER_data/FACED"
FOUNDATION_DIR="/home/connectome/justin/DIVER_QML/DIVER_data/dmodel256_mp_rank_00_model_states.pt"
BASE_MODEL_DIR="/home/connectome/justin/DIVER_QML/DIVER_data/qts_checkpoints/faced_sweep"

run_experiment() {
    local seed=$1
    local n_qubits=$2
    local degree=$3
    local n_ansatz_layers=$4

    local exp_name="seed${seed}_q${n_qubits}_d${degree}_l${n_ansatz_layers}"
    local model_dir="${BASE_MODEL_DIR}/${exp_name}"

    echo "Running experiment: ${exp_name}"

    python finetune_main.py \
        --seed ${seed} \
        --cuda 0 \
        --epochs 100 \
        --batch_size 32 \
        --lr 1.00e-03 \
        --weight_decay 1.00e-02 \
        --downstream_dataset FACED \
        --datasets_dir ${DATASETS_DIR} \
        --model_dir ${model_dir} \
        --foundation_dir ${FOUNDATION_DIR} \
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
        --qts_n_qubits ${n_qubits} \
        --qts_degree ${degree} \
        --qts_n_ansatz_layers ${n_ansatz_layers} \
        --qts_dropout 0.1

    echo "Finished experiment: ${exp_name}"
    echo "----------------------------------------"
}

# Run sweep
for seed in "${SEEDS[@]}"; do
    for n_qubits in "${N_QUBITS[@]}"; do
        for degree in "${DEGREES[@]}"; do
            for n_ansatz_layers in "${N_ANSATZ_LAYERS[@]}"; do
                run_experiment ${seed} ${n_qubits} ${degree} ${n_ansatz_layers}
            done
        done
    done
done

echo "All experiments completed!"
