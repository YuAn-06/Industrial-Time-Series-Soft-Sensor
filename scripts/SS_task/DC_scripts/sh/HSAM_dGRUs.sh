#!/bin/bash
set -e

echo "========================================"
echo "Run model: HSAM_dGRUs"
echo "Experiment: soft_sensor on DC"
echo "Window: seq_len=4, pred_len=1"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python -u run.py \
    --model 'HSAM_dGRUs' \
    --task 'soft_sensor' \
    --data_name 'DC' \
    --data_path './data/DC/debutanizer_column.csv' \
    --target 'y_1' \
    --data_aug \
    --num_workers 1 \
    --missing_rate 0 \
    --enc_in 13 \
    --dec_in 13 \
    --C_in 13 \
    --C_out 1 \
    --seq_len 4 \
    --label_len 16 \
    --pred_len 1 \
    --hidden_dim 8 \
    --d_model 32 \
    --n_heads 4 \
    --e_layers 1 \
    --d_layers 1 \
    --dropout 0.05 \
    --activation 'GELU' \
    --batch_size 128 \
    --learning_rate 0.01 \
    --epoch 100 \
    --patience 30 \
    --lradj 'type2' \
    --inverse \
    --use_cuda \
    --device 'cuda' \
    --gpu 0 \
    --seed 2021 \
    --device_ids 0
