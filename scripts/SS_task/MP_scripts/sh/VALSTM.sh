#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../../.."

python -u run.py \
    --model 'VALSTM' \
    --task 'soft_sensor' \
    --data_name 'MP' \
    --data_path './data/MP/MP_data.csv' \
    --target '% Silica Concentrate' \
    --num_workers 1 \
    --missing_rate 0 \
    --enc_in 22 \
    --dec_in 22 \
    --C_in 22 \
    --C_out 1 \
    --seq_len 4 \
    --label_len 4 \
    --pred_len 1 \
    --embed 'TimeF' \
    --freq 'h' \
    --factor 1 \
    --hidden_dim 128 \
    --dropout 0.05 \
    --activation 'gelu' \
    --batch_size 64 \
    --learning_rate 0.001 \
    --epoch 300 \
    --patience 10 \
    --lradj 'cosine' \
    --weight_decay 0.0 \
    --inverse \
    --use_cuda \
    --device 'cuda' \
    --gpu 0 \
    --seed 2021 \
    --device_ids 0
