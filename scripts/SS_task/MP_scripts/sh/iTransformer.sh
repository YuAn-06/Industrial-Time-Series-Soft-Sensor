#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../../.."

python -u run.py \
    --model 'iTransformer' \
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
    --seq_len 16 \
    --label_len 16 \
    --pred_len 1 \
    --embed 'TimeF' \
    --freq 'h' \
    --factor 1 \
    --d_model 64 \
    --d_ff 256 \
    --n_heads 4 \
    --e_layers 1 \
    --d_layers 1 \
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
