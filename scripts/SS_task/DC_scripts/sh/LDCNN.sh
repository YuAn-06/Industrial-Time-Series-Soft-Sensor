#!/bin/bash
set -e

echo "========================================"
echo "Run model: LDCNN"
echo "Experiment: soft_sensor on DC"
echo "Window: seq_len=16, pred_len=1"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python -u run.py \
    --model 'LDCNN' \
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
    --seq_len 16 \
    --label_len 16 \
    --pred_len 1 \
    --embed 'TimeF' \
    --freq 's' \
    --factor 1 \
    --n_components 5 \
    --d_model 128 \
    --n_heads 8 \
    --e_layers 4 \
    --d_layers 1 \
    --d_ff 128 \
    --dropout 0.2 \
    --activation 'gelu' \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --epoch 300 \
    --patience 10 \
    --lradj 'cosine' \
    --inverse \
    --use_cuda \
    --device 'cuda' \
    --gpu 0 \
    --seed 2021 \
    --device_ids 0
