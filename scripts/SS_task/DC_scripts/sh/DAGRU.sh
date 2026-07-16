#!/bin/bash
set -e

echo "========================================"
echo "Run model: DAGRU"
echo "Experiment: soft_sensor on DC"
echo "Window: seq_len=4, pred_len=1"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python -u run.py \
    --model 'DAGRU' \
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
    --embed 'TimeF' \
    --freq 's' \
    --d_model 32 \
    --d_ff 64 \
    --batch_size 64 \
    --learning_rate 0.0005 \
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
