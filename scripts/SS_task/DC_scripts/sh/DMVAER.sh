#!/bin/bash
set -e

echo "========================================"
echo "Run model: DMVAER"
echo "Experiment: soft_sensor on DC"
echo "Window: seq_len=16, pred_len=6"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python -u run.py \
    --model 'DMVAER' \
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
    --pred_len 6 \
    --embed 'TimeF' \
    --freq 's' \
    --n_components 3 \
    --z_global_dim 32 \
    --z_local_dim 32 \
    --DMVAER_loss_weight 0.1 1 1 1 0.001 \
    --d_model 64 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --epoch 300 \
    --patience 10 \
    --lradj 'cosine' \
    --inverse \
    --use_cuda \
    --device 'cuda' \
    --gpu 0 \
    --seed 2021 \
    --device_ids 0
