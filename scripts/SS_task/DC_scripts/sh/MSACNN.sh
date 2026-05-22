#!/bin/bash
set -e

# Best seen: dm16 reduction_ratio32 stride1 dropout0.05 bt64 lr0.001 wd0.0001
for reduction_ratio in 16 32; do
    for weight_decay in 0.0 0.0001 0.001; do
        echo "===== Running MSACNN | fixed best dm16/stride1/dropout0.05 | reduction_ratio=${reduction_ratio} | weight_decay=${weight_decay} ====="

        python -u run.py \
            --model 'MSACNN' \
            --task 'soft_sensor' \
            --data_name 'DC' \
            --data_path './data/DC/debutanizer_column.csv' \
            --target 'y_1' \
            --num_workers 1 \
            --missing_rate 0 \
            --enc_in 13 \
            --dec_in 13 \
            --C_in 13 \
            --C_out 1 \
            --seq_len 16 \
            --label_len 16 \
            --pred_len 1 \
            --d_model 16 \
            --stride 1 \
            --dropout 0.05 \
            --activation 'GELU' \
            --reduction_ratio ${reduction_ratio} \
            --batch_size 64 \
            --learning_rate 0.001 \
            --weight_decay ${weight_decay} \
            --epoch 400 \
            --patience 20 \
            --lradj 'cosine' \
            --inverse \
            --use_cuda \
            --device 'cuda' \
            --gpu 0 \
            --seed 2021 \
            --data_aug

        sleep 5
    done
done

wait
