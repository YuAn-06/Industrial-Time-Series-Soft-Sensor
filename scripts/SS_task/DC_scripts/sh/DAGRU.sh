#!/bin/bash
set -e

# Best seen: dm32 dff64 nh4 el1 dl1 dropout0.05 bt64 lr0.0005
for d_model in 32 64; do
    for d_ff in 64 128; do
        for lr in 0.0005 0.001; do
            echo "===== Running DAGRU | fixed best nh4/el1/dl1/dropout0.05 | d_model=${d_model} | d_ff=${d_ff} | lr=${lr} ====="

            python -u run.py \
                --model 'DAGRU' \
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
                --seq_len 4 \
                --label_len 4 \
                --pred_len 1 \
                --embed 'TimeF' \
                --freq 's' \
                --factor 1 \
                --d_model ${d_model} \
                --d_ff ${d_ff} \
                --n_heads 4 \
                --e_layers 1 \
                --d_layers 1 \
                --dropout 0.05 \
                --activation 'gelu' \
                --batch_size 64 \
                --learning_rate ${lr} \
                --weight_decay 0.0 \
                --epoch 300 \
                --patience 10 \
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
done

wait
