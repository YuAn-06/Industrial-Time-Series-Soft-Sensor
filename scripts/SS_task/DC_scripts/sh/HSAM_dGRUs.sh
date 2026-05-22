#!/bin/bash
set -e

# Best seen: hidden_dim8 d_model32 nh4 el1 dl1 lr0.01, but batch is kept at 64 by current rule
for hidden_dim in 4 8; do
    for d_model in 32 64; do
        for lr in 0.001 0.01; do
            echo "===== Running HSAM_dGRUs | fixed best nh4/el1/dl1 | hidden_dim=${hidden_dim} | d_model=${d_model} | batch=64 | lr=${lr} ====="

            python -u run.py \
                --model 'HSAM_dGRUs' \
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
                --hidden_dim ${hidden_dim} \
                --d_model ${d_model} \
                --d_ff 64 \
                --n_heads 4 \
                --e_layers 1 \
                --d_layers 1 \
                --dropout 0.05 \
                --activation 'gelu' \
                --batch_size 64 \
                --learning_rate ${lr} \
                --weight_decay 0.0 \
                --epoch 300 \
                --patience 30 \
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
