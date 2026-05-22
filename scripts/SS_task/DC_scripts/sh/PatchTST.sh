#!/bin/bash
set -e

# Best seen: patch4 stride2 dm128 dff128 nh4
for d_ff in 128 256; do
    for dropout in 0.0 0.05; do
        for lr in 0.0005 0.001; do
            echo "===== Running PatchTST | fixed best patch4/st2/dm128/nh4 | d_ff=${d_ff} | dropout=${dropout} | lr=${lr} ====="

            python -u run.py \
                --model 'PatchTST' \
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
                --label_len 1 \
                --pred_len 1 \
                --patch_len 4 \
                --stride 2 \
                --embed 'TimeF' \
                --freq 's' \
                --factor 1 \
                --d_model 128 \
                --d_ff ${d_ff} \
                --n_heads 4 \
                --e_layers 1 \
                --d_layers 1 \
                --dropout ${dropout} \
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
