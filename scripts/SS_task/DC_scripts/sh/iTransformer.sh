#!/bin/bash
set -e

# Best seen: dm64 dff512 nh8 el1 dl1 dp0.05 bt64 lr0.001
for dropout in 0.0 0.05; do
    for lr in 0.0005 0.001; do
        echo "===== Running iTransformer | fixed best dm64/dff512/nh8 | dropout=${dropout} | lr=${lr} ====="

        python -u run.py \
            --model 'iTransformer' \
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
            --embed 'TimeF' \
            --freq 's' \
            --factor 1 \
            --d_model 64 \
            --d_ff 512 \
            --n_heads 8 \
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

wait
