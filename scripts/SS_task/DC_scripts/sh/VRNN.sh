#!/bin/bash
set -e

# Best seen: dm32 dff64 x_embed8 z_embed8 z_dim8 ReLU lr0.001, but batch is kept at 64 by current rule
for d_model in 32 64; do
    for z_dim in 4 8 16; do
        echo "===== Running VRNN | fixed best data_aug=1/C_in=13/dff64/xed8/zed8/ReLU | d_model=${d_model} | z_dim=${z_dim} | batch=64 | lr=0.001 ====="

        python -u run.py \
            --model 'VRNN' \
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
            --d_ff 64 \
            --x_embed_dim 8 \
            --z_embed_dim 8 \
            --z_dim ${z_dim} \
            --n_heads 4 \
            --e_layers 1 \
            --d_layers 1 \
            --dropout 0.05 \
            --activation 'ReLU' \
            --num_landmarks 4 \
            --batch_size 64 \
            --learning_rate 0.001 \
            --weight_decay 0.0 \
            --epoch 100 \
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
