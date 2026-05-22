#!/bin/bash
set -e

# Best seen: dm128 el4 dropout0.2 bt64 lr0.0005
for dropout in 0.1 0.2 0.3; do
    for lr in 0.0003 0.0005 0.001; do
        echo "===== Running LDCNN | fixed best data_aug=1/C_in=13/dm128/el4 | dropout=${dropout} | lr=${lr} ====="

        python -u run.py \
            --model 'LDCNN' \
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
            --d_model 128 \
            --e_layers 4 \
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
