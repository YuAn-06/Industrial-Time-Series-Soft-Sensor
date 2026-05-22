#!/bin/bash
set -e

# Best seen: num_channels=4,8,16 kernel2 dropout0.0 bt64 lr0.001
for kernel_size in 2 3; do
    for dropout in 0.0 0.05; do
        for lr in 0.0005 0.001; do
            echo "===== Running TCN | fixed best channels=4,8,16 | kernel=${kernel_size} | dropout=${dropout} | lr=${lr} ====="

            python -u run.py \
                --model 'TCN' \
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
                --num_channels '4,8,16' \
                --kernel_size ${kernel_size} \
                --dropout ${dropout} \
                --batch_size 64 \
                --learning_rate ${lr} \
                --weight_decay 0.0 \
                --epoch 300 \
                --patience 15 \
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
