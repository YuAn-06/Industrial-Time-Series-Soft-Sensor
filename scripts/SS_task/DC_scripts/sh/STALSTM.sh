#!/bin/bash
set -e

# Best seen: ci13 augTrue SA60 TA90 dropout0.05 bt64 lr0.001
for SA_dim in 60 90; do
    for TA_dim in 60 90 120; do
        echo "===== Running STALSTM | fixed best data_aug=1/C_in=13/dropout0.05 | SA_dim=${SA_dim} | TA_dim=${TA_dim} ====="

        python -u run.py \
            --model 'STALSTM' \
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
            --SA_dim ${SA_dim} \
            --TA_dim ${TA_dim} \
            --dropout 0.05 \
            --activation 'gelu' \
            --batch_size 64 \
            --learning_rate 0.001 \
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
