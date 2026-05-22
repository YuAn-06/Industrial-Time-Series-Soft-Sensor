#!/bin/bash
set -e

# Best seen: sl16 dm32 zg32 zl32 nc3 bt64 lr0.0005
for latent_pair in 32:32 32:16 16:32; do
    z_global_dim="${latent_pair%%:*}"
    z_local_dim="${latent_pair##*:}"
    for lr in 0.0001 0.0005 0.001; do
        echo "===== Running DMVAER | fixed best data_aug=1/C_in=13/dm32/nc3 | z_global=${z_global_dim} | z_local=${z_local_dim} | batch=64 | lr=${lr} ====="

        python -u run.py \
            --model 'DMVAER' \
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
            --d_model 32 \
            --d_ff 64 \
            --z_global_dim ${z_global_dim} \
            --z_local_dim ${z_local_dim} \
            --z_dim ${z_local_dim} \
            --n_components 3 \
            --n_heads 4 \
            --e_layers 1 \
            --d_layers 1 \
            --dropout 0.05 \
            --activation 'GELU' \
            --DMVAER_loss_weight '0.1,1,1,1,0.001' \
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
