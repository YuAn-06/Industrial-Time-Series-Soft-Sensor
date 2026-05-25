#!/usr/bin/env bash

    python -u run.py \
        --model 'GCT' \
        --task 'soft_sensor' \
        --data_name 'SRU' \
        --data_path './data/SRU/SRU_data.csv' \
        --target 'SO2' \
        --num_workers 1 \
        --data_aug \
        --missing_rate 0 \
        --enc_in 21 \
        --dec_in 21 \
        --C_in 21 \
        --C_out 1 \
        --seq_len 16 \
        --embed 'TimeF' \
        --freq 's' \
        --factor 1 \
        --label_len 16 \
        --pred_len 6 \
        --d_model 64 \
        --d_ff 2048 \
        --kernel_size 3 \
        --n_heads 8 \
        --e_layers 1 \
        --d_layers 1 \
        --dropout 0.05 \
        --activation 'gelu' \
        --batch_size 64 \
        --learning_rate 0.001 \
        --epoch 300 \
        --patience 10 \
        --lradj 'cosine' \
        --inverse \
        --use_cuda \
        --device 'cuda' \
        --gpu 0 \
        --seed 1998

    sleep 5
done
done

wait
