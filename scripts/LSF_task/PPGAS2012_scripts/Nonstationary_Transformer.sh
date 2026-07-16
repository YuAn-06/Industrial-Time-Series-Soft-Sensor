#!/bin/bash

# Transformer model training script
python -u run.py --model 'Nonstationary_Transformer' \
                 --task 'short_term_forecasting' \
                 --data_name "PPGAS2011" \
                 --data_path './data/PPGAS/gt_2011.csv' \
                 --target 'NOX' \
                 --num_workers 1 \
                 --missing_rate 0 \
                 --enc_in 10 \
                 --dec_in 10 \
                 --C_in 10 \
                 --C_out 1 \
                 --seq_len 16 \
                 --embed 'TimeF' \
                 --freq 'h' \
                 --factor 1 \
                 --label_len 16 \
                 --pred_len 6 \
                 --d_model 128 \
                 --d_ff 256 \
                 --n_heads 8 \
                 --e_layers 1 \
                 --d_layers 1 \
                 --dropout 0.05 \
                 --activation 'gelu' \
                 --batch_size 64 \
                 --learning_rate 0.001 \
                 --epoch 200 \
                 --patience 10 \
                 --lradj 'cosine' \
                 --device "cuda" \
                 --gpu 0 \
                 --seed 2021 \



