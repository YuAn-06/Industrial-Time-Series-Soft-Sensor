
            python -u run.py \
                         --model 'HSAM_dGRUs' \
                         --task 'soft_sensor' \
                         --data_name "SRU" \
                         --data_path './data/SRU/SRU_data.csv' \
                         --target 'SO2' \
                         --num_workers 1 \
                         --missing_rate 0 \
                         --data_aug \
                         --enc_in 20 \
                         --dec_in 20 \
                         --C_in 20 \
                         --C_out 1 \
                         --seq_len 4 \
                         --embed 'TimeF' \
                         --freq 's' \
                         --factor 1 \
                         --label_len 4 \
                         --pred_len 1 \
                         --d_model 64 \
                         --n_heads 3 \
                         --hidden_dim 2 \
                         --num_layers 1 \
                         --dropout 0.05 \
                         --activation 'GELU' \
                         --batch_size 64 \
                         --learning_rate 0.001 \
                         --epoch 300 \
                         --patience 10 \
                         --lradj 'cosine' \
                         --inverse \
                         --use_cuda  \
                         --device "cuda" \
                         --gpu 0 \
                         --seed 2021 \

        sleep 5

wait






