
            python -u run.py --model 'DAGRU' \
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
                         --label_len 4 \
                         --pred_len 1 \
                         --embed 'TimeF' \
                         --node_dim 8 \
                         --freq 's' \
                         --factor 1 \
                         --d_model 256 \
                         --d_ff 64 \
                         --dropout 0.05 \
                         --activation 'gelu' \
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






