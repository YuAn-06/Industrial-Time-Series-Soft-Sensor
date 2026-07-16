
            python -u run.py --model 'MSGNet' \
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
                         --seq_len 16 \
                         --label_len 16 \
                         --pred_len 6 \
                         --embed 'TimeF' \
                         --node_dim 8 \
                         --conv_channel 12 \
                         --skip_channel 6 \
                         --propalpha 0.1 \
                         --gcn_depth 2 \
                         --freq 's' \
                         --d_model 64 \
                         --d_ff 512 \
                         --n_heads 4 \
                         --e_layers 1 \
                         --dropout 0.05 \
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






