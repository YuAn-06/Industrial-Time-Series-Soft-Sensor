
            python -u run.py 
                         --model 'TCN' \
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
                         --pred_len 1 \
                         --embed 'TimeF' \
                         --freq 's' \
                         --kernel_size 4 \
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