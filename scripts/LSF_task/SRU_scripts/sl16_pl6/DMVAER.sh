
            echo "========================================"
            echo "Run model: DMVAER"
            echo "Experiment: short_term_forecasting on SRU"
            echo "Input data: ./data/SRU/SRU_data.csv"
            echo "Output target: SO2"
            echo "Window: seq_len=16, pred_len=6"
            echo "Loop params: z_global_dim=${z_global_dim}, z_dim=${z_dim}"
            echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "========================================"
            python -u run.py \
                         --model 'DMVAER' \
                         --task 'short_term_forecasting' \
                         --data_name "SRU" \
                         --data_path './data/SRU/SRU_data.csv' \
                         --target 'SO2' \
                         --num_workers 1 \
                         --missing_rate 0 \
                         --enc_in 6 \
                         --dec_in 6 \
                         --C_in 6 \
                         --C_out 1 \
                         --seq_len 16 \
                         --embed 'TimeF' \
                         --freq 's' \
                         --factor 1 \
                         --label_len 16 \
                         --pred_len 6 \
                         --n_heads 8 \
                         --e_layers 1 \
                         --d_layers 1 \
                         --z_global_dim ${z_global_dim} \
                         --z_local_dim ${z_global_dim} \
                         --z_dim ${z_dim} \
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

done
done
wait
