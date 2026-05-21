
                        python -u run.py \
                        --model 'TimeKAN' \
                        --task 'short_term_forecasting' \
                        --data_name "SRU" \
                        --data_path './data/SRU/SRU_data.csv' \
                        --target 'SO2' \
                        --num_workers 1 \
                        --missing_rate 0 \
                        --enc_in 6 \
                        --C_in 6 \
                        --C_out 1 \
                        --seq_len 16 \
                        --embed 'TimeF' \
                        --freq 's' \
                        --label_len 16 \
                        --pred_len 6 \
                        --moving_avg 3 \
                        --down_sampling_window 8 \
                        --down_sampling_layers 1 \
                        --d_model 16 \
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
done
wait 