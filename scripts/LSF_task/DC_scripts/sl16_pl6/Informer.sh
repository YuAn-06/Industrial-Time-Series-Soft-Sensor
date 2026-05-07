for e_layers in 1 2 3 4; do
for d_layers in 1 2 3 4; do
        {
            python -u run.py --model 'Informer' \
                         --task 'short_term_forecasting' \
                         --data_name "DC" \
                         --data_path './data/DC/debutanizer_column.csv' \
                         --target 'y_1' \
                         --num_workers 1 \
                         --missing_rate 0 \
                         --enc_in 8 \
                         --dec_in 8 \
                         --C_in 8 \
                         --C_out 1 \
                         --seq_len 16 \
                         --label_len 16 \
                         --pred_len 6 \
                         --embed 'TimeF' \
                         --freq 's' \
                         --factor 1 \
                         --distil \
                         --d_model 64 \
                         --d_ff 128 \
                         --n_heads 4 \
                         --e_layers ${e_layers}\
                         --d_layers ${d_layers} \
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
        }
done
done
wait