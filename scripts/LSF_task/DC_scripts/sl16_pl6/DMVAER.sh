
for z_global_dim in 8 16 32 64 ; do
    for z_dim in 8 16 32 64 ; do

            python -u run.py \
                         --model 'DMVAER' \
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
                         --if_valid False \
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






