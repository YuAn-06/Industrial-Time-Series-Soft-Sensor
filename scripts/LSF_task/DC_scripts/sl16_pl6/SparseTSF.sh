
for d_model in 32 64 128 256; do
        for period_len in 2 4 6; do
            python -u run.py \
                         --model 'SparseTSF' \
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
                         --d_model ${d_model} \
                         --period_len ${period_len} \
                         --pred_len 6 \
                         --embed 'TimeF' \
                         --freq 's' \
                         --factor 1 \
                         --model_type 'mlp' \
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
done
done
wait






