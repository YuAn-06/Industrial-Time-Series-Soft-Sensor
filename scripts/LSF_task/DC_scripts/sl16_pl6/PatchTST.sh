

GPU=0
for d_model in 32 64 128 256; do
    for d_ff in 64 128 256 512; do
        {
            python -u run.py --model 'PatchTST' \
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
                         --d_model ${d_model} \
                         --d_ff ${d_ff} \
                         --n_heads 4 \
                         --e_layers 1 \
                         --d_layers 1 \
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






