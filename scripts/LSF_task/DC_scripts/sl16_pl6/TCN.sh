for kernel_size in 2 4 6 8 10 12 14 16; do

        {
            python -u run.py --model 'TCN' \
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
                         --label_len 8 \
                         --pred_len 6 \
                         --moving_avg 3 \
                         --embed 'TimeF' \
                         --freq 's' \
                         --factor 1 \
                         --kernel_size ${kernel_size} \
                         --e_layers 1\
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