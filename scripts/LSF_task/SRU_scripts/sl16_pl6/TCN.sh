
            echo "========================================"
            echo "Run model: TCN"
            echo "Experiment: short_term_forecasting on SRU"
            echo "Input data: ./data/SRU/SRU_data.csv"
            echo "Output target: SO2"
            echo "Window: seq_len=16, pred_len=6"
            echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "========================================"
            python -u run.py --model 'TCN' \
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
                         --label_len 8 \
                         --pred_len 6 \
                         --embed 'TimeF' \
                         --freq 's' \
                         --kernel_size 8 \
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
