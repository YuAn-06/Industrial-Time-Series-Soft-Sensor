
                        echo "========================================"
                        echo "Run model: VRNN"
                        echo "Experiment: short_term_forecasting on DC"
                        echo "Input data: ./data/DC/debutanizer_column.csv"
                        echo "Output target: y_1"
                        echo "Window: seq_len=16, pred_len=6"
                        echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
                        echo "========================================"
                        python -u run.py \
                        --model 'VRNN' \
                        --task 'short_term_forecasting' \
                        --data_name "DC" \
                        --data_path './data/DC/debutanizer_column.csv' \
                        --target 'y_1' \
                        --num_workers 1 \
                        --missing_rate 0 \
                        --C_in 8 \
                        --C_out 1 \
                        --seq_len 16 \
                        --label_len 16 \
                        --pred_len 6 \
                        --d_model 16 \
                        --activation 'GELU' \
                        --x_embed_dim 32 \
                        --z_embed_dim 16 \
                        --z_dim 8 \
                        --batch_size 64 \
                        --learning_rate 0.001 \
                        --epoch 300 \
                        --patience 10 \
                        --lradj 'cosine' \
                        --use_cuda  \
                        --device "cuda" \
                        --gpu 0 \
                        --seed 2021 \
