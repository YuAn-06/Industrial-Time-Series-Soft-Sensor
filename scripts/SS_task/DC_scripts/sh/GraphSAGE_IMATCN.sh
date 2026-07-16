#!/bin/bash
set -e

# Baseline values come from scripts/SS_task/DC_scripts/yaml/GraphSAGE_IMATCN.yaml.
# Add more candidates to these arrays when tuning.
seq_lens=(128)
d_models=(64)
n_heads_list=(8)
num_channels_list=("64 64") # "64 64"
kernel_sizes=(2)
dropouts=(0.05)
graph_build_methods=("mi")
graph_thresholds=(0.4)
graph_sample_sizes=(64)
batch_sizes=(64)
learning_rates=(0.001)
weight_decays=(0.0001)
seeds=(2021)

run_id=0

echo "========================================"
echo "Run model: GraphSAGE_IMATCN"
echo "Experiment: soft_sensor on DC"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

for seq_len in "${seq_lens[@]}"; do
for d_model in "${d_models[@]}"; do
for n_heads in "${n_heads_list[@]}"; do
for num_channels in "${num_channels_list[@]}"; do
for kernel_size in "${kernel_sizes[@]}"; do
for dropout in "${dropouts[@]}"; do
for graph_build_method in "${graph_build_methods[@]}"; do
for graph_threshold in "${graph_thresholds[@]}"; do
for graph_sample_size in "${graph_sample_sizes[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
for learning_rate in "${learning_rates[@]}"; do
for weight_decay in "${weight_decays[@]}"; do
for seed in "${seeds[@]}"; do
    run_id=$((run_id + 1))

    echo "----------------------------------------"
    echo "Run ${run_id}: seq_len=${seq_len}, d_model=${d_model}, n_heads=${n_heads}, num_channels=${num_channels}, kernel_size=${kernel_size}, dropout=${dropout}, graph=${graph_build_method}/${graph_threshold}, lr=${learning_rate}, wd=${weight_decay}, seed=${seed}"
    echo "----------------------------------------"

    python -u run.py \
        --model 'GraphSAGE_IMATCN' \
        --task 'soft_sensor' \
        --data_name 'DC' \
        --data_path './data/DC/debutanizer_column.csv' \
        --target 'y_1' \
        --data_aug \
        --num_workers 1 \
        --missing_rate 0 \
        --enc_in 13 \
        --dec_in 13 \
        --C_in 13 \
        --C_out 1 \
        --seq_len "${seq_len}" \
        --label_len "${seq_len}" \
        --pred_len 1 \
        --num_channels ${num_channels} \
        --kernel_size "${kernel_size}" \
        --d_model "${d_model}" \
        --n_heads "${n_heads}" \
        --dropout "${dropout}" \
        --graph_build_method "${graph_build_method}" \
        --graph_threshold "${graph_threshold}" \
        --graph_sample_size "${graph_sample_size}" \
        --batch_size "${batch_size}" \
        --learning_rate "${learning_rate}" \
        --epoch 300 \
        --patience 10 \
        --lradj 'cosine' \
        --weight_decay "${weight_decay}" \
        --inverse \
        --use_cuda \
        --device 'cuda' \
        --gpu 0 \
        --seed "${seed}" \
        --device_ids 0
done
done
done
done
done
done
done
done
done
done
done
done
done

echo "========================================"
echo "Finished ${run_id} run(s)"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
