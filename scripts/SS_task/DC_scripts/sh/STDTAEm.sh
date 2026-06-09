#!/bin/bash
set -e

# Baseline values come from scripts/SS_task/DC_scripts/yaml/STDTAEm.yaml.
# Add more candidates to these arrays when tuning.
seq_lens=(16)
label_lens=(16)
hidden_dims=(8 6 16)
latent_dims=(4 8)
std_windows=(4)
tae_hidden_dims_list=("16" "16 32")
tae_backbones=("mlp")
freeze_tae=True
dropouts=(0.1)
tae_betas=(0.5)
triplet_margins=(0.3)
tae_noise_stds=(0.01)
tae_mask_ratios=(0.1)
batch_sizes=(32)
pretrain_learning_rates=(0.001)
finetune_learning_rates=(0.0005)
weight_decays=(0.0)
seeds=(2021)

base_yaml="scripts/SS_task/DC_scripts/yaml/STDTAEm.yaml"
tmp_dir="scripts/SS_task/DC_scripts/yaml/.tmp_STDTAEm_tuning"
mkdir -p "${tmp_dir}"

run_id=0

echo "========================================"
echo "Run model: STDTAEm"
echo "Experiment: soft_sensor on DC"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

for seq_len in "${seq_lens[@]}"; do
for label_len in "${label_lens[@]}"; do
for hidden_dim in "${hidden_dims[@]}"; do
for latent_dim in "${latent_dims[@]}"; do
for std_window in "${std_windows[@]}"; do
for tae_hidden_dims in "${tae_hidden_dims_list[@]}"; do
for tae_backbone in "${tae_backbones[@]}"; do
for dropout in "${dropouts[@]}"; do
for tae_beta in "${tae_betas[@]}"; do
for triplet_margin in "${triplet_margins[@]}"; do
for tae_noise_std in "${tae_noise_stds[@]}"; do
for tae_mask_ratio in "${tae_mask_ratios[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
for pretrain_lr in "${pretrain_learning_rates[@]}"; do
for finetune_lr in "${finetune_learning_rates[@]}"; do
for weight_decay in "${weight_decays[@]}"; do
for seed in "${seeds[@]}"; do
    run_id=$((run_id + 1))
    tmp_yaml="${tmp_dir}/STDTAEm_run${run_id}.yaml"

    echo "----------------------------------------"
    echo "Run ${run_id}: seq_len=${seq_len}, hidden_dim=${hidden_dim}, latent_dim=${latent_dim}, std_window=${std_window}, tae_hidden_dims=${tae_hidden_dims}, dropout=${dropout}, pre_lr=${pretrain_lr}, ft_lr=${finetune_lr}, seed=${seed}"
    echo "----------------------------------------"

    python -c "import yaml; src=r'${base_yaml}'; dst=r'${tmp_yaml}'; cfg=yaml.safe_load(open(src, encoding='utf-8')); p=cfg['params']; p.update({'seq_len': int('${seq_len}'), 'label_len': int('${label_len}'), 'hidden_dim': int('${hidden_dim}'), 'latent_dim': int('${latent_dim}'), 'std_window': int('${std_window}'), 'tae_hidden_dims': [int(x) for x in '${tae_hidden_dims}'.split()], 'tae_backbone': '${tae_backbone}', 'freeze_tae': ${freeze_tae}, 'dropout': float('${dropout}'), 'tae_beta': float('${tae_beta}'), 'triplet_margin': float('${triplet_margin}'), 'tae_noise_std': float('${tae_noise_std}'), 'tae_mask_ratio': float('${tae_mask_ratio}'), 'batch_size': int('${batch_size}'), 'pretrain_learning_rate': float('${pretrain_lr}'), 'finetune_learning_rate': float('${finetune_lr}'), 'learning_rate': float('${finetune_lr}'), 'weight_decay': float('${weight_decay}'), 'seed': int('${seed}')}); yaml.safe_dump(cfg, open(dst, 'w', encoding='utf-8'), sort_keys=False, allow_unicode=True)"

    python -u run_pretrain_finetune.py --yaml "${tmp_yaml}"
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
done
done
done
done

echo "========================================"
echo "Finished ${run_id} run(s)"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
