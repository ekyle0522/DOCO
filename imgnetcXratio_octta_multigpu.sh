#!/bin/bash

set -Eeuo pipefail
trap 'echo "Error at line $LINENO: $BASH_COMMAND"; exit 1' ERR

export HF_HUB_OFFLINE=1

CONDA_ENV_NAME="${CONDA_ENV_NAME:-doco}"
DATA_DIR="${DATA_ROOT:-/mnt/d/stamp_lib/datasets}"
SAVE_DIR_BASE="${SAVE_DIR_BASE:-../output_imgnetc}"

BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_ID_SAMPLES="${NUM_ID_SAMPLES:-5000}"

# Edit this list, or override from the command line:
#   bash imgnetcXratio_octta_multigpu.sh 1:doco,dpcore 2:eata,source 7:vida
GPU_METHOD_GROUPS=(
    "0:doco"
    "1:dpcore"
    "2:eata,eatacome"
    "3:eataunient"
    "4:stamp"
    "5:cotta"
    "6:vida"
    "7:source,tent,ostta"
    "8:sar"
    "9:sarcome"
)

if [ "$#" -gt 0 ]; then
    GPU_METHOD_GROUPS=("$@")
fi

OOD_DATASETS=(
    "places365"
    "textures"
    "inaturalist"
    "sun"
    "ssb-hard"
    "ninco_ood_classes"
)

OOD_PROPORTIONS=(
    50
    # 10
    # 20
    # 30
    # 40
)

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found."
    exit 1
fi

conda activate "$CONDA_ENV_NAME"

echo "========================================================"
echo "Starting multi-GPU ImageNet-C ratio suite at $(date)"
echo "Assignments: ${GPU_METHOD_GROUPS[*]}"
echo "========================================================"

cd imagenet || { echo "Failed to enter the imagenet directory."; exit 1; }
mkdir -p "$SAVE_DIR_BASE"

compute_num_ood_samples() {
    local ood_ratio_percent="$1"
    if [ "$ood_ratio_percent" -eq 0 ]; then
        echo 0
    else
        awk -v p="$ood_ratio_percent" -v n="$NUM_ID_SAMPLES" 'BEGIN {
            r = p / 100.0
            printf "%.0f", (r * n) / (1 - r)
        }'
    fi
}

run_gpu_lane() {
    local gpu_id="$1"
    local method_csv="$2"
    local assigned_methods=()
    local method_raw
    local method
    local ood_ratio_percent
    local num_ood_samples
    local dataset
    local config_file
    local run_save_dir

    IFS=',' read -r -a assigned_methods <<< "$method_csv"

    echo "[GPU ${gpu_id}] Starting methods: ${method_csv}"

    for ood_ratio_percent in "${OOD_PROPORTIONS[@]}"
    do
        num_ood_samples="$(compute_num_ood_samples "$ood_ratio_percent")"

        echo ""
        echo "[GPU ${gpu_id}] ########################################################"
        echo "[GPU ${gpu_id}] Current OOD ratio: ${ood_ratio_percent}%"
        echo "[GPU ${gpu_id}] Computed OOD samples: ${num_ood_samples}"
        echo "[GPU ${gpu_id}] ########################################################"

        for method_raw in "${assigned_methods[@]}"
        do
            method="${method_raw//[[:space:]]/}"
            if [ -z "$method" ]; then
                continue
            fi

            config_file="./cfgs/ood_${method}.yaml"
            if [ ! -f "$config_file" ]; then
                echo "[GPU ${gpu_id}] Warning: Config file $config_file not found. Skipping method $method."
                continue
            fi

            for dataset in "${OOD_DATASETS[@]}"
            do
                run_save_dir="${SAVE_DIR_BASE}/ratio_${ood_ratio_percent}/${method}/${dataset}"
                mkdir -p "$run_save_dir"

                echo ""
                echo "[GPU ${gpu_id}] --------------------------------------------------------"
                echo "[GPU ${gpu_id}] Running: ratio=${ood_ratio_percent}% / method=${method} / dataset=${dataset}"
                echo "[GPU ${gpu_id}] --------------------------------------------------------"

                CUDA_VISIBLE_DEVICES="$gpu_id" python octta_main.py \
                    --cfg "$config_file" \
                    --data_dir "$DATA_DIR" \
                    RNG_SEED 1 \
                    CORRUPTION.ID_BENCHMARK "imagenet_c" \
                    CORRUPTION.OOD_DATASET "$dataset" \
                    CORRUPTION.NUM_OOD_SAMPLES "$num_ood_samples" \
                    CORRUPTION.NUM_EX "$NUM_ID_SAMPLES" \
                    TEST.BATCH_SIZE "$BATCH_SIZE" \
                    OPTIM.DOCO_BETA 0.5 \
                    OPTIM.DOCO_PROMPT_NUM 8 \
                    SAVE_DIR "$run_save_dir"

                echo "[GPU ${gpu_id}] Done: ratio=${ood_ratio_percent}% / method=${method} / dataset=${dataset}"
            done
        done
    done

    echo "[GPU ${gpu_id}] Finished methods: ${method_csv}"
}

pids=()
for group in "${GPU_METHOD_GROUPS[@]}"
do
    if [[ "$group" != *:* ]]; then
        echo "Invalid assignment '$group'. Expected format: GPU_ID:method1,method2"
        exit 1
    fi

    gpu_id="${group%%:*}"
    method_csv="${group#*:}"
    run_gpu_lane "$gpu_id" "$method_csv" &
    pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"
do
    if ! wait "$pid"; then
        exit_code=1
    fi
done

echo ""
echo "========================================================"
if [ "$exit_code" -eq 0 ]; then
    echo "All multi-GPU ImageNet-C ratio tasks completed at $(date)"
else
    echo "One or more multi-GPU ImageNet-C ratio tasks failed at $(date)"
fi
echo "========================================================"

exit "$exit_code"
