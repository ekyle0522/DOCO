#!/bin/bash

set -Eeuo pipefail
trap 'echo "Error at line $LINENO: $BASH_COMMAND"; exit 1' ERR

export HF_HUB_OFFLINE=1

CONDA_ENV_NAME="${CONDA_ENV_NAME:-doco}"
DATA_DIR="${DATA_ROOT:-/mnt/d/stamp_lib/datasets}"
SAVE_DIR_BASE="${SAVE_DIR_BASE:-../output_closed_benchmarks}"
TEMP_CFG_DIR="${TEMP_CFG_DIR:-./cfgs/__temp_closed_runtime}"

NUM_ID_SAMPLES="${NUM_ID_SAMPLES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Edit this list, or override from the command line:
#   bash closed_imgnetCLASR_multigpu.sh 1:doco,dpcore 2:eata,source 7:vida
GPU_METHOD_GROUPS=(
    "0:doco"
    "1:dpcore"
    "2:eata,eatacome"
    "3:eataunient"
    "4:ostta"
    "5:cotta"
    "6:vida"
    "7:source"
    "8:sar"
    "9:sarcome"
)

if [ "$#" -gt 0 ]; then
    GPU_METHOD_GROUPS=("$@")
fi

DATASETS=(
    "imagenet-c"
    "laion-c"
    "imagenet-a"
    "imagenet-sketch"
    "imagenet-r"
)

# Only applies to LAION-C.
LAION_SEVERITY="${LAION_SEVERITY:-3}"

ORDER_1='
  TYPE:
    - "mosaic"
    - "vertical_lines"
    - "glitched"
    - "luminance"
    - "geometric_shapes"
    - "sticker"
'

LAION_ORDER_NAME="${LAION_ORDER_NAME:-ORDER_1}"

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
echo "Starting multi-GPU closed-set evaluation at $(date)"
echo "Assignments: ${GPU_METHOD_GROUPS[*]}"
echo "========================================================"

cd imagenet || { echo "Failed to enter the imagenet directory."; exit 1; }
mkdir -p "$SAVE_DIR_BASE"
mkdir -p "$TEMP_CFG_DIR"

write_laion_temp_config() {
    local config_file="$1"
    local temp_config_file="$2"
    local order_contents_var_name="$LAION_ORDER_NAME"
    local order_contents="${!order_contents_var_name}"
    local new_corruption_block

    new_corruption_block=$(cat <<EOM
CORRUPTION:
  ID_BENCHMARK: 'laion_c'
  DATASET: 'laion-c'
  NUM_OOD_SAMPLES: 0
  NUM_EX: ${NUM_ID_SAMPLES}
  SEVERITY:
    - ${LAION_SEVERITY}
${order_contents}
EOM
)

    {
        awk '
            /^\s*CORRUPTION:/ { in_block=1; next; }
            in_block && /^[A-Z_]+:/ { in_block=0; }
            !in_block { print; }
        ' "$config_file"
        echo "$new_corruption_block"
    } > "$temp_config_file"
}

run_gpu_lane() {
    local gpu_id="$1"
    local method_csv="$2"
    local assigned_methods=()
    local method_raw
    local method
    local dataset
    local config_file
    local run_save_dir
    local temp_config_file
    local status

    IFS=',' read -r -a assigned_methods <<< "$method_csv"

    echo "[GPU ${gpu_id}] Starting methods: ${method_csv}"

    for dataset in "${DATASETS[@]}"
    do
        echo ""
        echo "[GPU ${gpu_id}] ########################################################"
        echo "[GPU ${gpu_id}] Current dataset: ${dataset}"
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

            if [ "$dataset" = "laion-c" ]; then
                run_save_dir="${SAVE_DIR_BASE}/${dataset}/sev${LAION_SEVERITY}/${LAION_ORDER_NAME}/${method}"
                temp_config_file="${TEMP_CFG_DIR}/temp_${dataset}_sev${LAION_SEVERITY}_${LAION_ORDER_NAME}_${method}_gpu${gpu_id}_${BASHPID}.yaml"
                mkdir -p "$run_save_dir"
                write_laion_temp_config "$config_file" "$temp_config_file"

                echo ""
                echo "[GPU ${gpu_id}] --------------------------------------------------------"
                echo "[GPU ${gpu_id}] Running closed-set evaluation: ${dataset} / ${method}"
                echo "[GPU ${gpu_id}] --------------------------------------------------------"

                if CUDA_VISIBLE_DEVICES="$gpu_id" python octta_closed_main.py \
                    --cfg "$temp_config_file" \
                    --data_dir "$DATA_DIR" \
                    RNG_SEED 1 \
                    CORRUPTION.DATASET "laion-c" \
                    CORRUPTION.ID_BENCHMARK "laion_c" \
                    CORRUPTION.NUM_OOD_SAMPLES 0 \
                    CORRUPTION.NUM_EX "$NUM_ID_SAMPLES" \
                    TEST.BATCH_SIZE "$BATCH_SIZE" \
                    SAVE_DIR "$run_save_dir"
                then
                    rm -f "$temp_config_file"
                else
                    status="$?"
                    rm -f "$temp_config_file"
                    return "$status"
                fi
            else
                run_save_dir="${SAVE_DIR_BASE}/${dataset}/${method}"
                mkdir -p "$run_save_dir"

                echo ""
                echo "[GPU ${gpu_id}] --------------------------------------------------------"
                echo "[GPU ${gpu_id}] Running closed-set evaluation: ${dataset} / ${method}"
                echo "[GPU ${gpu_id}] --------------------------------------------------------"

                CUDA_VISIBLE_DEVICES="$gpu_id" python octta_closed_main.py \
                    --cfg "$config_file" \
                    --data_dir "$DATA_DIR" \
                    RNG_SEED 1 \
                    CORRUPTION.DATASET "$dataset" \
                    CORRUPTION.ID_BENCHMARK "imagenet_c" \
                    CORRUPTION.NUM_OOD_SAMPLES 0 \
                    CORRUPTION.NUM_EX "$NUM_ID_SAMPLES" \
                    TEST.BATCH_SIZE "$BATCH_SIZE" \
                    SAVE_DIR "$run_save_dir"
            fi

            echo "[GPU ${gpu_id}] Done: ${dataset} / ${method}"
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
    echo "All multi-GPU closed-set evaluation tasks completed at $(date)"
else
    echo "One or more multi-GPU closed-set evaluation tasks failed at $(date)"
fi
echo "========================================================"

exit "$exit_code"
