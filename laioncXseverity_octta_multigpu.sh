#!/bin/bash

set -Eeuo pipefail
trap 'echo "Error at line $LINENO: $BASH_COMMAND"; exit 1' ERR

export HF_HUB_OFFLINE=1

CONDA_ENV_NAME="${CONDA_ENV_NAME:-doco}"
DATA_DIR="${DATA_ROOT:-/mnt/d/stamp_lib/datasets}"
SAVE_DIR_BASE="${SAVE_DIR_BASE:-../output_laionc}"
TEMP_CFG_DIR="${TEMP_CFG_DIR:-./cfgs/__temp_laion_runtime}"

NUM_ID_SAMPLES="${NUM_ID_SAMPLES:-5000}"
NUM_OOD_SAMPLES="${NUM_OOD_SAMPLES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

SEVERITIES=(
    1
    3
)

# Edit this list, or override from the command line:
#   bash laioncXseverity_octta_multigpu.sh 1:doco,dpcore 2:eata,source 7:vida
GPU_METHOD_GROUPS=(
    "0:doco"
    "1:dpcore"
    "2:eata,eatacome"
    "3:eataunient"
    "4:ostta"
    "5:cotta"
    "7:source,tent"
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
    "ninco_ood_classes"
    "ssb-hard"
)

ORDER_CONTENTS='
  TYPE:
  - "mosaic"
  - "vertical_lines"
  - "glitched"
  - "luminance"
  - "geometric_shapes"
  - "sticker"
'

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda initialization script not found."
    exit 1
fi

conda activate "$CONDA_ENV_NAME"

echo "========================================================"
echo "Starting multi-GPU LAION-C severity suite at $(date)"
echo "Assignments: ${GPU_METHOD_GROUPS[*]}"
echo "========================================================"

cd imagenet || { echo "Failed to enter the imagenet directory."; exit 1; }

mkdir -p "$SAVE_DIR_BASE"
mkdir -p "$TEMP_CFG_DIR"

write_temp_config() {
    local base_config_file="$1"
    local temp_config_file="$2"
    local severity="$3"
    local ood_dataset="$4"
    local new_corruption_block

    new_corruption_block=$(cat <<EOM
CORRUPTION:
  ID_BENCHMARK: 'laion_c'
  DATASET: 'imagenet'
  OOD_DATASET: '${ood_dataset}'
  NUM_OOD_SAMPLES: ${NUM_OOD_SAMPLES}
  NUM_EX: ${NUM_ID_SAMPLES}
  SEVERITY:
    - ${severity}
${ORDER_CONTENTS}
EOM
)

    {
        awk '
            /^\s*CORRUPTION:/ { in_block=1; next; }
            in_block && /^[A-Z_]+:/ { in_block=0; }
            !in_block { print; }
        ' "$base_config_file"
        echo "$new_corruption_block"
    } > "$temp_config_file"
}

run_gpu_lane() {
    local gpu_id="$1"
    local method_csv="$2"
    local assigned_methods=()
    local method_raw
    local method
    local severity
    local ood_dataset
    local base_config_file
    local run_save_dir
    local temp_config_file
    local status

    IFS=',' read -r -a assigned_methods <<< "$method_csv"

    echo "[GPU ${gpu_id}] Starting methods: ${method_csv}"

    for severity in "${SEVERITIES[@]}"
    do
        echo ""
        echo "[GPU ${gpu_id}] ########################################################"
        echo "[GPU ${gpu_id}] Current severity: ${severity}"
        echo "[GPU ${gpu_id}] ########################################################"

        for method_raw in "${assigned_methods[@]}"
        do
            method="${method_raw//[[:space:]]/}"
            if [ -z "$method" ]; then
                continue
            fi

            base_config_file="./cfgs/ood_${method}.yaml"
            if [ ! -f "$base_config_file" ]; then
                echo "[GPU ${gpu_id}] Warning: Config file $base_config_file not found. Skipping method $method."
                continue
            fi

            for ood_dataset in "${OOD_DATASETS[@]}"
            do
                run_save_dir="${SAVE_DIR_BASE}/sev${severity}/${method}/${ood_dataset}"
                temp_config_file="${TEMP_CFG_DIR}/temp_sev${severity}_${method}_${ood_dataset}_gpu${gpu_id}_${BASHPID}.yaml"

                mkdir -p "$run_save_dir"
                write_temp_config "$base_config_file" "$temp_config_file" "$severity" "$ood_dataset"

                echo ""
                echo "[GPU ${gpu_id}] --------------------------------------------------------"
                echo "[GPU ${gpu_id}] Running: severity=${severity} / method=${method} / ood_dataset=${ood_dataset}"
                echo "[GPU ${gpu_id}] --------------------------------------------------------"

                if CUDA_VISIBLE_DEVICES="$gpu_id" python octta_main.py \
                    --cfg "$temp_config_file" \
                    --data_dir "$DATA_DIR" \
                    RNG_SEED 1 \
                    CORRUPTION.DATASET "imagenet" \
                    CORRUPTION.ID_BENCHMARK "laion_c" \
                    CORRUPTION.OOD_DATASET "$ood_dataset" \
                    CORRUPTION.NUM_OOD_SAMPLES "$NUM_OOD_SAMPLES" \
                    CORRUPTION.NUM_EX "$NUM_ID_SAMPLES" \
                    TEST.BATCH_SIZE "$BATCH_SIZE" \
                    OPTIM.DOCO_BETA 0.0 \
                    OPTIM.DOCO_PROMPT_NUM 9 \
                    SAVE_DIR "$run_save_dir"
                then
                    rm -f "$temp_config_file"
                else
                    status="$?"
                    rm -f "$temp_config_file"
                    return "$status"
                fi

                echo "[GPU ${gpu_id}] Done: severity=${severity} / method=${method} / ood_dataset=${ood_dataset}"
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
    echo "All multi-GPU LAION-C severity tasks completed at $(date)"
else
    echo "One or more multi-GPU LAION-C severity tasks failed at $(date)"
fi
echo "========================================================"

exit "$exit_code"
