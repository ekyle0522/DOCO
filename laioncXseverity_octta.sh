#!/bin/bash

set -e
trap 'echo "Error at line $LINENO: $BASH_COMMAND"; exit 1' ERR

export HF_HUB_OFFLINE=1

CONDA_ENV_NAME="doco"
CUDA_DEVICE_ID="3"

DATA_DIR="${DATA_ROOT:-/mnt/d/stamp_lib/datasets}"
SAVE_DIR_BASE="../output_laionc"
TEMP_CFG_DIR="./cfgs/__temp_laion_runtime"

NUM_ID_SAMPLES=5000
NUM_OOD_SAMPLES=5000
BATCH_SIZE=64

SEVERITIES=(
    1
    3
)

METHODS=(
    "source"
    "eatacome"
    "eata"
    "dpcore"
    "doco"
    "tent"
    "eataunient"
    "sar"
    "sarcome"
    "cotta"
    "ostta"
    "vida"
    "stamp"
)

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
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_ID"

echo "Starting LAION-C experiments at $(date)"

cd imagenet || { echo "Failed to enter the imagenet directory."; exit 1; }

mkdir -p "$SAVE_DIR_BASE"
mkdir -p "$TEMP_CFG_DIR"

for severity in "${SEVERITIES[@]}"
do
    for method in "${METHODS[@]}"
    do
        BASE_CONFIG_FILE="./cfgs/ood_${method}.yaml"

        if [ ! -f "$BASE_CONFIG_FILE" ]; then
            echo "Config file not found: $BASE_CONFIG_FILE. Skipping method $method."
            continue
        fi

        for OOD_DATASET in "${OOD_DATASETS[@]}"
        do
            RUN_SAVE_DIR="${SAVE_DIR_BASE}/sev${severity}/${method}/${OOD_DATASET}"
            TEMP_CONFIG_FILE="${TEMP_CFG_DIR}/temp_sev${severity}_${method}_${OOD_DATASET}.yaml"

            mkdir -p "$RUN_SAVE_DIR"

            echo "Running: severity=${severity}, method=${method}, ood_dataset=${OOD_DATASET}"

            new_corruption_block=$(cat <<EOM
CORRUPTION:
  ID_BENCHMARK: 'laion_c'
  DATASET: 'imagenet'
  OOD_DATASET: '${OOD_DATASET}'
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
                ' "$BASE_CONFIG_FILE"
                echo "$new_corruption_block"
            } > "$TEMP_CONFIG_FILE"

            python octta_main.py \
                --cfg "$TEMP_CONFIG_FILE" \
                --data_dir "$DATA_DIR" \
                RNG_SEED 1 \
                CORRUPTION.DATASET "imagenet" \
                CORRUPTION.ID_BENCHMARK "laion_c" \
                CORRUPTION.OOD_DATASET "$OOD_DATASET" \
                CORRUPTION.NUM_OOD_SAMPLES "$NUM_OOD_SAMPLES" \
                CORRUPTION.NUM_EX "$NUM_ID_SAMPLES" \
                TEST.BATCH_SIZE "$BATCH_SIZE" \
                OPTIM.DOCO_BETA 0.0 \
                OPTIM.DOCO_PROMPT_NUM 9 \
                SAVE_DIR "$RUN_SAVE_DIR"

            rm -f "$TEMP_CONFIG_FILE"
            echo "Done: severity=${severity}, method=${method}, ood_dataset=${OOD_DATASET}"
        done
    done
done

echo "All LAION-C experiments finished at $(date)"
