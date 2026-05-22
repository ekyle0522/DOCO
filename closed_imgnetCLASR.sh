#!/bin/bash

set -e
trap 'echo "Error at line $LINENO: $BASH_COMMAND"; exit 1' ERR

export HF_HUB_OFFLINE=1

CONDA_ENV_NAME="doco"
CUDA_DEVICE_ID="9"
DATA_DIR="${DATA_ROOT:-/mnt/d/stamp_lib/datasets}"
SAVE_DIR_BASE="../output_closed_benchmarks"
TEMP_CFG_DIR="./cfgs/__temp_closed_runtime"


NUM_ID_SAMPLES=5000
BATCH_SIZE=64

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

DATASETS=(
    "imagenet-c"
    "laion-c"
    "imagenet-a"
    "imagenet-sketch"
    "imagenet-r"
)

# Only applies to LAION-C
LAION_SEVERITY=3

ORDER_1='
  TYPE:
    - "mosaic"
    - "vertical_lines"
    - "glitched"
    - "luminance"
    - "geometric_shapes"
    - "sticker"
'

LAION_ORDER_NAME="ORDER_1"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found."
    exit 1
fi

conda activate "$CONDA_ENV_NAME"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID

cd imagenet || { echo "Failed to enter the imagenet directory."; exit 1; }
mkdir -p "$SAVE_DIR_BASE"
mkdir -p "$TEMP_CFG_DIR"

echo "========================================================"
echo "Starting closed-set evaluation"
echo "Start time: $(date)"
echo "========================================================"

for dataset in "${DATASETS[@]}"
do
    for method in "${METHODS[@]}"
    do
        CONFIG_FILE="./cfgs/ood_${method}.yaml"
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "Warning: Config file $CONFIG_FILE not found. Skipping method $method."
            continue
        fi

        if [ "$dataset" = "laion-c" ]; then
            order_contents_var_name="$LAION_ORDER_NAME"
            order_contents="${!order_contents_var_name}"
            RUN_SAVE_DIR="${SAVE_DIR_BASE}/${dataset}/sev${LAION_SEVERITY}/${LAION_ORDER_NAME}/${method}"
            TEMP_CONFIG_FILE="${TEMP_CFG_DIR}/temp_${dataset}_sev${LAION_SEVERITY}_${LAION_ORDER_NAME}_${method}.yaml"
            mkdir -p "$RUN_SAVE_DIR"

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
                ' "$CONFIG_FILE"
                echo "$new_corruption_block"
            } > "$TEMP_CONFIG_FILE"

            echo "Running closed-set evaluation: $dataset / $method"

            python octta_closed_main.py \
                --cfg "$TEMP_CONFIG_FILE" \
                --data_dir "$DATA_DIR" \
                RNG_SEED 1 \
                CORRUPTION.DATASET "laion-c" \
                CORRUPTION.ID_BENCHMARK "laion_c" \
                CORRUPTION.NUM_OOD_SAMPLES 0 \
                CORRUPTION.NUM_EX "$NUM_ID_SAMPLES" \
                TEST.BATCH_SIZE "$BATCH_SIZE" \
                SAVE_DIR "$RUN_SAVE_DIR"

            rm -f "$TEMP_CONFIG_FILE"
        else
            RUN_SAVE_DIR="${SAVE_DIR_BASE}/${dataset}/${method}"
            mkdir -p "$RUN_SAVE_DIR"

            echo "Running closed-set evaluation: $dataset / $method"

            python octta_closed_main.py \
                --cfg "$CONFIG_FILE" \
                --data_dir "$DATA_DIR" \
                RNG_SEED 1 \
                CORRUPTION.DATASET "$dataset" \
                CORRUPTION.ID_BENCHMARK "imagenet_c" \
                CORRUPTION.NUM_OOD_SAMPLES 0 \
                CORRUPTION.NUM_EX "$NUM_ID_SAMPLES" \
                TEST.BATCH_SIZE "$BATCH_SIZE" \
                SAVE_DIR "$RUN_SAVE_DIR"
        fi

        echo "Done: $dataset / $method"
    done
done

echo ""
echo "========================================================"
echo "All closed-set evaluation tasks completed at $(date)"
echo "========================================================"
