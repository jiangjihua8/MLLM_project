#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU (Ascend) training script
# Qwen3-VL-8B LLM (auto-extract) + DINOv3 + DeepStack
# Saves the best checkpoint by eval_loss.
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd "$SCRIPT_DIR"
echo "Script path: $SCRIPT_PATH"
echo "Script folder path: $SCRIPT_DIR"
echo "Current working path: $PWD"

# ====================== NPU environment ======================
export ASCEND_CUSTOM_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp

workerID=$(echo "$HOSTNAME" | awk -F'-' '{print $(NF-1)"-"$NF}')
echo "${workerID}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
sudo chmod -R 777 /usr/local/Ascend/ascend-toolkit/
source /usr/local/Ascend/nnal/atb/set_env.sh

# ====================== moxing upgrade ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> changing moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
USE_MEMARTS=0 python -c "import moxing; moxing.file.copy('obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
pip uninstall moxing-framework -y
pip cache purge
pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
export MOX_PROFILE=1
export MOX_RECORD_OBS=1
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>> moxing change finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== dependencies ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Installing dependencies >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

pip install torch==2.7.1
pip install torch_npu==2.7.1rc1

python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
pip install --force-reinstall /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl

pip install sentencepiece
pip install tiktoken
pip install "transformers>=4.51.0"
pip install "tokenizers>=0.21"
pip install accelerate==1.6.0
pip install deepspeed==0.14.4
pip install safetensors
pip install packaging
pip install Pillow
pip install torchvision==0.22.1
pip install shortuuid
pip install peft
pip install pydantic
pip install 'markdown2[all]'
pip install 'numpy>=1.26'
pip install 'scikit-learn>=1.2'
pip install 'gradio>=5.0'
pip install requests
pip install uvicorn
pip install fastapi
pip install 'einops>=0.6'
pip install 'einops-exts>=0.0.4'
pip install 'timm>=0.9.0'
pip install "huggingface-hub>=0.25.1" --force-reinstall
pip install urllib3==1.26.15

echo "========== key deps =========="
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_npu; print('torch_npu', torch_npu.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
echo "==============================="

pip list
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Dependencies installed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== distributed parameters ======================
if [[ -z "${MA_VJ_NAME}" ]]; then
    NNODES=1
    NODE_RANK=0
    NPROC_PER_NODE=8
    MASTER_ADDR=localhost
else
    NNODES="$MA_NUM_HOSTS"
    NODE_RANK="$VC_TASK_INDEX"
    NPROC_PER_NODE="$MA_NUM_GPUS"
    MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
fi

MASTER_PORT="${MASTER_PORT:-6060}"
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID="${RDZV_ID:-1234}"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> machine information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== HCCL & NPU settings ======================
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export INF_NAN_MODE_ENABLE=1
export HCCL_ASYNC_ERROR_HANDLING=0
export WITHOUT_JIT_COMPILE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE
export COMBINED_ENABLE=1
export OMP_NUM_THREADS=1
export LLAVA_LOG_RANK0_ONLY=${LLAVA_LOG_RANK0_ONLY:-1}

# ====================== output management ======================
CLUSTER_SAVE=${OUTPUT_URL}
OSB_SHARE_PATH="$CLUSTER_SAVE"
echo "System defined obs share path: $OSB_SHARE_PATH"

LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-/cache/local_model_save_path}
mkdir -p "$LOCAL_MODEL_SAVE_PATH"

if [[ "$NODE_RANK" == 0 ]]; then
    OUTPUT_PATH=$OSB_SHARE_PATH
else
    OUTPUT_PATH=$LOCAL_MODEL_SAVE_PATH
fi

# ====================== OBS paths ======================
OBS_CACHE=${OBS_CACHE:-/cache}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/MLLM20260427_rc_jjh.zip}

DINOV3_PATH=${DINOV3_PATH:-${OBS_CACHE}/checkpoints/facebook_dinov3-vitl16-pretrain-lvd1689m}
Qwen3VL_PATH=${Qwen3VL_PATH:-${OBS_CACHE}/checkpoints/Qwen3-VL-8B-Instruct}

DATASET_PATH=${DATASET_PATH:-/cache/MLLM20260427_rc_jjh}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}

# ====================== download ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading models >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/facebook_dinov3-vitl16-pretrain-lvd1689m', '${DINOV3_PATH}')"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/Qwen3-VL-8B-Instruct', '${Qwen3VL_PATH}')"

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>> Downloading dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${OBS_CACHE}/dataset.zip')"

cd /cache
unzip -o dataset.zip
cd "$SCRIPT_DIR"

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Expected dataset directory $DATASET_PATH not found after unzip."
    ls -l /cache/
    exit 1
fi

TRAIN_PATH=${TRAIN_PATH:-${DATASET_PATH}/train.jsonl}
TEST_PATH=${TEST_PATH:-${DATASET_PATH}/test.jsonl}
DEFAULT_VAL_PATH="${DATASET_PATH}/val.jsonl"

if [ -z "${EVAL_PATH:-}" ]; then
    if [ -f "$DEFAULT_VAL_PATH" ]; then
        EVAL_PATH="$DEFAULT_VAL_PATH"
    elif [ -f "$TEST_PATH" ]; then
        EVAL_PATH="$TEST_PATH"
        echo "WARN: ${DEFAULT_VAL_PATH} not found; using TEST_PATH as EVAL_PATH for best eval_loss selection."
    else
        EVAL_PATH="$DEFAULT_VAL_PATH"
    fi
fi
EVAL_IMAGE_FOLDER=${EVAL_IMAGE_FOLDER:-${IMAGE_FOLDER}}

if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: $TRAIN_PATH not found"
    exit 1
fi
if [ ! -f "$EVAL_PATH" ]; then
    echo "ERROR: EVAL_PATH $EVAL_PATH not found"
    exit 1
fi

echo "DATASET_PATH:       $DATASET_PATH"
echo "TRAIN_PATH:         $TRAIN_PATH"
echo "EVAL_PATH:          $EVAL_PATH"
echo "IMAGE_FOLDER:       $IMAGE_FOLDER"
echo "EVAL_IMAGE_FOLDER:  $EVAL_IMAGE_FOLDER"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish moxing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# ====================== auto gradient accumulation ======================
tar_equal_batch_size=${TARGET_GLOBAL_BATCH_SIZE:-64}
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}

total_gpus=$(( NNODES * NPROC_PER_NODE ))
gas=$((tar_equal_batch_size / (total_gpus * per_device_train_batch_size) ))
if [ "$gas" -lt 1 ]; then
    gradient_accumulation_steps=1
else
    gradient_accumulation_steps=$gas
fi

echo ">>> Target global batch: ${tar_equal_batch_size}"
echo ">>> Per-device batch: ${per_device_train_batch_size}, Total GPUs: ${total_gpus}"
echo ">>> Gradient accumulation steps: ${gradient_accumulation_steps}"

# ====================== training ======================
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---------- Training params ----------
MM_VISION_SELECT_LAYER=${MM_VISION_SELECT_LAYER:--2}
MM_PROJECTOR_TYPE=${MM_PROJECTOR_TYPE:-mlp2x_gelu}
UNFREEZE_MM_VISION_TOWER=${UNFREEZE_MM_VISION_TOWER:-True}
DEEPSTACK_VISUAL_INDEXES=${DEEPSTACK_VISUAL_INDEXES:-"6 12 18 23"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}
NUM_EPOCHS=${NUM_EPOCHS:-8}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_STEPS=${WARMUP_STEPS:-50}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
EVAL_STEPS=${EVAL_STEPS:-300}
SAVE_STEPS=${SAVE_STEPS:-${EVAL_STEPS}}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAMPLE_SEED=${SAMPLE_SEED:-42}
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-False}
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-3000}
BEST_TRAIN_LOSS_DIR=${BEST_TRAIN_LOSS_DIR:-best}

if [ $((SAVE_STEPS % EVAL_STEPS)) -ne 0 ]; then
    echo "ERROR: SAVE_STEPS (${SAVE_STEPS}) must be a multiple of EVAL_STEPS (${EVAL_STEPS}) when load_best_model_at_end=True."
    exit 1
fi

EVAL_STRATEGY_ARG=$(python - << 'PY'
import inspect
from transformers import TrainingArguments
params = inspect.signature(TrainingArguments.__init__).parameters
print("--eval_strategy" if "eval_strategy" in params else "--evaluation_strategy")
PY
)

# ---------- DeepStack ----------
DEEPSTACK_ARGS=()
if [[ "${DISABLE_DEEPSTACK:-False}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
    DEEPSTACK_ARGS=(--disable_deepstack True)
    DEEPSTACK_LABEL="disabled"
    GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
elif [ -n "${DEEPSTACK_VISUAL_INDEXES}" ]; then
    DEEPSTACK_ARGS=(--deepstack_visual_indexes ${DEEPSTACK_VISUAL_INDEXES})
    DEEPSTACK_LABEL="${DEEPSTACK_VISUAL_INDEXES}"
    GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
else
    DEEPSTACK_LABEL="disabled"
    GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
fi

echo "============================================================"
echo "Model:             ${Qwen3VL_PATH} (Qwen3-VL-8B -> auto-extract LLM)"
echo "ViT:               ${DINOV3_PATH}"
echo "DeepStack:         ${DEEPSTACK_LABEL}"
echo "Grad ckpt:         ${GRADIENT_CHECKPOINTING}"
echo "DeepSpeed:         ${DEEPSPEED_CONFIG}"
echo "Eval strategy arg: ${EVAL_STRATEGY_ARG}"
echo "Eval steps:        ${EVAL_STEPS}"
echo "Save steps:        ${SAVE_STEPS}"
echo "Best metric:       eval_loss (lower is better)"
echo "Best train loss:   ${SAVE_BEST_TRAIN_LOSS}, start_step=${BEST_TRAIN_LOSS_START_STEP}, dir=${BEST_TRAIN_LOSS_DIR}"
echo "Output path:       ${OUTPUT_PATH}"
echo "============================================================"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m llava.train.train_qwen \
    --model_name_or_path "${Qwen3VL_PATH}" \
    --version conv_qwen_3_Dinov2_huawei \
    --vision_tower "${DINOV3_PATH}" \
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER}" \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --unfreeze_mm_vision_tower "${UNFREEZE_MM_VISION_TOWER}" \
    "${DEEPSTACK_ARGS[@]}" \
    --data_path "${TRAIN_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --eval_data_path "${EVAL_PATH}" \
    --eval_image_folder "${EVAL_IMAGE_FOLDER}" \
    --sample_seed "${SAMPLE_SEED}" \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "${OUTPUT_PATH}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${per_device_train_batch_size}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --learning_rate "${LR}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-True}" \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    "${EVAL_STRATEGY_ARG}" steps \
    --eval_steps "${EVAL_STEPS}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --save_best_train_loss "${SAVE_BEST_TRAIN_LOSS}" \
    --best_train_loss_start_step "${BEST_TRAIN_LOSS_START_STEP}" \
    --best_train_loss_dir "${BEST_TRAIN_LOSS_DIR}" \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to none \
    --ddp_find_unused_parameters False \
    --ddp_backend hccl \
    --deepspeed "${DEEPSPEED_CONFIG}"

echo "=== Training finished ==="
echo "Best checkpoint is recorded in: ${OUTPUT_PATH}/trainer_state.json"
if [[ "$NODE_RANK" == 0 && -f "${OUTPUT_PATH}/trainer_state.json" ]]; then
    python - "${OUTPUT_PATH}/trainer_state.json" << 'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    state = json.load(f)
print("best_model_checkpoint:", state.get("best_model_checkpoint"))
print("best_metric:", state.get("best_metric"))
PY
fi
