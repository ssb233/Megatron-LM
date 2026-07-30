#!/usr/bin/env bash
set -euo pipefail

CONFIG_ID="${1:-}"
MODE="${2:-}"
RUN_DIR="${3:-}"
ORDER_FILE="${4:-}"
DEPENDENCY_FILE="${5:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/configs.json"
PYTHON="/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python"

if ! "${PYTHON}" - "${CONFIG_FILE}" "${CONFIG_ID}" <<'PY'
import json
import sys

configs = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if sys.argv[2] in configs else 1)
PY
then
  echo "unknown configuration '${CONFIG_ID}'" >&2
  exit 2
fi

case "${MODE}" in
  CALIBRATE|1F1B) ;;
  MAGELLAN)
    if [[ -z "${ORDER_FILE}" || -z "${DEPENDENCY_FILE}" ]]; then
      echo "MAGELLAN requires order and dependency JSON" >&2
      exit 2
    fi
    if [[ ! -s "${ORDER_FILE}" || ! -s "${DEPENDENCY_FILE}" ]]; then
      echo "MAGELLAN schedule inputs must be non-empty files" >&2
      exit 2
    fi
    ;;
  *)
    echo "unknown mode '${MODE}'" >&2
    exit 2
    ;;
esac

if [[ -z "${RUN_DIR}" ]]; then
  echo "run directory is required" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="/home/songxb26/mnist/crosspipe-old/data"
DATA_PREFIX="${DATA_ROOT}/output_text_document_text_document"
VOCAB_FILE="${DATA_ROOT}/gpt2-vocab.json"
MERGE_FILE="${DATA_ROOT}/gpt2-merges.txt"

mapfile -t CONFIG_VALUES < <(
  "${PYTHON}" - "${CONFIG_FILE}" "${CONFIG_ID}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
for key in ("hidden", "ffn", "heads", "seq", "mbs", "gbs"):
    print(config[key])
print("" if config.get("experts") is None else config["experts"])
print("" if config.get("topk") is None else config["topk"])
PY
)

HIDDEN_SIZE="${CONFIG_VALUES[0]}"
FFN_HIDDEN_SIZE="${CONFIG_VALUES[1]}"
NUM_HEADS="${CONFIG_VALUES[2]}"
SEQ_LENGTH="${CONFIG_VALUES[3]}"
MICRO_BATCH_SIZE="${CONFIG_VALUES[4]}"
GLOBAL_BATCH_SIZE="${CONFIG_VALUES[5]}"
NUM_EXPERTS="${CONFIG_VALUES[6]}"
MOE_TOPK="${CONFIG_VALUES[7]}"

TRAIN_ITERS="${CDC_OVERHEAD_TRAIN_ITERS:-20}"
EXP_TEST_ITERS="${CDC_OVERHEAD_EXP_TEST_ITERS:-17}"
MASTER_PORT="${CDC_OVERHEAD_MASTER_PORT:-29630}"
SHARED_CACHE="${REPO_ROOT}/runs/magellan_comm_dependency_overhead_pp4/shared_data_cache/${CONFIG_ID}"

mkdir -p "${RUN_DIR}" "${RUN_DIR}/triton_cache" "${SHARED_CACHE}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="${RUN_DIR}/triton_cache"
unset CUSTOM_SCHEDULE_TRACE_FLUSH_EACH_EVENT

COMMON_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 4
  --num-layers-per-virtual-pipeline-stage 2
  --num-layers 8
  --hidden-size "${HIDDEN_SIZE}"
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
  --num-attention-heads "${NUM_HEADS}"
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings "${SEQ_LENGTH}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}"
  --transformer-impl local
  --normalization LayerNorm
  --position-embedding-type learned_absolute
  --tokenizer-type GPT2BPETokenizer
  --data-path "${DATA_PREFIX}"
  --vocab-file "${VOCAB_FILE}"
  --merge-file "${MERGE_FILE}"
  --data-cache-path "${SHARED_CACHE}"
  --split 949,50,1
  --fp16
  --loss-scale 1
  --lr 3e-4
  --min-lr 3e-5
  --lr-decay-style cosine
  --lr-warmup-iters 0
  --weight-decay 0.1
  --adam-beta1 .9
  --adam-beta2 .95
  --adam-eps 1e-8
  --clip-grad 1
  --init-method-std .02
  --seed 1234
  --log-interval 1
  --eval-iters 0
  --tensorboard-dir "${RUN_DIR}"
  --no-check-for-nan-in-loss-and-grad
  --no-barrier-with-level-1-timing
  --ckpt-format torch
  --distributed-timeout-minutes 2
  --head_tail_as_one_layer
  --num_subparts 1
  --no-align-grad-reduce
  --no-align-param-gather
  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
  --ddp-bucket-size 100000000000
  --cdc_profile_iter 2
  --cdc_exp_logging
  --cdc_exp_test_start_iter 3
  --cdc_exp_tf_block_size 2
  --cdc_verbose_print 1
  --num_dc 1
  --cdc_exp_per_cfg_test_iters "${EXP_TEST_ITERS}"
  --cdc_latency_bandwidth_delay_as_F_stage 0,0
)

if [[ -n "${NUM_EXPERTS}" ]]; then
  COMMON_ARGS+=(
    --num-experts "${NUM_EXPERTS}"
    --moe-router-topk "${MOE_TOPK}"
    --expert-model-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.01
    --disable-bias-linear
  )
fi

SCHEDULE_ARGS=()
if [[ "${MODE}" == "MAGELLAN" ]]; then
  SCHEDULE_ARGS+=(
    --custom-pipeline-schedule "${ORDER_FILE}"
    --custom-comm-dependency "${DEPENDENCY_FILE}"
  )
else
  SCHEDULE_ARGS+=(--enable_cdcpp_scheduler --static_schedule 1F1B)
fi

COMMAND=(
  "${PYTHON}" -m torch.distributed.run
  --standalone
  --nproc_per_node=4
  --master_port="${MASTER_PORT}"
  pretrain_gpt.py
  "${COMMON_ARGS[@]}"
  "${SCHEDULE_ARGS[@]}"
)

{
  echo "config_id=${CONFIG_ID}"
  echo "mode=${MODE}"
  echo "run_dir=${RUN_DIR}"
  echo "order_file=${ORDER_FILE}"
  echo "dependency_file=${DEPENDENCY_FILE}"
  echo "num_dc=1"
  echo "delay_pairs=0,0"
  echo "train_iters=${TRAIN_ITERS}"
  echo "measured_iterations=6-19"
  echo "exp_test_iters=${EXP_TEST_ITERS}"
  echo "python=${PYTHON}"
  printf "command="
  printf "%q " "${COMMAND[@]}"
  printf "\n"
} | tee "${RUN_DIR}/run.info"

cd "${REPO_ROOT}"
"${COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
echo "completed config=${CONFIG_ID} mode=${MODE}" | tee "${RUN_DIR}/completed.txt"
