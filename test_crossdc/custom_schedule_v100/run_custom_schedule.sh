#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 <A|B|C|C_TRACE> [run-root]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="/home/songxb26/mnist/.venvs/crosspipi-magellan/bin/python"
DATA_ROOT="/home/songxb26/mnist/crosspipe-old/data"
DATA_PREFIX="${DATA_ROOT}/output_text_document_text_document"
VOCAB_FILE="${DATA_ROOT}/gpt2-vocab.json"
MERGE_FILE="${DATA_ROOT}/gpt2-merges.txt"
ORDER_FILE="${REPO_ROOT}/tests/unit_tests/pipeline_parallel/fixtures/replay_order_pp4_n8_star.json"
DEPENDENCY_FILE="${REPO_ROOT}/tests/unit_tests/pipeline_parallel/fixtures/notification_deps_pp4_n8_star.json"
DEFAULT_RUN_ROOT="${REPO_ROOT}/runs/custom_schedule_v100/$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${2:-${CUSTOM_SCHEDULE_RUN_ROOT:-${DEFAULT_RUN_ROOT}}}"

case "${MODE}" in
  A)
    RUN_NAME="A_default_1f1b"
    MASTER_PORT=29610
    ;;
  B)
    RUN_NAME="B_custom_order"
    MASTER_PORT=29611
    ;;
  C)
    RUN_NAME="C_custom_order_dependency"
    MASTER_PORT=29612
    ;;
  C_TRACE)
    RUN_NAME="C_trace"
    MASTER_PORT=29613
    ;;
  *)
    echo "Unknown mode '${MODE}'; expected A, B, C, or C_TRACE." >&2
    exit 2
    ;;
esac

RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/data_cache" "${RUN_DIR}/triton_cache"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="${RUN_DIR}/triton_cache"

COMMON_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 4
  --num-layers-per-virtual-pipeline-stage 2
  --num-layers 8
  --hidden-size 512
  --ffn-hidden-size 2048
  --num-attention-heads 8
  --seq-length 256
  --max-position-embeddings 256
  --micro-batch-size 1
  --global-batch-size 8
  --train-iters 20
  --transformer-impl local
  --normalization LayerNorm
  --position-embedding-type learned_absolute
  --tokenizer-type GPT2BPETokenizer
  --data-path "${DATA_PREFIX}"
  --vocab-file "${VOCAB_FILE}"
  --merge-file "${MERGE_FILE}"
  --data-cache-path "${RUN_DIR}/data_cache"
  --split 949,50,1
  --fp16
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
  --distributed-timeout-minutes 1
  --head_tail_as_one_layer
  --num_subparts 1
  --no-align-grad-reduce
  --no-align-param-gather
  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
  --ddp-bucket-size 100000000000
  --num_dc 1
  --cdc_profile_iter 2
  --cdc_exp_logging
  --cdc_exp_test_start_iter 3
  --cdc_exp_per_cfg_test_iters 8
  --cdc_exp_tf_block_size 2
  --cdc_exp_dump_execution_plan
  --cdc_latency_bandwidth_delay_as_F_stage 0,0
  --cdc_verbose_print 1
)

SCHEDULE_ARGS=()
case "${MODE}" in
  A)
    SCHEDULE_ARGS+=(--enable_cdcpp_scheduler --static_schedule 1F1B)
    ;;
  B)
    SCHEDULE_ARGS+=(--custom-pipeline-schedule "${ORDER_FILE}")
    ;;
  C)
    SCHEDULE_ARGS+=(
      --custom-pipeline-schedule "${ORDER_FILE}"
      --custom-comm-dependency "${DEPENDENCY_FILE}"
    )
    ;;
  C_TRACE)
    SCHEDULE_ARGS+=(
      --custom-pipeline-schedule "${ORDER_FILE}"
      --custom-comm-dependency "${DEPENDENCY_FILE}"
      --custom-schedule-trace-dir "${RUN_DIR}/trace"
    )
    ;;
esac

{
  echo "mode=${MODE}"
  echo "run_dir=${RUN_DIR}"
  echo "order_file=${ORDER_FILE}"
  echo "dependency_file=${DEPENDENCY_FILE}"
  echo "command_python=${PYTHON}"
} | tee "${RUN_DIR}/run.info"

cd "${REPO_ROOT}"
"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  pretrain_gpt.py \
  "${COMMON_ARGS[@]}" \
  "${SCHEDULE_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "Completed ${MODE}: ${RUN_DIR}"
