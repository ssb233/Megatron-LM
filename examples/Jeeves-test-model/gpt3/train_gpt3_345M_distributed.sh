#!/bin/bash

# Runs the "345M" parameter model

# 限制每个GPU的最大并发连接数，避免TP时资源竞争
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2 #每个节点GPU数
# Change for multinode config
MASTER_ADDR=localhost   #主节点地址，用于多节点通信
MASTER_PORT=6000
NUM_NODES=1 #节点数
NODE_RANK=0 #当前节点序号
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES)) #总进程数,总GPU数=节点数*每节点GPU数

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$5 #<Specify path and file prefix>_text_document

# 运行目录为根目录，参数样例如下
# ./example/Jeeves-test-model/ckpt/  ./example/Jeeves-test-model/tensorBoardLog/  ./myData/gpt2-vocab.json ./myData/gpt2-merges.txt ./myData/my-gpt2_text_document

# torchrun的启动参数
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)
# 模型结构参数，修改这个来修改模型大小
GPT_MODEL_ARGS=(
    --num-layers 12 
    --hidden-size 512 
    --num-attention-heads 8 
    --seq-length 1024 
    --max-position-embeddings 2048 
    --attention-backend auto # Can use (flash/fused/unfused/local)，注意力实现方式
)
# 训练参数,减少iter加速训练，重点在trace分析
TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 128 # batch size = micro * dp * num_micro_batches, 2048
    # --rampup-batch-size 4 4 5000 # 逐步增加batch size，初值，步长，总步数, 指定iter情况下不能设置此参数
    --train-iters 10 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 10
)
# 模型并行参数
MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 90,9,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 5
    --save-interval 5 
    --eval-interval 1
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 5
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# 添加perfetto的profiler配置参数，实现trace生成
PERFETTO_PROFILE_ARGS=(
    --profile-log-ranks 0 1  #在哪些rank上生成trace
)

# 添加cross dc 配置参数
CROSS_DC_ARGS=(
    --use-cross-dc True #true，启用cross dc
    --cross-dc-delay 20.0
    --dc-size 1
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PERFETTO_PROFILE_ARGS[@]} \
    ${CROSS_DC_ARGS[@]}


# bash ./examples/Jeeves-test-model/gpt3/train_gpt3_345M_distributed.sh ./example/Jeeves-test-model/ckpt/  ./example/Jeeves-test-model/tensorBoardLog/  ./myData/gpt2-vocab.json ./myData/gpt2-merges.txt ./myData/my-gpt2_text_document