# 1. 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU 版 PyTorch 路径
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch 安装路径
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch 路径
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # 添加 NPU 库路径
export HCCL_CONNECT_TIMEOUT=7200
# 2. 加载环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=INFO
export ASDOPS_LOG_TO_FILE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export ATB_ACLNN_CACHE_GLOABL_COUNT=0
export INF_NAN_MODE_FORCE_DISABLE=1

# 3. 清理旧日志
\rm -rf core.*

# 4. 启动分布式服务
export ASCEND_RT_VISIBLE_DEVICES=8
MODEL_PATH="/export/home/models/Qwen3-14B"
MASTER_NODE_ADDR="127.0.0.1:19310"                  # Master 节点地址（需全局一致）

START_PORT=18119                                  # 服务起始端口
START_DEVICE=0                                     # 起始 NPU 逻辑设备号
NNODES=1                                         # 单卡单进程
PHY_ID=8                                          # 物理 NPU 卡号（与 ASCEND_RT_VISIBLE_DEVICES 一致）
TRANSFER_START_PORT=25550
export HCCL_IF_BASE_PORT=43595  # HCCL 通信基础端口
LOG_DIR="${LOG_DIR:-./eval_log_slo}"
OVERHEAD_DIR="${OVERHEAD_DIR:-.}"                  # map/unmap profiling 输出目录
ENABLE_FORWARD_ADMISSION="${ENABLE_FORWARD_ADMISSION:-false}"
mkdir -p "$LOG_DIR"
for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  TRANSFER_PORT=$((TRANSFER_START_PORT + i))
  export NPU_PHY_ID=$PHY_ID
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --npu_phy_id=$PHY_ID \
    --transfer_listen_port=$TRANSFER_PORT \
    --max_memory_utilization=0.95 \
    --max_tokens_per_batch=100000 \
    --max_seqs_per_batch=32 \
    --max_concurrent_requests=1000 \
    --enable_mla=false \
    --enable_shm=false \
    --block_size=128 \
    --dp_size=1 \
    --xtensor_master_node_addr=127.0.0.1:19886 \
    --enable_xtensor=true \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --priority_level=2 \
    --enable_watermark_degrade_restore_mvp=true \
    --priority_ttft_slo_ms=10000 \
    --priority_tpot_slo_ms=80 \
    --layer_offload_low_watermark_ratio=0 \
    --load_model_slo_violation_rate=100 \
    --layer_offload_high_watermark_ratio=0.25 \
    --priority_window_size=5000 \
    --kv_prealloc_min_free_ratio=0.10 \
    --num_threads=9 \
    --enable_prism=false \
    --enable_map_unmap_profiling=true \
    --map_unmap_profiling_dir=$OVERHEAD_DIR \
    --map_unmap_profiling_flush_ms=500 \
    --node_rank=$i >>$LOG_FILE 2>&1 &
done
 #   --enable_activation_pooling=true \
 #   --host_blocks_factor=1.1 \
 #   --max_global_ttft_ms=1000 \
 #   --max_global_tpot_ms=50 \
 #   --enable_latency_aware_schedule=true \
 #   --enable_profile_step_time=true \
 #   --profile_max_prompt_length=2048 \