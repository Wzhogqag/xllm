#!/usr/bin/env bash
# map/unmap overhead sweep for single-card Qwen3-14B on the forward-mapping
# (feat/final_multi_model) build. For each QPS point it restarts a fresh xllm
# server with --enable_map_unmap_profiling, replays trace_1_100.csv, and
# collects map/unmap aggregates + server/client TTFT/TPOT into a timestamped
# subdirectory under overhead/.
#
# Usage:
#   bash overhead/run_overhead.sh                # sweep default QPS points
#   QPS_LIST="2 2.5" bash overhead/run_overhead.sh
#   PHY_ID=6 bash overhead/run_overhead.sh       # pin a known-good card
#
# Detach from Claude's process group so it survives the session:
#   setsid nohup bash overhead/run_overhead.sh > overhead/run_overhead.out 2>&1 &

set -u

# ------------------------- config -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO" || exit 1

MODEL_PATH="/export/home/models/Qwen3-14B"
MODEL_NAME="Qwen3-14B"
XLLM_BIN="./build/xllm/core/server/xllm"
SENDER="$REPO/overhead/sender.py"
TRACE_CSV="$REPO/trace_1_100.csv"

QPS_LIST="${QPS_LIST:-1 1.5 2 2.5 3}"
LIMIT="${LIMIT:-1000}"
INPUT_SCALE="${INPUT_SCALE:-1}"
MAX_TOKENS_SCALE="${MAX_TOKENS_SCALE:-20}"
MAX_CONTEXT_CAP="${MAX_CONTEXT_CAP:-8192}"

TTFT_SLO_MS="${TTFT_SLO_MS:-10000}"   # target: TTFT <= 10s
TPOT_SLO_MS="${TPOT_SLO_MS:-80}"      # target: TPOT <= 80ms

PORT="${PORT:-13212}"
MASTER_NODE_ADDR="127.0.0.1:9929"
TRANSFER_PORT="${TRANSFER_PORT:-25550}"
XTENSOR_MASTER_ADDR="127.0.0.1:15534"

HBM_FREE_THRESHOLD_MB="${HBM_FREE_THRESHOLD_MB:-6000}"  # card considered free below this
NPU_WAIT_TIMEOUT_S="${NPU_WAIT_TIMEOUT_S:-180}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
DRAIN_EXTRA_S="${DRAIN_EXTRA_S:-90}"   # extra observation after sender finishes

TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="$REPO/overhead/sweep_${TS}"
mkdir -p "$SWEEP_DIR"
SWEEP_SUMMARY="$SWEEP_DIR/sweep_summary.md"

# ------------------------- env (mirrors start.sh) -------------------------
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=7200
# Ascend env scripts reference unbound vars (e.g. ZSH_VERSION); disable nounset
# around them so `set -u` does not abort the sweep on source.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
set -u
export ASDOPS_LOG_TO_STDOUT=0
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_FILE=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export INF_NAN_MODE_ENABLE=0
export ATB_ACLNN_CACHE_GLOABL_COUNT=0
export INF_NAN_MODE_FORCE_DISABLE=1
export MC_MS_AUTO_DISC=0
export HCCL_IF_BASE_PORT=45532

log() { echo "[run_overhead $(date +%H:%M:%S)] $*"; }

# hbm used (MB) for a physical NPU id, parsed from `npu-smi info`.
# The per-chip row is identified by its PCI bus address in column 3, e.g.:
#   | 0     6         | 0000:91:00.0 | 0    0 / 0    3164 / 65536 |
# The HBM usage is the LAST "used / total" pair in the 4th column (total is the
# big HBM number, e.g. 65536MB), NOT the first pair (which is AICore util 0/0).
# Prints exactly one integer (the first matching chip row), or -1 if not found.
npu_hbm_used_mb() {
  local phy="$1"
  npu-smi info 2>/dev/null | awk -v id="$phy" '
    $0 ~ /[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]/ {
      n = split($0, a, "|")
      split(a[2], c, " ")
      if (c[2] == id) {
        s = a[4]; used = -1
        while (match(s, /([0-9]+)[ ]*\/[ ]*([0-9]+)/, m)) {
          used = m[1]                      # last pair wins -> HBM used
          s = substr(s, RSTART + RLENGTH)
        }
        print used; exit
      }
    }
    END { if (!seen) { } }'
  return 0
}

# pick a physical NPU id that stays free across 3 samples
pick_free_npu() {
  if [[ -n "${PHY_ID:-}" ]]; then
    echo "$PHY_ID"; return 0
  fi
  local candidates=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
  for phy in "${candidates[@]}"; do
    local ok=1
    for s in 1 2 3; do
      local used; used=$(npu_hbm_used_mb "$phy")
      [[ "$used" =~ ^[0-9]+$ ]] || used=-1
      if [[ "$used" -lt 0 || "$used" -ge "$HBM_FREE_THRESHOLD_MB" ]]; then ok=0; break; fi
      sleep 1
    done
    if [[ "$ok" -eq 1 ]]; then echo "$phy"; return 0; fi
  done
  echo "-1"; return 1
}

wait_npu_free() {
  local phy="$1"; local t=0
  while [[ "$t" -lt "$NPU_WAIT_TIMEOUT_S" ]]; do
    local used; used=$(npu_hbm_used_mb "$phy")
    [[ "$used" =~ ^[0-9]+$ ]] || used=-1
    if [[ "$used" -ge 0 && "$used" -lt "$HBM_FREE_THRESHOLD_MB" ]]; then return 0; fi
    sleep 3; t=$((t+3))
  done
  log "WARN: NPU $phy still not free after ${NPU_WAIT_TIMEOUT_S}s (used=$(npu_hbm_used_mb "$phy")MB)"
  return 1
}

kill_xllm() {
  pkill -9 xllm 2>/dev/null
}

# --------------------- run one QPS point ---------------------
run_one_qps() {
  local qps="$1"; local phy="$2"
  local rundir="$SWEEP_DIR/${TS}_qps${qps}"
  mkdir -p "$rundir"
  local node_log="$rundir/node_0.log"

  log "QPS=$qps -> $rundir (phy=$phy)"

  export ASCEND_RT_VISIBLE_DEVICES="$phy"
  export NPU_PHY_ID="$phy"

  \rm -rf core.* 2>/dev/null

  # snapshot the exact launch command
  cat > "$rundir/start.sh.snapshot" <<EOF
# QPS=$qps phy=$phy model=$MODEL_PATH ts=$TS
ASCEND_RT_VISIBLE_DEVICES=$phy NPU_PHY_ID=$phy \\
$XLLM_BIN --model $MODEL_PATH --devices=npu:0 --port $PORT \\
  --master_node_addr=$MASTER_NODE_ADDR --nnodes=1 \\
  --max_memory_utilization=0.86 --max_tokens_per_batch=16184 --max_seqs_per_batch=256 \\
  --enable_mla=false --block_size=128 --dp_size=1 --enable_xtensor=true \\
  --npu_phy_id=$phy --transfer_listen_port=$TRANSFER_PORT --communication_backend=hccl \\
  --enable_prefix_cache=true --enable_chunked_prefill=true --enable_schedule_overlap=false \\
  --xtensor_master_node_addr=$XTENSOR_MASTER_ADDR --priority_level=3 --priority_window_size=1000 \\
  --host_blocks_factor=1 --kv_cache_transfer_mode=PULL --node_rank=0 \\
  --enable_map_unmap_profiling=true --map_unmap_profiling_dir=$rundir --map_unmap_profiling_flush_ms=500
EOF

  "$XLLM_BIN" \
    --model "$MODEL_PATH" \
    --devices="npu:0" \
    --port "$PORT" \
    --master_node_addr="$MASTER_NODE_ADDR" \
    --nnodes=1 \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=16184 \
    --max_seqs_per_batch=256 \
    --enable_mla=false \
    --block_size=128 \
    --dp_size=1 \
    --enable_xtensor=true \
    --npu_phy_id="$phy" \
    --transfer_listen_port="$TRANSFER_PORT" \
    --communication_backend="hccl" \
    --enable_prefix_cache=true \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=false \
    --xtensor_master_node_addr="$XTENSOR_MASTER_ADDR" \
    --priority_level=3 \
    --priority_window_size=1000 \
    --host_blocks_factor=1 \
    --kv_cache_transfer_mode=PULL \
    --node_rank=0 \
    --enable_map_unmap_profiling=true \
    --map_unmap_profiling_dir="$rundir" \
    --map_unmap_profiling_flush_ms=500 \
    > "$node_log" 2>&1 &
  local server_pid=$!

  # wait for readiness
  local t=0; local ready=0
  while [[ "$t" -lt "$READY_TIMEOUT_S" ]]; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      log "ERROR: server process died during startup (QPS=$qps). See $node_log"
      break
    fi
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    sleep 3; t=$((t+3))
  done
  if [[ "$ready" -ne 1 ]]; then
    log "ERROR: server not ready for QPS=$qps, skipping"
    grep -E "available memory from PhyPagePool|Check failed|Segmentation" "$node_log" | tail -5
    kill_xllm; wait_npu_free "$phy"; echo "$qps,ERROR,,,,," >> "$SWEEP_DIR/sweep_raw.csv"
    return 1
  fi
  grep -m1 "available memory from PhyPagePool" "$node_log" | tee "$rundir/pool_size.txt" || true
  log "server ready for QPS=$qps"

  # replay
  python "$SENDER" \
    --trace-csv "$TRACE_CSV" \
    --target-qps "$qps" \
    --input-scale "$INPUT_SCALE" \
    --max-tokens-scale "$MAX_TOKENS_SCALE" \
    --max-context-cap="$MAX_CONTEXT_CAP" \
    --limit "$LIMIT" \
    --port "$PORT" \
    --requests-jsonl "$rundir/client_requests.jsonl" \
    > "$rundir/sender_stdout.log" 2>&1
  log "sender done for QPS=$qps"

  # let tail decode + a few profiler flush windows settle
  sleep "$DRAIN_EXTRA_S"

  # collect server-side slices
  grep -a "\[priority window metric\]" "$node_log" > "$rundir/priority_window_metrics.log" 2>/dev/null
  grep -a "\[memory sample\]" "$node_log" > "$rundir/memory_samples.log" 2>/dev/null
  # keep only first 3000 startup lines of node log to bound disk
  head -3000 "$node_log" > "$rundir/node_0.startup.log" 2>/dev/null

  kill_xllm
  wait_npu_free "$phy"

  # per-run README + append to sweep raw
  python3 "$REPO/overhead/summarize_run.py" \
    --rundir "$rundir" --qps "$qps" --model "$MODEL_NAME" \
    --ttft-slo "$TTFT_SLO_MS" --tpot-slo "$TPOT_SLO_MS" \
    --sweep-raw "$SWEEP_DIR/sweep_raw.csv" 2>>"$rundir/summarize.err" || \
    log "WARN: summarize_run.py failed for QPS=$qps (see $rundir/summarize.err)"
}

# ----------------------------- main -----------------------------
log "sweep dir: $SWEEP_DIR"
log "QPS list: $QPS_LIST"
kill_xllm; sleep 2
PHY=$(pick_free_npu)
if [[ "$PHY" -lt 0 ]]; then log "ERROR: no free NPU found"; exit 1; fi
log "using physical NPU $PHY"

echo "qps,status,pool_gb,ttft_p50_ms,ttft_p95_ms,tpot_p50_ms,tpot_p95_ms,map_count,map_total_ms,unmap_count,unmap_total_ms,meets_slo" > "$SWEEP_DIR/sweep_raw.csv"

for qps in $QPS_LIST; do
  wait_npu_free "$PHY"
  run_one_qps "$qps" "$PHY"
done

# final sweep summary
python3 "$REPO/overhead/summarize_sweep.py" \
  --sweep-raw "$SWEEP_DIR/sweep_raw.csv" \
  --out "$SWEEP_SUMMARY" \
  --ttft-slo "$TTFT_SLO_MS" --tpot-slo "$TPOT_SLO_MS" 2>>"$SWEEP_DIR/summarize_sweep.err" || \
  log "WARN: summarize_sweep.py failed (see $SWEEP_DIR/summarize_sweep.err)"

log "DONE. sweep summary: $SWEEP_SUMMARY"
