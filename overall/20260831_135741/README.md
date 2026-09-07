# Colocation experiment 20260831_135741

单卡（NPU3 chip0 = phy6）同时 serve **Qwen2.5-3B**（base）+ **Qwen2.5-7B**（fork，同进程同 PageAllocator）。

## 配置
- 卡：phy6 (NPU3 / chip0)，AICore Count=24/die，PhyPagePool 可用 52.30 GB
- base=Qwen2.5-3B（start.sh 启动），fork=Qwen2.5-7B#0（/fork_master，worker_rank=0，awake）
- **enable_prefix_cache=true**（关键：影响 KV 是否随请求结束释放）
- 压测：QPS=5，发送窗口 LIMIT=80s（注意是**秒**不是条数），--extend 开，--head=24900，cut_last=true（窗口结束+1s截断，不等积压排空）
- remap-order：Qwen2.5-3B,Qwen2.5-7B,Qwen3-14B,Qwen3-32B,Qwen2.5-14B,Qwen2.5-32B,Qwen2.5-0.5B,Qwen3-0.6B,Qwen3-1.7B,Qwen2.5-1.5B,Qwen3-4B
- 服务端指标采样间隔：--priority_window_size=1000ms（见 start.sh.snapshot）

## 文件说明（每项数据怎么来的）

> 注：保存**完整 node_0.log**（刷屏的 [token latency step] 调试打印已在源码 VLOG 门控关闭，日志不再膨胀）。另附两个便捷切片。

| 文件 | 内容 | 采集方式 |
|---|---|---|
| `node_0.log` | xLLM **完整日志**（启动+运行） | start.sh 重定向；含模型加载、PhyPagePool、fork、TTFT/TPOT、显存打点 |
| `node_0.priority_metrics.log` | **服务端 TTFT/TPOT**（每窗口一条） | grep `[priority window metric]`（request_metric_aggregator.cpp:463） |
| `node_0.memory_samples.log` | **显存打点**原始行 | grep `[memory sample]`（dynamic_hbm 的输入） |
| `start.sh.snapshot` | 本次启动的完整 FLAGS | 启动用的 start.sh 副本 |
| `bench_stdout.log` | **客户端** TTFT/TPOT/goodput 汇总 | xllm_benchmark.py 输出；per-model p50/p90/p95/p99 + goodput |
| `metrics/npu_usages.csv` | **AICore/AICube/AIVector 利用率** + HBM 带宽/用量 | npu_usages_sampler.py 每 1s 跑 `npu-smi info -t usages` |
| `metrics/dynamic_hbm.csv` | **真实显存**：weight/KV/activation 分模型分项（页 + GB） | parse_mem_samples.py 从 memory_samples 切片抽取 |
| `metrics/dynamic_hbm.png` | 动态显存堆叠时序图 | 同上 |
| `metrics/*.png` (ttft/itl) | 客户端 TTFT/ITL 的 CDF + 时间趋势 | xllm_benchmark.py |
| `client/traces/requests.jsonl` | 逐请求 trace | xllm_benchmark.py |

## 四类目标数据 & 复看命令

1. **xLLM 启动日志** → `node_0.log`（本目录，完整日志，开头即启动段）。

2. **TTFT / TPOT**（两个口径）：
   - 服务端（窗口聚合，权威）：`cat node_0.priority_metrics.log`
     字段：avg_ttft_ms / avg_tpot_ms / ttft_violation_rate / tpot_violation_rate（每 1000ms 一条，来源 request_metric_aggregator.cpp:463）。
   - 客户端（端到端）：见 `bench_stdout.log` 的 per-model 段。

3. **AICore / AICube / AIVector 利用率** → `metrics/npu_usages.csv`：
   - 列：chip0_aicore, _aicube（矩阵乘/tensor core）, _aivector, _hbm_bw, _hbm_use, _npu_util，单位 %。
   - **总量 vs 使用量表示**：每 die 有 **24 个 AICore**（`npu-smi info -t common` 的 Aicore Count）。等效在用核数 = 利用率% × 24 / 100。
     例：aicube=63% → ≈ 15 个 Cube 单元在忙（满配 24）。
   - 快速看峰值：`awk -F, 'NR>1{print $2,$3,$4}' metrics/npu_usages.csv | sort -rn | head`
   - AICore 是整核综合占用；AICube 是其中的矩阵乘单元（prefill 高）；AIVector 是向量单元（激活/norm）。

4. **真实显存（weight+KV+activation 之和）** → `metrics/dynamic_hbm.{csv,png}`：
   - 每行一个采样窗口（1000ms），列含各模型 weight_pages / kv_pages + activation_pages + total_pages + total_gb（页=2MiB）。
   - 这是**池内运行时真实占用**（随负载动态涨落），区别于 npu-smi 看到的恒定物理池占用。
   - 原始打点：`cat node_0.memory_samples.log`（来源 request_metric_aggregator.cpp:364 / page_allocator.get_model_memory_usage）。

## 结果速览
- benchmark 汇总：见 bench_stdout.log 末尾（success/failed、per-model TTFT/TPOT/goodput）。
- 命中确认：bench_stdout.log 里 Qwen2.5-3B 和 Qwen2.5-7B 的 n= 应 >0（其余模型未启动，落空为 Model not supported，属预期）。
