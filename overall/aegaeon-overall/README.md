# 两模型 (Qwen2.5-3B + 7B) observed 实验 — 20260901_093158

Aegaeon 单卡时分复用 serve 两个模型的观测实验。本文档说明**每个指标存在哪个文件的哪个字段**，以及读取时的注意事项。

- 硬件：A800-SXM4-80GB × 2（物理 GPU **2, 3**；prefill engine #0 → GPU 2，decode engine #1 → GPU 3）
- 显存软上限：`--gpu-memory-utilization 0.654` ≈ 52.3GB（对齐 NPU 52.3GB 硬限制的对比实验）
- 负载：`--qps 10 --limit 80 --cut-last`，发送窗口 79.9s，遥测覆盖 172.4s
- 结果：**92 次模型切换全部成功，零 CUDA 错误**

## 1. 指标索引

| 指标 | 文件 | 字段 | 粒度 |
|---|---|---|---|
| 权重显存 | `telemetry/qwen25-{3b,7b}__activation_usage.jsonl` | `model_weights_bytes` | 200ms 轮询 |
| KV 预留容量 | `telemetry/qwen25-{3b,7b}__kv_usage.jsonl` | `num_gpu_blocks` × `block_bytes` | 200ms 轮询 |
| KV 实际在用 | 同上 | `logical_in_use_kv_bytes` / `gpu_resident_kv_bytes` | 200ms 轮询 |
| **激活显存** | `step_events.jsonl` | `activation_peak_bytes` | **逐 step** |
| Tensor Core 利用率 | `telemetry/dcgm_prof.csv` | `tensor_active` | 200ms（DCGM 独立采样）|
| 每 token 时延（服务端） | `step_events.jsonl` | `compute_duration_ms` / `num_generated_tokens` | 逐 step |
| TTFT / TPOT（端到端） | `client/traces/requests.jsonl` | `first_token_wall_ns`、`end_wall_ns` | 逐请求 |
| 模型切换耗时 | `switch_events.jsonl` | `switch_duration_ms` | 逐次切换 |
| 整卡显存/功耗 | `engine_gpu_timeline.jsonl`、`gpu_util.csv` | `gpu_memory_used_bytes`、`power_w` | 200ms 轮询 |

### 读取前必须注意

1. **KV / 权重文件必须先过滤 `active == true`。** 模型未驻留 GPU 时仍会写占位行（`sample_state: "model_not_loaded_on_any_engine"`，各字段为 0）。这是"该时刻未加载"的记录，不是测量失败。不过滤会把统计量严重压低。
   - 注意还存在第三种状态：`active: true` 但 `logical_in_use_blocks: 0` — 模型仍驻留（预留 block 在），只是当前无请求。与"已卸载"含义不同。
2. **激活显存请用 `step_events.activation_peak_bytes`，不要用 `activation_usage.jsonl` 的 `activation_proxy_bytes`。** 后者定义为 `nvml_used - weights - kv`，其中 `nvml_used` 含 vLLM 预留但空闲的整个 block 池，因此算出 39–50 GiB，比真实激活（1 GiB 量级）高出两个数量级。字段名与 `telemetry_semantics: "aegaeon_activation_proxy"` 已标明是代理值。
3. **KV 峰值系低估。** 200ms 轮询下，decode step 仅 3.46ms — 每个采样间隔内约跑 65 个 step，间隔内的波动未被记录。适合看趋势，不适合作为容量上限依据。
4. **KV 与激活无法逐行对齐。** step_events 有 1866 条逐 step 记录，KV 仅 545 个采样时刻。联合分析须按时间窗口聚合，不可配对。
5. **DCGM 均值会误导，请看峰值与非零分布。** 200ms 采样 vs 数毫秒的 step，大量采样落在空闲间隙。`tensor_active` 1638 个样本中仅 341 个非零。
6. 采样间隔实测 p50 227ms（名义 200ms + 快照 RPC 开销），最大一次 1792ms — 大概率是切换时 controller 快照被阻塞，该窗口 KV 变化丢失。

## 2. 显存占用

### 权重（常量，实测）

| 模型 | 权重 |
|---|---|
| Qwen2.5-3B | **6.18 GiB** |
| Qwen2.5-7B | **14.50 GiB** |

### KV Cache

预留容量与实际在用需区分 — 前者由 `gpu_memory_utilization` 预切，全程恒定；后者随请求变化。

| 模型 | block 大小 | 预留容量 | 在用 p50 | p95 | 峰值 |
|---|---|---|---|---|---|
| Qwen2.5-3B | 576 KiB | 41.53 GiB | 1086.8 MiB | 1086.8 MiB | **1086.8 MiB** |
| Qwen2.5-7B | 896 KiB | 34.04 GiB | 318.5 MiB | 1868.1 MiB | **2020.4 MiB** |

`num_gpu_blocks` 各有两个取值（3B: 75595/51838，7B: 39834/33324），因 prefill 与 decode 引擎预算不同。3B 的 p50 = p95 = max 是采样恰好集中在满载态所致，非静态值 — `logical_in_use_blocks` 实测有 78 种取值（0→1932）。`moving_gpu_blocks`（3B 峰值 168，7B 峰值 204）非零，证实 KV 换入换出真实发生。

### 激活显存（逐 step 真实峰值）

| 阶段 | 模型 | p50 | p95 | 峰值 |
|---|---|---|---|---|
| prefill | 3B | 4.56 MiB | 957.40 MiB | **1060.3 MiB** |
| prefill | 7B | 6.52 MiB | 109.78 MiB | 183.1 MiB |
| decode | 3B | 1.01 MiB | 8.41 MiB | 1040.1 MiB |
| decode | 7B | 1.16 MiB | 10.86 MiB | 15.1 MiB |

p50 与 max 相差最多三个数量级（少数长 prompt step 拉出极值），**报告时须同时给出 p50 和 max**。

激活量跟随 batch size 而非时间单调增长 — decode 分段观测：batch p50 1 → 3 → 21 → 61 时，激活 p50 为 1.0 → 1.1 → 2.9 → 6.9 MiB。作图横轴宜用 `batch_size` 或 `num_input_tokens`。

### 整卡实测

`fb_used_mib` p50 **56300 MiB**、峰值 **56816 MiB**。高于 52.3GB 软上限，因 NCCL / cuBLAS / CUDA graph 显存绕过该限制（预期行为）。

## 3. Tensor Core 与 SM 利用率

`telemetry/dcgm_prof.csv`，1638 样本 / 165s。

| 指标 | 非零样本 | 非零 p50 | 峰值 |
|---|---|---|---|
| `tensor_active` | 341 / 1638 | 0.0770 | **0.5110** |
| `sm_active` | 410 / 1638 | 0.1420 | 0.6250 |
| `dram_active` | 701 / 1638 | 0.0460 | 0.4970 |
| `fp32_active` | 305 / 1638 | 0.0020 | 0.0280 |

`fp16_active` / `fp64_active` 全为 0。权重为 bf16，DCGM 将其计入 `tensor_active` 而非 `fp16_active`。

其余可用列：`sm_occupancy`、`gpu_util_pct`（峰值 100%，均值 20.75%）、`fb_used_mib`。

## 4. 时延

### 服务端计算耗时（`step_events.jsonl`，1866 steps）

| 阶段 | p50 | p95 | max |
|---|---|---|---|
| prefill | 13.45 ms | 44.31 ms | 286.0 ms |
| decode | 11.52 ms | 14.89 ms | 505.4 ms |

**decode 每 token：p50 3.460 ms，p95 11.847 ms**

### 端到端（`client/traces/requests.jsonl`）

| 指标 | 样本 | p50 | min | max |
|---|---|---|---|---|
| TTFT | 11 | 497.2 ms | 40.3 ms | 3085.0 ms |
| TPOT | 11 | 114.51 ms | 43.35 ms | 377.96 ms |

TTFT 可扩至 265 个样本 — 另有 254 个被 `cut_last` 截断的请求同样含 `first_token_wall_ns`（TPOT 因输出不完整不可用）。

服务端 3.46 ms 与端到端 114.51 ms 的差距来自排队与调度开销，二者分别回答"算得多快"与"用户等多久"，不矛盾。

### 模型切换（`switch_events.jsonl`）

92 次全部成功：p50 **343.9 ms**，p95 704.7 ms，max 1101.3 ms，累计 40.3s（占 172s 遥测窗口的 23.4%）。

转换构成：`3B→7B` 45 次，`7B→3B` 45 次，冷启动 `None→3B` 2 次。

## 5. 数据集与请求

数据集：`anonymized_chatbot_trace.parquet`，客户端参数 `--head 24900 --extend --max-inflight 1024 --cut-last`。

共发送 **800** 个请求，涵盖 10 种 model_id：

| model_id | 请求数 | 是否被服务 |
|---|---|---|
| Qwen2.5-3B | 152 | ✅ |
| Qwen2.5-7B | 129 | ✅ |
| Qwen3-32B | 100 | ❌ HTTP 400 |
| Qwen3-14B | 93 | ❌ |
| Qwen3-8B | 86 | ❌ |
| Qwen3-4B | 83 | ❌ |
| Qwen3-1.7B | 73 | ❌ |
| Qwen3-0.6B | 42 | ❌ |
| Qwen2.5-0.5B | 34 | ❌ |
| Qwen2.5-1.5B | 8 | ❌ |

### 结果构成

| 类别 | 数量 | 说明 |
|---|---|---|
| `http_non_200` | 519 | **预期行为**：`--window-model-remap-order` 仅将 top-2 频次模型 remap 到 3B/7B，其余 8 种必然 400 |
| `cut_last_truncated` | 270 | **预期行为**：`--cut-last` 在接收窗口结束时主动截断 |
| 完整成功 | 11 | 未被截断且完整返回 |

被服务的两模型共 281 个请求。**无任何失败源自 CUDA 错误或模型切换。**

### 长度分布

输入长度（服务端真实 tokenize 后，`step_events.num_input_tokens`）：

| 模型 | 请求数 | min | p50 | p95 | max | 合计 |
|---|---|---|---|---|---|---|
| 3B | 152 | 8 | 53 | 814 | 2290 | 23032 |
| 7B | 129 | 8 | 44 | 802 | 1032 | 19337 |

输出：`max_tokens` 请求区间 100–8000（p50 650）；被服务请求实际 `output_tokens` p50 51、max 790、合计 26944。

服务端生成 token 合计：3B **7367**（503 steps），7B **21735**（1082 steps）。

decode batch size：3B p50 1 / p95 104 / max 150；7B p50 7 / p95 80 / max 111。

### 模型驻留份额

| 模型 | active 样本 | 占比 |
|---|---|---|
| Qwen2.5-3B | 812 / 929 | 87.4% |
| Qwen2.5-7B | 278 / 662 | 42.0% |

分母不同系采样自各模型首次出现起算（7B 起步晚）。812 + 278 = 1090，恰为 `engine_gpu_timeline.jsonl` 行数 — 每个采样时刻两台引擎各驻留一个模型，两文件是同一事实的两种记法。3B 驻留时长约为 7B 两倍，但 7B 单位驻留时间内完成更多 step（1082 vs 503）。

## 6. 文件清单

```
aegaeon.log                              服务端完整日志
server_stdout.log / client_stdout.log    shim 与客户端 stdout
observation_monitor.log                  监控启动记录（含 local→physical GPU 映射）

step_events.jsonl          (1866)  逐 step：compute 耗时、激活峰值、batch、token 数
switch_events.jsonl        (92)    逐次模型切换耗时
engine_gpu_timeline.jsonl  (1090)  逐引擎：NVML 显存/利用率/功耗
active_model_timeline.jsonl(1090)  逐引擎：当前驻留模型
gpu_util.csv                       NVML 整卡采样

telemetry/dcgm_prof.csv                     (1638) DCGM：tensor/sm/dram active、fb_used
telemetry/qwen25-{3b,7b}__kv_usage.jsonl    block manager 快照（需过滤 active）
telemetry/qwen25-{3b,7b}__activation_usage.jsonl  权重 + 激活代理值（勿用后者）

client/traces/requests.jsonl  (800)  逐请求：TTFT/TPOT/状态/长度
benchmark_plots/                     TTFT/TPOT/ITL 的 CDF 与时间趋势
plots/                               引擎利用率、显存构成、切换 vs 计算时间线
```

## 7. 本次修复记录

此前该实验在首次 3B→7B 切换时崩溃，表现为"卡住 + GPU 利用率 0%"。

根因：`_register_host_in_chunks`（旧实现）在启动期注册失败时，将 48GB 主机缓存切成 8GB 小块分别 `cudaHostRegister`。CUDA 视每次注册为独立 pinned region 且不合并相邻区间，因此**任何跨接缝的 H2D 拷贝均被驱动拒绝**（`CUDA error: invalid argument`，抛出点 `aegaeon/loader/loader.py:293`）。3B 权重落在首块内故正常，7B 约 14GB 必然跨越 8GB 边界。worker 抛异常退出后，orchestrator 仍停留在 `wait_healthy` 等待永不到来的 `/health`，故表象为挂起。

修复：改为 `_register_host`（`aegaeon/llm.py:44`），整块注册，失败重试 5 次 × 2s，仍失败则报错而非退化为分块。四处调用点同步更新（`llm.py` cpu cache、`worker.py` kv_swap、`loader/loader.py` model_cache、`loader/cache.py` QuickCache）。

补充：触发分块的注册失败**并非 A800 尺寸限制** — 机器空闲时 48GB 单块注册成功，两个 48GB 同时注册亦成功。该失败源于主机内存压力，正确处置是释放内存而非缩小 chunk。
