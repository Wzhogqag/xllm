---
name: xllm-debug
description: Common debugging recipes and tracing techniques for xLLM. Use when the user reports a crash, hang, OOM, perf regression, schedule stall, multi-model issue, or asks "why is X happening" in a running deployment. Provides FLAGS to flip, log keys to grep, and the order to chase symptoms upstream.
---

# xLLM 调试 / 追踪手册

## 一、首先确认环境与运行参数

1. `start.sh` 在仓库根 —— 看正在用的实际命令。本分支默认开的关键 FLAGS：`--enable_xtensor=true`、`--enable_prism=false`（即 activation pooling + forward admission 路径，e5d15d43 合并后的统一开关）、`--priority_level=3`、`--enable_chunked_prefill=true`、`--enable_schedule_overlap=false`。
2. 日志：`node_*.log`（默认；`start.sh` 中 `LOG_FILE`）。glog 默认 `--alsologtostderr=true`，stderr 也会有。
3. 开高级日志：`--v=1` 或 `--v=2`；具体模块用 `--vmodule=page_allocator=2,layer_offload_manager=2,llm_engine=2`。
4. 环境变量备查：
   - `XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS` —— 多节点 process group test 超时（默认 4s）。
   - `ASCEND_RT_VISIBLE_DEVICES` / `NPU_PHY_ID` —— NPU 卡选择。
   - `NPU_MEMORY_FRACTION=0.98` —— 物理显存使用上限。
   - `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` —— torch NPU allocator 模式。
   - `ATB_*` —— ATB 算子库行为。

## 二、按症状索引

### 症状：进程启动失败 / OOM-during-init

- 看 `XTensor initialized with N physical pages`。N≤0 即 `init_phy_page_pools` 失败 —— 通常是 `--max_memory_utilization` 太高或别的进程没让出显存。
- `Failed to init model from: <path>` —— `LLMEngine::init_model` 失败；进一步看 `hf_model_loader` 错误。
- `process_group_test` 超时 —— 多节点 brpc 通讯没连上；检查 `--master_node_addr`、`--xtensor_master_node_addr` 与防火墙。
- `Failed to create master, backend is X` —— 不支持的 backend（仅 llm/vlm/dit/rec）。

### 症状：请求 stuck / 长时间不返回

按这个顺序检查：
1. **API 层**：grep `Model not supported` / `Master for model not found` / `503` / `UNAVAILABLE`，看 `*_service_impl.cpp::process_async_impl` 是否路由失败。
2. **rate limiter**：`get_num_concurrent_requests`、`try_set_sleeping` 失败说明 master 被卡 sleep。
3. **schedule_blocked**：grep `is_model_schedule_blocked`；`PageAllocator::block_model_schedule` 已加锁该 model。
4. **PageAllocator wait**：grep `wait_enough_pages` 或 `prealloc_running_` / `eviction_in_progress_` —— 物理页等不到。
5. **EngineForwardAdmissionController**：grep `pending_enqueue_len`、`active_batches_inflight`、`deadline_sort_key_ms`；按 deadline 排队过长。
6. **Worker step**：grep `step_async` / `forward` 卡在某层；NPU 端 `aclrtSynchronizeStream` 失败常见于 layer offload 竞争。

### 症状：突发性能下降

- 看 `engine_latency_seconds`（Prometheus 指标） + `step time` profile。
- `LayerOffloadManager::monitor_loop` 频繁触发 → `--layer_offload_low_watermark_ratio` 设太高 / 模型副本太多。
- `prealloc_worker` 频繁触发驱逐 → KV reserved pool 太小，看 `min/max_reserved_pages` 是否被 `priority_level` 压缩。
- ATB buffer 没释放 → 多模型 TP 下要释放 ATB buffer，参考 commit `eb6fa277`、`e5d15d43`（"keep ATB buffer when decode-only" 已收住）。

### 症状：OOM at runtime（而非启动）

- `emergency_eviction` 日志：会汇报触发原因、worker_rank、pages_needed。
- `GlobalXTensor::wait_enough_pages`：物理页耗尽 + 没释放 → 看右端 `free_offset_` 是否还在退（lazy unmap 线程是否阻塞）。
- 多模型副本未及时 sleep：grep `pick_lowest_priority_awake` 与其返回；若 nullptr 说明候选都不可 offload（所有模型都 `is_sleeping` 或 `request_blocked`）。
- ATB workspace 与激活池竞争：看 `ATB_WORKSPACE_MEM_ALLOC_*` 环境变量。

### 症状：精度/输出错乱

- 优先排除 sampler：`Sampler` / `BeamSearcher`，`process_sample_output(result, false)` 中第二参数是 `enable_schedule_overlap`，false 时附加 real token，true 时假 token，bug 多发于此 commit 改动后。
- 张量 dtype 不匹配：`--kv_cache_dtype` 与模型 dtype 错配。
- 多模态预处理：`processors/` 与 `framework/request/mm_*`；commit `ec0c913f`、`31609f24` 是这类 bug 的典型修复。

### 症状：PD 分离链路问题

- 实例发现：`disagg_pd_service*.cpp` + etcd 注册；看 `enable_service_routing` 是否启用 + etcd_addr 配置。
- KV transfer：`kv_cache_transfer.cpp` + Mooncake；`--kv_cache_transfer_mode=PUSH/PULL`。
- max_reqs_p2d_once / enable_batch_response 影响吞吐 vs 延迟。

### 症状：fork master 失败

- `fork_master requires xtensor to be enabled`：开 `--enable_xtensor`。
- `Duplicate runtime model_id`：worker_rank 给重了；副本 ID 是 `base#worker_rank`。
- `Cannot sleep model with in-flight requests`：CAS 失败，等并发降为 0 再试。
- **fork 后 `tp_size=0 / world_size=0` + `xtensor_allocator.cpp:744] KV tensors not created` → 随即 SIGSEGV**：collective 没建起来。两大根因：(1) 复用了 base 的 `master_node_addr`（:9929）→ CollectiveServer bind 冲突，每个 fork 必须用 distinct addr；(2) base 的 world 不够大，`worker_rank+nnodes > physical_world_size` 过不了 `dist_manager.cpp:143` 的 CHECK——跨卡多副本要先用 `nnodes=N` 的占位模型撑起 world。正确配方见 skill `xllm-multimodel` 6.1 节 / 记忆 fork-master-quirks。
- **fork POST 打到 `node_rank≥1` 进程的 `/v1/models` 得 404**：非 0 节点只暴露 `/fork_master`（`xllm_server.cpp:38-60`），探活用裸 TCP 连端口，别 GET /v1/models。
- **fork 请求带了 `"trigger_offload": false` 却没 fork**：该字段语义反向，present-and-false = "这是 reload 命令"会短路 fork（`api_service.cpp:896`）。想 fork awake 副本就**别带这个字段**。

### 症状：模型 layer offload 不触发 / 总在打架

- 确认 `--enable_watermark_degrade_restore_mvp=true`。
- 看 `PageAllocator::start_layer_offload_monitor` 日志，monitor 线程是否在跑。
- `pick_lowest_priority_awake` 候选为空 → 所有 awake 模型都是同优先级；区分 priority。
- 重复抢占：调高 `--layer_offload_high_watermark_ratio` 与 `--layer_offload_restore_watermark_ratio` 间距。

## 三、必备 grep 套路

```bash
# 多模型注册情况
grep -nE "register_model|sleep_model|wakeup_model|fork_master|ForkMaster" -r xllm/ | head

# Layer offload 决策痕迹
grep -nE "pick_lowest_priority_awake|pick_highest_priority_degraded|offload_internal|load_layers|trigger_load_for_base_model" -r xllm/

# SLO admission 关键
grep -nE "EngineForwardAdmissionController|ScopedEngineForwardAdmission|compute_batch_deadline_ms" xllm/core/distributed_runtime/llm_engine.cpp

# 任何 FLAGS_xxx 的用处
grep -rn "FLAGS_enable_xtensor" xllm/

# Priority 派发
grep -rn "get_replica_dispatch_weights|RequestMetricAggregator" xllm/
```

## 四、本分支 commit → 涉及代码区域

把 commit 当作"已知改动点"用 `git log --oneline --grep '<关键词>'` 反查：

| 关键词              | 提交                                                        |
| ------------------- | ----------------------------------------------------------- |
| activation pooling  | `eee8f82f`, `8184e6e5`, `e3ec6e5e`, `eb6fa277`, `e9a21b58`  |
| weight xtensor unify| `eae69ddd`                                                  |
| layer offload       | `fd7870cd`, `0efbcd2c`, `8e6225ef`, `bb8cf4fb`, `c893f615`, `8eb9ee99` |
| forward admission   | `6a4fe80e`                                                  |
| priority calc       | `f7655e76`, `e5d15d43`                                      |
| Mooncake D2D weight | `94edfca9`, `fa16c68f`                                      |
| fork master         | `2582c073`, `928f0ba5`                                      |
| multi-model PA      | `7c4102bb`, `a16b1d2a`, `217450a9`, `5aaa78aa`              |
| prefix cache (multi)| `334837a9`, `87a431be`, `f197098a`                          |

## 五、Profiling

- 模型 step time：`--enable_profile_step_time=true` → 由 `scheduler/profile/profile_manager.cpp` 记录。
- TTFT/TPOT 上报：`RequestMetricAggregator::add_sample` 自动；用 `--priority_window_size` 控制窗口。
- Prometheus metrics：`core/common/metrics.{h,cpp}` 定义 counters / gauges / histograms；HTTP 端口 `/metrics`。
- NPU timeline：`tools/` 目录有解析脚本（`acl_graph` / `ATB` 时间线）。

## 六、最常用的临时 patch 思路

- "想看具体哪个序列被抢占了" → 在 `ContinuousScheduler::handle_abnormal_request` 加 LOG。
- "想看某次 emergency eviction 因谁" → 在 `PageAllocator::emergency_eviction` 入口 LOG `model_id, num_layers_on_device, weight_pages_allocated`。
- "想看每个模型当前状态" → 调 `PageAllocator::get_model_memory_usage()` + `RequestMetricAggregator::get_model_priority()`，可在 `MasterServiceImpl` 暴露 HTTP debug 端点。

## 七、单测在哪

- 仅部分模块带 `*_test.cpp`：`block_manager_test`, `chunked_prefill_scheduler_test`, `continuous_scheduler_test`, `rate_limiter_test`, `vlm_master_test`, `mapping_npu_test`, `acl_graph_executor_test`, `cuda_graph_executor_test`, `mlu_graph_executor_test`, `anthropic_protocol_test`, `eplb_policy_test`, `shared_vmm_allocator_test`, `prefix_cache_test`, `chat_json_utils_test`, `npu_dp_ep_padding_test`, `batch_test`, `blocking_counter_test`, `threadpool_test`.
- 没有覆盖到的模块改动后建议手动起 `start.sh` 单跑或加临时 demo `examples/`。
