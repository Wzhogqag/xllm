---
name: xllm-scheduler
description: xLLM scheduler subsystem. Read when working on request scheduling, batch formation, priority queues, chunked prefill, PD disagg scheduling, SLO/latency-aware batching, or anything under `xllm/core/scheduler/`. Skip for pure model-layer or worker-layer changes.
---

# xLLM Scheduler 子系统

调度器把请求队列变成一个个可送到 Engine 的 `std::vector<Batch>`。本目录用**策略工厂**模式，由 `scheduler_factory.cpp` 按 FLAGS 选具体实现，所有实现共享 `ContinuousScheduler` 的优先级队列 / kv_cache_manager / response_processor 基础设施。

## 一、策略工厂选择路径

`scheduler_factory.cpp::create_continuous_scheduler`：

```
if FLAGS_use_mix_scheduler:        return MixScheduler           # ProSched 多优先级混队
elif options.enable_disagg_pd():
    if options.enable_pd_ooc():    return PDOOCScheduler
    else:                          return DisaggPDScheduler
elif options.enable_chunked_prefill():
    if options.num_speculative_tokens()>0:
                                   return PrefillOnlyScheduler   # 投机的 prefill 实例
    else:                          return ChunkedPrefillScheduler# **多模型分支默认**
elif FLAGS_use_zero_evict:         return ZeroEvictionScheduler
else:                              return ContinuousScheduler

# DiT 单独工厂：
create_dit_scheduler → DiTDynamicBatchScheduler
```

## 二、ContinuousScheduler 数据结构（基类）

`continuous_scheduler.h` 中的关键字段（**所有派生类都继承**）：

| 字段                              | 用途                                                       |
| --------------------------------- | ---------------------------------------------------------- |
| `request_queue_` (folly MPMCQueue) | API 层 `add_request` 投入，scheduler 线程取出              |
| `waiting_priority_queue_`         | 在线请求等待队列（priority_queue 按 `priority_strategy`）  |
| `waiting_priority_queue_offline_` | 离线请求等待队列（同上）                                   |
| `running_queue_` / `running_queue_offline_` | `DecodePriorityQueue` —— decode 阶段反饥饿优先级           |
| `running_requests_` / `running_sequences_` | 当前 step 选中的请求 / 序列                                |
| `running_sequences_budgets_`      | 每序列 token 配额（chunked prefill 用）                    |
| `preemptable_requests_`           | 已分块未结束、可被抢占的序列                               |
| `last_batch_` / `last_running_*`  | `enable_schedule_overlap=true` 下保留上一步引用              |
| `kv_cache_manager_`               | `BlockManagerPool` 或 `HierarchyBlockManagerPool` 或 `XTensorManagerPool` |
| `response_processor_`             | 异步推回客户端（`AsyncResponseProcessor`）                 |
| `profile_manager_`                | 时延/budget 预测（`scheduler/profile/`）                   |
| `instance_info_`                  | 自身实例元信息（PD 分离 etcd 注册用）                      |

## 三、优先级策略（priority_comparator.h）

FLAGS：`--priority_strategy={fcfs|priority|deadline|sjf|density|urgency_density|urgency_priority|decode_*}`

| Comparator                              | 行为                                          |
| --------------------------------------- | --------------------------------------------- |
| `FCFSComparator`                        | 默认；按到达时间                              |
| `StrictPriorityComparator`              | 严格 HIGH > MEDIUM > LOW                      |
| `DeadlineComparator`                    | TTFT 截止越近越优先                           |
| `SJFComparator`                         | 短作业优先                                    |
| `DensityComparator`                     | density = 剩余 token / 剩余时间               |
| `DecodeDeadlineComparator`              | decode 用的截止                              |
| `DecodeDensityWithAntiStarveComparator` | decode + 反饥饿                              |
| `UrgencyDensityComparator`              | density × urgency 组合，新提交                |
| `UrgencyPriorityComparator`             | priority + urgency 组合，新提交               |
| `DecodeUrgencyDensityComparator`        | decode 版                                     |

工厂：`create_comparator(strategy, reverse)`。

## 四、Scheduler::step 的一帧

```
ContinuousScheduler::step(timeout)
  ├─ 若 enable_schedule_overlap → step_with_schedule_overlap (双 batch 流水)
  ├─ 否则 → schedule_request(timeout)
  │     ├─ pop_from_request_queue_ → push_to_waiting_priority_queue_
  │     ├─ prepare_batch()  ← 派生类可 override
  │     │   ├─ handle_prefill_requests(latency_budget, …, waiting_queue)
  │     │   │   - 把 waiting_priority_queue 弹出来分配 KV、放进 running_queue
  │     │   ├─ handle_decode_requests(…, running_queue)
  │     │   │   - 计算 decode 子集，token_budget/seq_budget
  │     │   ├─ handle_abnormal_request → 抢占 / 缩短
  │     │   └─ 返回 vector<Batch> (按 dp_rank)
  │     └─ engine_->step(batches)
  │         ↘ LLMEngine::step → ScopedEngineForwardAdmission → workers
  ├─ update_token_latency_metrics（计 TTFT/TPOT 上报 RequestMetricAggregator）
  └─ process_batch_output → 拉取 ForwardOutput → process_sample_output / process_beam_search_output → response_processor_->push
```

## 五、ChunkedPrefillScheduler（多模型默认）

在基类基础上：
- prefill 阶段按 `max_tokens_per_chunk_for_prefill` 切块；超出部分序列回 `waiting_priority_queue_`（变 preemptable）。
- 大头逻辑：`allocate_blocks_for`、`check_if_enough_to_evict`（基类提供）。

## 六、MixScheduler（ProSched / 多优先级混队）

- 不再区分 prefill_queue / decode_queue，单一 `running_queue_` 混排。
- 关键私有方法：
  - `handle_running_queue_requests` 主循环
  - `get_latency_budget_and_request_order` 算 SLO 余量
  - `get_max_chunk` 单序列单次最大可推 token 数（支持二次公式 latency model）
  - `get_max_copy_block_num` / `get_needed_copy_block_num` H2D 块数预算
- 需开 `enable_chunked_prefill=true`。

## 七、DisaggPDScheduler / PDOOCScheduler（PD 分离）

- 实例分为 Prefill/Decode 两组（`InstanceRole::PREFILL/DECODE`）。
- 调度路径：
  - **Prefill 实例**：跑完 prefill，把 KV blocks 通过 KVCacheTransfer 推/拉给 Decode 实例。
  - **Decode 实例**：从 P 实例拉 KV（PULL）或被推（PUSH）；只做 decode。
- 关键：`kv_cache_transfer_mode={PUSH,PULL}`、`max_reqs_p2d_once`、`enable_batch_response`。
- OOC（Out-Of-Cache）版用于 KV 不在本地的快速回源。
- 服务面：`distributed_runtime/disagg_pd_service*.cpp`, `pd_ooc_service*.cpp`。

## 八、PrefillOnlyScheduler / ZeroEvictionScheduler / FixedStepsScheduler

- PrefillOnly：投机推理场景，主实例仅做 prefill，draft 在别处。
- ZeroEvict：保证只要进入就跑完，不回滚（实验用）。
- FixedSteps：固定 step 数，profile 用。

## 九、调度器与 Profile

- `scheduler/profile/profile_manager.{h,cpp}`：实际推理时序记录 → 拟合 latency model。
- `perf_model.{h,cpp}`：单 step 时延预测函数（陆续被改进）。
- FLAGS：`--enable_profile_step_time` `--enable_profile_token_budget` `--enable_latency_aware_schedule` `--profile_max_prompt_length` `--enable_profile_kv_blocks` `--disable_ttft_profiling`。

## 十、与多模型的接合点

- **每个 Master 持有自己的 scheduler**（fork 出的副本各有一个）；scheduler 不知道彼此存在。
- 共享通过 `PageAllocator`（KV 分配会失败/阻塞）传导。
- `MasterStaus::LIGHT_SLEEP/DEEP_SLEEP` 切换由 Master 调用 `scheduler_->step` 前后；schedule_blocked 的模型其 scheduler 通常进入 stoped_/running_ 控制位。
- TTFT/TPOT 上报：scheduler 在收到 token 时调 `RequestMetricAggregator::add_sample`（详见 `update_token_latency_metrics`）。

## 十一、改 / 调 scheduler 的快速通道

| 目标                              | 入手点                                                              |
| --------------------------------- | ------------------------------------------------------------------- |
| 改 priority 策略                  | `priority_comparator.{h,cpp}` 加 struct + 注册到 `create_comparator` |
| 改 prefill / decode token budget  | `ChunkedPrefillScheduler::handle_*_requests`                        |
| 改 PD 分离传输策略                | `DisaggPDScheduler::step` + `kv_cache_transfer.cpp`                 |
| 改抢占规则                        | `check_if_enough_to_evict`（基类）                                  |
| 添加新调度器                      | 子类 ContinuousScheduler + 在 `scheduler_factory.cpp` 注册分支         |
| 跑单测                            | `*_test.cpp`（如 `continuous_scheduler_test.cpp`、`chunked_prefill_scheduler_test.cpp`） |
