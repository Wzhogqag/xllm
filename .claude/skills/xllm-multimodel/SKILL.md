---
name: xllm-multimodel
description: Multi-model deployment subsystem of xLLM (feat/final_multi_model branch). Read this whenever the work touches multiple co-located models, model sleep/wakeup/fork, weight offload/load, activation pooling, model priority, SLO admission, or replica dispatch. Skips harmlessly for plain single-model bug fixes that don't cross PageAllocator / RequestMetricAggregator / api_service.
---

# 多模型部署机制（feat/final_multi_model 核心知识）

本分支以 **xtensor 虚拟连续显存** 为底盘，实现"一卡多模型 + 动态重平衡"。三件套：

1. **统一池化**（Activation + KV + Weight 同住一块物理页池）
2. **权重分层加卸载**（按 transformer layer 粒度 H2D/D2H/D2D，由水位线驱动）
3. **优先级 + SLO 感知调度**（请求派发 + Engine 准入 + 模型选择 degrade/restore）

## 一、总体拓扑

```
        ┌─────────────────────────────────────────────────────────────────┐
        │   多 Master 共存（主 Master + 通过 ForkMaster API 派生的副本）     │
        │   每个 Master 持有自己的 Engine/Scheduler/Worker，但 …              │
        └─────────────────────────────────────────────────────────────────┘
                  │ 调度面（每模型独立）         │ 资源面（全卡共享）
                  ▼                              ▼
        ┌──────────────────────┐   ┌────────────────────────────────────┐
        │ ContinuousScheduler   │   │  PageAllocator 单例（master 进程）  │
        │  + DecodePriorityQueue│◀──│  ├─ ModelState[model_id]             │
        │  + waiting offline 队列│   │  ├─ worker_pages_used_[worker_rank]  │
        └───────────┬──────────┘   │  ├─ LayerOffloadManager（子组件）     │
                    │              │  └─ 异步驱逐线程 + KV 预分配线程        │
                    ▼              └────────────────────────────────────┘
        ┌──────────────────────┐                ▲
        │ LLMEngine.step()      │      RPC      │  worker 侧
        │  + ScopedEngineForward│   ┌──────────▼─────────────┐
        │    Admission (SLO)    │   │  GlobalXTensor 单例    │
        └───────────┬──────────┘   │  + PhyPagePool         │
                    │              │  + XTensorAllocator    │
                    ▼              │  + worker_impl 内的     │
        ┌──────────────────────┐   │    weight_xtensors_     │
        │ WorkerImpl::step      │   │    activation_xtensor   │
        │ + offload/load layer  │◀──┤    kv xtensor          │
        │ + sync_npu_stream     │   └────────────────────────┘
        └──────────────────────┘
```

## 二、xtensor 显存底盘（无脑必读）

**核心抽象**（`xllm/core/framework/xtensor/`）：
- `PhyPage`：2MB 物理页，封装 NPU/CUDA VMM `MemMap` 句柄。
- `PhyPagePool`（worker 侧）：管理某一 device 上所有物理页的分配/归还。
- `XTensor`：一段连续虚拟地址 + `unordered_map<page_id, PhyPage*>` 映射表。可逐页 `map/unmap`。
- `XTensorAllocator`（worker 侧 singleton）：初始化 `PhyPagePool` + 广播分发的 RPC 客户端。
- `GlobalXTensor`（worker 侧 singleton）：一块超大（默认 128GB 段，可拼多段）的虚拟空间，所有模型的权重/激活都映射到其偏移；左端 `allocate_offset_` 长出，右端 `free_offset_` 收缩。带 lazy unmap（异步线程）。
- `PageAllocator`（**master 侧 singleton**）：唯一对外的 KV/权重 page 分配口，所有 model_id 都在这里登记；带优先级、reserved pages、layer offload 子组件、async eviction、prealloc 线程。
- `LayerOffloadManager`（PageAllocator 拥有）：master 侧，按水位线/SLO 触发对最低优先级 awake 模型逐层 offload，对最高优先级 degraded 模型逐层 load。
- `XTensorBlockManagerImpl` / `XTensorManagerPool`：作为 `KVCacheManager` 的实现接入 `BlockManagerPool` 抽象，让 scheduler 用同一套 `allocate(sequence)` API。

**入口条件**：`--enable_xtensor=true`。多模型场景所有路径都默认走 xtensor，关掉 xtensor 等同于退回到单模型经典 block 路径。

## 三、激活池化 (Activation Pooling)

- FLAG：`--enable_prism=false`（默认即开 activation pooling；commit `e5d15d43` 把原 `enable_activation_pooling` 与 `enable_forward_admission` 合并为单个 `enable_prism`，**语义反向** —— prism=true 切到 prism 策略，prism=false 走经典 pooling+admission 路径）。
- 思路：把过去 PyTorch caching allocator 中的"激活临时内存"也接到 `GlobalXTensor.allocate_from_left()`。`NPUPluggableAllocator.cpp` 实现 `my_custom_alloc` / `my_custom_free`，对应 `torch::npu::NPUPluggableAllocator::createCustomAllocator`，挂载到 NPU caching allocator。
- 关键路径：
  - 启动：`xllm.cpp` 中 `FLAGS_enable_xtensor && !FLAGS_enable_prism` 分支 → 注册 pluggable allocator。
  - 运行时：所有 torch::Tensor 临时分配落入 `GlobalXTensor`。完全用满后会触发 `GlobalXTensor::wait_enough_pages` 或 emergency eviction（释放 idle 模型的权重/KV）。
- 调试要点：
  - `GlobalXTensor::map_miss_time` 监控 map 失败次数；高即激活压力大。
  - "memory cooperation" 是指激活不够时主动让出 reserved KV（commit `a16b1d2a`）。
  - 多模型 + TP 下要释放 ATB buffer，见 `eb6fa277` 提交。

## 四、KV Cache 池化（多模型共享物理页）

- 入口：`ContinuousScheduler::prepare_batch` → `KVCacheManager::allocate(sequence)` → `XTensorManagerPool::allocate` → 经 RPC 到 worker 的 `XTensorManager` 调 `PhyPagePool` 取页。
- 计费：`PageAllocator::ModelState.dp_group_pages[dp_rank]` 维护 `free / reserved / allocated` virt page 列表；`worker_pages_used_[worker_rank]` 汇总每个 worker 的占用。
- 每模型有 `priority` (25/50/75/100，由 `--priority_level=1..4` 映射，**映射位置在 `llm_engine.cpp:992-1019`，不在 page_allocator 里**)，决定 `min_reserved_pages` / `max_reserved_pages`；外部 setter 是 `PageAllocator::update_model_reserved_pages`（`page_allocator.cpp:1270`）。**注意：没有独立的 `update_reserved_pages_with_utilization` 函数 —— 利用率驱动的动态调整逻辑内联在 `PageAllocator::prealloc_worker`（`page_allocator.cpp:1318-1417`），且需要 `--enable_dynamic_reserved_pages=true` 才生效。**
- 预留：reserved 是"映射好物理页等待复用"的 virt page；trim 时 unmap 还回 PhyPagePool。
- 紧急驱逐：`PageAllocator::emergency_eviction(pages_needed, worker_rank)` 会通过 LayerOffloadManager 选模型 offload；最后兜底 `release_all_reserved_pages_for_models`。

## 五、权重分层加卸载 (Layer Offload / Restore)

- FLAGS（在 `global_flags.cpp` 第 626~656 行附近）：
  - `--enable_watermark_degrade_restore_mvp`：**默认 false**（`global_flags.cpp:628`），需显式 `=true` 才开启 offload 监控 — 该 FLAG 不开，layer offload 整条路径都不工作
  - `--layer_offload_low_watermark_ratio=0.10`：低于该比例触发 offload（实际触发阈值是 `0.5 * low_watermark_ratio`，见 `layer_offload_manager.cpp:113`）
  - `--layer_offload_high_watermark_ratio=0.15`：高于该比例停止 offload
  - `--layer_offload_restore_watermark_ratio=0.25`：高于该比例触发 restore — **当前 monitor_loop 的 RESTORE 分支整段被注释掉**（`layer_offload_manager.cpp:172-214`），此 FLAG 实际未生效；恢复只通过显式 `trigger_load_for_base_model`（API fallback + fork 路径）
  - `--offload_chunk_layers=2` / `--load_chunk_layers=2`
  - `--layer_offload_poll_interval_ms=100`
- 启动：`LLMEngine::init` → `PageAllocator::start_layer_offload_monitor()` → 起 `LayerOffloadManager::monitor_loop` 线程。
- 决策：
  - **offload**：`pick_lowest_priority_awake` 选模型，调 `state.offload_fn(layer_id)` 经 worker 解映射对应 layer 的权重 xtensor pages。
  - **load**：`pick_highest_priority_degraded` 选模型，调 `state.load_fn(layer_id)` 重新分配 + H2D 拷贝；之后 `npu_sync_fn()` 同步 stream。
- D2D 加速：模型权重支持通过 **Mooncake** 在节点间直接拷贝（`mooncake_weight_transfer.cpp`、`feat: support weight D2D via Mooncake`）。
- 状态：模型有 `num_layers_on_device` 与 `is_degraded` 标记；schedule_blocked 时新请求会 hold（`block_model_schedule`），step 在 wait_and_mark_model_step_begin 上等。
- API 触发：`api_service.cpp::ForkMasterHttp` 在 `trigger_offload=false` 时显式调 `PageAllocator::load_model(model_path)` 主动唤醒。

## 六、模型 Sleep / Wakeup / Fork

- 状态枚举：`MasterStaus`（`master.h`）— 注意拼写：
  - `WAKEUP = 0` 正常服务
  - `LIGHT_SLEEP = 1` 仅释放 KV，权重保留
  - `DEEP_SLEEP = 2` 权重也释放（依赖 D2D / 本地重新加载）
- 接口：
  - `LLMMaster::sleep()` / `wakeup(WakeupOptions)` → `LLMEngine::sleep(status)` → 每个 `worker_clients_[i]->sleep_async(status)`。
  - REST：`/sleep`、`/wakeup`、`/link_d2d`、`/unlink_d2d`（见 `api_service.h`）。
- Fork：`ForkMasterHttp` 收到 fork 请求 → `make_model_instance_id(base, idx)` 生成 `base#N` → `fork_master(master_, options)`（master.cpp:302）创建新 Master、独立 scheduler/engine、共享 PageAllocator。
- CAS：`master->get_rate_limiter()->try_set_sleeping()` 必须 0 并发请求才允许 sleep。

### 6.1 跨卡多副本部署配方（实测跑通，别再踩坑）

想在 N 张卡各放一个目标模型副本（`Qwen3-8B#0`@卡0、`Qwen3-8B#1`@卡1 …），**不是**直接 fork N 次目标模型，正确套路（`bench/replica_dispatch_ab/run_trial.sh` 已验证）：

1. **起 N 个 xllm 进程，每卡一个**，全部加载**同一个占位模型**（Qwen3-0.6B），`--nnodes=N --dp_size=1`，构成 world_size=N 的 collective。占位模型撑起 world + 让 PageAllocator 预留 N 个 worker 计费槽。
2. **对目标模型 fork N 次**，第 k 次 `worker_rank=k` + **各自 distinct `master_node_addr`**（:9930/:9931…，别复用 base 的 :9929 否则 CollectiveServer bind 冲突）；**同一 fork 的 POST 同时发给全部 N 个进程**（`asyncio.gather`）。
3. fork 完 **DEEP_SLEEP 占位模型** 释放权重，留下 N 个目标副本各占一卡。
4. 压测发 `model="Qwen3-8B"`，dispatch 在 `#0/#1` 间选。

**bounds check**（`dist_manager.cpp:143`）：`CHECK_LE(worker_rank_base + world_size, physical_world_size)`，`physical_world_size = each_node_ranks * nnodes`——`worker_rank` 能给多大取决于 base 的 world，所以必须先用占位模型把 nnodes 撑到 N。

**⚠️ 订正历史误判**：早期以为"同卡 fork 必 SIGSEGV"。真相是当时 base `nnodes=1`（world=1）却想 fork 到 base 占用的 `:9929`，collective 建不起 → `tp_size=0` → KV 除零 → 段错。按上面配方就不会。段错日志锚点：`tp_size=0, world_size=0` + `xtensor_allocator.cpp:744] KV tensors not created`。详见记忆 fork-master-quirks。

### 6.2 进程 / 线程模型 & HCCL rendezvous（你会反复问到）

**每张卡上到底跑几个进程？** 默认 **1 个 xllm 进程/卡**，卡内一切都在进程内的线程里：

- **Master = 同进程多实例**（vLLM 里没有这个概念）：每 fork 一个副本就在**同一进程**内多一个 Master，各自持有独立 Engine/Scheduler + 一个 `loop_thread_` 跑 `scheduler->step`。N 个副本 = N 个 Master 对象 = N 个 loop 线程，**不是** N 个进程。
- **Worker 默认是线程不是进程**：`WorkerServer` 持 `worker_thread_`（`worker_server.h:93`）。**只有 offline_inference** 才走 `posix_spawnp` 起独立进程（`dist_manager.cpp:202` 的 `use_spawn_worker = enable_offline_inference && idx>0`）。在线服务里 worker 就是本进程的一个线程，`WorkerClient` 本地直调、跨节点才走 brpc。
- **CollectiveServer = HCCL rendezvous 交汇点**：类比 torch `init_process_group` 的 `MASTER_ADDR`。`node_rank==0` 进程在 `master_node_addr` 上起 `CollectiveService`（`dist_manager.cpp:225-237`）并 `wait()` 收齐所有 worker 的地址；`node_rank≥1` 进程起 WorkerServer 线程去连它。收齐后交换 HCCL rank/addr，建成 process group。
- **谁进 dispatch 表**：只有 `node_rank==0` 的 master 被 `add_model_master`（`api_service.cpp:960-963`）。`node_rank≥1` 进程**只暴露 `/fork_master`**，不暴露 `/v1/models`（`xllm_server.cpp:38-60`）——所以探活 node_1 得用裸 TCP 而非 GET /v1/models。

一句话对照 vLLM：vLLM 每卡 1 个 worker 进程、无 Master 概念、调度在 engine 进程；xLLM 每卡 1 进程内含【1 个占位/多个 fork 的 Master + 各自 loop 线程 + worker 线程 + PageAllocator 单例】，卡间靠 CollectiveServer 会合。

## 七、副本调度与优先级派发

API 层选副本（`*_service_impl.cpp::process_async_impl`）的算法：
1. 取 `LLMModelMasters{vector<Master*>, rr}` 表，过滤掉 sleeping / schedule_blocked。
2. 取 `RequestMetricAggregator::get_replica_dispatch_weights(base_id, replica_ids)`：
   - 按各副本所属 worker 的 `worker_pages_used_` 反比加权（内存压力越大权重越小）。
   - 加权随机抽样（`std::mt19937_64`）一个 master。
3. 若全部不可用 → 调 `LayerOffloadManager::trigger_load_for_base_model` 唤醒一个 degraded 副本；仍失败返回 503。

**计费语义（易误解，务必看）**：`get_worker_pages_used(rank)` = `worker_pages_used_[rank] + worker_reported_pages_used_[rank]`（`page_allocator.cpp:170-171`），**只按 `worker_rank`（= 物理卡）索引，没有 model_id 维度**。因此：
- **同一张卡上多个模型实例的占用是叠加的、跨模型不去重**——这是**设计如此**，因为 dispatch 关心的是"这张卡还剩多少物理页"，而非某个模型单独用了多少。卡0 上 `Qwen3-8B#0` + `Qwen3-1.7B#0` 的页会 `+=` 到同一个 `worker_pages_used_[0]`。
- 跨进程聚合靠 `/dev/shm/xllm_activation_phy_pages_used_<port-node_rank>`：每进程把自己 worker 的占用 `__atomic_store` 进 shm（`:148`），读时 `sync_reported_phy_pages_from_shm_locked` 拉回来累加（`:161`）。所以卡上是"别的进程 fork 的副本"也能被算进这张卡的压力，无需去重——不同来源写不同 rank 槽，同 rank 的本进程分配走 `worker_pages_used_`、外部报告走 `worker_reported_pages_used_`，两者相加即该卡真实占用。

## 八、SLO 感知 Engine 准入（forward admission）

- `core/distributed_runtime/llm_engine.cpp` 的 `EngineForwardAdmissionController` 单例（约 232~756 行）：
  - 每个 forward batch 入队 `QueuedBatch{deadline_sort_key, predicted_cost, …}`（`llm_engine.cpp:292`）。
  - `prefill_contained_queue_` 与 `decode_only_queue_` 分队；排序键 = `deadline_sort_key_ms - predicted_cost`（latest-start-time-first，是 EDF + cost preemption 的变体，见 `QueuedBatchCompare`，`llm_engine.cpp:311`）。
  - 截止期（`compute_batch_deadline_ms`，`llm_engine.cpp:365`）：prefill = `min(recv_ts + ttft_slo_ms)`；decode = `min(start_ts + tpot_slo_ms * (gen_tokens+1))`。
  - `predicted_cost` 是模型特定二次型 `a + b*x + c*x*x`（已为 Qwen3 变体打表，否则 fallback `prefill_len*3 + decode_bs`，`llm_engine.cpp:108-228`）。
- 进入路径：`LLMEngine::step` → `ScopedEngineForwardAdmission`（RAII；析构自动 `mark_forward_done`，`llm_engine.cpp:764`）。
- **关键现状（重要）**：gate 在 `llm_engine.cpp:1740-1751` 写成 `if (FLAGS_enable_prism && !FLAGS_enable_prism)` —— **恒为 false，admission 路径当前完全不生效**。代码还在，但 `enable_prism=false` 默认下不触发。要做对比实验，得手动 patch 这一行（例如改为 `if (!FLAGS_enable_prism)` 或加新 FLAG）。
- 仅考虑 TTFT 违反触发模型加载，见提交 `e5d15d43`（"consider only TTFT violation when loading model"）。

## 九、SLO 监控与模型优先级

- 聚合器：`RequestMetricAggregator`（`request/request_metric_aggregator.{h,cpp}`，单例）。
  - `add_sample(model_id, ttft_ms, tpot_ms)` 由 scheduler 输出后调用；窗口大小 `--priority_window_size=5000ms`。
  - `get_model_priority` = f(平均时延 / SLO，副本数)；越违反 SLO 越高分。
  - `get_replica_dispatch_weights` 见上一节。
- 全局 SLO：`--priority_ttft_slo_ms` `--priority_tpot_slo_ms`（默认 INT_MAX，需显式设置）。
- 阈值：`--load_model_slo_violation_rate=50`（窗口违反率）→ 触发 LayerOffloadManager load。
- 配套 commits：`8e6225ef`（slo monitor），`f7655e76`（priority calculation update），`6a4fe80e`（forward admission），`e5d15d43`（unify + TTFT-only loading）。

## 十、关键不变量与常见坑

1. **PageAllocator 是 master 节点单例**：fork 出的子 master 共享同一份 `model_states_`。不要 per-Master 持有副本。
2. **model_id 在 PageAllocator 里就是 base#N（运行时 ID）**：API 入参可能是 base，需通过 `master_instances_` 二级表映射；不要混淆 base 与 runtime id。
3. **MasterStaus 拼写**：是 `Staus` 不是 `Status`，所有引用都按这个；如要改名要全仓改。
4. **`enable_prism && !enable_prism`**：`llm_engine.cpp:1740-1751` 这个 gate **恒为 false**，是 `unify some features for prism`（e5d15d43）后留下的占位 —— Engine forward admission 实际未启用。如果你看到 admission 路径像没触发——确实没触发；要实验需手动 patch 这一行。
5. **GlobalXTensor 不能在 `Date.now()` 等不确定函数中触发**——它假设状态可重入，关键路径必须 lock `mtx_`/`page_map_mtx_`。
6. **Worker 与 Master 通过 brpc + RPC 单 dispatch**，同进程时是直接调用（`WorkerClient` 抽象）；TP 内的 driver 才做 sample。
7. **共享内存指标**：`worker_reported_pages_used_` 通过 `/dev/shm` 拉，看 `init_reported_phy_pages_shm_if_needed` / `sync_reported_phy_pages_from_shm_locked`。

## 十一、调试 / 改动入手清单

- **查多模型当前显存**：grep `get_model_memory_usage` `get_all_worker_free_pages`；加 `--v=1` 看 PageAllocator 的 VLOG。
- **怀疑 layer offload 没触发**：先确认 `--enable_watermark_degrade_restore_mvp=true`（**默认 false**，不显式开整条路径都不工作）；然后看 `LayerOffloadManager::monitor_loop` 日志，确认 `pick_lowest_priority_awake` 找到了候选。注意 monitor_loop 的 RESTORE 分支当前是注释掉的，restore 只能通过 `trigger_load_for_base_model` 触发。
- **怀疑 SLO admission 阻塞**：grep `EngineForwardAdmissionController` 日志中的 `pending_enqueue_len`、`active_batches_inflight`。
- **添加新的优先级策略**：在 `priority_comparator.{h,cpp}` 加一个 struct，更新 `create_comparator` 工厂。
- **改 fork/sleep 行为**：从 `api_service.cpp::ForkMasterHttp/SleepHttp/WakeupHttp` 入手，跟到 `master.cpp::fork_master`，再到 `LLMMaster::sleep/wakeup`。
- **改副本派发**：`*ServiceImpl::process_async_impl` + `RequestMetricAggregator::get_replica_dispatch_weights`。
