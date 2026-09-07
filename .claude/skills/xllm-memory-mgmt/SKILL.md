---
name: xllm-memory-mgmt
description: VMM-based memory subsystem (xtensor) of xLLM. Open this when working on KV cache allocation, weight loading, activation pooling, page faults, PD disagg memory, NPU/CUDA VMM APIs, OOM debugging, or anything that touches the `xllm/core/framework/xtensor/` directory or `--enable_xtensor` paths.
---

# xtensor 显存管理子系统

xtensor 用 NPU/CUDA 的 **VMM API**（`cuMemMap` / NPU 的 `aclrtMallocPhysical`+`aclrtMapMem`）把虚拟地址与物理页解耦，从而：
- KV cache 物理上离散、虚拟上连续（攻击大 block 限制）
- 权重 / 激活 / KV 共享一套物理页池（攻击多模型同卡部署）
- 支持按页 lazy unmap、left/right 双端分配、跨节点 Mooncake 传输

## 一、关键类与责任

| 类                              | 文件                                                  | 单例？      | 职责                                                       |
| ------------------------------- | ----------------------------------------------------- | ----------- | ---------------------------------------------------------- |
| `PhyPage`                       | `phy_page.h`                                          | 否          | 一个 2MB 物理页（NPU/CUDA VMM 句柄 + page_id）             |
| `PhyPagePool`                   | `phy_page_pool.{h,cpp}`                               | worker 单例 | 给一个 device 用的物理页池                                 |
| `XTensor`                       | `xtensor.{h,cpp}`                                     | 否          | 一段虚拟地址 + 局部 page_id→PhyPage 映射；线程不安全       |
| `XTensorAllocator`              | `xtensor_allocator.{h,cpp}`                           | worker 单例 | 初始化 `PhyPagePool`，分发 map/unmap 广播 RPC              |
| `GlobalXTensor`                 | `global_xtensor.{h,cpp}`                              | worker 单例 | 全局 128GB×N 段虚拟空间；权重/激活的 bump 分配器           |
| `PageAllocator`                 | `page_allocator.{h,cpp}`                              | master 单例 | KV / 权重 page 分配 + 多模型计费 + 优先级 / reserved pages |
| `LayerOffloadManager`           | `layer_offload_manager.{h,cpp}`                       | 由 PageAllocator 持有 | 按水位线驱动逐 layer offload/load                          |
| `XTensorBlockManagerImpl`       | `xtensor_block_manager_impl.{h,cpp}`                  | 否          | 把 XTensor 适配为 `BlockManager`（让 scheduler 不感知）    |
| `XTensorManagerPool`            | `xtensor_manager_pool.{h,cpp}`                        | 否          | 多 dp_rank 的 manager 聚合，对外做 `KVCacheManager`        |
| `XTensorDistClient/Server/Service` | `xtensor_dist_{client,server,service}.{h,cpp}`     | 否          | 跨节点 page map/unmap、emergency eviction 协议             |

## 二、地址布局（GlobalXTensor）

```
0                                                                  total_size_ (= n*segment_size_, segment_size_=128GB)
│◄────────── allocate_offset_ (atomic, 左端长) ──┐    ┌── free_offset_ (右端缩) ──────────────►│
│ init_arena (initialization, 启动加载阶段) │ in-use weight / activation │ 空闲（unmap 后回这里）│
│                                            │                          │                       │
└────────────────────────────────────────────┴──────────────────────────┴───────────────────────┘

infer_arena_start_                            allocate_offset_ ────►        ◄──── free_offset_
init_allocate_offset_  (init_arena 内的偏移)
```

- 左端 `allocate_from_left()` / `allocate_init_from_left()`：长出权重段、激活段；调用者拿 raw void*。
- 右端 `free_to_right_async`：归还的物理页堆右端等待 lazy unmap 线程异步真正 unmap。
- 迁移：`migration_in_flight_` 期间可触发 `move_one_page(src_addr, dst_offset)` 把 src 的页直接 remap 到 dst（commit `eae69ddd`：unify weight xtensor virtual memory space）。

## 三、KV / 权重虚 / 物理对应关系

对每个**模型**：
- 非连续模式：每个 layer 独立 K 与 V 各一个 XTensor，每个 XTensor 大小 = `mem_size_per_layer = total_phy_mem / (2 * num_layers)`。
- 单个**虚拟 page**（VirtPage）= 一个 layer 内一份 K 或 V 的 page；一次 `alloc_kv_cache_page` 实际消耗 `phy_pages_per_virt_page = 2 * num_layers` 个物理页（K/V × num_layers）。
- Block ↔ VirtPage 转换：`PageAllocator::get_virt_page_id(block_id, block_mem_size)`；offset 转换 `get_offset(virt_page_id)`。

## 四、Page Allocator 的状态机（最易踩坑）

每个 `ModelState`（按 `model_id` 索引）有这些字段：
- 资源：`num_layers`, `num_total_virt_pages`, `phy_pages_per_virt_page`, `dp_group_pages[dp_rank]{free, reserved, allocated}`
- 模式：`is_sleeping`, `pending_map_ops` (CAS 安全 sleep 用)
- 优先级：`priority`（25/50/75/100），`min_reserved_pages`, `max_reserved_pages`, `base_min/max_reserved_pages`
- 分层 offload：`num_layers_on_device`, `layer_offloaded_phy_pages`
- 并行：`model_dp_size`, `model_tp_size`, `model_worker_rank_base`, `model_world_size`（fork 不同 strategy 时设置）
- 调度门控：`request_blocked`, `schedule_blocked`, `step_inflight`, `inflight_requests`

**API 速查**：
- 注册：`register_model(model_id, num_layers, master_status, priority, min_reserved, max_reserved)`
- 唤眠：`sleep_model(model_id)` / `wakeup_model(model_id)`
- KV：`alloc_kv_cache_page(model_id, dp_rank)` / `free_kv_cache_pages` / `trim_kv_cache`
- 权重：`alloc_weight_pages(model_id, num_pages)` / `free_weight_pages` / `set_weight_pages_count`
- 计费查询：`get_free_phy_pages_for_model`、`get_all_worker_free_pages`、`get_model_memory_usage`
- 紧急驱逐：`emergency_eviction(pages_needed, worker_rank)`
- 调度门控：`block_model_schedule` / `unblock_model_schedule` / `wait_and_mark_model_step_begin` / `mark_model_step_end`
- 优先级派发辅助：`pick_best_loadable_model_for_base(base_model_id)` —— 取压力最小的可加载 degraded 副本
- 分层 offload 启停：`start_layer_offload_monitor()` / `stop_layer_offload_monitor()`
- 异步驱逐线程：`start_async_eviction_thread()` / `stop_async_eviction_thread()`

## 五、控制流：一次 KV 分配

```
Sequence.allocate(num_tokens)
  └─ BlockManagerPool::allocate(sequence) — block_manager_pool.cpp
       └─ XTensorBlockManagerImpl::allocate(sequence)  (--enable_xtensor)
            └─ PageAllocator::alloc_kv_cache_page(model_id, dp_rank)
                 ├─ has_enough_phy_pages_for_dp？ 否 → trigger_preallocation + 等
                 ├─ 取一个 free_virt_page_id（lazy 复用 reserved_virt_page_list 内的）
                 ├─ map_virt_pages(model_id, dp_rank, page_ids) — 向所属 worker 广播 RPC
                 │   └─ Worker side: XTensorManagerServer.MapPages → XTensor::map(offset)
                 └─ 更新 worker_pages_used_[worker_range] += phy_pages_per_virt_page
```

## 六、Mooncake 集成（D2D / 跨节点权重传输）

- 模块：`framework/kv_cache/mooncake_weight_transfer.{h,cpp}`、`mooncake_transfer_engine.{h,cpp}`、`mooncake_kv_cache_transfer.{h,cpp}`、`framework/xtensor/...mooncake_registered_`
- 启动：通过 `--store_protocol --store_master_server_address --store_metadata_server --store_local_hostname` 配置
- 用途：
  - 权重 D2D：fork 副本时直接从另一 GPU/NPU pull 权重，避免重新加载
  - KV swap：层级 KV 缓存的 host↔device 传输（结合 `HierarchyBlockManagerPool`）
- 注意：`GlobalXTensor::mooncake_registered_` 是幂等注册标志（commit `3d074792`：idempotent initialization）

## 七、并发与生命周期警示

- 多锁层级：`PageAllocator::mtx_` >> `GlobalXTensor::mtx_` >> `GlobalXTensor::page_map_mtx_`（shared）；务必按此序拿，否则可能死锁。
- `unmap_worker` 线程：`GlobalXTensor` 有一个独立 unmap 线程持续从 `unmap_queue_` 取地址做实际 unmap；调用者只 enqueue。
- `pending_map_ops`：sleep_model 前必须等 0；否则可能 unmap 一个还在 map 中的 page。
- `wait_enough_pages`：分配右端预算耗尽时阻塞；要排查死锁先看是否两个模型在等同一片右端空间。
- 启动顺序：`xllm.cpp` 必须 `XTensorAllocator::init` → `init_phy_page_pools` → `GlobalXTensor::init`（隐式由 `XTensorAllocator` 引出）→ `create_master`。
- PD 分离 + xtensor：见 commit `0f0fb85b`（"PD separation in virtual memory management"）；KV transfer 走 page id 而非 ptr，避免地址在两端不一致。

## 八、看 / 改 xtensor 的高效切入点

1. **加新分配语义**：编辑 `PageAllocator`，复用 `consume_phy_pages_for_dp` / `release_phy_pages_for_dp` 计费；新增字段时同步更新 `ModelState` + `dp_group_pages`。
2. **调整 reserved pages 策略**：见 `update_model_reserved_pages` 与 `kv_prealloc_min_free_ratio` 配套 FLAG；动态调整逻辑在 `prealloc_worker`。
3. **修复某 OOM 路径**：先看 `emergency_eviction` 的兜底 + LayerOffloadManager 的 `kEmergencyEviction` 上下文；其次 `release_all_reserved_pages_for_models`。
4. **跨节点 page 操作**：`XTensorDistClient` 是 caller，`XTensorDistServer` 是 callee；broadcast 是经 `XTensorAllocator::map_virt_pages_async` 系列。
5. **看实际显存占用**：`get_model_memory_usage()` 与 `get_all_worker_free_pages()`；这俩值会被 etcd / xservice 报上去（commit `9f156b66`）。

## 九、相关 docs（中文）

- `docs/zh/features/xtensor_memory.md` — 设计意图（VMM + Page 池）
- `docs/zh/features/global_kvcache.md` — 全局 KV cache 管理
- `docs/zh/dev_guide/code_arch.md` — 全局代码结构
