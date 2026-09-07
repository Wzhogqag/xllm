---
name: xtensor-memory
description: xtensor 显存子系统的关键抽象、地址布局和并发约束。
metadata:
  node_type: memory
  type: project
---

xtensor 用 VMM API 把虚拟地址/物理页解耦，是多模型分支的资源底盘。

## 关键类（`xllm/core/framework/xtensor/`）

| 类                            | 单例？      | 作用                                                       |
| ----------------------------- | ----------- | ---------------------------------------------------------- |
| `PhyPage`                     | 否          | 一个物理页（2MB 默认，封装 VMM 句柄 + page_id）             |
| `PhyPagePool`                 | worker 单例 | 单 device 物理页池                                          |
| `XTensor`                     | 否          | 一段虚拟地址 + page_id→PhyPage 映射（**线程不安全**）       |
| `XTensorAllocator`            | worker 单例 | 初始化 PhyPagePool + map/unmap 广播 RPC                     |
| `GlobalXTensor`               | worker 单例 | 全局连续虚拟地址空间（默认每段 128GB，可拼多段）            |
| `PageAllocator`               | **master 单例** | 唯一 KV / 权重 page 分配口，按 model_id 计费                |
| `LayerOffloadManager`         | 由 PA 持有 | 水位线驱动的逐 layer offload/load                           |
| `XTensorBlockManagerImpl`     | 否          | 把 XTensor 适配为 `BlockManager` 接口                       |
| `XTensorManagerPool`          | 否          | 多 dp_rank 聚合，对外作为 `KVCacheManager`                  |

## GlobalXTensor 地址布局

```
[0 ......... allocate_offset_ (atomic, 左端长) ←→ free_offset_ (右端缩) ......... total_size_]
 ├── init_arena (启动期分配区)
 ├── infer_arena (推理期：权重/激活段)
 └── 右侧空闲区（unmap 后回归这里，lazy unmap 线程异步处理）
```

- `allocate_from_left(count)` / `allocate_init_from_left(count)` 长出 → 返回 raw void*
- `free_to_right_async(pages)` 归还 → enqueue lazy unmap
- 迁移：`move_one_page(src, dst)` 在 `migration_in_flight_` 期间重映射（commit `eae69ddd`）

## KV / 权重虚-物理对应

- 每模型每 layer 各有独立的 K 与 V XTensor。
- `mem_size_per_layer = total_phy_mem / (2 * num_layers)`。
- 一次 `alloc_kv_cache_page` 实际消耗 `2 * num_layers` 个物理页（每 layer 一对 K/V）。

## PageAllocator 状态机字段

每个 `ModelState`（按 `model_id`）：
- 资源：`num_layers`, `num_total_virt_pages`, `phy_pages_per_virt_page`, `dp_group_pages[dp_rank]{free,reserved,allocated}`
- 模式：`is_sleeping`, `pending_map_ops`
- 优先级：`priority` (25/50/75/100), `min/max_reserved_pages`, `base_min/max_reserved_pages`
- 分层 offload：`num_layers_on_device`, `layer_offloaded_phy_pages`
- 并行：`model_dp_size/tp_size/worker_rank_base/world_size`
- 调度门控：`request_blocked`, `schedule_blocked`, `step_inflight`, `inflight_requests`

## 锁层级（按此序拿，否则死锁）

`PageAllocator::mtx_` >> `GlobalXTensor::mtx_` >> `GlobalXTensor::page_map_mtx_`（shared）

## 启动顺序

1. `xllm.cpp::run` 解析 FLAGS
2. 若 `--enable_xtensor && !--enable_prism`：注册 NPUPluggableAllocator
3. `XTensorAllocator::get_instance().init(device)`
4. （多节点）`setup_multi_node_xtensor_dist`
5. `init_phy_page_pools(max_memory_utilization, max_cache_size)`
6. `create_master / fork_master` → `Master::run`

## Mooncake 集成

- `framework/kv_cache/mooncake_*.{h,cpp}`、`framework/xtensor/...mooncake_registered_`
- 用途：权重 D2D（fork 副本跳过重新加载）、KV swap（host↔device 层级缓存）
- 启动：`--store_protocol --store_master_server_address --store_metadata_server --store_local_hostname`
- 注册幂等性：`GlobalXTensor::mooncake_registered_`（commit `3d074792`）

**Why:** 多模型分支所有"显存够不够""map 失败""OOM""sleep 卡住"问题最终都落到这里。

**How to apply:** 任何 OOM / map 失败 / hang 时，先看 `wait_enough_pages`、`emergency_eviction`、`lazy unmap 队列`、锁竞争是不是有踩坑。改这块代码前确认是否影响其他单例的状态。详见 skill `xllm-memory-mgmt`、[[multimodel-mechanism]]。
