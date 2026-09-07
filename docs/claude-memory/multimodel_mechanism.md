---
name: multimodel-mechanism
description: 多模型同卡部署机制（feat/final_multi_model 核心）：池化、分层加卸载、优先级 SLO 调度。
metadata:
  node_type: memory
  type: project
---

`feat/final_multi_model` 分支在 main 之上加了 89 个 commit，三大组件：

## 1. 统一池化（Memory Pooling）

- 底盘：xtensor 子系统（`xllm/core/framework/xtensor/`）基于 NPU/CUDA VMM API。
- 单例：
  - `PageAllocator`（master 侧，唯一 KV/权重计费入口，按 `model_id` 分账）
  - `GlobalXTensor`（worker 侧，128GB×N 连续虚拟空间，权重/激活落在此处）
  - `XTensorAllocator`（worker 侧，PhyPagePool + RPC 广播）
- 激活池化：默认开启，开关是 `--enable_prism=false`（commit `e5d15d43` 把原 `enable_activation_pooling` 与 `enable_forward_admission` 合并到 `enable_prism`，**语义反向**：prism=false 走 pooling+admission 经典路径，prism=true 切到 prism 策略）。靠 `NPUPluggableAllocator.cpp` 注册到 torch NPU caching allocator。
- KV cache：经 `XTensorManagerPool`（一种 `KVCacheManager` 实现）调 `PageAllocator::alloc_kv_cache_page(model_id, dp_rank)`。
- 权重：每模型一段 `GlobalXTensor` 内的 bump-allocated 区域；按 layer 拆分。

## 2. 权重分层加卸载（Layer Offload/Restore）

- `LayerOffloadManager`（`xllm/core/framework/xtensor/layer_offload_manager.{h,cpp}`），由 `PageAllocator` 持有。
- 触发：`--enable_watermark_degrade_restore_mvp=true`。
  - 低于 `--layer_offload_low_watermark_ratio` → 选最低优先级 awake 模型 offload `--offload_chunk_layers` 层。
  - 高于 `--layer_offload_restore_watermark_ratio` → 选最高优先级 degraded 模型 load `--load_chunk_layers` 层。
- 每模型 `PerModelState` 持 `offload_fn(layer_id)` / `load_fn(layer_id)` / `npu_sync_fn()` 回调。
- D2D：`mooncake_weight_transfer.{h,cpp}` 允许直接从邻居 NPU pull 权重（commit `94edfca9`, `fa16c68f`）。
- 主线 commit：`fd7870cd`, `8eb9ee99`, `c893f615`, `bb8cf4fb`, `8e6225ef`, `e5d15d43`。

## 3. 优先级 + SLO 调度

- 模型优先级 = `--priority_level∈{1,2,3,4} × 25`，映射到 `min/max_reserved_pages`（LOW/MEDIUM/HIGH/CRITICAL）。
- 请求层 SLO：`--priority_ttft_slo_ms / priority_tpot_slo_ms`，由 `RequestMetricAggregator::add_sample` 在每 token 后更新窗口（`--priority_window_size=5000ms`）。
- 副本派发：`RequestMetricAggregator::get_replica_dispatch_weights(base, replicas)` 返回基于 worker 内存压力的反比加权；`*ServiceImpl::process_async_impl` 用加权随机抽样选 master。
- Engine forward admission：`EngineForwardAdmissionController`（`llm_engine.cpp` 232-756 行）单例，按 batch deadline 排队（prefill: recv+ttft_slo，decode: start+tpot_slo*tokens）。
- 模型加载触发：`--load_model_slo_violation_rate=50`（窗口违反率）→ `LayerOffloadManager::trigger_load_for_base_model`。仅 TTFT 违反触发加载（commit `e5d15d43`）。

## Master / Fork / Sleep / Wakeup

- `MasterStaus` 枚举（**拼写就是 Staus**，不要"修正"）：`WAKEUP=0, LIGHT_SLEEP=1, DEEP_SLEEP=2`。
- API：`/sleep`, `/wakeup`, `/link_d2d`, `/unlink_d2d`, `/fork_master`（通过 `api_service.cpp`）。
- Fork 派生副本：`make_model_instance_id(base, idx) = "base#N"`，由 `fork_master(master, opts)` 创建独立 Master/Engine/Scheduler，共享 `PageAllocator`。
- CAS：sleep 必须 0 并发请求（`RateLimiter::try_set_sleeping`）。

## 不变量（注意！）

1. `PageAllocator::get_instance()` 是 master 进程单例，跨所有 fork 出来的 Master 共享同一份状态。
2. `model_id` 在 PageAllocator 内是 runtime ID (`base#N`)；API 入参常是 base，由 `master_instances_` 二级映射转换。
3. `fork_master` 强依赖 `FLAGS_enable_xtensor`，否则 nullptr。
4. `enable_prism && !enable_prism` 在 `llm_engine.cpp` 多处是 unify 后留的占位逻辑，运行时不触发。
5. NPU pluggable allocator 与 prism 互斥：`if (FLAGS_enable_xtensor && !FLAGS_enable_prism)` 才注册。

**Why:** 该机制是当前分支的全部存在意义；用户后续大概率围绕这里做需求/debug。

**How to apply:** 任何"模型为什么 sleep/wakeup""为什么副本被选/不被选""为什么 OOM""权重为什么换""SLO 为什么超"问题都走这条。详细 trace 路径见 [[xtensor-memory]] 与 skill `xllm-multimodel`、`xllm-memory-mgmt`。
