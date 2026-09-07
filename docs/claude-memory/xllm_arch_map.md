---
name: xllm-arch-map
description: xLLM 顶层架构与一次推理请求的端到端调用链，含每层职责。
metadata:
  node_type: memory
  type: project
---

xLLM 总体分 7 大层（自顶向下）：

1. **入口**：`xllm.cpp::main → run()`（注册 NPU pluggable allocator、初始化 XTensorAllocator/PhyPagePool、起 brpc HttpServer）。
2. **API 层**：`xllm/api_service/`。`APIService` 路由 OpenAI/Anthropic 兼容 HTTP；每 base_model_id 维护 `LLMModelMasters{vector<Master*>, atomic rr}` 多副本表；通过 `ParseModelReplicaSpec` 解析 `base#N`。
3. **控制面 Master**：`core/distributed_runtime/{llm,vlm,dit,rec}_master.cpp`。`Master::run` 起 loop_thread_ 跑 `scheduler->step` 循环；多 master 同进程共存（`fork_master` 派生）。
4. **调度面 Scheduler**：`core/scheduler/`。`scheduler_factory.cpp` 按 FLAGS 选具体实现；多模型默认是 `ChunkedPrefillScheduler`，新加的多优先级 ProSched 走 `MixScheduler`。
5. **引擎 Engine**：`core/distributed_runtime/{llm,vlm,...}_engine.cpp`。`Engine::step(batches)` 单步入口；多模型分支在此植入 `ScopedEngineForwardAdmission`（SLO 准入）。
6. **执行面 Worker / Executor**：`core/runtime/`。`Worker` 包装本地/远程；`WorkerImpl` 抽象 + 各种 `*_worker_impl.cpp`；`Executor` 体系含 `BaseExecutorImpl/AclGraphExecutor/CudaGraphExecutor/MluGraphExecutor`。
7. **数据/资源底盘 Framework**：`core/framework/`。`xtensor/` 是多模型核心（VMM 抽象）；`block/` `kv_cache/` `prefix_cache/` `request/` `batch/` `sampling/` `parallel_state/` `eplb/` `model/` `tokenizer/` `state_dict/` `chat_template/` `dit_cache/` 各管一摊。

一次 chat 请求的完整链路：

```
HTTP → APIService::ChatCompletionsHttp → ChatServiceImpl::process_async_impl
  → 按 RequestMetricAggregator::get_replica_dispatch_weights 选 LLMMaster
  → LLMMaster::handle_request → 构 Request → scheduler_->add_request
LLMMaster::loop_thread_:
  → Scheduler::step
    → prepare_batch (KVCacheManager 分配 KV: XTensorManagerPool→PageAllocator)
    → LLMEngine::step
       → prepare_inputs
       → ScopedEngineForwardAdmission（SLO deadline 排队）
       → worker_clients_[i]->step_async
           → WorkerImpl::step → Executor::forward → 模型 layer 前向 → Sampler
    → process_batch_output → AsyncResponseProcessor → 回调
```

**Why:** 任何跨层的问题都需要按这条链 trace；不知道入口就只能搜全仓。

**How to apply:** 用户问"为什么 X 发生"先按这条链定位是哪一层，再用对应 skill 深入。多模型副本选择/SLO 等 4-6 层逻辑见 [[multimodel-mechanism]]；显存相关见 [[xtensor-memory]]。
