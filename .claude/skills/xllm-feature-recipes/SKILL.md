---
name: xllm-feature-recipes
description: Cookbook for adding a new feature to xLLM (new model, new scheduler, new layer, new API, new FLAG, new metric, etc.). Read when the user says "I want to add / implement / extend X" — it points to the canonical files to copy and the registration touchpoints.
---

# 加新东西到 xLLM 的"作法"清单

每个 recipe 都列：**入口文件 → 必改文件 → 注册点 → 常见忘记的地方**。

---

## Recipe 1: 加一个新 FLAG

1. `xllm/core/common/global_flags.h` 加 `DECLARE_<type>(name);`
2. `xllm/core/common/global_flags.cpp` 加 `DEFINE_<type>(name, default, "help");`
3. 在 `xllm/core/common/options.h` 加对应 `PROPERTY(T, name)` 并在 `xllm.cpp::run()` 里把 `FLAGS_name → options.name(...)` 串起来。
4. 用 `core/common/help_formatter.h` 暴露给 `--help`（可选）。

容易忘：步骤 1（只在 cpp 定义会编译过但其他模块取不到）。

---

## Recipe 2: 加一个新的 Scheduler 策略

1. 新文件 `xllm/core/scheduler/my_scheduler.{h,cpp}` —— 继承 `ContinuousScheduler`，必须 `override prepare_batch()`，可选 override `handle_prefill_requests/handle_decode_requests/if_queue_not_empty`。
2. 在 `scheduler/CMakeLists.txt` 加源文件。
3. 在 `scheduler_factory.cpp::create_continuous_scheduler` 加分支：```cpp
   if (FLAGS_use_my_scheduler) return std::make_unique<MyScheduler>(engine, options);
   ```
4. 加 FLAG `FLAGS_use_my_scheduler`（按 Recipe 1）。
5. 单测：拷一份 `chunked_prefill_scheduler_test.cpp` 改名。

模板：`chunked_prefill_scheduler.{h,cpp}`。

---

## Recipe 3: 加一个新的优先级策略（priority_comparator）

1. 在 `framework/request/priority_comparator.h` 加 struct `MyComparator : PriorityComparator`。
2. 在 `priority_comparator.cpp` 实现 `operator()`。
3. 在 `create_comparator` 工厂函数中加 `if (strategy=="my") return MyComparator{};`。
4. 写好后传 `--priority_strategy=my` 即可。

---

## Recipe 4: 加一个新的 Worker 类型（如 RecVLMWorker）

1. 新文件 `runtime/my_worker_impl.{h,cpp}` —— 继承 `WorkerImpl`，必须 override：`init_model`, `step`。
2. 在 `runtime/CMakeLists.txt` 加源。
3. 在 `runtime/worker.cpp::create_worker_impl` （或类似工厂）按 `WorkerType` 选自己的实现。
4. 在 `common/types.h` 的 `WorkerType` 枚举加项。
5. Engine 侧选择 worker_type 时分支：`distributed_runtime/llm_engine.cpp::setup_workers`（或 `rec_engine.cpp` 等）。

模板：`llm_worker_impl.{h,cpp}`。

---

## Recipe 5: 加一个新的 Executor / Graph 模式

1. 新文件 `runtime/my_graph_executor_impl.{h,cpp}` —— 继承 `Executor` (或 `BaseExecutorImpl`)。
2. 在 `runtime/executor_impl_factory.cpp` 加分支（按设备 / graph 模式选）。
3. 平台条件编译：用 `#if defined(USE_NPU)` 等。

模板：`acl_graph_executor_impl.{h,cpp}`（NPU）/ `cuda_graph_executor_impl.{h,cpp}`。

---

## Recipe 6: 加一个新的模型

LLM：
1. 新文件 `xllm/models/llm/my_model.h` —— 用既有 `LLM_MODEL_REGISTER(...)` 之类宏注册（看 `models.h` / `model_registry.h` 的现成实现）。
2. 实现自己的 decoder layer：`xllm/core/layers/my_decoder_layer.{h,cpp}` + 平台特化 `npu/my_decoder_layer_impl.{h,cpp}`。
3. 如需新的 vision 编码器：`vlm/` 下加；processors 加图像处理（如 `qwen2_vl_image_processor.cpp` 模板）。
4. 在 `xllm/models/llm/CMakeLists.txt`、`models.h` 上注册。
5. 模型 type → backend 映射在 `model_registry.cpp::get_model_backend`。

DiT/Rec/Embedding 类似，分别看 `models/dit/`、`models/llm/{glm4_moe_*}`、`models/llm/qwen3_embedding.h`。

参考最新 commit `3c74517f`（Llama3.2 / Llama3 manual loader）。

---

## Recipe 7: 加一个新的 API 端点（HTTP/gRPC）

1. 在 `xllm/core/proto/xllm_service.proto`（或对应 `chat.proto`、`multimodal.proto` 等）加 rpc 方法和 message 定义；重生成 pb。
2. `xllm/api_service/api_service.h` 在 `APIService` 类下加 `MyEndpoint` 和 `MyEndpointHttp` 方法。
3. `api_service.cpp` 实现：通常 `Http` 版做 JSON↔PB 转换后转给原版。
4. 若是业务路由型，加 `MyServiceImpl` 到 `api_service/` 并在 `APIService` 构造里初始化。
5. brpc 自动按 method 名做 URL 路由（小写 + `_http`）。

参考：`ChatCompletionsHttp` / `ForkMasterHttp`。

---

## Recipe 8: 暴露一个新的 Prometheus metric

1. `xllm/core/common/metrics.h` 加 `COUNTER_DEFINE`/`GAUGE_DEFINE`/`HISTOGRAM_DEFINE`。
2. 在用到的地方 `COUNTER_ADD(name, value)` / `GAUGE_SET(name, value)`。
3. 自动通过 `/metrics` 暴露。

例：`COUNTER_ADD(engine_latency_seconds, timer.elapsed_seconds());`

---

## Recipe 9: 修改 PageAllocator 的状态（如新增一种资源计费）

1. `framework/xtensor/page_allocator.h` 中 `ModelState` 加字段（如 `int32_t my_resource_usage = 0;`）。
2. `register_model` 初始化。
3. 在分配/释放路径加 update（注意拿 `mtx_`）。
4. 在 `sleep_model` / `wakeup_model` / `emergency_eviction` 路径同步处理。
5. 若需要跨 worker 同步：参考 `worker_pages_used_` 的 SHM/RPC 模式（`init_reported_phy_pages_shm_if_needed` 等）。

---

## Recipe 10: 改 PD 分离传输策略

1. `framework/kv_cache/kv_cache_transfer.{h,cpp}` —— 基础接口。
2. `mooncake_kv_cache_transfer.{h,cpp}` / `hierarchy_kv_cache_transfer.{h,cpp}` —— 具体实现。
3. Scheduler 端：`DisaggPDScheduler` 决策何时 push/pull；`PDOOCScheduler` 处理 cache miss 回源。
4. brpc 服务：`distributed_runtime/disagg_pd_service*.{h,cpp}`、`pd_ooc_service*.{h,cpp}`。

---

## Recipe 11: 改激活池化 / pluggable allocator

- 文件：`xllm/NPUPluggableAllocator.cpp`（顶层！注意是 `xllm/NPUPluggableAllocator.cpp` 而非 `xllm/core/...`）。
- 关键函数：`my_custom_alloc` / `my_custom_free`，挂到 `c10_npu::NPUCachingAllocator::NPUAllocator`。
- 启动注册：`xllm.cpp::run()` 中 `FLAGS_enable_xtensor && !FLAGS_enable_prism` 分支。
- 关闭时回退：设 `enable_prism=true`（e5d15d43 后用单一开关；prism=true 时跳过 pluggable allocator 注册，让 torch 自己用 caching allocator）。

---

## Recipe 12: 修改 Engine.step / forward 路径

- 一致入口：`Engine::step(std::vector<Batch>& batch) → ForwardOutput`。
- 各实现：`llm_engine.cpp` / `vlm_engine.cpp` / `dit_engine.cpp` / `rec_engine.cpp` / `speculative_engine.cpp`。
- 经过的关键步骤（LLM 为例）：
  1. `prepare_inputs(batch)` → `vector<RawForwardInput>`（按 dp_rank）。
  2. `build_engine_forward_batch_metrics`（用于 SLO admission）。
  3. `ScopedEngineForwardAdmission`（RAII 准入）。
  4. 每 worker `step_async(raw_forward_inputs[dp_rank])`。
  5. `folly::collectAll(futures).get()` 等结果。
  6. `process_sample_output / process_beam_search_output` 写回 Batch。

不要打破"engine 只跑一步"这条线 —— 多步循环属于 scheduler 的事。

---

## Recipe 13: 加一个新的 multimodal modality（如 audio）

1. `framework/request/mm_*.{h,cpp}` 体系：加 `MmAudioDataItem`、视配 `MmDataVisitor`。
2. `processors/` 加音频处理器（按 image 处理器模板）。
3. 模型 layer：`framework/model/causal_vlm.h` 派生 `causal_alm.h`（或扩 visitor）。
4. proto 加 `multimodal.proto` 中 audio 字段。
5. `api_service/mm_service_utils.h` 加解码。

参考：commit `06a28eac`、`acc25afd`、`cc7b060d`（最近的 multimodal refactor）。

---

## 共通建议

- **改完跑 `pre-commit run -a`**；clang-format 会自动格式化。
- 在 commit 前**确认是否需要更新 docs/zh/**；本仓库文档与代码同仓。
- 新加文件**先看是否有 CMakeLists.txt** 要登记。
- 多模型相关改动**务必先 grep 现有 `model_id` 路径**，确认 base vs runtime 语义。
