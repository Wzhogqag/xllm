---
name: xllm-arch
description: xLLM repository architecture map. Use BEFORE any non-trivial exploration of this codebase — it tells you which directory owns which responsibility, the canonical control flow (Server → Master → Engine → Scheduler → Worker → Executor → Model), and where to grep for a given concept. Read this skill whenever the user asks "where is X handled", asks you to add a feature, or asks you to debug behavior crossing layers.
---

# xLLM 架构地图（feat/final_multi_model 分支）

xLLM 是一个面向国产芯片（NPU 优先，CUDA/MLU/MUSA 兼容）的 LLM/VLM/DiT/Rec 多后端推理引擎。本分支在主线之上叠加了**多模型同卡部署**（激活+KV+权重统一池化、权重分层加卸载、优先级/SLO 感知调度）。

## 一、顶层分层（自上而下）

```
xllm.cpp                                # main() — 解析 flags、初始化 XTensor/PhyPagePool、create_master、起 brpc HTTP server
└── api_service/                        # BRPC 服务实现（OpenAI 兼容 + 自定义 ForkMaster/Sleep/Wakeup）
    │   api_service.cpp                 # 路由总入口；维护 masters_ / master_instances_ 多副本表
    │   {chat,completion,embedding,rerank,image_generation}_service_impl.{h,cpp}
    │                                   # 每种业务一个 *ServiceImpl；内部按 base_model_id → LLMModelMasters{vector<Master*>, rr} 派发
    │   anthropic_service_impl.*        # Claude API 兼容
    └── service_impl_factory.h          # 模板化创建
core/distributed_runtime/               # Master / Engine 层（控制面）
    │   master.{h,cpp}                  # Master 基类 + MasterStaus{WAKEUP/LIGHT_SLEEP/DEEP_SLEEP} + create_master/fork_master
    │   {llm,vlm,dit,rec}_master.*      # 4 类后端 Master，run() 起 scheduler loop_thread_
    │   {llm,vlm,dit,rec,speculative}_engine.*  # Engine.step() 是单次 forward 调度入口
    │   dist_manager.*                  # 多节点 brpc 集群协调；UniqueId 派发 / 进程组建立
    │   worker_server.* worker_service.* remote_worker.*  # 远程 worker brpc 服务
    │   collective_service.* comm_channel.* shm_channel.* # 集合通信抽象层
    │   disagg_pd_service*.* pd_ooc_service*.*            # PD 分离专用服务面
core/scheduler/                         # 请求/批次调度（调度面）
    │   scheduler.h                     # Scheduler 抽象接口（add_request / step / generate）
    │   scheduler_factory.cpp           # 策略选择：mix>disagg_pd>chunked_prefill>zero_evict>continuous
    │   continuous_scheduler.*          # 连续批；多优先级队列（waiting + offline）+ DecodePriorityQueue
    │   chunked_prefill_scheduler.*     # 分块预填
    │   mix_scheduler.*                 # ProSched：prefill/decode 混队，多优先级
    │   disagg_pd_scheduler.* pd_ooc_scheduler.*  # PD 分离 / OOC 调度
    │   zero_eviction_scheduler.* fixed_steps_scheduler.* prefill_only_scheduler.*
    │   dit_scheduler.*                 # DiT 动态批
    │   perf_model.* profile/           # 时延预测（profile token budget / step time）
    │   decode_priority_queue.h         # decode 阶段反饥饿优先级队列
    │   async_response_processor.*      # 异步输出回调处理
core/runtime/                           # Worker / Executor 层（执行面）
    │   worker.{h,cpp}                  # Worker 包装（local 或 remote 调用统一）
    │   worker_impl.{h,cpp}             # WorkerImpl 抽象：init_model/step/sleep/wakeup/transfer/layer offload
    │   {llm,vlm,embed,embed_vlm,mm_embed_vlm,rec,eagle3,mtp,speculative}_worker_impl.*
    │   executor.* executor_impl*.*     # 抽象 Executor + Factory
    │   base_executor_impl.*            # 基础执行器
    │   {acl,cuda,mlu}_graph_executor_impl.*  # 各平台图模式执行器
    │   {dit,vlm}_executor*.*           # DiT / VLM 专用
    │   forward_params.h                # ForwardInput/Output、ForwardMetrics 定义
    │   forward_shared_memory_manager.* # 跨进程共享内存输入面（启用 enable_shm 时）
    │   params_utils.{h,cpp}            # 输入/输出参数序列化
    │   xservice_client.*               # XService（etcd 注册/上报）客户端
    │   worker_client.*                 # 本地 / 远程 worker 统一抽象
core/framework/                         # 数据结构/算法/资源管理（共享底盘）
    │   model/                          # 模型抽象：CausalLM/CausalVLM/EmbeddingLM/DiTModel + model_args/input_params
    │   model_context.*  model_loader.* hf_model_loader.* dit_model_loader.*  # 加载 + 上下文
    │   request/                        # Request/Sequence/SequencesGroup + 多模态 MM* 全家桶 + priority_comparator
    │   batch/                          # Batch、BatchInputBuilder（含 RecMultiRoundBatchInputBuilder）+ beam_search + mposition
    │   sampling/                       # Sampler、BeamSearcher、logits_processor
    │   block/                          # 经典 block 化 KV 缓存：BlockManagerPool / HierarchyBlockManagerPool（带 host 层级）
    │   kv_cache/                       # KVCache 张量 + 传输（KVCacheTransfer/Mooncake/SpecKVCache + Store + Hierarchy）
    │   xtensor/                        # 【多模型核心】VMM 虚拟连续显存子系统 — 见 xllm-memory-mgmt skill
    │   prefix_cache/                   # Trie 前缀缓存 + Global 版本 + Upload + 多模态前缀
    │   parallel_state/                 # TP/DP/PP 进程组（NPU/CUDA/MLU/MUSA 各自实现 + ProcessGroup）
    │   eplb/                           # MoE 专家负载均衡（动态调整 redundant experts）
    │   chat_template/                  # 基于 minja 的 jinja 模板
    │   tokenizer/                      # tokenizer + tokenizer_args
    │   state_dict/                     # 权重读取/解析
    │   dit_cache/                      # DiT 缓存优化（feature cache）
    │   quant_args.h                    # 量化参数
core/layers/                            # 模型层硬件实现（npu/cuda/mlu/musa/ilu + common）
    │   qwen2_decoder_layer.* qwen3_*decoder_layer.* qwen{2_5,2,3}_vision_layer.*
    │   npu/  cuda/  mlu/  musa/  ilu/  npu_torch/  common/
core/platform/                          # 平台/设备抽象（device.* stream.* vmm_api.* shared_vmm_allocator.*）
    │   cuda/ npu/                      # 各家私有 API
core/common/                            # 通用：global_flags（所有 FLAGS_xxx）、options、types、metrics、etcd_client、rate_limiter、help_formatter
core/util/                              # 工具：threadpool/timer/uuid/net/hash_util/json_reader/scope_guard/concurrent_queue …
core/kernels/                           # 设备特定 kernel（如 NPU 自定义算子）
models/                                 # 各模型注册（model_registry.* 自动注册到 .h 定义的 model_class）
    │   llm/{deepseek_v2,v3,v32,mtp; glm4*; kimi_k2; llama,llama3; oxygen; qwen2,3,3_moe,3_eagle3,3_embedding; npu/}
    │   vlm/{qwen2_vl,qwen2_5_vl,qwen3_vl,qwen3_vl_moe,qwen2_vl_embedding; npu/}
    │   dit/{pipeline_flux,flowmatch_euler,transformer_flux,t5_encoder,clip_text_model,autoencoder_kl, …}
processors/                             # 多模态预处理（clip/glm4v/qwen2_vl/minicpmv 图像处理器）
parser/                                 # function_call + reasoning detector/parser
function_call/                          # tool call 解析（位于 xllm/function_call/ 顶层）
proto/                                  # gRPC/brpc 协议定义（chat.proto / completion.proto / multimodal.proto / common.proto …）
pybind/                                 # 主要给 launch_xllm.py 用的薄包装
server/                                 # xllm_server.* — 直接被 xllm.cpp 调用注册 brpc 服务
```

## 二、控制流（一次推理请求的完整生命周期）

阅读源码时按这条链跳转。注意 **多模型分支引入了多个 Master 并存** 和 **PageAllocator 单例跨 Master 共享**。

```
HTTP/gRPC 请求 (OpenAI / Anthropic 兼容)
  └─ xllm_server (brpc HttpServer) - server/xllm_server.cpp
      └─ APIService::{ChatCompletions, Completions, Embeddings, Rerank, ImageGeneration, ModelsHttp, ForkMasterHttp, Sleep/WakeupHttp, LinkD2D/UnlinkD2DHttp}
           - api_service/api_service.{h,cpp}
           - 解析 model_id: base#N 形式 → ParseModelReplicaSpec → 选 master_instances_[base#N]
      └─ {Chat,Completion,…}ServiceImpl::process_async_impl
           - 取出 LLMModelMasters{vector<Master*> masters, atomic<size_t> rr}
           - 调 RequestMetricAggregator::get_replica_dispatch_weights → memory-aware 加权选 master
           - 过滤掉 is_model_schedule_blocked / is_sleeping 的副本
           - 必要时调 LayerOffloadManager::trigger_load_for_base_model 唤醒一个 degraded 副本
      └─ LLMMaster::handle_request (or VLMMaster / DiTMaster / RecMaster)
           - distributed_runtime/llm_master.cpp
           - chat_template_->apply（如适用）→ tokenizer encode
           - generate_request 构造 Request 对象（包含 RequestParams、ttft/tpot SLO、priority）
           - scheduler_->add_request(req)
LLMMaster::loop_thread_
  └─ Scheduler::step(timeout) — chunked_prefill_scheduler.cpp / continuous_scheduler.cpp / mix_scheduler.cpp
       ├─ prepare_batch() — 从优先级队列取请求，规划 token_budget / seq_budget；分配 KV 块
       │      KVCacheManager 分两套实现：
       │        BlockManagerPool（默认）/ HierarchyBlockManagerPool（host 层级）
       │        XTensorManagerPool（enable_xtensor=true — 多模型默认路径）
       │      Block 分配走 PageAllocator::alloc_kv_cache_page；按 model_id + dp_rank 计费
       ├─ engine_->step(batches) — Engine::step
       │     └─ LLMEngine::step → prepare_inputs → ScopedEngineForwardAdmission (SLO 准入)
       │           - distributed_runtime/llm_engine.cpp：EngineForwardAdmissionController 单例
       │           - 按 batch 截止期 (deadline = recv_ts + ttft_slo  /  start_ts + tpot_slo*tokens) 排队
       │           - 每 worker_clients_[i]->step_async(raw_forward_inputs[dp_rank])
       │     └─ Worker::step_async (本地直接走 WorkerImpl，远程走 brpc)
       │           - runtime/worker.cpp / worker_client.cpp
       │     └─ WorkerImpl::step (llm_worker_impl.cpp / vlm_worker_impl.cpp / mtp_worker_impl.cpp / mm_embed_vlm_worker_impl.cpp / …)
       │           - update_input_by_last_step_output（schedule_overlap）
       │           - model_executor_->forward → BaseExecutor / AclGraphExecutor / CudaGraphExecutor / MluGraphExecutor
       │                 - runtime/{base,acl,cuda,mlu}_graph_executor_impl.cpp
       │           - sampler_->sample（仅 driver/dp_driver）
       │           - 返回 ForwardOutput
       ├─ process_batch_output / process_sample_output / process_beam_search_output
       └─ AsyncResponseProcessor 异步推送回 OutputCallback
```

**关键文件 → 责任** 速查：

| 想看什么                                | 看哪里                                                          |
| --------------------------------------- | --------------------------------------------------------------- |
| 启动顺序 / FLAGS 注入 / Master 类型选择 | `xllm/xllm.cpp` 函数 `run()`                                    |
| Master 工厂 / fork                      | `core/distributed_runtime/master.cpp` `create_master` `fork_master` |
| HTTP 路由 + 副本选择 + 权重派发         | `xllm/api_service/api_service.cpp` + `*_service_impl.cpp` 中的 `process_async_impl` |
| Engine 单步入口 + SLO admission         | `core/distributed_runtime/llm_engine.cpp` (≈800-1800 行)        |
| 调度策略工厂                            | `core/scheduler/scheduler_factory.cpp`                          |
| 默认调度器 step                         | `chunked_prefill_scheduler.cpp` / `continuous_scheduler.cpp`    |
| KV 块分配（多模型）                     | `core/framework/xtensor/page_allocator.{h,cpp}`（**单例**）     |
| 权重分层加卸载                          | `core/framework/xtensor/layer_offload_manager.{h,cpp}`          |
| 模型优先级 / 副本派发权重               | `core/framework/request/request_metric_aggregator.{h,cpp}`      |
| 全局 FLAGS                              | `core/common/global_flags.{h,cpp}`                              |
| 所有可调选项的 setter/getter            | `core/common/options.h`（`PROPERTY(T, name)` 宏展开自动生成）   |
| 模型注册表                              | `xllm/models/model_registry.{h,cpp}` + `models.h`               |
| 启动脚本（参数样例）                    | `start.sh`（根目录）                                            |
| 中文设计文档                            | `docs/zh/features/*.md`，特别是 `xtensor_memory.md`、`overview.md`、`xllm_service_overview.md` |

## 三、读代码时的"找北针"

- "**这个 FLAG 谁定义/谁用？**" → `grep -nE "DEFINE_(string|int|bool|double|int32|int64|uint32)" xllm/core/common/global_flags.cpp` 找定义；`grep -rn FLAGS_xxx xllm/` 找用处。
- "**这个 Option 怎么传进来？**" → `options.h` 中 `PROPERTY` 宏自动生成同名方法；调用链一律 `options.foo()` 或 `options.foo(value)` 链式赋值。
- "**多模型相关的代码在哪？**" → 任何含 `model_id` 入参的方法都是多模型语境；`PageAllocator` 几乎所有 API 都按 `model_id` 分账。
- "**这个类有几种实现？**" → Worker/Executor/Scheduler 大量使用 Impl 模式，看对应的 `_impl_factory.cpp` 或 `_factory.cpp`。
- "**多平台/多硬件代码在哪？**" → 见 `layers/{npu,cuda,mlu,musa,ilu}/` 与 `platform/{npu,cuda}/`；条件编译用 `USE_NPU` / `USE_CUDA` / `USE_MLU` / `USE_MUSA`。

## 四、命名速查（详见 xllm-conventions skill）

- 抽象基类裸名 `Foo` ↔ 默认实现 `FooImpl` ↔ 工厂 `FooFactory`；Pool 类一般是"所有 dp/worker 的聚合容器"。
- 文件命名一律 `snake_case.{h,cpp}`，类名 `PascalCase`，方法/变量 `snake_case`；成员变量带尾下划线 `foo_`。
- 选项类内部用 `PROPERTY(T, name)` 宏生成（见 `core/common/macros.h`）。
- 多模型 instance id：`base_model_id + "#" + replica_index`（`api_service.cpp::make_model_instance_id`）。
- 状态机：`MasterStaus`（注意：是 `Staus` 拼写而非 `Status`，已固化）= `WAKEUP(0) / LIGHT_SLEEP(1) / DEEP_SLEEP(2)`。

## 五、不要在这里找的东西

- 训练代码 —— xLLM 是**推理引擎**，没有训练。
- Python 模型实现 —— 模型层在 C++（`xllm/models/`、`xllm/core/layers/`）；Python 只是薄入口（`launch_xllm.py`）。
- 用户提示词管理 —— 仅做模板渲染（`chat_template/`），不存历史。
