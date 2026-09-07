---
name: xllm-conventions
description: Naming, code style, and codebase conventions for xLLM. Read before writing or refactoring any C++/Python in this repo — it covers the PROPERTY macro, file/class/method naming, namespace rules, Impl pattern, FLAGS/Options/CHECK conventions, build & lint setup, and the gotchas (e.g. `MasterStaus` spelling, `model_id` semantics).
---

# xLLM 代码规范速查

## 一、文件与命名

- **C++ 文件**：`snake_case.{h,cpp}`。一个类一对文件（极少多类同文件，例外是 `priority_comparator.h` 这类小 struct 集合）。
- **类**：`PascalCase`（`PageAllocator`, `LayerOffloadManager`）。
- **函数 / 方法**：`snake_case`（`alloc_kv_cache_page`, `wait_and_mark_model_step_begin`）。
- **变量**：
  - 局部 / 形参：`snake_case`（`num_pages`, `dp_rank`）。
  - 类成员：`snake_case_` 末尾下划线（`model_states_`, `worker_pages_used_`）。
  - 静态常量 / 全局常量：`kPascalCase` 或 `ALL_CAPS_SNAKE`（混用，比如 `MIN_RESERVED_PAGES`、`PAGE_PREALLOC_ENABLED`）。
- **枚举**：`PascalCase` 类型 + `ALL_CAPS` 项（`MasterStaus { WAKEUP, LIGHT_SLEEP, DEEP_SLEEP }`，注意 `Staus` 拼写延续）。
- **命名空间**：`xllm`（顶层），`xllm::util` / `xllm::xtensor` / `xllm::runtime` 等子模块。
- **Python**：`snake_case.py`，函数/变量 `snake_case`，类 `PascalCase`。yapf 风格（见 `.style.yapf`）。

## 二、Options 类与 `PROPERTY` 宏

`core/common/macros.h`：

```cpp
#define PROPERTY(T, property)                                                 \
 public:                                                                      \
  [[nodiscard]] const T& property() const& noexcept { return property##_; }   \
  [[nodiscard]] T& property() & noexcept { return property##_; }              \
  [[nodiscard]] T&& property() && noexcept { return std::move(property##_); } \
  auto property(const T& value) & -> decltype(*this) { property##_ = value; return *this; } \
  auto property(T&& value) & -> decltype(*this) { property##_ = std::move(value); return *this; } \
  void property(const T& value) && = delete;                                  \
  void property(T&& value) && = delete;                                       \
  T property##_
```

用法：
```cpp
class Options { public:
  PROPERTY(std::string, model_path);
  PROPERTY(int32_t, max_tokens_per_batch) = 20480;   // 默认值
  PROPERTY(std::optional<std::string>, devices);
};
// 调用：
Options opts;
opts.model_path("…").max_tokens_per_batch(8192);   // 链式
opts.model_path();                                  // getter
opts.model_path() = "…";                            // 直接改引用
```
**不要写自己的 setter/getter**，全靠这个宏。

## 三、Impl + Factory + Pool 模式

xLLM 大量复用三件套：
- `Foo`：用户面公开抽象（pure virtual / 暴露最小 API）。
- `FooImpl` / 具体派生（如 `LlmWorkerImpl`, `ChunkedPrefillScheduler`）：实现细节。
- `FooFactory` / `FooFactory::create(...)`：根据 FLAGS 或 Options 选具体实现。
- `FooPool` / `FooManagerPool`：聚合多个 Foo（按 dp_rank / worker_rank / device）。

例：`Worker(裸抽象)` → `WorkerImpl(基类)` → `LlmWorkerImpl/VlmWorkerImpl/MtpWorkerImpl/...`；`Executor` → `BaseExecutorImpl` → `AclGraphExecutorImpl/CudaGraphExecutorImpl/MluGraphExecutorImpl/DiTExecutor`；`Scheduler` → `ContinuousScheduler` → `ChunkedPrefillScheduler/MixScheduler/...`；`BlockManager` → `BlockManagerImpl/ConcurrentBlockManagerImpl/XTensorBlockManagerImpl` → 由 `BlockManagerPool/HierarchyBlockManagerPool/XTensorManagerPool` 聚合。

## 四、FLAGS（global_flags）

- 文件：`core/common/global_flags.{h,cpp}`（**所有** gflag 都在这两个文件）。
- 命名：`FLAGS_<snake_case>`；定义用 `DEFINE_<type>(name, default, help)`；声明在 `.h` 里用 `DECLARE_<type>(name);`。
- 改默认值前先看是否被 `start.sh` / docker / CI 覆盖。
- 新增 FLAG 后**记得在 .h 也加 DECLARE**，否则其他翻译单元用不到。
- `--help` 自定义：`xllm/core/common/help_formatter.h`。

## 五、Logging / CHECK / Macros

- 用 `LOG(INFO|WARNING|ERROR|FATAL) << ...` (glog)。`LOG(FATAL)` 会终止进程，仅用于真无法继续。
- 高频路径用 `VLOG(level)`，配 `--v=N`。
- 断言：`CHECK(...)`, `CHECK_EQ/NE/GT/...`, `DCHECK*`（Release 不生效）。
- 自定义宏（`core/common/macros.h`）：
  - `DISALLOW_COPY_AND_ASSIGN(T)`：禁止拷贝。
  - `NOT_IMPLEMENTED()`：抛 LOG(FATAL)。
  - `CALLBACK_WITH_ERROR(code, msg)`：常用于 API 层回调。
  - `CHECK_ACL_SUCCESS(expr, msg)`：NPU ACL 调用包装。
  - `ADD_VECTOR_TO_PROTO(proto_field, vec)`：proto 批量塞。

## 六、错误处理 / 异常

- C++ 习惯 **不抛异常**（构建关 `-fno-exceptions` 的代码路径少，但风格上避免）。
- 失败用 bool 返回 + LOG(ERROR)；不可恢复用 LOG(FATAL)。
- 唯一被广泛抛的：`ForwardInterruptedException`（layer 中断时）。
- folly `Try<T>` / `SemiFuture<T>` 用于 worker_client_ 异步调用。

## 七、并发 / 同步

- 锁：`std::mutex` / `std::shared_mutex` 命名 `mtx_` / `xxx_mtx_`。
- 条件变量：`std::condition_variable` 命名 `cond_` / `cv_xxx_`。
- 原子：`std::atomic<…>`，命名 `xxx_` 同普通成员；初始化用 `{}` 形式 `std::atomic<bool> running_{false};`。
- 自定义同步工具：`util/blocking_counter.h`, `util/closure_guard.h`, `util/spin_lock.h`, `util/double_buffer.h`, `util/concurrent_queue.h`（moodycamel）, `util/lightweightsemaphore.h`。
- 线程池：`util/threadpool.h` —— 每个池命名要传到 `ThreadPool(name, size)`，避免 monitor 时混乱（commit `ddcb539d`）。
- 异步 API 习惯命名 `xxx_async` 返回 `folly::SemiFuture<T>`；同名同步版直接 `.get()`。

## 八、Proto / RPC（brpc）

- 文件：`xllm/core/{api_service,distributed_runtime}/*.proto`（编译进 `*.pb.{h,cc}`）。
- 服务类继承 `proto::XxxService`，方法签名 `(controller, request, response, done)`。
- HTTP 路由：`*Http` 后缀的方法是 brpc 的 HTTP gateway 版，内部转回 protobuf。
- arena: `auto arena = response->GetArena(); auto req_pb = google::protobuf::Arena::CreateMessage<...>(arena);`。
- JSON↔PB：`json2pb::JsonToProtoMessage` / `ProtoToJsonMessage`。

## 九、构建 / 第三方依赖

- 顶层 `CMakeLists.txt` + 每子目录小 `CMakeLists.txt`；用 `add_library`/`target_link_libraries` 组合。
- 平台条件：`USE_NPU`, `USE_CUDA`, `USE_MLU`, `USE_MUSA`（在 cmake 与代码 `#if defined(USE_NPU)`）。
- 第三方 submodule（`.gitmodules`）：brpc / cpprestsdk / minja / sentencepiece / smhasher / xllm_ops / etcd_cpp_apiv3 / spdlog / Mooncake / torch_npu_ops。
- vcpkg.json 管理 host 依赖。

## 十、Pre-commit / Lint

- `.pre-commit-config.yaml`：clang-format（cpp）+ yapf（py）。
- `.clang-format`：Google base, indent 2, column 80, `PointerAlignment: Left`。
- `.style.yapf`：Google style for Python。
- 提交前推荐 `pre-commit run -a`，CI 会复跑。

## 十一、Commit message 风格（必看）

本仓库严格遵循前缀：
- `feat:` 新功能
- `bugfix:` 修 bug
- `refactor:` 重构
- `optimization:` 性能优化

短描述用动词原形 + 必要时点号分多条用阿拉伯数字编号（参考 `e5d15d43` 的 6 条 refactor）。常见后缀引用：`(#1234)` 引用 PR；多步联动用 `(1/3)`, `(2/3)`, `(3/3)`。

## 十二、最易忽视的坑

1. `MasterStaus`（不是 Status）—— 全仓拼写一致，不要"修正"。
2. `model_id` 在 PageAllocator 里默认是 **runtime ID**（`base#N`），在 API 入参里默认是 **base ID**；中间在 `api_service.cpp::make_model_instance_id` 转换。
3. 多模型场景共享单例：`PageAllocator::get_instance()`、`GlobalXTensor::get_instance()`、`XTensorAllocator::get_instance()`、`RequestMetricAggregator::instance()`、`EngineForwardAdmissionController::instance()`、`DeviceMonitor::get_instance()`、`InstanceName::name()`、`InterruptionBus::get_instance()`。**不要为某个 Master 持有独立副本**。
4. `enable_prism && !enable_prism`（在 `llm_engine.cpp` 多处）：是 unify 后的占位，运行时不触发；如新增逻辑要触发，要先决定 prism 是开是关。
5. fork_master 强依赖 `FLAGS_enable_xtensor`，否则直接返回 nullptr。
6. NPU pluggable allocator 与 prism 互斥：见 `xllm.cpp` 中 `if (FLAGS_enable_xtensor && !FLAGS_enable_prism)`。
7. C 模板风格：避免引入新模板复杂度；现有大量代码继承自 ScaleLLM（双 Copyright Header），改时保留版权头。
8. License header：每个新 .h/.cpp 都要带 `Copyright 2025/2026 The xLLM Authors` Apache-2.0 头（看任意现有文件第 1-14 行复制粘贴）。

## 十三、推荐文件作为模板

- **新增一个 Scheduler**：参考 `chunked_prefill_scheduler.{h,cpp}`。
- **新增一个 WorkerImpl**：参考 `llm_worker_impl.{h,cpp}`。
- **新增一个 Executor**：参考 `base_executor_impl.{h,cpp}`。
- **新增一个 API 路由**：参考 `api_service.cpp::ChatCompletionsHttp` + `chat_service_impl.cpp::process_async_impl`。
- **新增一个 model 注册**：参考 `xllm/models/llm/qwen3.h` + `models.h` 注册宏。
- **新增一个 layer 实现**：参考 `xllm/core/layers/qwen3_decoder_layer.h` + 平台目录下 `*_impl.{h,cpp}`。
