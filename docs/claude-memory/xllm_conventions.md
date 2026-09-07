---
name: xllm-conventions
description: xLLM 代码规范、命名约定、Options/PROPERTY 宏、Impl/Factory/Pool 模式、易踩坑拼写。
metadata:
  node_type: memory
  type: project
---

## 命名

- 文件：`snake_case.{h,cpp}`，一类一对文件。
- 类：`PascalCase`；函数/方法/局部变量：`snake_case`；类成员带末尾下划线 `foo_`。
- 枚举：类型 `PascalCase`，项 `ALL_CAPS`。
- 命名空间：`xllm`（顶层），子模块如 `xllm::util` `xllm::xtensor`。

## Options/PROPERTY 宏

`core/common/macros.h` 中：

```cpp
PROPERTY(T, name);   // 自动生成 const& / & / && getter + 链式 setter
```

用法 `opts.name(value).other(...)`。所有 Options 类内字段都用这个宏，不要写自己的 setter/getter。

`Scheduler::Options` / `BlockManagerPool::Options` 等内嵌 Options 同样。

## Impl + Factory + Pool 模式

- `Foo`：抽象基类（pure virtual）
- `FooImpl` 或具体派生：实现
- `FooFactory` 或 `create_foo`：工厂
- `FooPool` / `FooManagerPool`：多 dp/worker 聚合

例：`Scheduler` → `ContinuousScheduler` → `ChunkedPrefillScheduler/MixScheduler/...`；`Worker` → `WorkerImpl` → `LlmWorkerImpl/VlmWorkerImpl/...`。

## FLAGS

- 文件：`xllm/core/common/global_flags.{h,cpp}`（所有 gflag 集中在此）。
- 加新 FLAG 务必同时在 `.h` 加 `DECLARE_<type>`、`.cpp` 加 `DEFINE_<type>`，否则其他模块取不到。

## 日志 / 断言

- glog：`LOG(INFO|WARNING|ERROR|FATAL)`，`VLOG(level)` 配 `--v=N`。
- `CHECK_*` 总是 enable；`DCHECK_*` 仅 Debug。
- 自定义宏：`DISALLOW_COPY_AND_ASSIGN`, `NOT_IMPLEMENTED`, `CALLBACK_WITH_ERROR`, `CHECK_ACL_SUCCESS`。

## 并发

- 锁：`std::mutex` 命名 `mtx_`；条件变量 `cond_`/`cv_`；原子加初始化大括号。
- 异步：`folly::SemiFuture<T>`，命名 `xxx_async`；同步版直接 `.get()`。
- 工具：`util/threadpool.h`（要传 pool name，commit `ddcb539d`），`util/blocking_counter.h`, `util/concurrent_queue.h`(moodycamel)。

## License Header

每个 .h/.cpp 都加 `Copyright 2025/2026 The xLLM Authors` Apache-2.0 头；多数文件还有 `Copyright 2024 The ScaleLLM Authors`（继承自上游）。

## Commit 前缀

`feat:` `bugfix:` `refactor:` `optimization:` —— 严格遵守。多步联动用 `(1/3)/(2/3)/(3/3)`。

## 必记的坑

1. `MasterStaus`（不是 Status）—— 全仓拼写一致，不要"修正"。
2. `model_id` 在 PageAllocator 内部是 runtime ID (`base#N`)，API 入参常是 base，通过 `master_instances_` 二级映射。
3. 单例：`PageAllocator/GlobalXTensor/XTensorAllocator/RequestMetricAggregator/EngineForwardAdmissionController/DeviceMonitor/InstanceName/InterruptionBus` 都是 master 进程级单例，不要 per-Master 持有副本。
4. `enable_prism && !enable_prism` 是 unify 后的占位，运行时不触发；改前先 `git blame`。
5. `fork_master` 强依赖 `FLAGS_enable_xtensor=true`。
6. NPU pluggable allocator 与 prism 互斥。

## 风格工具

- `.clang-format`：Google base, indent 2, column 80, `PointerAlignment: Left`。
- `.style.yapf`：Python Google style。
- `.pre-commit-config.yaml` 已配，提交前 `pre-commit run -a` 自动 fix。

**Why:** 写代码不照规范，PR 一定被打回；尤其本仓库有几处反直觉的命名（Staus）不能"修正"。

**How to apply:** 改/加任何文件前先扫一眼这条；新文件复制现有文件的版权头与命名风格。详见 skill `xllm-conventions`。
