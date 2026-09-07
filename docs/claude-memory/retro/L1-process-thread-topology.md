---
name: l1-process-thread-topology
description: "多模型 colocation 的进程/线程/RPC 拓扑(代码实测):单卡=1进程,每个 fork 出的模型=1个 Master(含 Engine + scheduler 线程),worker 是隔着 brpc 的独立服务单元(单机=brpc server 线程,离线多卡=posix_spawn 进程)。真正的竞态不在 engine↔worker(有 RPC 隔离),而在多 Master 共享同进程的 PageAllocator/GlobalXTensor 单例。第一层所有竞态笔记的公共背景。"
metadata:
  node_type: memory
  type: project
  tags:
    - retro
    - multimodel
    - architecture
    - topology
    - layer1
---

# 进程 / 线程 / RPC 拓扑(第一层背景板)

回答"谁和谁真并发"——这是 [[L1-global-xtensor-ring]]、[[L1-free-offset-dual-meaning]]、[[L1-map-page-check-then-act]]、[[L1-online-migration-race]] 所有竞态的公共前提。**代码实测,非记忆**。

## 单机单卡默认拓扑

```
单卡 = 1 个进程
└── 进程内多个线程:
    ├── Master A ── Engine A ── scheduler_thread_ A  ┐
    ├── Master B ── Engine B ── scheduler_thread_ B  ┤ fork 出的每个模型一个 Master,
    ├── Master C ── ...                              ┘ 同进程共存
    │
    ├── WorkerServer 线程(内含 brpc server)
    │     ↑ engine 侧通过 RemoteWorker + brpc channel RPC 调用(即使同进程也隔着 RPC!)
    │
    └── 共享单例(多 Master 线程 + 后台线程真并发访问):
        ├── PageAllocator(master 侧单例)
        ├── GlobalXTensor(worker 侧单例)+ unmap_thread_ + migration(线程池)
        └── PageAllocator 的 async eviction / prealloc 线程
```

## 关键事实(逐条对应代码)

| 组件 | 载体 | 代码 |
|---|---|---|
| 每个模型一个 Master | fork 出的 `Master`,同进程共存 | `master.cpp:302 fork_master` |
| Engine | Master 持有,同线程上下文 | `master.cpp:245` |
| Scheduler | **独立线程** | `llm_engine.cpp:761 std::thread scheduler_thread_` |
| Worker(单机默认) | **brpc server 线程**,不是普通线程 | `worker_server.cpp:256 "start worker in a thread"` |
| Worker(离线多卡) | **posix_spawn 独立进程** | `worker_server.cpp:237 create_spawn_server` → `:181 posix_spawnp` |
| engine↔worker 通信 | **brpc RPC**(RemoteWorker + channel),即使同机 | `dist_manager.cpp:249-253` |
| 同机 worker 数据面 | shared memory(`/dev/shm`) | `worker_server.cpp:250-253 prepare_shm` |

## 两处最易记错的点(校准)

1. **worker 不是"普通线程直调",而是隔着 brpc 的独立服务单元。** 单机默认它是"跑了个 brpc server 的线程",engine 经 `RemoteWorker` + channel RPC 调它;离线多卡才 `posix_spawn` 成真进程。**engine↔worker 之间永远有 RPC 边界**,不是共享内存函数调用。
2. **多机(nnodes>1)时 worker 才跨物理节点**;单机 nnodes=1 时 worker 是同进程 brpc 线程。

## 为什么这决定了竞态在哪(串起整个第一层)

- **engine↔worker 有 RPC 隔离** → 这条链**不是**竞态来源(类似 vLLM 用消息边界隔离 engine/worker)。
- **真正的竞态在共享单例层**:多个 Master 线程 + unmap/migration/eviction 后台线程,**同进程真并发**读写 `PageAllocator` / `GlobalXTensor`。这就是难点2/3([[L1-map-page-check-then-act]]、[[L1-online-migration-race]])的根。
- **对比 vLLM/SGLang**:它们把 KV/block 分配收在**单个 scheduler 线程/进程**里串行做(单写者,消灭竞态);而这里是**多 Master 共享同进程单例**,被迫用锁 + 手写 lazy unmap 对抗竞态。这是"没能守住单写者"的直接后果。

关联:心智模型 [[L1-global-xtensor-ring]];竞态后果 [[L1-online-migration-race]]、[[L1-map-page-check-then-act]];底盘 [[xtensor_memory]]、机制总览 [[multimodel_mechanism]]。
