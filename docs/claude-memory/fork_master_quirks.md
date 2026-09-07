---
name: fork-master-quirks
description: "fork_master HTTP API 的真实语义 + 经实验验证的跨卡多副本部署配方（内存-均衡 A/B 实验中打通）。含 trigger_offload 反向语义、bounds-check、以及为什么早期'同卡 fork 必 segfault'的结论其实是拓扑用错。"
metadata:
  node_type: memory
  type: project
---

`POST /fork_master`（`api_service.cpp::ForkMasterHttp`）在这个分支上有几个非显然点，
**副本派发 A/B 实验里已经把正确用法跑通**（`bench/replica_dispatch_ab/run_trial.sh`）。

## ✅ 经验证的正确用法：跨卡多副本部署配方（canonical pattern）

想在 N 张卡上各放一个目标模型副本（`Qwen3-8B#0` 在卡0、`Qwen3-8B#1` 在卡1 …），
**不是**直接对目标模型 fork N 次，而是：

1. **起 N 个 xllm 进程，每卡一个**，全部加载**同一个占位模型**（如 Qwen3-0.6B），
   `--nnodes=N --dp_size=1`，构成 world_size=N 的 collective。
   - `node_rank=0` 进程暴露 HTTP API（`/v1/models` + `/fork_master`）
   - `node_rank≥1` 进程**只暴露 `/fork_master`**（`xllm_server.cpp:38-60`）——探活得用裸 TCP，不能 GET /v1/models
   - 占位模型的作用：占住 collective 的 N 个 slot + 让 PageAllocator 预留好 N 个 worker 的计费槽。
2. **对目标模型 fork N 次**，第 k 次用 `worker_rank=k`，**每次一个 distinct `master_node_addr`**（:9930, :9931 …），
   且**同一个 fork 的 POST 要同时发给全部 N 个 xllm 进程**（`asyncio.gather`）。
   - node_0 进程：在该 addr 上起 **CollectiveServer**（HCCL 交汇点）
   - node_k 进程：起一个 WorkerServer 线程去连那个 addr
   - `dist_manager.cpp:181-187` 按 `actual_rank/each_node_ranks == node_rank` 过滤，
     每个进程只建自己那张卡的 worker，彼此在 collective 上会合。
   - 只有 `node_rank==0` 的 master 进 dispatch 表（`api_service.cpp:960-963`）。
3. fork 完 **DEEP_SLEEP 占位模型**（`/sleep` status=2）释放其权重，留下 N 个目标副本各占一卡。
4. 压测发给 `model="Qwen3-8B"`，dispatch 在 `Qwen3-8B#0/#1` 间按策略选。

**Why placeholder：** fork 的 bounds check（见下）要求 `worker_rank + nnodes ≤ base 的 world_size`；
不先用一个 `nnodes=N` 的 base 把 world 撑到 N，`worker_rank=1` 这样的 fork 过不了检查。
**Why distinct addr：** 每个 fork 会起自己的 CollectiveServer，复用 base 的 :9929 会 bind 冲突。

## 1. Bounds check（`api_service.cpp:855` / `dist_manager.cpp:143`）

```cpp
CHECK_LE(worker_rank_base + world_size, physical_world_size);  // dist_manager.cpp:143
```
`physical_world_size = each_node_ranks * nnodes`。所以 `worker_rank` 能给多大，**取决于 base 起了多大的 world**。
单卡 base（nnodes=1）只有 `worker_rank=0` 过得了；要 `worker_rank=1` 就得 base nnodes≥2。

## 2. `trigger_offload` JSON 字段语义反向（最坑）

`api_service.cpp:896`：`if (node_rank==0 && has_trigger_offload() && !trigger_offload()) { load_model(path); return; }`

- JSON 里写 `"trigger_offload": false` **不是**"fork 出来别 offload"，而是 **"这是个 reload 命令"**（短路 fork）。
- 想 **fork 一个 awake 副本：JSON 里绝对不要出现 `trigger_offload` 字段**（present-and-false 都不行）。
- `xllm_client.fork_replica` 默认 omit；要 fork+offload 传 `True`，纯 reload 传 `False`。

## 3. ⚠️ 订正：早期"同卡 fork 必 SIGSEGV"是**拓扑用错**，不是 fork 本身的 bug

实验初期我误判成"同节点同卡 fork 必段错误"。**真相**：当时 base 用 `nnodes=1`（单卡 world_size=1），
却想 fork 到已被 base 占用的 `:9929` → 新 DistManager 建不起 collective → `tp_size=0` → KV 除零 → SIGSEGV。
根因是**没有用 placeholder 把 world 撑起来 + 复用了 base 的 addr**，
按上面的正确配方（nnodes=N 占位 + distinct addr）就不会段错。段错的日志特征（留作诊断锚点）：
```
Set model parallel strategy for Qwen3-8B#0: dp_size=1, tp_size=0, worker_rank_base=0, world_size=0
E xtensor_allocator.cpp:744] KV tensors not created for model Qwen3-8B#0
Segmentation fault (core dumped)
```
看到 `tp_size=0 / world_size=0` 就是 collective 没建起来——查 addr 冲突或 bounds/placeholder。

## 关联

- 完整可跑脚本：`bench/replica_dispatch_ab/run_trial.sh`（占位+双 fork+DEEP_SLEEP）、`xllm_client.py`（fork_replica/wait_tcp_open/sleep_model）。
- 进程/HCCL 拓扑细节见 skill `xllm-multimodel` 的"进程模型 & HCCL rendezvous"节。
- 长实验脱离进程组见 [[long-running-tasks-detach]]；NPU 僵尸显存见 [[build-and-run]]。
