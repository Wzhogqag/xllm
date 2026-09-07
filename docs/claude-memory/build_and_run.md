---
name: build-and-run
description: xLLM 编译与本地启动姿势（包含 NPU 物理卡选择流程）。
metadata:
  node_type: memory
  type: project
---

## 编译

```bash
python setup.py build
```

直接在 host 上跑即可；NPU 头/库由 `set_env.sh` 提供（start.sh 第 9-10 行已 source）。如果只想跑构建容器，用 `cibuild/build_npu.sh python setup.py build --device a2`。

## 启动

入口脚本：**`start.sh`**（仓库根目录）。改完代码后：

```bash
npu-smi info               # 1. 看哪些物理卡空闲（Memory-Usage 列接近 0、AICore 0%）
# 2. 用文本编辑器把空闲卡 ID 同步到两个位置：
#    - start.sh:13  export ASCEND_RT_VISIBLE_DEVICES=<id>
#    - start.sh:49  phy_ids=(<id>)
#   两处必须**一致**；NNODES=1 时只填一个 id，多节点时空格分隔多个 id
bash start.sh              # 3. 后台起服务，日志落到 ./node_<i>.log
```

**Why:** 该机器多人共用，ID 不能写死；两处只要不一致就会 mismatch 启不来或踩别人的卡。

**How to apply:** 任何"重新跑服务""换一张卡再试""压测前重启 xllm"都按这三步。

## 服务参数速记（来自 start.sh 默认）

- 端口：`START_PORT=13212`（HTTP API），`TRANSFER_START_PORT=25550`（D2D），`HCCL_IF_BASE_PORT=45532`
- 模型：`MODEL_PATH=/export/home/models/Qwen3-8B`（其他可选模型在 32-40 行注释里）
- 主要 FLAGS：`--enable_xtensor=true --priority_level=3 --max_memory_utilization=0.86 --block_size=128 --enable_prefix_cache=true --enable_chunked_prefill=true --kv_cache_transfer_mode=PULL`
- 日志：`./node_<rank>.log`（rank 从 0 起）

## 多副本 / Fork

start.sh 启的是 *单个* base master；多副本要起服务后通过 `/fork_master` HTTP API 派生（见 [[multimodel-mechanism]] 第六节）。

## NPU 僵尸显存（SIGKILL / coredump 后的坑）

xllm 进程被 `SIGKILL`（`pkill -9`）或 segfault coredump 后，**NPU HBM 不会立刻回收**——`npu-smi info` 里那张卡的 Memory-Usage 仍显示几 GB 占用。紧接着复用同一张卡起新进程会 `Insufficient memory for PhyPagePool` / num_pages 变负 / 直接 OOM-during-init。

**Why:** 实测踩坑：副本派发 A/B 的 3×3 实验里，trial 崩溃（coredump）后没等显存释放就起下一 trial，导致 trial 2-9 级联失败。`sleep 5` 不够，coredump 尤其慢。

**How to apply:**
- 清理只用 `pkill -9 xllm`（用户指定命令）；**绝不 `pkill -f xllm`**——会匹配当前 bash argv 自杀。要按 comm+user 过滤时用 `ps -eo pid,user,comm | awk '$2==me && $3=="xllm"'`。
- 起新进程/新 trial 前**轮询等 HBM 真降下来**再上，别用固定 sleep。`bench/replica_dispatch_ab/run_all.sh` 里的 `wait_npu_free()` / `npu_hbm_used_mb()` 就是干这个的（阈值 6000MB，超时 180s）。
- 910C 是 **2 chip / NPU，共 16 张卡（phy 0-15）**；选卡时扫全部 16 张找空闲的，别只盯着 0-7。
- **读 `free` 别看错单位**：本机内存是 **3.0 TiB**（`free -g` 显示 total=3067 是 **GB**，不是 MB）；available 常有 2+ TB。别把 3067 误读成 3GB（我踩过，据此误判了根因）。

## ⚠️ PhyPagePool 大小在启动瞬间锁定 + 跨容器显存争抢

**症状 A**：fork#1 分配 8B 权重时 `page_allocator.cpp:991] Not enough physical pages for weight allocation: requested 7822, min_available 1722`。日志上游有 `XTensor mode: available memory from PhyPagePool: 20.12 GB (pages: 10299)` —— 池子只建了 20GB 而非满 62GB，放下 fork#0 后就不够 fork#1。
**症状 B**：node 启动最早期 `xllm.cpp:320] Failed to initialize PhyPagePool` 直接崩，模型都没加载。

**根因**：`--max_memory_utilization=0.86` 按 xllm **进程启动那一瞬间**卡上的空闲 HBM 计算 PhyPagePool 大小，**之后锁定不变**。本机 NPU 是**跨容器共享物理硬件**，别的容器的负载 HBM 占用随时波动。如果某张卡在你启动瞬间正被别人占着（哪怕 npu-smi 下一秒又显示 0MB），你的池子就建小了 → 放不下 2 个 8B 副本 → fork#1 必失败；极端情况连 PhyPagePool 都建不起来。

**How to apply**：
- **`npu-smi` 此刻显示 0MB ≠ 这张卡可靠**。别人的负载是脉冲式的。选卡要挑**持续多次采样都空闲**的卡，别信单次快照。
- **已验证稳定的卡**（同一实验里成功建满 62GB 池、跑通多 trial 的）远比"npu-smi 现在 0MB"的新卡可靠。换卡不如复用验证过的卡串行重跑。
- `wait_npu_free` 阈值 6000MB 太松——卡上有 20GB 占用时（<6000 的判断基于别的采样口径可能漏）仍可能放行导致池子建小。真要稳，启动后立刻 grep `available memory from PhyPagePool` 确认拿到了预期 GB 数，不够就换卡重启。
- 教训：副本派发 A/B 并行跑时，random 流在 phy2,3 和 phy14,15 反复栽（池子建小 / PhyPagePool 初始化失败），而 phy0,1 / 4,5 稳定成功。别盲目换卡试错，回到验证过的卡。
- **池太小崩溃 → 客户端 `wait_until_ready` 死等 900s 的坑**：占位模型池建小(如只 247 页)时，fork 目标模型在 `page_allocator.cpp:991 Not enough physical pages for weight allocation` 崩溃(`llm_master.cpp:52 Check failed engine_->init`)，但 fork 客户端的 `wait_until_ready(timeout_s=900)` 会**死等到 900s 超时**才返回失败、才触发 run_all 的重试 —— 白白卡 15 分钟。**应对**：发现某流长时间卡在 `waiting for node_0 /v1/models` 且 node_0.log 已有 `Check failed engine_->init` / `main process disappeared`，直接 `kill -9` 该流端口的 xllm(`pgrep -f "port $P"`)，让 `wait_until_ready` 立即连接失败 → run_all 重试逻辑接管，在此刻空闲的卡上重建满池子。实测这样能立即恢复(attempt=2 SUCCESS pool=52GB)，不用干等超时。根治要缩短 `xllm_client.py` 的 `wait_until_ready` 超时或加"node_0 进程已死"探测。

## ⚠️ mlock 墙：host_blocks_factor>1 + 并行多实例必崩

**症状**：fork 第 2 个副本（fork#1）时 node 崩溃，日志 `hierarchy_kv_cache_transfer.cpp:534] aclrtHostRegister fail: 207001` → `create_page_aligned_host_cache` → `Failed to initialize kv cache` → `fork_master` CHECK 失败。

**根因**：`--host_blocks_factor>1`（如 2）时，每个副本要在 host 上 `mmap+mlock+aclrtHostRegister` 一块 page-aligned 的 host KV 镜像。而容器 **`ulimit -l`（锁页内存）只有 64MB**（`ulimit -l` = 65536 KB）。单实验勉强过，**并行跑多个实例（或同一实例 fork 多副本）叠加 mlock 立即超限**。不是物理内存不足（内存有 3TB），是**锁页配额**不足。

**How to apply（跑副本派发 A/B 这类多实例实验）**：
- **设 `--host_blocks_factor=1`**（默认本就是 0.0，纯 device KV，无 host 镜像）。经 subagent 读码确认：`>1` 才走 `HierarchyBlockManagerPool` + host offload；`=1` 完全跳过 mlock。**对 dispatch 实验无语义影响**——device 端 KV 分配、`worker_pages_used_` 计费、`get_replica_dispatch_weights` 派发权重全部与 host_blocks_factor 无关（`page_allocator.cpp:170-171` 只统计 device 侧）。唯一副作用：KV 超压时无 host offload 兜底，可能更早 preemption/503，但三策略同等施加不破坏对照。
- 想保留 host offload 又要并行：得 `ulimit -l unlimited`（需权限）或错开各流的 fork 时间。一般实验不值当，直接 `=1`。

## 并行跑多策略（效率）

`bench/replica_dispatch_ab/launch_parallel.sh`：3 策略各占独立卡对（phy 0,1 / 2,3 / 4,5）+ 独立端口块（132xx/133xx/134xx）+ 独立 master_node_addr（99xx 段）+ 独立 /dev/shm key（按 port-node_rank 天然隔离），setsid 脱离。墙钟 ≈ 单流（~35-40 min 跑 3 trial），而非 9 trial 串行。清理按**端口 scope** `pgrep -f "port $P"` + `kill -9`（HCCL 卡死的 node_1 忽略 SIGTERM），不误伤兄弟流。

## 实验分支（exp/replica-dispatch-ab）追加 FLAGS

跑副本派发 A/B 时在 xllm 命令行尾部追加：
```
--replica_dispatch_policy={memory_inverse|random|slo_weighted}
--replica_dispatch_slo_alpha_x100=100      # 仅 slo_weighted
--dispatch_viol_window_ms=1000
--experiment_jsonl=/path/exp.jsonl
```
不加这些 FLAG 时行为与 main 完全一致（policy 默认 memory_inverse，sink 关闭）。
