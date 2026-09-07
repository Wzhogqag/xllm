---
name: container-pid-isolation
description: 运行环境是带独立 PID namespace 的 Docker 容器——pkill -9 xllm 安全（够不到别的容器/宿主进程）；但 NPU 显存是跨容器共享硬件，选卡仍需避让别人。
metadata:
  node_type: memory
  type: project
---

xLLM 开发环境是一个 **Docker 容器，带独立 PID namespace**（实测确认：`/.dockerenv` 存在、PID 1 = bash、`/proc/self/ns/pid` 与 `/proc/1/ns/pid` 同属容器专属 namespace `pid:[4026...]`、`ps -e` 只可见 ~43 个进程）。

**How to apply:**

1. **`pkill -9 xllm` 是安全的**（用户当初给的清理命令就是它）。`pkill`/`kill` 只在本容器 PID namespace 内遍历 `/proc`，**物理上够不到其他容器 / 宿主机的进程**，不会误伤别人。清理自己的 xllm 直接 `pkill -9 xllm` 即可，不必再搞 `ps ... $user==me` 那套按 user 过滤（而且本环境里所有进程都显示 root，按 user 过滤本来也无意义）。

2. **不要被 npu-smi 的 PID 迷惑**：`npu-smi info` 的 "Process id" 列是**宿主机视角的 PID**（NPU 驱动在内核层，不受 PID namespace 约束），那些大 PID（如 2455093）在容器内 `ps`/`kill` 里根本不存在，你既看不到也 kill 不到——它们是**别的容器**的进程。

3. **进程隔离 ≠ 显存隔离**：NPU 是**跨容器共享的物理硬件**。别的容器的 xllm 会真实占用 HBM，`npu-smi` 能看到全局占用。所以：
   - **选卡前必须看 npu-smi 全局占用**，避让别的容器正在用的卡（这是"不影响别人"的真正含义——争抢显存而非杀进程）。
   - 我自己实验的 `wait_npu_free` 轮询避让逻辑仍然必要（防跨容器抢占 + 防自己上一轮的僵尸显存）。

**Why:** 一次实验清理时我误判成"机器全 root、pkill 会杀别人进程"，基于宿主机心智模型。用户提醒"我在容器里"，实测后证明进程被 PID namespace 隔离、pkill 安全；但显存确实跨容器共享。两件事不矛盾：**显存共享、进程隔离**。

关联：僵尸显存与选卡见 [[build-and-run]]；长任务脱离见 [[long-running-tasks-detach]]。
