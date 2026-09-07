---
name: long-running-tasks-detach
description: 跑超过 10 分钟的后台任务时，默认用 setsid+nohup 脱离 Claude Code 进程组，避免 session 结束/ssh 断开 cascade 杀掉任务。
metadata:
  node_type: memory
  type: feedback
---

任何运行时间超过 **~10 分钟** 的后台任务（实验、长压测、批量数据处理、CI 构建等），**不要**直接用 `Bash run_in_background=true` —— 这种方式起的子进程跟 Claude Code agent 同生死。一旦 Claude Code 退出（用户 Cmd+Q、ssh 断、Mac 睡眠等），SIGHUP/SIGTERM 会 cascade 杀掉整条 process chain，**任务半途死掉、JSONL 截断、必须重跑**。

**Why:** 实测踩坑：dual-card 3×3 实验跑了 9 分钟（trial 1 measure phase 中段），用户问"锁屏会不会断"，才意识到 `Bash run_in_background=true` 的所有 child 都属于 Claude 的进程组。

**How to apply:**

**默认姿势 — 用 `setsid nohup`**（不只是 `nohup`！）：
```bash
setsid nohup bash -c "<your long cmd>" > /path/to/log 2>&1 < /dev/null &
echo $! > /path/to/pid
```
- `setsid` → 新 session + 新 pgid，跟父进程组解耦
- `nohup` → 额外挡 SIGHUP
- `< /dev/null` → 切断 stdin（防 EIO）

**验证脱离成功**：
```bash
ps -o pid,ppid,sid,pgid -p $PID
# 期望 PPID=1（init）、SID=PGID=$PID（自己当头）
```

**对 xLLM 实验**：已写好 `bench/replica_dispatch_ab/launch_detached.sh`，封装了 setsid+nohup+pid 落盘+kill 提示，未来跑全量实验直接调它而不是 `bash run_all.sh`。

**杀任务时**：
- 杀 launcher 用进程组（负 pid）：`kill -- -<PGID>`；若 launcher 无子进程（处在 trial 间隙）则直接 `kill -9 <pid>`。
- **清 xllm 直接 `pkill -9 xllm` 即可，安全**——本环境是带独立 PID namespace 的 Docker 容器，pkill 够不到别的容器/宿主进程，不会误伤别人（见 [[container-pid-isolation]]，实测确认）。
- **`-f` 自匹配自杀坑（`pkill -f` 和 `pgrep -f` 都中招）**：`-f` 匹配完整命令行。只要你用来清理的那条命令的 argv 里**出现了同一个模式串**，就会匹配到自己 → `pkill -f` 直接自杀（exit 144/143），`pgrep -f` 也会把自己算进去。本会话两个都踩了：`pkill -f "ROOT=/tmp/xllm_ab_smoke"`、`pgrep -f rerun_random` 都自匹配。
  - **规避 1**：清 xllm 用不带 `-f` 的 `pkill -9 xllm`（按 comm 精确匹配，最稳）。
  - **规避 2**：非清 xllm、必须按命令行模式清理时（如清某个 launcher/脚本），把模式串**用变量拆开拼接**避免它完整出现在 argv 里：`PAT="rerun""_random"; pgrep -f "$PAT"`。
  - **规避 3**：按端口 scope 清理 `pgrep -f "port $P"`（端口号不会出现在清理逻辑的其它地方）。
- ~~按 `$user==me` 过滤~~ 无必要：本容器内所有进程都显示 root，按 user 过滤无意义，且 PID namespace 已隔离好了。

**别用 `source <脚本>` 来"测试/查看"有副作用的脚本**：本会话为了自检 launch_matrix.sh 的 FLAG 包，用了 `source launch_matrix.sh`，结果**把整个编排器真跑了一遍**——spawn 了一批真 xllm 进程(得紧急清理)。`source`/`. ` 会在当前 shell 执行脚本全部语句,不是"读一下"。要验证脚本逻辑用 `bash -n`(仅语法)或把要测的函数**单独复制**到一个 `bash -c '...'` 里跑,绝不 source 会启动进程/删文件的脚本。这和 `pkill -f` 自杀同属"命令自身副作用"坑。

**清理前务必确认任务真结束了（别看假象就动手）**：本会话踩过——看到日志打出 `SUCCESS 2/2` 就以为全完了，立即 `kill` 清理，结果**误杀了一个还在 measure 阶段的 trial**，client.jsonl 只落了 187 行（正常 3810）半截数据，得重跑。
- **判据看数据完整度，不看单行日志**：trial 完整 = client.jsonl 行数达标（本实验 ~3810）+ runner.log 出现 `[bench] DONE`。别拿"某个 SUCCESS 计数"当全局完成信号。
- setsid 脱离的后台脚本，其真实子进程**不在你能 `pgrep -g <launcher_pgid>` 到的组里**（setsid 让 trial 自成 session），所以"launcher 进程数=0"不代表任务停了——要查**端口上的 xllm** 或 **数据文件行数**才准。

**对 Claude tool 调用的影响**：
- `Bash run_in_background=true` 仍可以用于**短任务** + 需要观察输出的场景
- 但凡 ETA > 10 min 的任务，优先 setsid + 把日志写到固定路径 + 让我下次 `tail` 观察

**反例（不要这样写）**：
- 纯 `bash cmd &` ← session 一断就死
- 仅 `nohup bash cmd &` ← 挡了 SIGHUP 挡不住 SIGTERM cascade
- `tmux/screen` ← 需要交互终端、Claude tool 用不了
