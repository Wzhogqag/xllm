---
name: overhead-map-unmap-experiment
description: map/unmap 开销实验的埋点设计与 run_overhead.sh 编排脚本的踩坑（vmm 咽喉聚合 profiler、Ascend set_env 与 set -u 冲突、npu-smi HBM 解析）。
metadata:
  node_type: memory
  type: project
---

单卡 Qwen3-14B forward-mapping 版本的 map/unmap 运行时开销实验（2026-09-02 建立）。

## 埋点设计（已实现并验证）

- **一个 profiler 单例** `xllm/core/platform/map_unmap_profiler.{h,cpp}`：只在 `vmm::map`/`vmm::unmap`（`platform/vmm_api.cpp` 唯一驱动咽喉）埋 2 处，用 RAII `ScopedProfile` 计时。覆盖所有类别（KV/激活/权重/迁移/同步/异步），因为它们最终都汇聚到这一对函数。
- 只记**聚合**：map/unmap 各 count/total_ns/min/max + 24 桶对数直方图（base 100ns）。**不记** model/reason/时间戳/逐事件——用户明确要求精简（只有一个模型）。热路径仅 ~1 次 relaxed atomic 自增。
- FLAG 门控：`--enable_map_unmap_profiling`（默认 false，关闭零开销）、`--map_unmap_profiling_dir`、`--map_unmap_profiling_flush_ms`（默认 500）。后台线程周期性**原子覆盖写** `map_unmap_summary.json`（tmp+rename），因为 xllm 常被 pkill -9 不触发析构。
- 范式抄 `RequestMetricAggregator`（单例 + FLAG + 后台 worker_loop）。
- **实测数字**（14B，52.3GB 池）：加载+推理期 map mean≈125µs、unmap mean≈265µs；一个逻辑 KV 页 = 40 层 × 2(K/V) = **80 次** aclrtMapMem。count 随请求单调增，验证正确。

## sender.py 增强

- 新增 `--requests-jsonl`：设置后走 SSE 流式（迭代 `resp.content` 找首个含 `choices[0].text` 的 chunk = TTFT，末 chunk 推 TPOT），逐请求写 JSONL + 打印 per-model p50/p90/p95/p99。**不设则行为完全不变**（原 `await resp.text()`）。SSE 格式确认：`data: {"choices":[{"index":0,"text":" the"}]}`。

## run_overhead.sh 编排（overhead/，含 summarize_run.py + summarize_sweep.py）

每 QPS 点（默认 1/1.5/2/2.5/3）独立重启 server + 独立时间戳目录，跑 sender，收集 summary/priority_metrics/memory_samples，pkill 等 HBM 回落，写 README + sweep_summary.md（标出满足 TTFT≤10s & TPOT≤80ms 的最高 QPS）。`PHY_ID=6 bash overhead/run_overhead.sh` 可跳过选卡。

## ⚠️ 两个踩坑（都已修）

1. **`set -u` + `source /usr/local/Ascend/nnal/atb/set_env.sh` 必崩**：该脚本 line 43 引用未绑定的 `ZSH_VERSION` → nounset 下整脚本**静默退出**（无任何输出，exit 1，极难诊断）。**修复：source 前后 `set +u` / `set -u` 包起来**。任何带 `set -u` 的 xllm 启动脚本都要注意这个。
2. **npu-smi HBM 解析取错列**：chip 行 `| 0  6 | 0000:91:00.0 | 0  0 / 0  3164 / 65536 |` 第 4 列有**两个** `x / y`——第一个是 AICore util(0/0)，**最后一个才是 HBM used/total**。必须循环 match 取最后一对，且 awk 要 `exit` 只输出一行（否则多行输出撑爆 `[[ -lt ]]`）。用 PCI 总线地址正则 `[0-9A-Fa-f]{4}:..` 认 chip 行，别用 `^\| [0-9]+ [0-9]+`（会误匹配进程行）。

**Why:** 这套实验要"记录 map/unmap 耗时/次数但不影响总时延"。**How to apply:** 复用该 profiler 加任何新驱动级计时；跑实验用 `overhead/run_overhead.sh`；改任何 xllm bash 启动脚本先想 set_env 的 set -u 坑。详见 [[build-and-run]]（僵尸显存/选卡）、[[xtensor-memory]]（map 路径）。
