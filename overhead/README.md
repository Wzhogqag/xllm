# overhead — map/unmap 开销 & 单卡时延实验

单卡 serve Qwen3-14B 时，测量运行时 VMM `map`/`unmap`（`aclrtMapMem`/`aclrtUnmapMem`）开销，并采集端到端 TTFT/TPOT/总时延，用于和另一版本对比。

## 脚本（工具，非数据）

| 文件 | 作用 |
|---|---|
| `run_overhead.sh` | 一键扫描多个 QPS：每个 QPS 点独立重启 xLLM（带 profiling）→ 跑 sender → 收集数据 → 写各点 README 和 sweep 汇总。自动选空闲 NPU。 |
| `sender.py` | 压测客户端：按 trace 回放请求；`--requests-jsonl` 开启后用 SSE 流式逐请求记录 TTFT/TPOT/总时延。 |
| `start111.sh` | 单卡启动 xLLM 的脚本快照（对齐对比版本的配置）。 |
| `summarize_run.py` | 解析单个 run 目录，生成该点的 `README.md` 和一行 `sweep_raw.csv`。 |
| `summarize_sweep.py` | 汇总所有 QPS 点的 `sweep_raw.csv`，输出跨 QPS 对比表 `sweep_summary.md`，标出满足 SLO 的最高 QPS。 |
| `plot_cdf_compare.py` | 画 TTFT/TPOT/总时延的 CDF 对比图（本次 run vs 另一版本的 latency CSV）。 |

## 实验结果目录

命名规则 `start111_qps<QPS>_<时间戳>/`（单点）或 `sweep_<时间戳>/`（多点扫描）。每个 run 目录内的文件：

| 文件 | 内容 |
|---|---|
| `start111.sh.snapshot` | 本次启动 xLLM 的完整命令快照（复现用） |
| `node_0.log` | 服务端完整日志 |
| `client_requests.jsonl` | **逐请求** TTFT/TPOT/总时延（每行一条，字段见各 run 的 README） |
| `sender_stdout.log` | 客户端汇总：per-model TTFT/TPOT 的 p50/p90/p95/p99 |
| `map_unmap_summary.json` | **map/unmap 聚合开销**：次数、总/均/min/max 耗时、对数直方图 |
| `priority_window_metrics.log` | 服务端窗口聚合指标（avg_ttft/avg_tpot/violation_rate，每窗口一条） |
| `memory_samples.log` | 显存打点：weight/kv/activation 页数（1 页=2MiB） |
| `cdf_compare/` | 与另一版本对比的 CDF 图（ttft/tpot/total + combined） |
| `README.md` | 该次实验的详细说明（部分 run 有） |

`sweep_*/` 目录额外含 `sweep_raw.csv`（每 QPS 一行）和 `sweep_summary.md`（跨 QPS 对比）。

## 现有实验

- `start111_qps1.8_20260902_201716/` — start111 配置，QPS=1.8（含详细 README）
- `start111_qps1.7_20260903_132328/` — start111 配置，QPS=1.7
- `sweep_20260902_172237/` — 早期用 `run_overhead.sh` 跑的单点（QPS=1.8，配置与 start111 不同：util=0.86 + chunked_prefill/prefix_cache 开）

> 注：`xllm基线性能数据.csv` 是外部对比版本的逐请求时延数据（列 `ttft_ms,tpot_ms,total_latency_ms`），供 `plot_cdf_compare.py` 作对照。

## 关键点

- profiling 由 `--enable_map_unmap_profiling` 开关，关闭时零开销；只在 `vmm::map`/`vmm::unmap` 唯一咽喉埋点，覆盖 KV/激活/权重全部路径。
- 14B 一个逻辑 KV 页 = 40 层 × 2(K/V) = 80 次驱动 `map`。
- `client_requests.jsonl` 里只有 `model=Qwen3-14B` 且 `status=200` 的是真实数据；其余 model 名是 trace remap 产物，返回空响应（`out_tokens=0`），统计时应过滤。
