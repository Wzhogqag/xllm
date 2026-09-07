# qps=1.8 单卡 Qwen3-14B 实验（start111.sh 配置）

单卡（NPU phy8）serve Qwen3-14B，用 `start111.sh` 启动，qps=1.8 回放 `trace_1_100.csv`。
本次开启了 map/unmap 埋点，用于统计运行时 VMM map/unmap 开销，同时采集端到端 TTFT/TPOT。

- 启动脚本：见 `start111.sh.snapshot`（本次启动命令的完整快照）
- 数据集/发送命令：`python sender.py --trace-csv trace_1_100.csv --target-qps 1.8 --input-scale 1 --max-tokens-scale 20 --max-context-cap=8192 --limit 1000 --port 18119 --requests-jsonl client_requests.jsonl`
- 关键配置：`max_memory_utilization=0.95`（池 57.81GB）、`enable_prefix_cache=false`、`enable_chunked_prefill=false`、`priority_level=2`、`priority_window_size=5000`

## 文件说明

| 文件 | 内容 |
|---|---|
| `start111.sh.snapshot` | 本次启动 xLLM 的完整脚本快照（复现用） |
| `node_0.log` | 服务端完整日志（含启动、建池、推理） |
| `client_requests.jsonl` | **逐请求**记录，每行一条：见下方字段说明 |
| `sender_stdout.log` | 客户端汇总输出：per-model TTFT/TPOT 的 p50/p90/p95/p99 |
| `priority_window_metrics.log` | 服务端**窗口聚合**指标（每 5000ms 一条），字段 `avg_ttft_ms/avg_tpot_ms/ttft_violation_rate/tpot_violation_rate` |
| `memory_samples.log` | 显存打点：`weight_phy_pages`（权重）、`kv_cache_phy_pages`（KV）、`activation_allocated_pages`（激活），单位=页（1 页=2MiB） |
| `map_unmap_summary.json` | **本实验核心**：map/unmap 驱动调用（aclrtMapMem/aclrtUnmapMem）的聚合统计，见下 |
| `cdf_compare/` | 与另一版本（`Qwen3-14B_node_0_latency.csv`）的 TTFT/TPOT/总时延 CDF 对比图 |

## client_requests.jsonl 字段

每行一条请求（只有 `model=Qwen3-14B` 且 `status=200` 的是实际部署模型的真实数据；其余 model 名是 trace remap 产物，返回空响应 `out_tokens=0`，统计时应过滤掉）：

- `req_id`：请求序号
- `model`：请求的模型名
- `status`：HTTP 状态（200=成功）
- `ttft_ms`：Time To First Token，首 token 时延（ms）
- `tpot_ms`：Time Per Output Token，decode 阶段平均每 token 时延（ms）= (末token时刻 - 首token时刻)/(生成token数-1)
- `total_ms`：整条请求端到端总时延（ms）
- `out_tokens`：实际生成的 token 数
- `max_tokens`：请求设定的最大生成 token 数

## map_unmap_summary.json 字段

map 与 unmap 各一块（`map` = aclrtMapMem，`unmap` = aclrtUnmapMem）。14B 一个逻辑 KV 页会展开成 40 层 × 2(K/V) = 80 次驱动 map。

- `count`：总调用次数
- `total_ns` / `total_ms`：总耗时
- `mean_us`：单次平均耗时（微秒）
- `min_ns` / `max_ns`：单次最短 / 最长耗时
- `p50_ge_ns` / `p90_ge_ns` / `p99_ge_ns`：由对数直方图推算的分位数（返回所在桶的下界，近似值）
- `hist`：对数直方图，24 桶。`bucket 0 = [0, 100ns)`，`bucket i = [100ns<<(i-1), 100ns<<i)`

## 本次结果速览

- TTFT p50=82.5ms / p95=122.9ms；TPOT p50=41.2ms / p95=51.9ms；均满足 SLO（TTFT≤10s, TPOT≤80ms）。
- map 277,289 次 / 32.1s（均值 116µs）；unmap 247,690 次 / 71.4s（均值 288µs）。unmap 走后台异步线程，不在推理关键路径。
- 与另一版本（同配置、同数据集、同 QPS）对比：TTFT/TPOT/总时延分布基本重合（p50/p95 差异 <2%），详见 `cdf_compare/`。
