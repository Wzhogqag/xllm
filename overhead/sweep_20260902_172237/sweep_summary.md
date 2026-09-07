# map/unmap overhead sweep 汇总 — 单卡 Qwen3-14B

目标：TTFT p95 ≤ 10000 ms 且 TPOT p95 ≤ 80 ms

**满足 SLO 的最高 QPS = 1.8** (TTFT p95=94.5ms, TPOT p95=49.27ms)

## 跨 QPS 对比

| QPS | status | pool(GB) | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | map次数 | map总(ms) | unmap次数 | unmap总(ms) | 达标 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.8 | OK | 52.29 | 65.2 | 94.5 | 39.85 | 49.27 | 300034 | 32037.4 | 273259 | 80513.0 | YES |

## 说明

- map/unmap 次数/耗时来自各 run 的 `map_unmap_summary.json`（驱动级 aclrtMapMem/aclrtUnmapMem 聚合）。
- TTFT/TPOT p50/p95 来自各 run 的 `client_requests.jsonl`（SSE 流式逐请求）。
- 每个 QPS 点的详情见 `*_qps<QPS>/README.md`。
