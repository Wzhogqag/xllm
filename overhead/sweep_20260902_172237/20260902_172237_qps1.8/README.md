# QPS=1.8 单卡 Qwen3-14B map/unmap 开销

- served requests (200): 1000 / 1000
- served-model requests: 493
- PhyPagePool: 52.29 GB

## 目标达成

- TTFT SLO ≤ 10000 ms, TPOT SLO ≤ 80 ms
- **达标: YES**

## 端到端 TTFT/TPOT（客户端，served model）

| 指标 | n | p50 | p95 |
|---|---|---|---|
| TTFT (ms) | 493 | 65.2 | 94.5 |
| TPOT (ms) | 493 | 39.85 | 49.27 |

## 服务端窗口指标（[priority window metric]）

| 指标 | 窗口数 | 均值 |
|---|---|---|
| avg_ttft_ms | 1039 | 73.2 |
| avg_tpot_ms | 1039 | 38.51 |
| ttft_violation_rate | 419 | 0.000 |
| tpot_violation_rate | 1039 | 0.000 |

## map/unmap 开销（map_unmap_summary.json）

| op | 次数 | 总耗时(ms) | 均值(us) | min(us) | max(us) |
|---|---|---|---|---|---|
| map | 300034 | 32037.4 | 106.78 | 28.93 | 20967.78 |
| unmap | 273259 | 80513.0 | 294.64 | 217.66 | 16889.11 |

> map 每次 = 一次 aclrtMapMem；14B 一个逻辑 KV 页 = 40 层 × 2(K/V) = 80 次 map。
