#!/usr/bin/env python3
"""Aggregate all per-QPS rows in sweep_raw.csv into a cross-QPS comparison
table and identify the highest QPS meeting both TTFT and TPOT SLOs."""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ttft-slo", type=float, default=10000.0)
    ap.add_argument("--tpot-slo", type=float, default=80.0)
    args = ap.parse_args()

    rows = []
    with open(args.sweep_raw) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    def to_f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    rows.sort(key=lambda r: to_f(r.get("qps")))

    meeting = [r for r in rows if r.get("meets_slo") == "YES"]
    best = None
    if meeting:
        best = max(meeting, key=lambda r: to_f(r.get("qps")))

    with open(args.out, "w") as f:
        f.write("# map/unmap overhead sweep 汇总 — 单卡 Qwen3-14B\n\n")
        f.write(f"目标：TTFT p95 ≤ {args.ttft_slo:.0f} ms 且 TPOT p95 ≤ {args.tpot_slo:.0f} ms\n\n")

        if best is not None:
            f.write(
                f"**满足 SLO 的最高 QPS = {best['qps']}** "
                f"(TTFT p95={best['ttft_p95_ms']}ms, TPOT p95={best['tpot_p95_ms']}ms)\n\n"
            )
        else:
            f.write("**没有 QPS 点同时满足 TTFT & TPOT SLO。**\n\n")

        f.write("## 跨 QPS 对比\n\n")
        f.write(
            "| QPS | status | pool(GB) | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | "
            "map次数 | map总(ms) | unmap次数 | unmap总(ms) | 达标 |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r.get('qps')} | {r.get('status')} | {r.get('pool_gb')} | "
                f"{r.get('ttft_p50_ms')} | {r.get('ttft_p95_ms')} | "
                f"{r.get('tpot_p50_ms')} | {r.get('tpot_p95_ms')} | "
                f"{r.get('map_count')} | {r.get('map_total_ms')} | "
                f"{r.get('unmap_count')} | {r.get('unmap_total_ms')} | "
                f"{r.get('meets_slo')} |\n"
            )
        f.write("\n")

        f.write("## 说明\n\n")
        f.write(
            "- map/unmap 次数/耗时来自各 run 的 `map_unmap_summary.json`（驱动级 "
            "aclrtMapMem/aclrtUnmapMem 聚合）。\n"
            "- TTFT/TPOT p50/p95 来自各 run 的 `client_requests.jsonl`（SSE 流式逐请求）。\n"
            "- 每个 QPS 点的详情见 `*_qps<QPS>/README.md`。\n"
        )

    print(f"[summarize_sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
