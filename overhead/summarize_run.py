#!/usr/bin/env python3
"""Summarize one QPS run: parse map/unmap aggregates, client TTFT/TPOT, and
server-side window metrics into a per-run README.md and append one row to the
sweep raw CSV."""
import argparse
import json
import os
import re
import statistics


def load_map_unmap(rundir):
    path = os.path.join(rundir, "map_unmap_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_client_trace(rundir):
    path = os.path.join(rundir, "client_requests.jsonl")
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def parse_server_windows(rundir):
    """Parse [priority window metric] lines for avg_ttft/avg_tpot and violation."""
    path = os.path.join(rundir, "priority_window_metrics.log")
    ttft, tpot, ttft_viol, tpot_viol = [], [], [], []
    if not os.path.exists(path):
        return ttft, tpot, ttft_viol, tpot_viol
    with open(path) as f:
        for line in f:
            def grab(key):
                m = re.search(key + r"=([0-9.]+)", line)
                return float(m.group(1)) if m else None
            a = grab("avg_ttft_ms")
            b = grab("avg_tpot_ms")
            c = grab("ttft_violation_rate")
            d = grab("tpot_violation_rate")
            if a is not None and a > 0:
                ttft.append(a)
            if b is not None and b > 0:
                tpot.append(b)
            if c is not None:
                ttft_viol.append(c)
            if d is not None:
                tpot_viol.append(d)
    return ttft, tpot, ttft_viol, tpot_viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--qps", required=True)
    ap.add_argument("--model", default="Qwen3-14B")
    ap.add_argument("--ttft-slo", type=float, default=10000.0)
    ap.add_argument("--tpot-slo", type=float, default=80.0)
    ap.add_argument("--sweep-raw", required=True)
    args = ap.parse_args()

    mu = load_map_unmap(args.rundir)
    recs = load_client_trace(args.rundir)

    ok = [r for r in recs if r.get("status") == 200]
    ttft_vals = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    tpot_vals = [r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None]

    # only the served model (client remaps top model -> this served model)
    model_ok = [r for r in ok if r.get("model") == args.model]
    m_ttft = [r["ttft_ms"] for r in model_ok if r.get("ttft_ms") is not None]
    m_tpot = [r["tpot_ms"] for r in model_ok if r.get("tpot_ms") is not None]
    # fall back to all-model if the served model name differs
    if not m_ttft:
        m_ttft, m_tpot = ttft_vals, tpot_vals

    ttft_p50 = pct(m_ttft, 50)
    ttft_p95 = pct(m_ttft, 95)
    tpot_p50 = pct(m_tpot, 50)
    tpot_p95 = pct(m_tpot, 95)

    s_ttft, s_tpot, s_ttft_v, s_tpot_v = parse_server_windows(args.rundir)

    pool_gb = ""
    pool_path = os.path.join(args.rundir, "pool_size.txt")
    if os.path.exists(pool_path):
        with open(pool_path) as f:
            m = re.search(r"([0-9.]+) GB", f.read())
            if m:
                pool_gb = m.group(1)

    def op(name):
        if not mu or name not in mu:
            return (0, 0.0, 0.0, 0.0, 0.0)
        o = mu[name]
        return (
            o.get("count", 0),
            o.get("total_ms", 0.0),
            o.get("mean_us", 0.0),
            o.get("min_ns", 0) / 1000.0,
            o.get("max_ns", 0) / 1000.0,
        )

    map_count, map_total_ms, map_mean_us, map_min_us, map_max_us = op("map")
    unmap_count, unmap_total_ms, unmap_mean_us, unmap_min_us, unmap_max_us = op("unmap")

    meets = (
        (ttft_p95 <= args.ttft_slo)
        and (tpot_p95 <= args.tpot_slo)
        and len(m_ttft) > 0
    )

    # append raw row
    with open(args.sweep_raw, "a") as f:
        f.write(
            f"{args.qps},OK,{pool_gb},"
            f"{ttft_p50:.1f},{ttft_p95:.1f},{tpot_p50:.2f},{tpot_p95:.2f},"
            f"{map_count},{map_total_ms:.1f},{unmap_count},{unmap_total_ms:.1f},"
            f"{'YES' if meets else 'NO'}\n"
        )

    # per-run README
    readme = os.path.join(args.rundir, "README.md")
    with open(readme, "w") as f:
        f.write(f"# QPS={args.qps} 单卡 {args.model} map/unmap 开销\n\n")
        f.write(f"- served requests (200): {len(ok)} / {len(recs)}\n")
        f.write(f"- served-model requests: {len(model_ok)}\n")
        f.write(f"- PhyPagePool: {pool_gb} GB\n\n")

        f.write("## 目标达成\n\n")
        f.write(f"- TTFT SLO ≤ {args.ttft_slo:.0f} ms, TPOT SLO ≤ {args.tpot_slo:.0f} ms\n")
        f.write(f"- **达标: {'YES' if meets else 'NO'}**\n\n")

        f.write("## 端到端 TTFT/TPOT（客户端，served model）\n\n")
        f.write("| 指标 | n | p50 | p95 |\n|---|---|---|---|\n")
        f.write(f"| TTFT (ms) | {len(m_ttft)} | {ttft_p50:.1f} | {ttft_p95:.1f} |\n")
        f.write(f"| TPOT (ms) | {len(m_tpot)} | {tpot_p50:.2f} | {tpot_p95:.2f} |\n\n")

        if s_ttft or s_tpot:
            f.write("## 服务端窗口指标（[priority window metric]）\n\n")
            f.write("| 指标 | 窗口数 | 均值 |\n|---|---|---|\n")
            if s_ttft:
                f.write(f"| avg_ttft_ms | {len(s_ttft)} | {statistics.mean(s_ttft):.1f} |\n")
            if s_tpot:
                f.write(f"| avg_tpot_ms | {len(s_tpot)} | {statistics.mean(s_tpot):.2f} |\n")
            if s_ttft_v:
                f.write(f"| ttft_violation_rate | {len(s_ttft_v)} | {statistics.mean(s_ttft_v):.3f} |\n")
            if s_tpot_v:
                f.write(f"| tpot_violation_rate | {len(s_tpot_v)} | {statistics.mean(s_tpot_v):.3f} |\n")
            f.write("\n")

        f.write("## map/unmap 开销（map_unmap_summary.json）\n\n")
        if mu:
            f.write("| op | 次数 | 总耗时(ms) | 均值(us) | min(us) | max(us) |\n")
            f.write("|---|---|---|---|---|---|\n")
            f.write(
                f"| map | {map_count} | {map_total_ms:.1f} | {map_mean_us:.2f} | "
                f"{map_min_us:.2f} | {map_max_us:.2f} |\n"
            )
            f.write(
                f"| unmap | {unmap_count} | {unmap_total_ms:.1f} | {unmap_mean_us:.2f} | "
                f"{unmap_min_us:.2f} | {unmap_max_us:.2f} |\n\n"
            )
            f.write(
                "> map 每次 = 一次 aclrtMapMem；14B 一个逻辑 KV 页 = 40 层 × 2(K/V) = 80 次 map。\n"
            )
        else:
            f.write("map_unmap_summary.json 未找到或解析失败。\n")

    print(f"[summarize_run] wrote {readme}")


if __name__ == "__main__":
    main()
