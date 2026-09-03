#!/usr/bin/env python3
"""Plot CDF comparison of TTFT / TPOT / total latency between this
forward-mapping run and another xLLM version's latency CSV."""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_mine(path, model):
    ttft, tpot, tot = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("model") != model or r.get("status") != 200:
                continue
            if r.get("ttft_ms") is not None:
                ttft.append(float(r["ttft_ms"]))
            if r.get("tpot_ms") is not None:
                tpot.append(float(r["tpot_ms"]))
            tot.append(float(r["total_ms"]))
    return ttft, tpot, tot


def load_other(path):
    ttft, tpot, tot = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ttft.append(float(row["ttft_ms"]))
            tpot.append(float(row["tpot_ms"]))
            tot.append(float(row["total_latency_ms"]))
    return ttft, tpot, tot


def cdf_xy(values):
    s = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(s) + 1) / len(s)
    return s, y


def plot_one(ax, mine, other, title, xlabel, logx=False):
    for data, label, color in [
        (mine, "forward-mapping (this run)", "#1f77b4"),
        (other, "other version", "#d62728"),
    ]:
        x, y = cdf_xy(data)
        ax.plot(x, y, label=f"{label} (n={len(data)})", color=color, linewidth=2)
        # p50/p95 markers
        p50 = np.percentile(data, 50)
        p95 = np.percentile(data, 95)
        ax.axvline(p50, color=color, linestyle=":", alpha=0.4, linewidth=1)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True, help="client_requests.jsonl")
    ap.add_argument("--other", required=True, help="other version latency csv")
    ap.add_argument("--model", default="Qwen3-14B")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    m_ttft, m_tpot, m_tot = load_mine(args.mine, args.model)
    o_ttft, o_tpot, o_tot = load_other(args.other)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) TTFT
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_one(ax, m_ttft, o_ttft, "TTFT CDF", "TTFT (ms)")
    fig.tight_layout()
    p = os.path.join(args.out_dir, "cdf_ttft.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)

    # 2) TPOT
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_one(ax, m_tpot, o_tpot, "TPOT CDF", "TPOT (ms)")
    fig.tight_layout()
    p = os.path.join(args.out_dir, "cdf_tpot.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)

    # 3) total latency (log x, spans 4 orders of magnitude)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_one(ax, m_tot, o_tot, "Total latency CDF (log x)", "total latency (ms)", logx=True)
    fig.tight_layout()
    p = os.path.join(args.out_dir, "cdf_total_latency.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)

    # combined 1x3
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    plot_one(axes[0], m_ttft, o_ttft, "TTFT CDF", "TTFT (ms)")
    plot_one(axes[1], m_tpot, o_tpot, "TPOT CDF", "TPOT (ms)")
    plot_one(axes[2], m_tot, o_tot, "Total latency CDF (log x)", "total latency (ms)", logx=True)
    fig.suptitle(f"{args.model} latency CDF: forward-mapping vs other version", fontsize=13)
    fig.tight_layout()
    p = os.path.join(args.out_dir, "cdf_combined.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
