import argparse
import asyncio
import json
import os
import re
import subprocess
import threading
import time
import traceback
from collections import Counter

import aiohttp
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_WINDOW_MODEL_REMAP_ORDER = [
    "Qwen3-14B",
    "Qwen2.5-14B",
    "Qwen3-32B",
    "Qwen2.5-32B",
    "Qwen2.5-0.5B",
    "Qwen3-0.6B",
    "Qwen2.5-1.5B",
    "Qwen3-1.7B",
    "Qwen2.5-3B",
    "Qwen3-4B",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
]
# 默认多端口映射来自 080_static/run_080.sh 与 run_080_2.sh 的 start_port/model_path。
DEFAULT_MODEL_PORT_MAP = {
    "Qwen3-32B": 18119,
    "Qwen3-14B": 18121,
    "Qwen3-4B": 18123,
    "Qwen3-1.7B": 18124,
    "Qwen3-0.6B": 18125,
    "Qwen2.5-0.5B": 18126,
    "Qwen2.5-1.5B": 18127,
    "Qwen2.5-3B": 18128,
    "Qwen2.5-14B": 18130,
    "Qwen2.5-32B": 18131,
    "Qwen2.5-1.5B-Instruct": 18132,
    "Qwen2.5-3B-Instruct": 18133,
}
SOURCE_MODEL_ORDER = list("ABCDEFGHIJKL")
WARMUP_REQUESTS_PER_MODEL = 1
NPU_PLOT_AVG_WINDOW = 5


def parse_monitor_npu_ids(npu_ids_arg: str) -> list[int]:
    raw = [x.strip() for x in str(npu_ids_arg).split(",") if x.strip()]
    if not raw:
        raise ValueError("--monitor-npu-ids 至少需要 1 个 NPU ID")
    try:
        npu_ids = [int(x) for x in raw]
    except ValueError as e:
        raise ValueError("--monitor-npu-ids 必须是逗号分隔整数") from e
    if len(set(npu_ids)) != len(npu_ids):
        raise ValueError("--monitor-npu-ids 不能包含重复 ID")
    if any(x < 0 for x in npu_ids):
        raise ValueError("--monitor-npu-ids 中 ID 必须为非负整数")
    return npu_ids


def parse_model_ports_arg(ports_arg: str) -> dict[str, int]:
    text = str(ports_arg).strip()
    if not text:
        return {}

    if text.lower() == "default":
        return dict(DEFAULT_MODEL_PORT_MAP)

    items = [x.strip() for x in text.split(",") if x.strip()]
    if not items:
        return {}

    # 形式1: model=port,model=port
    if any("=" in x for x in items):
        out: dict[str, int] = {}
        for item in items:
            if "=" not in item:
                raise ValueError("--ports 混合格式非法；含 '=' 时每项都需为 model=port")
            model, port_str = item.split("=", 1)
            model = model.strip()
            if not model:
                raise ValueError("--ports 中 model 不能为空")
            try:
                port = int(port_str.strip())
            except ValueError as e:
                raise ValueError(f"--ports 端口非法: {item}") from e
            if port <= 0:
                raise ValueError(f"--ports 端口必须 > 0: {item}")
            out[model] = port
        return out

    # 形式2: 纯端口列表，按 DEFAULT_WINDOW_MODEL_REMAP_ORDER 对齐。
    try:
        ports = [int(x) for x in items]
    except ValueError as e:
        raise ValueError("--ports 必须是 default、model=port 列表或纯端口列表") from e
    if any(p <= 0 for p in ports):
        raise ValueError("--ports 中端口必须 > 0")
    if len(ports) > len(DEFAULT_WINDOW_MODEL_REMAP_ORDER):
        raise ValueError(
            f"--ports 端口数量({len(ports)})超过模型数量({len(DEFAULT_WINDOW_MODEL_REMAP_ORDER)})"
        )

    return {
        model: port
        for model, port in zip(DEFAULT_WINDOW_MODEL_REMAP_ORDER, ports)
    }


def parse_npu_usage_from_npu_smi(output_text: str, target_npu_ids: list[int]) -> dict[int, float]:
    usage_by_npu: dict[int, float] = {}
    target_set = set(target_npu_ids)

    for line in output_text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue

        id_match = re.match(r"^\|\s*(\d+)\b", line)
        if id_match:
            npu_id = int(id_match.group(1))
            if npu_id in target_set:
                pct_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", line)
                if pct_matches:
                    usage_by_npu[npu_id] = float(pct_matches[0])

        parts = line.split("|")
        if len(parts) < 4:
            continue

        first_col = parts[1].strip()
        second_col = parts[2].strip()
        metric_col = parts[3].strip()
        if not re.search(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]", second_col):
            continue

        first_col_ints = re.findall(r"\d+", first_col)
        if len(first_col_ints) < 2:
            continue

        phy_id = int(first_col_ints[1])
        if phy_id not in target_set:
            continue

        util_match = re.search(r"(\d+(?:\.\d+)?)", metric_col)
        if util_match:
            usage_by_npu[phy_id] = float(util_match.group(1))

    return usage_by_npu


def collect_npu_usage_once(target_npu_ids: list[int]) -> dict[int, float]:
    try:
        proc = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return {}

    output_text = f"{proc.stdout}\n{proc.stderr}" if proc.stderr else proc.stdout
    return parse_npu_usage_from_npu_smi(output_text, target_npu_ids)


def npu_usage_monitor_worker(
    stop_event: threading.Event,
    target_npu_ids: list[int],
    sample_interval_s: float,
    usage_lists_by_npu: dict[int, list[tuple[float, float]]],
) -> None:
    interval = max(0.1, float(sample_interval_s))
    while not stop_event.is_set():
        now_ts = time.time()
        usage_now = collect_npu_usage_once(target_npu_ids)
        for npu_id in target_npu_ids:
            usage = usage_now.get(npu_id, -1.0)
            usage_lists_by_npu[npu_id].append((now_ts, usage))
        stop_event.wait(interval)


def save_npu_usage_csv_and_plot(
    usage_lists_by_npu: dict[int, list[tuple[float, float]]],
    csv_path: str,
    plot_path: str,
) -> None:
    rows = []
    for npu_id in sorted(usage_lists_by_npu.keys()):
        for ts, usage in usage_lists_by_npu[npu_id]:
            rows.append(
                {
                    "timestamp_s": float(ts),
                    "npu_id": int(npu_id),
                    "npu_usage_pct": float(usage),
                }
            )

    if not rows:
        print("[sender] skip NPU usage export: no samples")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["timestamp_s", "npu_id"]).reset_index(drop=True)
    base_ts = float(df["timestamp_s"].min())
    df["time_offset_s"] = df["timestamp_s"] - base_ts

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[sender] NPU usage CSV saved: {csv_path}")

    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = 0
    for npu_id in sorted(df["npu_id"].unique().tolist()):
        sub = df[df["npu_id"] == npu_id]
        valid = sub[(sub["npu_usage_pct"] >= 0.0) & (sub["npu_usage_pct"] <= 100.0)]
        if valid.empty:
            continue

        # 每 5 个采样点做一次均值聚合，降低绘图抖动。
        valid = valid.reset_index(drop=True)
        valid["plot_group"] = valid.index // NPU_PLOT_AVG_WINDOW
        valid = (
            valid.groupby("plot_group", as_index=False)
            .agg(
                time_offset_s=("time_offset_s", "mean"),
                npu_usage_pct=("npu_usage_pct", "mean"),
            )
        )

        ax.plot(
            valid["time_offset_s"],
            valid["npu_usage_pct"],
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            label=f"NPU {npu_id}",
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("[sender] skip NPU usage plot: no valid usage points")
        return

    ax.set_xlabel("Time since monitor start (s)")
    ax.set_ylabel("NPU usage (%)")
    ax.set_title("NPU Usage Timeline")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"[sender] NPU usage plot saved: {plot_path}")

def request_sender(trace_df, target_qps,
                   input_scale=1.0, max_tokens_scale=1.0, min_max_tokens=0, max_context_cap=16384):
    """
    Generator that yields request dicts scaled to a desired average total QPS.

    Parameters
    ----------
    trace_df : pd.DataFrame
        DataFrame sorted by time with columns:
          time (int, relative seconds), model,
          input_tokens, output_tokens.
    target_qps : float
        Desired average total QPS for the output request stream.
        Timestamps are scaled by ``base_qps / target_qps``:
          scale > 1  →  slower stream;  scale < 1  →  faster stream.
    input_scale : float, default 1.0
        Multiply input_length by this factor before applying the cap.
        Use values < 1 to compress, > 1 to expand.
    max_tokens_scale : float, default 1.0
        Multiply output_length by this factor before applying the cap.
        Use values < 1 to compress, > 1 to expand.
    min_max_tokens : int, default 0
        Minimum allowed output_length after scaling by max_tokens_scale.
        If scaled output_length is smaller than this value, it is raised
        to min_max_tokens before applying proportional cap.
    max_context_cap : int, default 16384
        Maximum allowed total context length (input + output) per request.
        Requests exceeding the cap are truncated proportionally:
          input_length  = floor(input_length  * cap / total)
          output_length = floor(output_length * cap / total)
        Requests with total == 0 after scaling are skipped.

    Yields
    ------
    dict
        {"time": float, "model": str,
         "input_length": int, "output_length": int}
    """
    n         = len(trace_df)
    t_min     = int(trace_df["time"].iloc[0])
    t_max     = int(trace_df["time"].iloc[-1])
    time_span = t_max - t_min
    base_qps  = n / time_span
    scale     = base_qps / target_qps

    # ── Stats before yield ────────────────────────────────────────────────────
    avg_in  = trace_df["input_tokens"].mean()
    avg_out = trace_df["output_tokens"].mean()
    scaled_avg_in  = avg_in  * input_scale
    scaled_avg_out = avg_out * max_tokens_scale
    cap_fraction = min(1.0, max_context_cap / max(scaled_avg_in + scaled_avg_out, 1e-9))
    print(
        f"[request_sender] n={n:,}  "
        f"base_qps={base_qps:.4f} req/s  "
        f"target_qps={target_qps} req/s  "
        f"time_scale={scale:.4f}x\n"
        f"  avg input : {avg_in:.1f} tok  →  {scaled_avg_in:.1f} tok "
        f"(input_scale={input_scale})\n"
        f"  avg output: {avg_out:.1f} tok  →  {scaled_avg_out:.1f} tok "
        f"(max_tokens_scale={max_tokens_scale})\n"
        f"  avg total : {avg_in + avg_out:.1f} tok  →  "
        f"{scaled_avg_in + scaled_avg_out:.1f} tok  "
        f"(cap={max_context_cap}, ~{cap_fraction*100:.1f}% fit without truncation)"
    )

    for _, row in trace_df.iterrows():
        raw_in_len = int(round(int(row["input_tokens"]) * input_scale))
        raw_out_len = int(round(int(row["output_tokens"]) * max_tokens_scale))

        # 保持原有请求集合：原始缩放后 input/output 同时为 0 的请求仍然跳过。
        if (raw_in_len + raw_out_len) == 0:
            continue

        in_len = raw_in_len
        out_len = raw_out_len
        if out_len < min_max_tokens:
            out_len = int(min_max_tokens)
        total = in_len + out_len

        # Proportional cap
        if total > max_context_cap:
            in_len  = int(in_len  * max_context_cap / total)
            out_len = int(out_len * max_context_cap / total)

        yield {
            "time":          float((int(row["time"]) - t_min) * scale),
            "model":         str(row["model"]),
            "input_length":  in_len,
            "output_length": out_len,
        }


def rank_models_by_count(requests: list[dict]) -> list[str]:
    counts = Counter(r["model"] for r in requests)
    first_seen: dict[str, int] = {}
    for idx, req in enumerate(requests):
        model = req["model"]
        if model not in first_seen:
            first_seen[model] = idx

    for model in SOURCE_MODEL_ORDER:
        counts.setdefault(model, 0)
        first_seen.setdefault(model, 10**9)

    return sorted(
        counts.keys(),
        key=lambda m: (
            -counts[m],
            first_seen.get(m, 10**9),
            SOURCE_MODEL_ORDER.index(m) if m in SOURCE_MODEL_ORDER else len(SOURCE_MODEL_ORDER),
            m,
        ),
    )


def build_model_remap(requests: list[dict]) -> dict[str, str]:
    ranked_src = rank_models_by_count(requests)
    if len(ranked_src) > len(DEFAULT_WINDOW_MODEL_REMAP_ORDER):
        raise ValueError(
            f"源模型数量({len(ranked_src)})超过可映射目标模型数量({len(DEFAULT_WINDOW_MODEL_REMAP_ORDER)})"
        )
    return {
        src: dst
        for src, dst in zip(ranked_src, DEFAULT_WINDOW_MODEL_REMAP_ORDER)
    }


def build_prompt_from_input_length(input_length: int) -> str:
    if input_length <= 0:
        return ""
    # 用空格分隔的占位词近似控制输入 token 数量。
    return " ".join(["token"] * input_length)


def build_payload(req: dict, model_remap: dict[str, str]) -> dict:
    src_model = req["model"]
    dst_model = model_remap.get(src_model, src_model)
    return {
        "model": dst_model,
        "model_id": dst_model,
        "prompt": build_prompt_from_input_length(int(req["input_length"])),
        "max_tokens": int(req["output_length"]),
        "ignore_eos": True,
        "stream": True,
        "temperature": 0.0,
    }


def build_warmup_payload(
    req: dict,
    model_remap: dict[str, str],
    input_scale: float,
    max_tokens_scale: float,
) -> dict:
    # warmup 只为触发模型加载，去掉 input/max-tokens 的放大倍数以节省预热时间。
    warmup_in = int(req["input_length"])
    if input_scale > 0:
        warmup_in = max(0, int(round(warmup_in / input_scale)))
    warmup_out = int(req["output_length"])
    if max_tokens_scale > 0:
        warmup_out = max(1, int(round(warmup_out / max_tokens_scale)))
    warmup_req = {**req, "input_length": warmup_in, "output_length": warmup_out}
    return build_payload(warmup_req, model_remap)


def format_exception_details(exc: Exception) -> str:
    exc_type = type(exc).__name__
    exc_text = str(exc).strip()
    if not exc_text:
        exc_text = "<empty>"
    repr_text = repr(exc)
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    last_tb_line = ""
    if tb_lines:
        last_tb_line = tb_lines[-1].strip()
    return (
        f"exception_type={exc_type}"
        f" | message={exc_text}"
        f" | repr={repr_text}"
        f" | traceback_last={last_tb_line}"
    )


async def send_one(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    req_id: int,
    trace_records: list | None = None,
) -> tuple[bool, float, str]:
    # When trace_records is None, keep the original behavior (read whole body).
    # When a list is passed, iterate the SSE stream to capture per-request
    # TTFT/TPOT and append one record; this only runs for the real replay, not
    # warmup, so default behavior is unchanged.
    if trace_records is None:
        st = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                txt = await resp.text()
                latency = time.perf_counter() - st
                if resp.status != 200:
                    return (
                        False,
                        latency,
                        (
                            f"req_id={req_id}"
                            f" | model={payload.get('model')}"
                            f" | max_tokens={payload.get('max_tokens')}"
                            f" | http_status={resp.status}"
                            f" | response_head={txt[:200]!r}"
                        ),
                    )
                return True, latency, ""
        except Exception as e:  # noqa: BLE001
            latency = time.perf_counter() - st
            return (
                False,
                latency,
                (
                    f"req_id={req_id}"
                    f" | model={payload.get('model')}"
                    f" | max_tokens={payload.get('max_tokens')}"
                    f" | {format_exception_details(e)}"
                ),
            )

    st = time.perf_counter()
    first_token_t: float | None = None
    last_token_t: float | None = None
    n_token_chunks = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                txt = await resp.text()
                latency = time.perf_counter() - st
                trace_records.append(
                    {
                        "req_id": req_id,
                        "model": payload.get("model"),
                        "status": resp.status,
                        "ttft_ms": None,
                        "tpot_ms": None,
                        "total_ms": latency * 1000.0,
                        "out_tokens": 0,
                        "max_tokens": payload.get("max_tokens"),
                    }
                )
                return (
                    False,
                    latency,
                    (
                        f"req_id={req_id}"
                        f" | model={payload.get('model')}"
                        f" | max_tokens={payload.get('max_tokens')}"
                        f" | http_status={resp.status}"
                        f" | response_head={txt[:200]!r}"
                    ),
                )

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                # Each SSE data chunk that carries generated text counts as one
                # token-bearing step. First such chunk => TTFT.
                has_text = False
                try:
                    obj = json.loads(data)
                    choices = obj.get("choices") or []
                    if choices:
                        ch = choices[0]
                        text = ch.get("text")
                        if text is None:
                            delta = ch.get("delta") or {}
                            text = delta.get("content")
                        has_text = bool(text)
                except Exception:  # noqa: BLE001
                    # Non-JSON keepalive/other; ignore for token counting.
                    has_text = False
                if has_text:
                    now = time.perf_counter()
                    if first_token_t is None:
                        first_token_t = now
                    last_token_t = now
                    n_token_chunks += 1

            latency = time.perf_counter() - st
            ttft_ms = (first_token_t - st) * 1000.0 if first_token_t is not None else None
            tpot_ms = None
            if (
                first_token_t is not None
                and last_token_t is not None
                and n_token_chunks > 1
            ):
                tpot_ms = (last_token_t - first_token_t) * 1000.0 / (n_token_chunks - 1)
            trace_records.append(
                {
                    "req_id": req_id,
                    "model": payload.get("model"),
                    "status": 200,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "total_ms": latency * 1000.0,
                    "out_tokens": n_token_chunks,
                    "max_tokens": payload.get("max_tokens"),
                }
            )
            return True, latency, ""
    except Exception as e:  # noqa: BLE001
        latency = time.perf_counter() - st
        trace_records.append(
            {
                "req_id": req_id,
                "model": payload.get("model"),
                "status": "exception",
                "ttft_ms": None,
                "tpot_ms": None,
                "total_ms": latency * 1000.0,
                "out_tokens": n_token_chunks,
                "max_tokens": payload.get("max_tokens"),
            }
        )
        return (
            False,
            latency,
            (
                f"req_id={req_id}"
                f" | model={payload.get('model')}"
                f" | max_tokens={payload.get('max_tokens')}"
                f" | {format_exception_details(e)}"
            ),
        )


async def warmup_models(
    session: aiohttp.ClientSession,
    requests: list[dict],
    model_remap: dict[str, str],
    build_url,
    input_scale: float,
    max_tokens_scale: float,
) -> None:
    print(f"[sender] warmup enabled: {WARMUP_REQUESTS_PER_MODEL} request per model")
    print(
        "[sender] warmup payload de-scaled to save time: "
        f"input_scale={input_scale} max_tokens_scale={max_tokens_scale} removed"
    )
    first_req_by_dst: dict[str, dict] = {}
    for req in requests:
        src_model = req["model"]
        dst_model = model_remap.get(src_model, src_model)
        if dst_model not in first_req_by_dst:
            first_req_by_dst[dst_model] = req

    # 按 remap 顺序预热，避免被首个请求出现顺序影响。
    warmup_ordered_dst_models = list(dict.fromkeys(model_remap.values()))
    for dst_model in warmup_ordered_dst_models:
        req = first_req_by_dst.get(dst_model)
        if req is None:
            continue
        ok = 0
        bad = 0
        for i in range(WARMUP_REQUESTS_PER_MODEL):
            payload = build_warmup_payload(req, model_remap, input_scale, max_tokens_scale)
            url = build_url(payload["model"])
            success, _, _ = await send_one(session, url, payload, -(i + 1))
            if success:
                ok += 1
            else:
                bad += 1
        print(f"  - {dst_model}: warmup_ok={ok} warmup_failed={bad}")


async def replay_requests(
    requests: list[dict],
    model_remap: dict[str, str],
    host: str,
    port: int,
    ports: str,
    max_inflight: int,
    warmup: bool,
    input_scale: float,
    max_tokens_scale: float,
    enable_npu_monitor: bool,
    monitor_npu_ids: str,
    monitor_interval_s: float,
    npu_monitor_csv: str,
    npu_monitor_plot: str,
    dry_run: bool,
    requests_jsonl: str = "",
) -> None:
    if not requests:
        print("[sender] no requests to replay")
        return

    model_port_map = parse_model_ports_arg(ports)

    def build_url(dst_model: str) -> str:
        chosen_port = model_port_map.get(dst_model, port)
        return f"http://{host}:{chosen_port}/v1/completions"

    if model_port_map:
        print(f"[sender] endpoint default={build_url('__default__')}")
        print("[sender] model-specific ports enabled:")
        for model in DEFAULT_WINDOW_MODEL_REMAP_ORDER:
            if model in model_port_map:
                print(f"  {model} -> {host}:{model_port_map[model]}")
        # 打印 map 中但不在默认顺序内的项。
        for model in sorted(model_port_map.keys()):
            if model not in DEFAULT_WINDOW_MODEL_REMAP_ORDER:
                print(f"  {model} -> {host}:{model_port_map[model]}")
    else:
        print(f"[sender] endpoint={build_url('__default__')}")

    print("[sender] model remap (by request count desc):")
    for src, dst in model_remap.items():
        print(f"  {src} -> {dst}")

    if dry_run:
        print("[sender] dry-run mode, only print first 3 payloads")
        for i, req in enumerate(requests[:3], start=1):
            payload = build_payload(req, model_remap)
            url = build_url(payload["model"])
            print(
                f"  #{i} t={req['time']:.3f}s src_model={req['model']} dst_model={payload['model']} "
                f"input_len={req['input_length']} output_len={req['output_length']} url={url}"
            )
        return

    sem = asyncio.Semaphore(max_inflight)
    tasks = []
    trace_records: list[dict] = []
    capture_trace = bool(requests_jsonl)
    monitor_target_npu_ids: list[int] = []
    npu_usage_lists_by_npu: dict[int, list[tuple[float, float]]] = {}
    npu_monitor_stop_event = threading.Event()
    npu_monitor_thread: threading.Thread | None = None

    if enable_npu_monitor:
        monitor_target_npu_ids = parse_monitor_npu_ids(monitor_npu_ids)
        npu_usage_lists_by_npu = {npu_id: [] for npu_id in monitor_target_npu_ids}

    connector = aiohttp.TCPConnector(limit=max_inflight, limit_per_host=max_inflight)
    timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=None, sock_read=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        try:
            if warmup:
                await warmup_models(
                    session=session,
                    requests=requests,
                    model_remap=model_remap,
                    build_url=build_url,
                    input_scale=input_scale,
                    max_tokens_scale=max_tokens_scale,
                )
            
            if enable_npu_monitor:
                npu_monitor_thread = threading.Thread(
                    target=npu_usage_monitor_worker,
                    args=(
                        npu_monitor_stop_event,
                        monitor_target_npu_ids,
                        monitor_interval_s,
                        npu_usage_lists_by_npu,
                    ),
                    daemon=True,
                    name="sender-npu-usage-monitor",
                )
                npu_monitor_thread.start()
                print(
                    "[sender] NPU usage monitor started: "
                    f"npu_ids={monitor_target_npu_ids}, interval_s={monitor_interval_s}"
                )

            start_ts = time.perf_counter()
            for idx, req in enumerate(requests, start=1):
                target_ts = start_ts + float(req["time"])
                now = time.perf_counter()
                if target_ts > now:
                    await asyncio.sleep(target_ts - now)

                payload = build_payload(req, model_remap)

                async def _do_send(i: int, p: dict) -> tuple[bool, float, str]:
                    await sem.acquire()
                    try:
                        url = build_url(p["model"])
                        return await send_one(
                            session,
                            url,
                            p,
                            i,
                            trace_records=trace_records if capture_trace else None,
                        )
                    finally:
                        sem.release()

                tasks.append(asyncio.create_task(_do_send(idx, payload)))

            results = await asyncio.gather(*tasks)
        finally:
            if enable_npu_monitor:
                npu_monitor_stop_event.set()
                if npu_monitor_thread is not None:
                    npu_monitor_thread.join(timeout=5.0)
                print("[sender] NPU usage monitor stopped")
                for npu_id in monitor_target_npu_ids:
                    samples = npu_usage_lists_by_npu.get(npu_id, [])
                    print(f"  NPU {npu_id}: usage_samples={len(samples)}")

                save_npu_usage_csv_and_plot(
                    usage_lists_by_npu=npu_usage_lists_by_npu,
                    csv_path=npu_monitor_csv,
                    plot_path=npu_monitor_plot,
                )

    ok = sum(1 for success, _, _ in results if success)
    bad = len(results) - ok
    avg_latency_ms = (sum(lat for _, lat, _ in results) / len(results) * 1000.0) if results else 0.0
    print(f"[sender] done: total={len(results)} success={ok} failed={bad} avg_latency={avg_latency_ms:.2f}ms")
    if bad:
        failure_type_counter: Counter[str] = Counter()
        for success, _, err in results:
            if success:
                continue
            if "exception_type=" in err:
                failure_type = err.split("exception_type=", 1)[1].split(" |", 1)[0]
            elif "http_status=" in err:
                failure_type = "HTTP_NON_200"
            else:
                failure_type = "UNKNOWN"
            failure_type_counter[failure_type] += 1

        print("[sender] failure type summary:")
        for failure_type, count in failure_type_counter.most_common():
            print(f"  - {failure_type}: {count}")

        print("[sender] first 5 errors:")
        shown = 0
        for success, lat, err in results:
            if success:
                continue
            print(f"  - latency_ms={lat * 1000.0:.2f} | {err}")
            shown += 1
            if shown >= 5:
                break

    if capture_trace:
        write_request_trace_and_summary(trace_records, requests_jsonl)


def _percentiles(values: list[float], ps=(50, 90, 95, 99)) -> dict[int, float]:
    if not values:
        return {p: float("nan") for p in ps}
    s = sorted(values)
    out: dict[int, float] = {}
    for p in ps:
        # nearest-rank percentile
        k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
        out[p] = s[k]
    return out


def write_request_trace_and_summary(trace_records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for rec in trace_records:
            f.write(json.dumps(rec) + "\n")
    print(f"[sender] request trace saved: {path} ({len(trace_records)} records)")

    # Per-model TTFT/TPOT summary over successful (status==200) records.
    by_model: dict[str, dict[str, list[float]]] = {}
    for rec in trace_records:
        if rec.get("status") != 200:
            continue
        model = rec.get("model") or "<none>"
        d = by_model.setdefault(model, {"ttft": [], "tpot": []})
        if rec.get("ttft_ms") is not None:
            d["ttft"].append(float(rec["ttft_ms"]))
        if rec.get("tpot_ms") is not None:
            d["tpot"].append(float(rec["tpot_ms"]))

    if not by_model:
        print("[sender] no successful streamed requests for TTFT/TPOT summary")
        return

    print("[sender] per-model TTFT/TPOT (ms):")
    for model in sorted(by_model.keys()):
        ttft = by_model[model]["ttft"]
        tpot = by_model[model]["tpot"]
        tp = _percentiles(ttft)
        pp = _percentiles(tpot)
        print(
            f"  {model}: n_ttft={len(ttft)} n_tpot={len(tpot)}\n"
            f"    TTFT p50={tp[50]:.1f} p90={tp[90]:.1f} p95={tp[95]:.1f} p99={tp[99]:.1f}\n"
            f"    TPOT p50={pp[50]:.2f} p90={pp[90]:.2f} p95={pp[95]:.2f} p99={pp[99]:.2f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trace sender with model remap and real request replay")
    p.add_argument("--trace-csv", default="trace_1_100.csv")
    p.add_argument("--target-qps", type=float, default=30.0)
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--max-tokens-scale", type=float, default=1.0)
    p.add_argument("--min-max-tokens", type=int, default=0, help="output_tokens 经缩放后的最小下限")
    p.add_argument("--max-context-cap", type=int, default=16384)
    p.add_argument("--limit", type=int, default=0, help="仅发送前 limit 条请求；<=0 表示不限制")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18119)
    p.add_argument(
        "--ports",
        default="",
        help=(
            "可选多端口配置。空=使用 --port；"
            "default=使用 run_080/run_080_2 默认模型端口；"
            "或 model=port,model=port；"
            "或纯端口列表(按 DEFAULT_WINDOW_MODEL_REMAP_ORDER 顺序映射)。"
        ),
    )
    p.add_argument("--max-inflight", type=int, default=500)
    p.add_argument("--warmup", dest="warmup", action="store_true", default=True, help="发送前按模型预热，每模型固定 1 个请求")
    p.add_argument("--no-warmup", dest="warmup", action="store_false", help="关闭预热")
    p.add_argument("--enable-npu-monitor", action="store_true", help="启动后台线程监控 NPU 利用率")
    p.add_argument("--monitor-npu-ids", default="8,9,10,11,12,13,14,15", help="要监控的 NPU ID，逗号分隔，例如 0,1,2,3")
    p.add_argument("--monitor-interval-s", type=float, default=1.0, help="NPU 利用率采样间隔（秒）")
    p.add_argument("--npu-monitor-csv", default="", help="可选：NPU 利用率 CSV 输出路径；为空时默认写入当前目录 npu_usage_timeline.csv")
    p.add_argument("--npu-monitor-plot", default="", help="可选：NPU 利用率时间轴图输出路径；为空时默认写入当前目录 npu_usage_timeline.png")
    p.add_argument("--dry-run", action="store_true", help="只构造并打印 payload，不实际发送")
    p.add_argument(
        "--requests-jsonl",
        default="",
        help="可选：设置后以 SSE 流式发送并逐请求记录 TTFT/TPOT 到该 JSONL；为空时保持原始（读全量 body）行为不变。",
    )
    args = p.parse_args()
    if args.monitor_interval_s <= 0:
        raise ValueError("--monitor-interval-s 必须 > 0")
    if args.min_max_tokens < 0:
        raise ValueError("--min-max-tokens 必须 >= 0")
    if args.port <= 0:
        raise ValueError("--port 必须 > 0")
    # 仅做格式校验，实际映射在 replay_requests 中使用。
    _ = parse_model_ports_arg(args.ports)
    if not args.npu_monitor_csv:
        args.npu_monitor_csv = "npu_usage_timeline.csv"
    if not args.npu_monitor_plot:
        args.npu_monitor_plot = "npu_usage_timeline.png"
    return args

if __name__ == "__main__":
    args = parse_args()
    trace_df = pd.read_csv(args.trace_csv)
    requests = list(
        request_sender(
            trace_df,
            target_qps=args.target_qps,
            input_scale=args.input_scale,
            max_tokens_scale=args.max_tokens_scale,
            min_max_tokens=args.min_max_tokens,
            max_context_cap=args.max_context_cap,
        )
    )
    if args.limit > 0:
        requests = requests[:args.limit]
    print(f"[sender] effective_requests={len(requests)} (limit={args.limit})")

    model_remap = build_model_remap(requests)
    asyncio.run(
        replay_requests(
            requests=requests,
            model_remap=model_remap,
            host=args.host,
            port=args.port,
            ports=args.ports,
            max_inflight=args.max_inflight,
            warmup=args.warmup,
            input_scale=args.input_scale,
            max_tokens_scale=args.max_tokens_scale,
            enable_npu_monitor=args.enable_npu_monitor,
            monitor_npu_ids=args.monitor_npu_ids,
            monitor_interval_s=args.monitor_interval_s,
            npu_monitor_csv=args.npu_monitor_csv,
            npu_monitor_plot=args.npu_monitor_plot,
            dry_run=args.dry_run,
            requests_jsonl=args.requests_jsonl,
        )
    )