#!/usr/bin/env python3
"""Sweep vLLM serve launch args for MiniCPM5-2B and compare vs baseline."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

MODEL = "/root/models/MiniCPM5-2B"
HOST = "127.0.0.1"
PORT = 8000
OUT_DIR = Path("/root/bench_results/tuning_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fair fixed workloads
BENCH_CASES = [
    # name, input, output, concurrency, num_prompts
    ("latency_c1", 512, 128, 1, 16),
    ("throughput_c8", 512, 128, 8, 24),
]

COMMON_SERVE = [
    "vllm",
    "serve",
    MODEL,
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--host",
    "0.0.0.0",
    "--port",
    str(PORT),
    "--no-enable-log-requests",
]

# Each config: (id, description, extra_cli_args, env_overrides)
CONFIGS = [
    (
        "baseline",
        "Current baseline: max-model-len=2048, gmu=0.85",
        ["--max-model-len", "2048", "--gpu-memory-utilization", "0.85"],
        {},
    ),
    (
        "gmu95",
        "Higher GPU memory util 0.95",
        ["--max-model-len", "2048", "--gpu-memory-utilization", "0.95"],
        {},
    ),
    (
        "no_prefix_cache",
        "Disable prefix caching (FlagOS bench default)",
        [
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.85",
            "--no-enable-prefix-caching",
        ],
        {},
    ),
    (
        "enforce_eager",
        "Disable CUDA graphs (--enforce-eager)",
        [
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.85",
            "--enforce-eager",
        ],
        {},
    ),
    (
        "cudagraph_full",
        "FULL_AND_PIECEWISE cudagraph",
        [
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.85",
            "--compilation-config",
            '{"cudagraph_mode":"FULL_AND_PIECEWISE"}',
        ],
        {},
    ),
    (
        "batched_16k_seqs_256",
        "max-num-batched-tokens=16384, max-num-seqs=256",
        [
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.85",
            "--max-num-batched-tokens",
            "16384",
            "--max-num-seqs",
            "256",
        ],
        {},
    ),
    (
        "batched_8k_seqs_128",
        "max-num-batched-tokens=8192, max-num-seqs=128",
        [
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.85",
            "--max-num-batched-tokens",
            "8192",
            "--max-num-seqs",
            "128",
        ],
        {},
    ),
    (
        "flaggems_off",
        "USE_FLAGGEMS=0 (native ops)",
        ["--max-model-len", "2048", "--gpu-memory-utilization", "0.85"],
        {"USE_FLAGGEMS": "0"},
    ),
    (
        "combo_best_guess",
        "gmu=0.95 + no-prefix-cache + batched 16k/seqs 256",
        [
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.95",
            "--no-enable-prefix-caching",
            "--max-num-batched-tokens",
            "16384",
            "--max-num-seqs",
            "256",
        ],
        {},
    ),
]

METRIC_PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "req_s": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "out_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "peak_out_tok_s": r"Peak output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
    "mean_e2el_ms": r"Mean E2EL \(ms\):\s+([0-9.]+)",
    "median_e2el_ms": r"Median E2EL \(ms\):\s+([0-9.]+)",
    "p99_e2el_ms": r"P99 E2EL \(ms\):\s+([0-9.]+)",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def kill_vllm() -> None:
    # Kill serve processes carefully
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(2)
    # Force leftover engine cores
    subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], check=False)
    subprocess.run(["pkill", "-9", "-f", "vllm serve"], check=False)
    time.sleep(2)


def wait_ready(timeout_s: int = 300) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://{HOST}:{PORT}/v1/models"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and "MiniCPM5-2B" in r.stdout:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_server(cfg_id: str, extra_args: list[str], env_over: dict[str, str]) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(env_over)
    cmd = COMMON_SERVE + extra_args
    log_path = OUT_DIR / f"server_{cfg_id}.log"
    log(f"Starting server [{cfg_id}]: {' '.join(cmd)}")
    if env_over:
        log(f"  env overrides: {env_over}")
    fp = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=fp,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    # stash for cleanup
    proc._log_fp = fp  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            proc._log_fp.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    kill_vllm()


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pat in METRIC_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def run_bench(cfg_id: str, case_name: str, in_len: int, out_len: int, conc: int, n: int) -> dict:
    tag = f"{cfg_id}__{case_name}"
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        MODEL,
        "--tokenizer",
        MODEL,
        "--endpoint",
        "/v1/completions",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--dataset-name",
        "random",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--random-input-len",
        str(in_len),
        "--random-output-len",
        str(out_len),
        "--max-concurrency",
        str(conc),
        "--num-prompts",
        str(n),
        "--save-result",
        "--result-dir",
        str(OUT_DIR),
        "--result-filename",
        f"{tag}.json",
    ]
    log(f"Bench [{tag}] in={in_len} out={out_len} c={conc} n={n}")
    # warm-up
    subprocess.run(
        [
            "curl",
            "-s",
            f"http://{HOST}:{PORT}/v1/completions",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(
                {
                    "model": MODEL,
                    "prompt": "warmup",
                    "max_tokens": 8,
                    "temperature": 0,
                    "ignore_eos": True,
                }
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    log_text = (p.stdout or "") + "\n" + (p.stderr or "")
    (OUT_DIR / f"{tag}.log").write_text(log_text)
    metrics = parse_metrics(log_text)
    metrics["exit_code"] = float(p.returncode)
    metrics["case"] = case_name  # type: ignore[assignment]
    metrics["config"] = cfg_id  # type: ignore[assignment]
    metrics["input_len"] = float(in_len)
    metrics["output_len"] = float(out_len)
    metrics["concurrency"] = float(conc)
    metrics["num_prompts"] = float(n)
    if p.returncode != 0:
        log(f"  FAILED exit={p.returncode}")
    else:
        log(
            f"  TTFT_p50={metrics.get('median_ttft_ms')}  "
            f"TPOT_p50={metrics.get('median_tpot_ms')}  "
            f"out_tok/s={metrics.get('out_tok_s')}"
        )
    return metrics


def main() -> None:
    kill_vllm()
    all_rows: list[dict] = []
    summary_path = OUT_DIR / "summary.jsonl"

    for cfg_id, desc, extra, env_over in CONFIGS:
        log("=" * 60)
        log(f"CONFIG {cfg_id}: {desc}")
        proc = None
        try:
            proc = start_server(cfg_id, extra, env_over)
            if not wait_ready(360):
                log(f"Server failed to become ready for {cfg_id}")
                # capture tail of server log
                slog = OUT_DIR / f"server_{cfg_id}.log"
                if slog.exists():
                    log("Server log tail:\n" + "\n".join(slog.read_text().splitlines()[-40:]))
                row = {
                    "config": cfg_id,
                    "description": desc,
                    "status": "server_failed",
                    "extra_args": extra,
                    "env": env_over,
                }
                all_rows.append(row)
                with summary_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                continue

            log(f"Server ready for {cfg_id}")
            for case_name, in_len, out_len, conc, n in BENCH_CASES:
                m = run_bench(cfg_id, case_name, in_len, out_len, conc, n)
                row = {
                    "config": cfg_id,
                    "description": desc,
                    "status": "ok" if m.get("exit_code", 1) == 0 else "bench_failed",
                    "extra_args": extra,
                    "env": env_over,
                    **{k: v for k, v in m.items() if k not in ("config",)},
                }
                all_rows.append(row)
                with summary_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
        finally:
            stop_server(proc)

    (OUT_DIR / "summary.json").write_text(json.dumps(all_rows, indent=2))
    log(f"Done. Wrote {OUT_DIR / 'summary.json'}")
    print("TUNING_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
