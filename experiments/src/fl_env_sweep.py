#!/usr/bin/env python3
"""Sweep only vllm-plugin-FL env knobs; keep vLLM serve CLI fixed."""

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
OUT_DIR = Path("/root/bench_results/fl_env_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed vLLM CLI (not under test)
SERVE_CMD = [
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
    "--max-model-len",
    "2048",
    "--gpu-memory-utilization",
    "0.85",
    "--no-enable-log-requests",
]

BENCH_CASES = [
    ("latency_c1", 512, 128, 1, 16),
    ("throughput_c8", 512, 128, 8, 24),
]

# FL-only env configs. CLI stays identical.
# Keys we may set/clear between runs:
FL_ENV_KEYS = [
    "VLLM_FL_PREFER_ENABLED",
    "VLLM_FL_PREFER",
    "VLLM_FL_STRICT",
    "VLLM_FL_PER_OP",
    "VLLM_FL_FLAGOS_WHITELIST",
    "VLLM_FL_FLAGOS_BLACKLIST",
    "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    "VLLM_FL_OOT_ENABLED",
    "VLLM_FL_OOT_WHITELIST",
    "VLLM_FL_OOT_BLACKLIST",
    "USE_FLAGGEMS",
    "VLLM_FL_PLATFORM",
    "VLLM_FL_CONFIG",
    "VLLM_FL_DISPATCH_DEBUG",
]

CONFIGS = [
    (
        "baseline_flagos",
        "Default FL: prefer=flagos + FlagGems (metax.yaml)",
        {},
    ),
    (
        "prefer_vendor",
        "VLLM_FL_PREFER=vendor (MetaX vendor kernels preferred)",
        {"VLLM_FL_PREFER": "vendor"},
    ),
    (
        "prefer_reference",
        "VLLM_FL_PREFER=reference (PyTorch native preferred)",
        {"VLLM_FL_PREFER": "reference"},
    ),
    (
        "prefer_disabled",
        "VLLM_FL_PREFER_ENABLED=0 (disable FL dispatch/FlagGems prefer path)",
        {"VLLM_FL_PREFER_ENABLED": "0"},
    ),
    (
        "flaggems_off",
        "USE_FLAGGEMS=0 (keep prefer=flagos but disable FlagGems)",
        {"USE_FLAGGEMS": "0"},
    ),
    (
        "oot_off",
        "VLLM_FL_OOT_ENABLED=0 (disable OOT op registration)",
        {"VLLM_FL_OOT_ENABLED": "0"},
    ),
    (
        "per_op_fused_reference",
        "Force fused ops to reference via VLLM_FL_PER_OP",
        {
            "VLLM_FL_PER_OP": "silu_and_mul=reference;rms_norm=reference;rotary_embedding=reference"
        },
    ),
    (
        "per_op_fused_vendor",
        "Force fused ops to vendor via VLLM_FL_PER_OP",
        {
            "VLLM_FL_PER_OP": "silu_and_mul=vendor|reference;rms_norm=vendor|reference;rotary_embedding=vendor|reference;attention_backend=vendor:metax"
        },
    ),
    (
        "whitelist_fused_only",
        "Only silu_and_mul,rms_norm,rotary_embedding use FlagGems",
        {"VLLM_FL_FLAGOS_WHITELIST": "silu_and_mul,rms_norm,rotary_embedding"},
    ),
    (
        "blacklist_append_linear",
        "Append linear to FlagGems blacklist",
        {"VLLM_FL_FLAGOS_BLACKLIST_APPEND": "linear"},
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
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(2)
    subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], check=False)
    subprocess.run(["pkill", "-9", "-f", "vllm serve"], check=False)
    time.sleep(2)


def wait_ready(timeout_s: int = 360) -> bool:
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


def make_env(overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for k in FL_ENV_KEYS:
        env.pop(k, None)
    env.update(overrides)
    return env


def start_server(cfg_id: str, overrides: dict[str, str]) -> subprocess.Popen:
    env = make_env(overrides)
    log(f"Starting [{cfg_id}] FL env={overrides or '{}'}")
    fp = open(OUT_DIR / f"server_{cfg_id}.log", "w")
    proc = subprocess.Popen(
        SERVE_CMD,
        stdout=fp,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    proc._log_fp = fp  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=20)
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
    log(f"Bench [{tag}]")
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
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    (OUT_DIR / f"{tag}.log").write_text(text)
    metrics = parse_metrics(text)
    metrics["exit_code"] = float(p.returncode)
    metrics["case"] = case_name  # type: ignore[assignment]
    metrics["input_len"] = float(in_len)
    metrics["output_len"] = float(out_len)
    metrics["concurrency"] = float(conc)
    metrics["num_prompts"] = float(n)
    log(
        f"  status={p.returncode} TTFT_p50={metrics.get('median_ttft_ms')} "
        f"TPOT_p50={metrics.get('median_tpot_ms')} out_tok/s={metrics.get('out_tok_s')}"
    )
    return metrics


def main() -> None:
    kill_vllm()
    rows: list[dict] = []
    summary_path = OUT_DIR / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    for cfg_id, desc, overrides in CONFIGS:
        log("=" * 60)
        log(f"CONFIG {cfg_id}: {desc}")
        proc = None
        try:
            proc = start_server(cfg_id, overrides)
            if not wait_ready(360):
                log(f"Server failed for {cfg_id}")
                slog = OUT_DIR / f"server_{cfg_id}.log"
                if slog.exists():
                    log("tail:\n" + "\n".join(slog.read_text().splitlines()[-50:]))
                row = {
                    "config": cfg_id,
                    "description": desc,
                    "status": "server_failed",
                    "fl_env": overrides,
                }
                rows.append(row)
                summary_path.open("a").write(json.dumps(row) + "\n")
                continue

            for case_name, in_len, out_len, conc, n in BENCH_CASES:
                m = run_bench(cfg_id, case_name, in_len, out_len, conc, n)
                row = {
                    "config": cfg_id,
                    "description": desc,
                    "status": "ok" if m.get("exit_code", 1) == 0 else "bench_failed",
                    "fl_env": overrides,
                    **m,
                }
                rows.append(row)
                summary_path.open("a").write(json.dumps(row) + "\n")
        finally:
            stop_server(proc)

    (OUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
    log(f"Wrote {OUT_DIR / 'summary.json'}")
    print("FL_ENV_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
