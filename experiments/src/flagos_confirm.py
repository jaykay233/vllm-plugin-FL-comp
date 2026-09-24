#!/usr/bin/env python3
"""Confirm the winning FlagOS config is reproducible."""

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
OUT_DIR = Path("/root/bench_results/flagos_confirm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERVE_CMD = [
    "vllm", "serve", MODEL,
    "--dtype", "bfloat16", "--trust-remote-code",
    "--host", "0.0.0.0", "--port", str(PORT),
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.85",
    "--no-enable-log-requests",
]

BENCH_CASES = [
    ("latency_c1", 512, 128, 1, 16),
    ("throughput_c8", 512, 128, 8, 24),
]

BASE_ENV = {"VLLM_FL_PREFER": "flagos", "USE_FLAGGEMS": "1"}

CONFIGS = [
    ("A_default_rerun", "FlagOS all-GEMS default (re-run 1)", {}),
    ("B_default_rerun2", "FlagOS all-GEMS default (re-run 2)", {}),
    (
        "C_whitelist_fused",
        "whitelist fused only: silu_and_mul,rms_norm,rotary_embedding (re-run 1)",
        {"VLLM_FL_FLAGOS_WHITELIST": "silu_and_mul,rms_norm,rotary_embedding"},
    ),
    (
        "D_whitelist_fused2",
        "whitelist fused only (re-run 2)",
        {"VLLM_FL_FLAGOS_WHITELIST": "silu_and_mul,rms_norm,rotary_embedding"},
    ),
    (
        "E_whitelist_plus_fused_add",
        "whitelist + fused_add_rms_norm",
        {"VLLM_FL_FLAGOS_WHITELIST": "silu_and_mul,rms_norm,fused_add_rms_norm,rotary_embedding"},
    ),
    (
        "F_whitelist_rms_only",
        "whitelist rms_norm only",
        {"VLLM_FL_FLAGOS_WHITELIST": "rms_norm"},
    ),
]

FL_ENV_KEYS = [
    "VLLM_FL_PREFER_ENABLED", "VLLM_FL_PREFER", "VLLM_FL_STRICT", "VLLM_FL_PER_OP",
    "VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_FLAGOS_BLACKLIST", "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    "VLLM_FL_OOT_ENABLED", "VLLM_FL_OOT_WHITELIST", "VLLM_FL_OOT_BLACKLIST",
    "USE_FLAGGEMS", "VLLM_FL_PLATFORM", "VLLM_FL_CONFIG", "VLLM_FL_DISPATCH_DEBUG",
    "FLAGGEMS_ENABLE_OPLIST_PATH",
]

METRIC_PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "req_s": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "out_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
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
    "median_e2el_ms": r"Median E2EL \(ms\):\s+([0-9.]+)",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def kill_vllm() -> None:
    for pat in ("vllm serve", "vllm bench", "VLLM::EngineCore"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(3)


def wait_ready(timeout_s: int = 360) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = subprocess.run(["curl", "-sf", f"http://{HOST}:{PORT}/v1/models"],
                           capture_output=True, text=True)
        if r.returncode == 0 and "MiniCPM5-2B" in r.stdout:
            return True
        time.sleep(2)
    return False


def parse_metrics(text: str) -> dict[str, float]:
    out = {}
    for key, pat in METRIC_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def run_case(cfg_id, case_name, in_len, out_len, conc, n):
    tag = f"{cfg_id}__{case_name}"
    cmd = ["vllm", "bench", "serve", "--backend", "vllm", "--model", MODEL,
           "--tokenizer", MODEL, "--endpoint", "/v1/completions", "--host", HOST,
           "--port", str(PORT), "--dataset-name", "random", "--ignore-eos",
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
           "--random-input-len", str(in_len), "--random-output-len", str(out_len),
           "--max-concurrency", str(conc), "--num-prompts", str(n),
           "--save-result", "--result-dir", str(OUT_DIR), "--result-filename", f"{tag}.json"]
    subprocess.run(["curl", "-s", f"http://{HOST}:{PORT}/v1/completions",
                    "-H", "Content-Type: application/json", "-d",
                    json.dumps({"model": MODEL, "prompt": "warmup", "max_tokens": 16,
                                "temperature": 0, "ignore_eos": True})],
                   capture_output=True, text=True, timeout=180)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    (OUT_DIR / f"{tag}.log").write_text(text)
    m = parse_metrics(text)
    m["exit_code"] = float(p.returncode)
    m["case"] = case_name
    m["input_len"] = float(in_len); m["output_len"] = float(out_len)
    m["concurrency"] = float(conc); m["num_prompts"] = float(n)
    log(f"  [{tag}] TTFT_p50={m.get('median_ttft_ms')} TPOT_p50={m.get('median_tpot_ms')} "
        f"out_tok/s={m.get('out_tok_s')}")
    return m


def main() -> None:
    kill_vllm()
    rows = []
    summary = OUT_DIR / "summary.jsonl"
    if summary.exists():
        summary.unlink()

    for cfg_id, desc, extra in CONFIGS:
        log("=" * 60)
        log(f"CONFIG {cfg_id}: {desc}")
        env = os.environ.copy()
        for k in FL_ENV_KEYS:
            env.pop(k, None)
        overrides = dict(BASE_ENV)
        overrides.update(extra)
        oplist = OUT_DIR / f"oplist_{cfg_id}.txt"
        overrides["FLAGGEMS_ENABLE_OPLIST_PATH"] = str(oplist)
        env.update(overrides)
        if oplist.exists():
            oplist.unlink()

        proc = None
        try:
            fp = open(OUT_DIR / f"server_{cfg_id}.log", "w")
            proc = subprocess.Popen(SERVE_CMD, stdout=fp, stderr=subprocess.STDOUT,
                                    env=env, preexec_fn=os.setsid)
            if not wait_ready(360):
                log(f"Server failed for {cfg_id}")
                row = {"config": cfg_id, "description": desc, "status": "server_failed", "fl_env": overrides}
                rows.append(row); summary.open("a").write(json.dumps(row) + "\n")
                continue
            for case_name, i, o, c, n in BENCH_CASES:
                m = run_case(cfg_id, case_name, i, o, c, n)
                row = {"config": cfg_id, "description": desc,
                       "status": "ok" if m["exit_code"] == 0 else "bench_failed",
                       "fl_env": overrides, **m}
                rows.append(row); summary.open("a").write(json.dumps(row) + "\n")
        finally:
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
            kill_vllm()

    (OUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
    print("CONFIRM_DONE", flush=True)


if __name__ == "__main__":
    main()
