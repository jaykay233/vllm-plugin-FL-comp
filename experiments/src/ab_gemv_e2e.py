#!/usr/bin/env python3
"""End-to-end A/B: does the MetaX M=1 GEMV fast path actually speed up serving?

Runs `vllm serve` twice against the same build -- once with the fast path
disabled (FLAG_GEMS_METAX_GEMV=0) and once enabled -- and compares the metrics
`vllm bench serve` reports. Concurrency 1 keeps the decode batch at M=1, which
is the case the kernel targets; concurrency 8 exercises the fallback path.

Also checks the reachability marker, so a null result can be told apart from
"the fast path never ran".

Run:  conda activate mx && python /root/src/ab_gemv_e2e.py
"""

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
OUT_DIR = Path("/root/bench_results/gemv_ab")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# name, input len, output len, concurrency, num prompts
CASES = [
    ("latency_c1", 512, 128, 1, 24),
    ("throughput_c8", 512, 128, 8, 32),
]

SETTINGS = [
    ("gemv_off", {"FLAG_GEMS_METAX_GEMV": "0"}, "generic linear_kernel (baseline)"),
    ("gemv_on", {"FLAG_GEMS_METAX_GEMV": "1"}, "MetaX M=1 GEMV fast path"),
]

BASE_SERVE = [
    "vllm", "serve", MODEL,
    "--dtype", "bfloat16",
    "--trust-remote-code",
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "--no-enable-log-requests",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.85",
]

METRICS = {
    "req_s": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "out_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
    "mean_e2el_ms": r"Mean E2EL \(ms\):\s+([0-9.]+)",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def kill_vllm() -> None:
    subprocess.run(["pkill", "-9", "-f", "vllm serve"], check=False)
    subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], check=False)
    time.sleep(3)


def wait_ready(timeout_s: int = 420) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = subprocess.run(
            ["curl", "-sf", f"http://{HOST}:{PORT}/v1/models"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and "MiniCPM5-2B" in r.stdout:
            return True
        time.sleep(3)
    return False


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pat in METRICS.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def run_case(tag: str, in_len: int, out_len: int, conc: int, n: int) -> dict:
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", MODEL,
        "--tokenizer", MODEL,
        "--endpoint", "/v1/completions",
        "--host", HOST, "--port", str(PORT),
        "--dataset-name", "random",
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--random-input-len", str(in_len),
        "--random-output-len", str(out_len),
        "--max-concurrency", str(conc),
        "--num-prompts", str(n),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    (OUT_DIR / f"{tag}.log").write_text(text)
    m = parse_metrics(text)
    m["exit_code"] = p.returncode
    return m


def main() -> None:
    kill_vllm()
    results: dict[str, dict[str, dict]] = {}

    for cfg_id, env_over, desc in SETTINGS:
        log("=" * 72)
        log(f"SETTING {cfg_id}: {desc}  env={env_over}")
        trace = OUT_DIR / f"trace_{cfg_id}.txt"
        if trace.exists():
            trace.unlink()

        env = os.environ.copy()
        env.update(env_over)
        env["FLAG_GEMS_METAX_GEMV_TRACE"] = str(trace)

        slog_path = OUT_DIR / f"server_{cfg_id}.log"
        with open(slog_path, "w") as fp:
            proc = subprocess.Popen(
                BASE_SERVE, stdout=fp, stderr=subprocess.STDOUT, env=env,
                preexec_fn=os.setsid,
            )
        try:
            if not wait_ready():
                log(f"  server NOT ready; tail of log:")
                log("\n".join(slog_path.read_text().splitlines()[-30:]))
                results[cfg_id] = {"_status": "server_failed"}
                continue
            log(f"  server ready; marker={'yes' if trace.exists() else 'no'} "
                f"({trace.read_text().strip() if trace.exists() else '-'})")

            results[cfg_id] = {"_marker": trace.read_text().strip() if trace.exists() else None}
            for case, in_len, out_len, conc, n in CASES:
                log(f"  bench {case} (in={in_len} out={out_len} c={conc} n={n})")
                m = run_case(f"{cfg_id}__{case}", in_len, out_len, conc, n)
                results[cfg_id][case] = m
                log(f"    TPOT_p50={m.get('median_tpot_ms')}ms "
                    f"out_tok/s={m.get('out_tok_s')} "
                    f"TTFT_p50={m.get('median_ttft_ms')}ms")
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
            kill_vllm()

    # ---- report ----------------------------------------------------------
    log("=" * 72)
    log("A/B RESULT")
    log("=" * 72)
    off, on = results.get("gemv_off", {}), results.get("gemv_on", {})
    for case, *_ in CASES:
        o, n_ = off.get(case), on.get(case)
        if not o or not n_:
            continue
        print(f"\n  {case}")
        for k in ("median_tpot_ms", "mean_tpot_ms", "median_ttft_ms",
                  "out_tok_s", "total_tok_s", "mean_itl_ms"):
            if k in o and k in n_:
                a, b = o[k], n_[k]
                if a and b:
                    print(f"    {k:18s} off={a:10.3f}  on={b:10.3f}  "
                          f"{'x%.3f' % (b / a)}")
    print(f"\n  marker off: {off.get('_marker')}")
    print(f"  marker on : {on.get('_marker')}")
    (OUT_DIR / "ab_results.json").write_text(json.dumps(results, indent=2))
    log(f"Wrote {OUT_DIR / 'ab_results.json'}")
    print("AB_GEMV_DONE", flush=True)


if __name__ == "__main__":
    main()
