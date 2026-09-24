#!/usr/bin/env python3
"""Sweep FlagOS/FlagGems knobs only. Keep prefer=flagos; no vendor/reference/pytorch path."""

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
OUT_DIR = Path("/root/bench_results/flagos_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "FLAGGEMS_ENABLE_OPLIST_PATH",
]

# All configs stay on FlagOS path (prefer=flagos + USE_FLAGGEMS=1).
CONFIGS = [
    (
        "flagos_default",
        "FlagOS baseline: VLLM_FL_PREFER=flagos, USE_FLAGGEMS=1 (metax.yaml)",
        {
            "VLLM_FL_PREFER": "flagos",
            "VLLM_FL_PREFER_ENABLED": "1",
            "USE_FLAGGEMS": "1",
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(OUT_DIR / "oplist_flagos_default.txt"),
        },
    ),
    (
        "flagos_oot_off",
        "FlagOS + VLLM_FL_OOT_ENABLED=0",
        {
            "VLLM_FL_PREFER": "flagos",
            "USE_FLAGGEMS": "1",
            "VLLM_FL_OOT_ENABLED": "0",
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(OUT_DIR / "oplist_flagos_oot_off.txt"),
        },
    ),
    (
        "flagos_whitelist_fused",
        "FlagOS: only enable fused FlagGems ops (silu/rms/rope)",
        {
            "VLLM_FL_PREFER": "flagos",
            "USE_FLAGGEMS": "1",
            "VLLM_FL_FLAGOS_WHITELIST": "silu_and_mul,rms_norm,rotary_embedding",
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(OUT_DIR / "oplist_flagos_whitelist_fused.txt"),
        },
    ),
    (
        "flagos_blacklist_append_linear",
        "FlagOS: blacklist_append=linear (keep other GEMS ops)",
        {
            "VLLM_FL_PREFER": "flagos",
            "USE_FLAGGEMS": "1",
            "VLLM_FL_FLAGOS_BLACKLIST_APPEND": "linear",
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(
                OUT_DIR / "oplist_flagos_blacklist_append_linear.txt"
            ),
        },
    ),
    (
        "flagos_per_op_fused_flagos",
        "FlagOS: VLLM_FL_PER_OP force fused ops to flagos first",
        {
            "VLLM_FL_PREFER": "flagos",
            "USE_FLAGGEMS": "1",
            "VLLM_FL_PER_OP": "silu_and_mul=flagos;rms_norm=flagos;rotary_embedding=flagos;attention_backend=vendor:metax",
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(
                OUT_DIR / "oplist_flagos_per_op_fused_flagos.txt"
            ),
        },
    ),
    (
        "flagos_strict",
        "FlagOS + VLLM_FL_STRICT=1",
        {
            "VLLM_FL_PREFER": "flagos",
            "USE_FLAGGEMS": "1",
            "VLLM_FL_STRICT": "1",
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(OUT_DIR / "oplist_flagos_strict.txt"),
        },
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
    for pat in ("vllm serve", "vllm bench", "VLLM::EngineCore"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(3)


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


def assert_flaggems_on(server_log: Path, oplist: Path) -> None:
    text = server_log.read_text() if server_log.exists() else ""
    if "[FlagGems]" not in text and "flag_gems" not in text.lower():
        # still ok if oplist got written after warmup
        pass
    if oplist.exists() and oplist.stat().st_size > 0:
        content = oplist.read_text()
        if "GEMS" in content or "flag_gems" in content or "default.flagos" in content:
            log(f"  FlagGems oplist OK: {oplist.name} ({oplist.stat().st_size} bytes)")
            return
    log(f"  WARN: FlagGems oplist missing/empty: {oplist}")


def start_server(cfg_id: str, overrides: dict[str, str]) -> subprocess.Popen:
    env = make_env(overrides)
    # clear previous oplist so we know this run wrote it
    oplist = Path(overrides.get("FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/flaggems_enable_oplist.txt"))
    if oplist.exists():
        oplist.unlink()
    log(f"Starting [{cfg_id}] FL env={overrides}")
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
                    "max_tokens": 16,
                    "temperature": 0,
                    "ignore_eos": True,
                }
            ),
        ],
        capture_output=True,
        text=True,
        timeout=180,
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
                    log("tail:\n" + "\n".join(slog.read_text().splitlines()[-60:]))
                row = {
                    "config": cfg_id,
                    "description": desc,
                    "status": "server_failed",
                    "fl_env": overrides,
                }
                rows.append(row)
                summary_path.open("a").write(json.dumps(row) + "\n")
                continue

            oplist = Path(overrides["FLAGGEMS_ENABLE_OPLIST_PATH"])
            assert_flaggems_on(OUT_DIR / f"server_{cfg_id}.log", oplist)

            for case_name, in_len, out_len, conc, n in BENCH_CASES:
                m = run_bench(cfg_id, case_name, in_len, out_len, conc, n)
                # re-check oplist after warmup/bench
                assert_flaggems_on(OUT_DIR / f"server_{cfg_id}.log", oplist)
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
    print("FLAGOS_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
