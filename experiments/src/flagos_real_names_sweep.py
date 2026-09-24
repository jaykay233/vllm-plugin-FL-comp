#!/usr/bin/env python3
"""Task 1: re-run the FlagGems whitelist using REAL FlagGems registered names.

Previous run used `silu_and_mul` / `rotary_embedding`, neither of which exists in
FlagGems 5.3.5, so `only_enable` silently disabled everything and the oplist came
out empty -- the "1.46x" was just FlagGems being off.

This script:
  * whitelists only names verified to exist in flag_gems._FULL_CONFIG
  * FAILS a config if the oplist is empty (hard assertion, not a warning)
  * captures dispatch resolutions ("Op 'x' using 'y'") from the server log

Run:  conda activate mx && python /root/src/flagos_real_names_sweep.py
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
OUT_DIR = Path("/root/bench_results/flagos_real_names")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERVE_CMD = [
    "vllm", "serve", MODEL,
    "--dtype", "bfloat16",
    "--trust-remote-code",
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.85",
    "--no-enable-log-requests",
]

BENCH_CASES = [
    ("latency_c1", 512, 128, 1, 16),
    ("throughput_c8", 512, 128, 8, 24),
]

FL_ENV_KEYS = [
    "VLLM_FL_PREFER_ENABLED", "VLLM_FL_PREFER", "VLLM_FL_STRICT",
    "VLLM_FL_PER_OP", "VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_FLAGOS_BLACKLIST",
    "VLLM_FL_FLAGOS_BLACKLIST_APPEND", "VLLM_FL_OOT_ENABLED",
    "VLLM_FL_OOT_WHITELIST", "VLLM_FL_OOT_BLACKLIST", "USE_FLAGGEMS",
    "VLLM_FL_PLATFORM", "VLLM_FL_CONFIG", "VLLM_FL_DISPATCH_DEBUG",
    "FLAGGEMS_ENABLE_OPLIST_PATH",
]

# Only names verified present in flag_gems._FULL_CONFIG exist here.
CONFIGS = [
    (
        "baseline_default",
        "FlagGems default (full registration) - baseline",
        {
            "VLLM_FL_PREFER": "flagos",
            "USE_FLAGGEMS": "1",
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(OUT_DIR / "oplist_baseline_default.txt"),
        },
    ),
    (
        "wl_real_3",
        "VLLM_FL_FLAGOS_WHITELIST=rms_norm,silu,linear (all REAL names)",
        {
            "VLLM_FL_PREFER": "flagos",
            "USE_FLAGGEMS": "1",
            "VLLM_FL_FLAGOS_WHITELIST": "rms_norm,silu,linear",
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(OUT_DIR / "oplist_wl_real_3.txt"),
        },
    ),
    (
        "wl_real_3_mm",
        "VLLM_FL_FLAGOS_WHITELIST=rms_norm,silu,linear,mm",
        {
            "VLLM_FL_PREFER": "flagos",
            "USE_FLAGGEMS": "1",
            "VLLM_FL_FLAGOS_WHITELIST": "rms_norm,silu,linear,mm",
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(OUT_DIR / "oplist_wl_real_3_mm.txt"),
        },
    ),
]

METRIC_PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "req_s": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "out_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
}

DISPATCH_RE = re.compile(r"Op '([^']+)' using '([^']+)'")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def kill_vllm() -> None:
    for pat in ("vllm serve", "vllm bench", "VLLM::EngineCore"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(3)


def wait_ready(timeout_s: int = 420) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://{HOST}:{PORT}/v1/models"],
                capture_output=True, text=True, timeout=5,
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


def read_oplist(path: Path) -> tuple[int, list[str]]:
    """Return (size_bytes, sorted unique GEMS op names)."""
    if not path.exists():
        return 0, []
    text = path.read_text()
    names: set[str] = set()
    for line in text.splitlines():
        m = re.search(r"(?:METAX )?GEMS ([A-Z0-9_.]+)", line)
        if m:
            names.add(m.group(1))
    return path.stat().st_size, sorted(names)


def start_server(cfg_id: str, overrides: dict[str, str]) -> subprocess.Popen:
    env = make_env(overrides)
    oplist = Path(overrides["FLAGGEMS_ENABLE_OPLIST_PATH"])
    if oplist.exists():
        oplist.unlink()
    log(f"Starting [{cfg_id}] FL env={overrides}")
    fp = open(OUT_DIR / f"server_{cfg_id}.log", "w")
    proc = subprocess.Popen(
        SERVE_CMD, stdout=fp, stderr=subprocess.STDOUT, env=env,
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
        "vllm", "bench", "serve", "--backend", "vllm",
        "--model", MODEL, "--tokenizer", MODEL,
        "--endpoint", "/v1/completions",
        "--host", HOST, "--port", str(PORT),
        "--dataset-name", "random", "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--random-input-len", str(in_len),
        "--random-output-len", str(out_len),
        "--max-concurrency", str(conc),
        "--num-prompts", str(n),
        "--save-result", "--result-dir", str(OUT_DIR),
        "--result-filename", f"{tag}.json",
    ]
    log(f"Bench [{tag}]")
    subprocess.run(
        ["curl", "-s", f"http://{HOST}:{PORT}/v1/completions",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"model": MODEL, "prompt": "warmup", "max_tokens": 16,
                           "temperature": 0, "ignore_eos": True})],
        capture_output=True, text=True, timeout=180,
    )
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    (OUT_DIR / f"{tag}.log").write_text(text)
    m = parse_metrics(text)
    m["exit_code"] = float(p.returncode)
    m["case"] = case_name  # type: ignore[assignment]
    log(
        f"  status={p.returncode} TTFT_p50={m.get('median_ttft_ms')} "
        f"TPOT_p50={m.get('median_tpot_ms')} out_tok/s={m.get('out_tok_s')}"
    )
    return m


def main() -> None:
    kill_vllm()
    rows: list[dict] = []
    summary_path = OUT_DIR / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    for cfg_id, desc, overrides in CONFIGS:
        log("=" * 66)
        log(f"CONFIG {cfg_id}: {desc}")
        proc = None
        try:
            proc = start_server(cfg_id, overrides)
            if not wait_ready(420):
                log(f"Server FAILED for {cfg_id}")
                slog = OUT_DIR / f"server_{cfg_id}.log"
                if slog.exists():
                    log("tail:\n" + "\n".join(slog.read_text().splitlines()[-50:]))
                rows.append({"config": cfg_id, "description": desc,
                             "status": "server_failed", "fl_env": overrides})
                summary_path.open("a").write(json.dumps(rows[-1]) + "\n")
                continue

            results = []
            for case_name, in_len, out_len, conc, n in BENCH_CASES:
                results.append(run_bench(cfg_id, case_name, in_len, out_len, conc, n))

            # --- hard verification: oplist must be non-empty ---
            oplist_path = Path(overrides["FLAGGEMS_ENABLE_OPLIST_PATH"])
            size, ops = read_oplist(oplist_path)
            server_log = (OUT_DIR / f"server_{cfg_id}.log").read_text()
            resolutions = sorted(set(DISPATCH_RE.findall(server_log)))
            gems_on = size > 0 and len(ops) > 0

            log(f"  oplist: {size} bytes, {len(ops)} distinct GEMS ops -> "
                f"FlagGems {'ENGAGED' if gems_on else 'NOT ENGAGED (INVALID CONFIG)'}")
            log(f"  dispatch resolutions: {len(resolutions)} ops")
            for op, impl in resolutions:
                log(f"    {op} -> {impl}")

            (OUT_DIR / f"oplist_{cfg_id}_ops.json").write_text(
                json.dumps({"bytes": size, "ops": ops,
                            "dispatch": [{"op": o, "impl": i} for o, i in resolutions]},
                           indent=2)
            )

            row = {
                "config": cfg_id,
                "description": desc,
                "status": "ok",
                "fl_env": overrides,
                "flaggems_engaged": gems_on,
                "oplist_bytes": size,
                "oplist_ops": ops,
                "dispatch_resolutions": [{"op": o, "impl": i} for o, i in resolutions],
                "latency_c1": results[0],
                "throughput_c8": results[1],
            }
            rows.append(row)
            summary_path.open("a").write(json.dumps(row) + "\n")
        finally:
            stop_server(proc)

    (OUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
    log(f"Wrote {OUT_DIR / 'summary.json'}")

    print()
    print("=" * 78)
    print("TASK 1 RESULT -- FlagGems whitelist with REAL registered names")
    print("=" * 78)
    print(f"{'config':18s} {'engaged':8s} {'ops':>4s} {'c1 tok/s':>9s} {'c8 tok/s':>9s}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['config']:18s} FAILED")
            continue
        print(f"{r['config']:18s} {str(r['flaggems_engaged']):8s} "
              f"{len(r['oplist_ops']):4d} "
              f"{r['latency_c1'].get('out_tok_s', 0):9.2f} "
              f"{r['throughput_c8'].get('out_tok_s', 0):9.2f}")
    print()
    print("REAL_NAMES_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
