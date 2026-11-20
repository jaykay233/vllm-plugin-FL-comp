import random

import torch
from vllm import LLM, SamplingParams


def main():
    M = 4096
    NP = 64
    rng = random.Random(777001)
    prompts = [
        {"prompt_token_ids": [rng.randrange(1000, 120000) for _ in range(M)]}
        for _ in range(NP)
    ]

    llm = LLM(
        model="/workspace/MiniCPM5-2B",
        max_model_len=131072,
        gpu_memory_utilization=0.85,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        seed=0,
    )
    sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=0.95)

    warm = [
        {"prompt_token_ids": [rng.randrange(1000, 120000) for _ in range(1024)]}
        for _ in range(4)
    ]
    llm.generate(warm, SamplingParams(max_tokens=1))
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate(prompts, sp)
        torch.cuda.synchronize()

    rows = []
    for ev in prof.key_averages():
        if ev.self_device_time_total > 0:
            rows.append((ev.self_device_time_total, ev.count, ev.key))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"\ntotal device time: {tot / 1e3:.2f} ms "
          f"(262144 prefill tokens -> {262144 / tot * 1e3:.2f} k tok/s device)")
    for us, cnt, key in rows[:26]:
        print(f"{us / 1e3:9.2f} ms  x{cnt:5d}  {us / max(cnt, 1):8.1f} us  {key[:86]}")


if __name__ == "__main__":
    main()
