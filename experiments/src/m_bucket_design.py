"""Compare candidate M-bucketing functions by how many distinct ConfigCache
keys they produce over the M range a real prefill walks.

Context: a cold ConfigCache key costs one full autotune (~7-9 s measured on
MetaX, 8 configs each Triton-compiled then benchmarked under REPLAY). So the
bucket-set size is a direct proxy for worst-case hot-path tuning time.

M range is taken from vLLM's max_num_batched_tokens on this GPU (2048), which
is the largest M a single forward can contain here.
"""

import math

MMAX = 2048


def identity(k):
    return k


def align32(k):
    if k == 0:
        return 0
    if k < 32:
        return 2 ** math.ceil(math.log2(k))
    return math.ceil(k / 32) * 32


def log2_bucket(k):
    if k == 0:
        return 0
    return 2 ** math.ceil(math.log2(k))


def make_hybrid(threshold):
    def hybrid(k):
        if k == 0:
            return 0
        if k < 32:
            return 2 ** math.ceil(math.log2(k))
        if k <= threshold:
            return math.ceil(k / 32) * 32
        return 2 ** math.ceil(math.log2(k))
    return hybrid


def make_capped(cap):
    def capped(k):
        if k == 0:
            return 0
        if k < 32:
            return 2 ** math.ceil(math.log2(k))
        return min(math.ceil(k / 32) * 32, cap)
    return capped


CANDS = [
    ("identity (current bug)", identity),
    ("align32 (applied fix)", align32),
    ("log2", log2_bucket),
    ("hybrid@64", make_hybrid(64)),
    ("hybrid@128", make_hybrid(128)),
    ("hybrid@256", make_hybrid(256)),
    ("align32 cap@256", make_capped(256)),
    ("align32 cap@512", make_capped(512)),
]

print(f"M range 1..{MMAX}\n")
print(f"{'strategy':24s} {'buckets':>8s} {'worst-case tuning':>18s}")
print("-" * 54)

for name, fn in CANDS:
    buckets = sorted({fn(m) for m in range(1, MMAX + 1)})
    # 8.5 s is the measured midpoint of a single MetaX autotune
    est = len(buckets) * 8.5
    print(f"{name:24s} {len(buckets):>8d} {est:>15.0f} s")

print("\n各方案在若干代表性 M 上的取值：")
print(f"{'M':>6s} " + " ".join(f"{n.split()[0][:11]:>12s}" for n, _ in CANDS))
for m in (1, 2, 8, 33, 64, 100, 128, 256, 512, 1024, 1500, 2048):
    row = " ".join(f"{fn(m):>12d}" for _, fn in CANDS)
    print(f"{m:>6d} {row}")

print("\nhybrid@128 的完整桶集：")
print(" ", sorted({make_hybrid(128)(m) for m in range(1, MMAX + 1)}))
print("align32 的完整桶集（前 24 个 + 末尾）：")
b = sorted({align32(m) for m in range(1, MMAX + 1)})
print(" ", b[:24], "...", b[-4:])
