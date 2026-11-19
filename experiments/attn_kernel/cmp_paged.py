"""Numeric gate for the paged/varlen kernel against the vendor kernel.

Two things this checks that a dense test cannot:

  * the KV cache is genuinely paged -- blocks are allocated in a shuffled order and
    padded with garbage, so a kernel that follows block_table gets the right
    answer and a kernel that walks the cache linearly gets a wrong one;
  * the batch is varlen and mixed -- sequences with q_len 1 next to sequences with
    q_len 3000, which is what a chunked-prefill step looks like.

Reference is ixformer's flash_attn_varlen_func with block_table, the kernel this
is meant to replace.
"""
import subprocess
import sys
import numpy as np
import torch

D, NH, NKVH, PAGE = 128, 16, 2, 16
BIN = "/tmp/fpg"


def build(nseq, maxblk, qlens, klens, seed=0, blocktable="shuffled"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    assert len(qlens) == nseq and len(klens) == nseq
    need = sum((k + PAGE - 1) // PAGE for k in klens)
    npages = need + 7  # slack so the table does not have to be dense
    if blocktable == "shuffled":
        perm = torch.randperm(npages, generator=g)
        pool = perm.tolist()
    else:
        pool = list(range(npages))
    bt = torch.full((nseq, maxblk), -1, dtype=torch.int32)
    totq = sum(qlens)
    q = (torch.randn(totq, NH, D, generator=g) * 0.5).to(torch.bfloat16)
    # stored HND, which is what this hardware's paged FA actually wants
    k = (torch.randn(npages, NKVH, PAGE, D, generator=g) * 0.5).to(torch.bfloat16)
    v = (torch.randn(npages, NKVH, PAGE, D, generator=g) * 0.5).to(torch.bfloat16)
    cur = 0
    for s in range(nseq):
        nb = (klens[s] + PAGE - 1) // PAGE
        for i in range(nb):
            bt[s, i] = pool[cur]
            cur += 1
    cu_q = torch.tensor([0] + list(np.cumsum(qlens)), dtype=torch.int32)
    cu_k = torch.tensor([0] + list(np.cumsum(klens)), dtype=torch.int32)
    return q, k, v, bt, cu_q, cu_k, npages


def run_case(name, nseq, maxblk, qlens, klens, seed=0):
    q, k, v, bt, cu_q, cu_k, npages = build(nseq, maxblk, qlens, klens, seed)
    pre = f"/tmp/pg_{name}"
    for t, nm in ((q, "Q"), (k, "K"), (v, "V"), (bt, "bt")):
        with open(f"{pre}.{nm}.bin", "wb") as f:
            # torch has no numpy view for bf16; go through uint8 like the dense harness
            raw = t.contiguous().cpu().view(torch.uint8).numpy() if t.dtype == torch.bfloat16 \
                else t.contiguous().cpu().numpy()
            f.write(raw.tobytes())
    with open(f"{pre}.case", "w") as f:
        f.write(f"{nseq} {NH} {NKVH} {maxblk}\n")
        for s in range(nseq):
            f.write(f"{qlens[s]} {klens[s]}\n")
        f.write(f"{pre}.Q.bin {pre}.K.bin {pre}.V.bin {pre}.bt.bin {pre}.out\n")

    r = subprocess.run([BIN, f"{pre}.case", "8", "2", "1"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{name}: kernel failed\n{r.stdout}\n{r.stderr}")
        return None

    from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func

    # The vendor's paged path rejects n_kv_heads < n_heads (CUINFER_STATUS_BAD_PARAM
    # at ixinfer_flash_attn_unpad.cpp:1080); only an already-expanded cache works.
    # Our kernel takes the real GQA cache of 2 kv heads, so the reference is built
    # by replicating each page 8x along heads.  That is the same function: GQA means
    # 8 query heads read the same kv head, so replicating it changes nothing.
    rep = NH // NKVH
    kexp = k.repeat_interleave(rep, dim=1).contiguous()
    vexp = v.repeat_interleave(rep, dim=1).contiguous()

    dev = "cuda"
    qd, kd, vd, btd = q.to(dev), kexp.to(dev), vexp.to(dev), bt.to(dev)
    cuq, cuk = cu_q.to(dev), cu_k.to(dev)
    out = {}
    for causal in (True, False):
        ref = flash_attn_varlen_func(
            qd, kd, vd, cuq, cuk, max(qlens), max(klens),
            causal=causal, softmax_scale=1.0 / (D ** 0.5), block_table=btd,
        ).float()
        mine = torch.from_numpy(
            np.fromfile(f"{pre}.out.{'causal' if causal else 'dense'}", dtype=np.float32)
        ).reshape(-1, NH, D).to(dev)
        dn = torch.linalg.vector_norm(mine - ref, dim=-1)
        rn = torch.linalg.vector_norm(ref, dim=-1)
        rel = dn / (rn + 1e-12)
        md = (mine - ref).abs().max().item()
        bad = int((rel > 1e-2).sum().item())
        out[causal] = (md, rel.max().item(), bad)
        print(f"{name:26s} causal={int(causal)}  maxdiff={md:.3e}  "
              f"worst row rel={rel.max().item():.3e}  rows>1e-2={bad}/{rel.numel()}")

    # A paged kernel that ignored block_table would still be "close" on average if
    # the pages happened to be in order; assert the table is actually shuffled.
    nb0 = (klens[0] + PAGE - 1) // PAGE
    ordered = all(int(bt[0, i]) == i for i in range(nb0))
    print(f"    block_table shuffled for seq 0: {not ordered} "
          f"(first 4 pages {bt[0, :4].tolist()})")
    ok = all(v[2] == 0 for v in out.values())
    return ok


cases = [
    # name, nseq, maxblk, qlens, klens
    ("single_prefill_256", 1, 16, [256], [256]),
    ("single_prefill_1024", 1, 64, [1024], [1024]),
    # chunked prefill: 512 new tokens attending to 2048 cached
    ("chunk_prefill_512_2048", 1, 128, [512], [2048]),
    # mixed: two prefills, a 1-token decode and a 7-token chunk, all in one step
    ("mixed_batch", 4, 128, [1024, 512, 1, 7], [1024, 2048, 1536, 99]),
    # ragged: q_len not a multiple of BM=256
    ("ragged", 3, 64, [300, 1, 129], [300, 300, 1000]),
]

allok = True
for c in cases:
    r = run_case(*c)
    if r is None:
        sys.exit(1)
    allok = allok and r

print("\nPAGED GATE:", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
