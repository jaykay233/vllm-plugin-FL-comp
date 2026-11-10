"""Fast, fully vectorised reference + accuracy sweep for the decode kernel.

The reference in fp8_decode_run.py loops over batch and head in Python, which
makes a full (page size x sequence length x split count) sweep slow enough to
be killed by the sandbox's CPU watchdog.  This version does one einsum per
(page size, sequence length) and reuses it across split counts.
"""
import torch
import fp8_decode_run as R


@torch.no_grad()
def reference_fast(q, k, v, bt, lens, sm_scale):
    B, H, D = q.shape
    HKV, P = k.shape[2], k.shape[1]
    g = H // HKV
    out = torch.empty(B, H, D, device=q.device, dtype=torch.float32)
    for b in range(B):
        # Page ids in bt are global (arange over B*npg), not per batch.
        n = int(lens[b])
        pages = bt[b, :(n + P - 1) // P].long()
        kf = k[pages].reshape(-1, HKV, D).float()[:n]
        vf = v[pages].reshape(-1, HKV, D).float()[:n]
        kf = kf.repeat_interleave(g, dim=1)  # (T, H, D)
        vf = vf.repeat_interleave(g, dim=1)
        s = torch.einsum("hd,thd->ht", q[b].float(), kf) * sm_scale
        p = torch.softmax(s, dim=-1)
        out[b] = torch.einsum("ht,thd->hd", p, vf)
    return out


def check(q, k, v, bt, lens, S, mode=4096, ref=None):
    sm = q.shape[2] ** -0.5
    if ref is None:
        ref = reference_fast(q, k, v, bt, lens, sm)
    out = R.Runner(q, k, v, bt, lens, S, mode, sm)()
    e = (out.float() - ref).abs().max().item()
    c = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    return e, c


if __name__ == "__main__":
    torch.manual_seed(0)
    # Validate the vectorised reference against the original scalar one.
    q, k, v, bt, lens = R.make(2, 200, 2, 128, 32)
    a = reference_fast(q, k, v, bt, lens, 128 ** -0.5)
    b = R.reference(q, k, v, bt, lens, 128 ** -0.5).float()
    print(f"reference parity: maxabs={(a - b).abs().max().item():.6f}", flush=True)

    bad = 0
    import os
    for P in (int(os.environ["PAGE"]),):
        for SEQ in ([int(x) for x in os.environ["SEQS"].split(",")]):
            q, k, v, bt, lens = R.make(3, SEQ, 2, 128, P)
            # ragged lengths exercise the tail tile of every split
            lens = torch.tensor([SEQ, max(1, SEQ // 2), max(1, SEQ - 37)],
                                device="cuda", dtype=torch.int32)
            ref = reference_fast(q, k, v, bt, lens, 128 ** -0.5)
            for S in (1, 3, 8, 16, 17):
                e, c = check(q, k, v, bt, lens, S, ref=ref)
                ok = e < 2e-3 and c > 0.9999
                bad += (not ok)
                print(f"P={P:<2d} SEQ={SEQ:<4d} S={S:<3d} maxabs={e:.5f} "
                      f"cos={c:.6f}{'' if ok else '  <-- FAIL'}", flush=True)
    print("FAILURES:", bad, flush=True)
