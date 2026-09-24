import importlib
import os

os.environ.setdefault("USE_FLAGGEMS", "1")
import flag_gems  # noqa: F401

M = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

sm = M.get_sm_count()
l2 = M.get_l2_cache_size()
print(f"RESULT SM={sm} L2={l2} sm_budget={max(1, sm * 3 // 4)} l2_budget={max(1, l2 // 32)}")
shapes = [("qkv", 2560, 2048), ("gate_up", 12288, 2048), ("o_proj", 2048, 2048), ("down", 2048, 6144)]
for name, N, K in shapes:
    row = []
    for m in (2, 8, 32, 512, 2048):
        prog = M._max_general_mm_programs(m, N)
        tag = "SPLITK" if M.splitk_mm_scenario(m, N, K) else "general"
        row.append(f"M{m}:{tag}(prog={prog})")
    print(f"RESULT {name:8s} " + "  ".join(row))
