from vllm_fl.dispatch.config import get_flagos_whitelist
w = get_flagos_whitelist()
open("/workspace/vllm-plugin-FL-comp/experiments/bench_results/eval_4k_bucket_nowl_r2/wl_check.txt", "w").write(repr(w))
