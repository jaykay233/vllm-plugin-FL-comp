from vllm_fl.dispatch.config import get_flagos_whitelist
w = get_flagos_whitelist()
open("/workspace/vllm-plugin-FL-comp/experiments/bench_results/eval_4k_nobucket_wl_off/wl_check.txt", "w").write(repr(w))
