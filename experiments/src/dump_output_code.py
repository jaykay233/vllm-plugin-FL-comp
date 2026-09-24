import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams
llm = LLM(model="/root/models/MiniCPM5-2B", dtype="bfloat16", trust_remote_code=True,
          max_model_len=2048, gpu_memory_utilization=0.30,
          compilation_config={"mode": "VLLM_COMPILE"})
sp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
llm.generate(["The theory of general relativity describes gravity as"], sp, use_tqdm=False)
print("DUMP_OUTPUT_CODE_DONE", flush=True)
