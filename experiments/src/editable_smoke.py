import os
os.environ.setdefault("VLLM_PLUGINS", "fl")
from vllm import LLM, SamplingParams

def main():
    llm = LLM(model="/workspace/MiniCPM5-2B", max_model_len=2048,
              gpu_memory_utilization=0.80, enforce_eager=True)
    sp = SamplingParams(temperature=0, max_tokens=16)
    out = llm.generate(["1+1="], sp)
    print("OUT:", repr(out[0].outputs[0].text))

if __name__ == "__main__":
    main()
