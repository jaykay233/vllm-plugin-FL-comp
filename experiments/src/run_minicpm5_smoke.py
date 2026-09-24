from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

if __name__ == "__main__":
    print(f"Platform: {current_platform}")

    model = "/root/models/MiniCPM5-2B"
    prompts = [
        [{"role": "user", "content": "用一句话介绍你自己。"}],
    ]

    sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    outputs = llm.chat(prompts, sampling_params)
    for output in outputs:
        print("PROMPT:", output.prompt)
        print("OUTPUT:", output.outputs[0].text)
        print("---")
    print("SMOKE_TEST_OK")
