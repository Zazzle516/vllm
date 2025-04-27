# SPDX-License-Identifier: Apache-2.0
print("zazzle start import")
from vllm import LLM, SamplingParams

print("zazzle after import start SamplingParams")
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
print("zazzle after Sampling_params begine main")

def main():
    # Create an LLM.
    print("zazzle after main begine LLM")
    llm = LLM(model="/home/ubuntuhx/xirui.hao/Models/deepseek-1.5B/DeepSeek-R1-Distill-Qwen-1.5B", dtype="bfloat16", swap_space=0)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.

    print("zazzle after LLM start generate")
    outputs = llm.generate(prompts, sampling_params)
    print("zazzle after generate")

    # Print the outputs.
    # print("\nGenerated Outputs:\n" + "-" * 60)
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt:    {prompt!r}")
    #     print(f"Output:    {generated_text!r}")
    #     print("-" * 60)


if __name__ == "__main__":
    main()
