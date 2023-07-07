from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
"""
单轮对话，不具有对话历史的记忆功能
"""


def main():
    # model_name = 'YeungNLP/firefly-baichuan-7b-qlora-sft-merge'
    model_name = 'YeungNLP/firefly-ziya-13b-qlora-sft-merge'
    # model_name = 'YeungNLP/firefly-bloom-7b1-qlora-sft-merge'

    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    device = 'cuda'
    input_pattern = '<s>{}</s>'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    text = input('User：')
    while True:
        text = text.strip()
        text = input_pattern.format(text)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(text, "").replace('</s>', "").replace('<s>', "").strip()
        print("Firefly：{}".format(response))
        text = input('User：')


if __name__ == '__main__':
    main()
