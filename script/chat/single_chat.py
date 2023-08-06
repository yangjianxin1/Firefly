from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
"""
单轮对话，不具有对话历史的记忆功能
"""


def main():
    model_name = 'YeungNLP/firefly-baichuan-13b'
    # model_name = 'YeungNLP/firefly-baichuan-7b'
    # model_name = 'YeungNLP/firefly-ziya-13b'
    # model_name = 'YeungNLP/firefly-bloom-7b1'
    # model_name = 'YeungNLP/firefly-baichuan-7b'
    # model_name = 'YeungNLP/firefly-baichuan-13b'
    # model_name = 'YeungNLP/firefly-bloom-7b1'
    # model_name = 'YeungNLP/firefly-llama-30b'

    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    device = 'cuda'
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
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    text = input('User：')
    while True:
        text = text.strip()
        # chatglm使用官方的数据组织格式
        if model.config.model_type == 'chatglm':
            text = '[Round 1]\n\n问：{}\n\n答：'.format(text)
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        # 为了兼容qwen-7b，因为其对eos_token进行tokenize，无法得到对应的eos_token_id
        else:
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
            eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
            input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(tokenizer.eos_token, "").strip()
        print("Firefly：{}".format(response))
        text = input('User：')


if __name__ == '__main__':
    main()
