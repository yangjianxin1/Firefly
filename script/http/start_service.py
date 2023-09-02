from flask import Flask, request
import json
import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码


@app.route('/firefly', methods=['POST'])
def ds_llm():
    params = request.get_json()
    inputs = params.pop('inputs').strip()

    # chatglm使用官方的数据组织格式
    if model.config.model_type == 'chatglm':
        text = '[Round 1]\n\n问：{}\n\n答：'.format(inputs)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    # 为了兼容qwen-7b，因为其对eos_token进行tokenize，无法得到对应的eos_token_id
    else:
        input_ids = tokenizer(inputs, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
        eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
        input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)

    logger.info(params)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, **params)
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    # response = tokenizer.batch_decode(outputs)
    response = tokenizer.decode(outputs)
    response = response.strip().replace(tokenizer.eos_token, "").strip()

    result = {
        'input': inputs,
        'output': response
    }
    with open(log_file, 'a', encoding='utf8') as f:
        data = json.dumps(result, ensure_ascii=False)
        f.write('{}\n'.format(data))

    return result


if __name__ == '__main__':
    # 参数设置
    model_name_or_path = 'YeungNLP/firefly-baichuan-13b'
    log_file = 'service_history.txt'
    port = 8877
    device = 'cuda'
    logger.info(f"Starting to load the model {model_name_or_path} into memory")

    # 加载model和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    logger.info(f"Successfully loaded the model {model_name_or_path} into memory")

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    print("Total model params: %.2fM" % (total / 1e6))
    model.eval()

    app.run(port=port)
