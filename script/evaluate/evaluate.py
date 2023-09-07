from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

batch_size = 20
# 指定评测模型
model_name_or_paths = [
    'internlm/internlm-chat-7b',
    'baichuan-inc/Baichuan-13B-Chat',
    'THUDM/chatglm2-6b',
    'YeungNLP/firefly-baichuan-7b',
    'YeungNLP/firefly-baichuan-13b',
    'YeungNLP/firefly-internlm-7b',
    'YeungNLP/firefly-chatglm2-6b',
    'YeungNLP/firefly-ziya-13b',
    'YeungNLP/firefly-bloom-1b4',
    'YeungNLP/firefly-bloom-2b6-v2',
    'YeungNLP/firefly-qwen-7b',
    'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16',
    'OpenBuddy/openbuddy-llama2-13b-v11.1-bf16',
]

models = []
for model_name_or_path in model_name_or_paths:
    # baichuan-7b与qwen的pad_token_id为None，将无法正常评测
    if 'baichuan-7b' in model_name_or_path.lower():
        pad_token = '</s>'
    elif 'qwen' in model_name_or_path.lower():
        pad_token = '<|endoftext|>'
    else:
        pad_token = None

    abbr = model_name_or_path.split('/')[-1]
    model = dict(
        type=HuggingFaceCausalLM,
        abbr=abbr,
        path=model_name_or_path,
        tokenizer_path=model_name_or_path,
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              trust_remote_code=True,
                              pad_token=pad_token
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch_size,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        batch_padding=False,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=2, num_procs=2),
    )
    models.append(model)


# 指定评测集
with read_base():
    from .datasets.ceval.ceval_ppl import ceval_datasets
    from .summarizers.example import summarizer

datasets = [*ceval_datasets]


# python run.py configs/eval_demo.py -w outputs/firefly
