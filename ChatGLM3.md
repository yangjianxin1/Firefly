# ChatGLM3微调介绍
之所以单独把微调ChatGLM3整理成一个文档，是因为它原生支持function call，而我们在微调的时候希望继续保持function call的功能，所以单独对其进行了适配，主要在于数据预处理。

## 数据处理
为了保持ChatGLM3原始的chat能力和function call能力，在训练时，我们与官方的数据拼接格式保持一致。
对于ChatGLM3的详细的数据处理逻辑可查看：[数据处理逻辑](https://github.com/yangjianxin1/Firefly/blob/master/component/dataset.py#L107)

## 训练数据格式
微调时，我们采用与ChatGLM3一致的数据文件格式，下面是一个示例，也可查看data/dummy_data_chatglm3.jsonl。 [官方介绍](https://github.com/THUDM/ChatGLM3/tree/main/finetune_demo)。

在微调时，可以将function call与非function call的训练数据混合。如果此条数据中需要进行function call，则需要有`tools`字段，并且在conversations中需要出现`rool`为`tool`的数据。
当此条数据非function call数据时，则不应包含`tools`字段，并且在conversations中不应出现`rool`为`tool`的数据。
```json
{
    "tools":[
        {
            "name":"get_current_weather",
            "description":"Get the current weather in a given location",
            "parameters":{
                "type":"object",
                "properties":{
                    "location":{
                        "type":"string",
                        "description":"The city and state, e.g. San Francisco, CA"
                    },
                    "unit":{
                        "type":"string"
                    }
                },
                "required":[
                    "location"
                ]
            }
        }
    ],
    "conversations":[
        {
            "role":"user",
            "content":"北京今天天气如何？"
        },
        {
            "role":"tool",
            "name":"get_current_weather",
            "parameters":{
                "location":"beijing"
            },
            "observation":{
                "temperature":"20摄氏度",
                "wind force":"4级"
            }
        },
        {
            "role":"assistant",
            "content":"北京今天气温20摄氏度，风力4级"
        },
        {
            "role":"user",
            "content":"北京有什么旅游景点"
        },
        {
            "role":"assistant",
            "content":"北京天安门、故宫博物院、天坛、长城等都是值得游玩的景点。"
        }
    ]
}
```

## 数据格式转换
为了兼容ChatGLM3的function call微调，我们采用了其官方的数据格式，并且与firefly的数据格式差异较大。所以在训练ChatGLM3的时候，需要手动将firefly的训练数据，进行格式转换。

我们提供了一个简单的[数据转换脚本](https://github.com/yangjianxin1/Firefly/blob/master/script/convert_data_format.py)，可以将此前firefly开源的数据直接转换成ChatGLM3的训练格式。

## 微调ChatGLM3
训练配置参数均保存在[chatglm3-6b-sft-qlora.json](https://github.com/yangjianxin1/Firefly/blob/master/train_args/qlora/chatglm3-6b-sft-qlora.json)中。

单卡训练，可直接执行：
```bash
python train_qlora.py --train_args_file train_args/qlora/chatglm3-6b-sft-qlora.json
```

若是多卡，应执行：
```bash
torchrun --nproc_per_node={num_gpus} train_qlora.py --train_args_file train_args/qlora/chatglm3-6b-sft-qlora.json
```

**注意：chatglm3-6b-sft-qlora.json文件中的model_name_or_path的value值，必须要包含`chatglm3`，否则数据处理逻辑会出错。** 因为我们是根据model_name_or_path来对不同的模型进行数据处理，如下：
```python
# 加载ChatGLM2的训练集
if 'chatglm2' in args.model_name_or_path:
    train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length)
# 加载ChatGLM3的训练集
elif 'chatglm3' in args.model_name_or_path:
    train_dataset = ChatGLM3SFTDataset(args.train_file, tokenizer, args.max_seq_length)
# 按照firefly格式进行拼接
else:
    train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length)
```

## 推理
直接使用ChatGLM3官方的推理脚本即可：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_name_or_path = 'THUDM/chatglm3-6b'
adapter_name_or_path = 'path-to-adapter'

# 加载base model
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto',
)
# 加载adapter
if adapter_name_or_path is not None:
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```