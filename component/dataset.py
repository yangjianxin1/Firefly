import json
from loguru import logger
import ast
import astunparse
from typing import Dict
from torch.utils.data import Dataset
import os
from os.path import join
import pandas as pd
from tqdm import tqdm
import pickle


class SFTDataset(Dataset):
    """
    Firefly项目默认的数据组织格式
    """
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']

        # 收集多轮对话
        utterances = []
        for x in conversation:
            utterances.append(x['human'])
            utterances.append(x['assistant'])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
        input_ids = [self.bos_token_id]
        target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + 1)
            else:
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class ChatGLM2SFTDataset(SFTDataset):

    def __getitem__(self, index):
        """
        基本沿袭ChatGLM2的指令微调的格式，做了小修改，多轮对话如下。
        """
        # 每条数据格式为: [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']
        input_format = '[Round {}]\n\n问：{}\n\n答：'
        target_format = '{}'

        # 收集多轮对话
        utterances = []
        for i, x in enumerate(conversation):
            human = input_format.format(i+1, x['human'])
            assistant = target_format.format(x['assistant'])
            utterances += ([human, assistant])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        # 每条数据格式为: [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        input_ids = []
        target_mask = []  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += utterances_id
            # input部分
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id))
            # target部分
            else:
                input_ids += [self.eos_token_id]
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class ChatGLM3SFTDataset(SFTDataset):
    def __init__(self, file, tokenizer, max_seq_length):
        super(ChatGLM3SFTDataset, self).__init__(file, tokenizer, max_seq_length)
        self.FUNCTION_CALL_NAME = 'tool_call'
        self.FUNCTION_CALL_PREFIX = '```python\n'
        self.FUNCTION_CALL_POSTFIX = '\n```'
        self.TOOL_DEFINITION_PREFIX = 'Answer the following questions as best as you can. You have access to the following tools:\n'

    def format_function_call(self, function_name: str, parameters: Dict[str, str]):
        function_name = ast.Name(id=function_name)
        keywords = [
            ast.keyword(arg=arg_name, value=ast.Constant(arg_value))
            for arg_name, arg_value in parameters.items()
        ]
        func_call = ast.Call(func=function_name, args=[], keywords=keywords)
        return astunparse.unparse(func_call).strip()

    def __getitem__(self, index):
        """
        沿袭ChatGLM3的指令微调格式，并且支持function call微调
        """
        data = self.data_list[index]
        data = json.loads(data)
        conversations = data['conversations']

        gmask_token_id = self.tokenizer.get_command('[gMASK]')
        sop_token_id = self.tokenizer.get_command('sop')
        input_ids = [gmask_token_id, sop_token_id]  # 收集
        target_mask = [0] * 2

        # 此轮对话存在function call
        if 'tools' in data.keys():
            conversations.insert(
                0, {"role": "system", "content": self.TOOL_DEFINITION_PREFIX + json.dumps(data['tools'], indent=4, ensure_ascii=False)}
            )

        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            role = conv['role'].strip()
            if role == 'tool':
                # function call
                value = self.FUNCTION_CALL_PREFIX + self.format_function_call(self.FUNCTION_CALL_NAME, conv["parameters"]) + self.FUNCTION_CALL_POSTFIX
                token_ids = self.tokenizer.build_single_message("assistant", conv["name"], value) + [self.tokenizer.eos_token_id]
                input_ids += token_ids
                # 不计算<|assistant|>的loss
                target_mask += [0] + [1] * (len(token_ids)-1)

                # function call result
                value = conv.get('observation', None)
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                token_ids = self.tokenizer.build_single_message("observation", "", value)
                input_ids += token_ids
                target_mask += [0] * len(token_ids)
            else:
                token_ids = self.tokenizer.build_single_message(role, "", conv["content"])
                if role == 'system' or role == 'user':
                    input_ids += token_ids
                    target_mask += [0] * len(token_ids)
                # role=assistant
                else:
                    input_ids += token_ids + [self.tokenizer.eos_token_id]
                    # 不计算<|assistant|>的loss，需要计算eos_token_id的loss
                    target_mask += [0] + [1] * (len(token_ids)-1) + [1]

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class ZephyrSFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        数据拼接格式如下
        <|user|>
        How many helicopters can a human eat in one sitting?</s>
        <|assistant|>
        Ah, me hearty matey! But yer question be a puzzler! A human cannot eat a helicopter in one sitting, as helicopters are not edible. They be made of metal, plastic, and other materials, not food!</s>
        """
        data = self.data_list[index]
        data = json.loads(data)
        conversations = data['conversation']

        # 收集模型输入
        input_ids = []
        target_mask = []

        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            input_tokens = self.tokenizer.encode(f'<|user|>\n{human}', add_special_tokens=False) + [self.eos_token_id]
            output_tokens = self.tokenizer.encode(f'<|assistant|>\n{assistant}', add_special_tokens=False) + [self.eos_token_id]

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens)
            # <|assistant|>占6个token，不计算<|assistant|>这个文本的loss
            target_mask += [0] * 6 + [1] * (len(output_tokens) - 6)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class MistralSFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list
        self.inst_begin_tokens = tokenizer.encode('[INST]', add_special_tokens=False)
        self.inst_end_tokens = tokenizer.encode('[/INST]', add_special_tokens=False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        数据拼接格式如下：
        <s>[INST]你是谁?[/INST]我是大模型</s>[INST]背诵李白的《静夜思》[/INST]窗前明月光...</s>
        """
        data = self.data_list[index]
        data = json.loads(data)
        conversations = data['conversation']

        # 收集模型输入
        input_ids = [self.bos_token_id]
        target_mask = [0]

        for conv in conversations:
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            assistant_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_tokens = self.inst_begin_tokens + human_tokens + self.inst_end_tokens
            output_tokens = assistant_tokens + [self.eos_token_id]

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class QwenSFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.im_start_id = tokenizer.im_start_id
        self.im_end_id = tokenizer.im_end_id
        self.enter_token_ids = tokenizer.encode('\n')   # 回车键
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        数据拼接格式如下：
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        你好呀<|im_end|>
        <|im_start|>assistant
        你好，我是xxx，很高兴为您服务<|im_end|>
        """
        data = self.data_list[index]
        data = json.loads(data)
        if 'system' in data.keys():
            system = data['system'].strip()
        else:
            system = 'You are a helpful assistant.'
        conversations = data['conversation']

        # 收集模型输入
        system_text = f'<|im_start|>system\n{system}<|im_end|>\n'
        input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
        target_mask = [0] * len(input_ids)

        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            input_tokens = self.tokenizer.encode(f'<|im_start|>user\n{human}<|im_end|>\n', add_special_tokens=False)
            output_tokens = self.tokenizer.encode(f'<|im_start|>assistant\n{assistant}<|im_end|>\n', add_special_tokens=False)

            input_ids += input_tokens + output_tokens
            # input_tokens部分不计算loss
            target_mask += [0] * len(input_tokens)
            # '<|im_start|>assistant\n'占3个token，结尾的'\n'占1个token，不计算它们的loss
            target_mask += [0] * 3 + [1] * (len(output_tokens) - 4) + [0]

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class PretrainDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_seq_length, min_seq_length, window_step_size):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length     # 小于min_seq_length的序列，会被抛弃
        self.window_step_size = window_step_size    # 滑动窗口步长
        self.tokenize_batch = 1024
        logger.info('Loading pretraining data: {}'.format(data_path))

        # 创建缓存路径
        cache_dir = join(data_path, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        # 读取缓存
        cache_file = join(cache_dir, 'train.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data_list = pickle.load(f)
            logger.info(f'Loading tokenized cache from: {cache_file}')
        else:
            # 扫描所有jsonl文件
            logger.info('Scanning all the training file...')
            files = []
            for root, dir_names, file_names in os.walk(data_path):
                for file_name in file_names:
                    file = join(root, file_name)
                    if file_name.endswith('.jsonl'):
                        files.append(file)
            logger.info(f'Total num of training file: {len(files)}')

            # 加载训练数据
            logger.info('Loading all training data')
            train_texts = []
            for file in files:
                df = pd.read_json(file, lines=True)
                text_list = [x.strip() for x in df['text'].tolist()]
                train_texts += text_list
            logger.info(f'Total num of training text: {len(train_texts)}')

            # 对文本进行tokenize，并且使用窗口滑动进行截断
            logger.info(f'Start tokenizing data...')
            train_windows = []  # 窗口截断之后的input_ids
            for i in tqdm(range(0, len(train_texts), self.tokenize_batch)):
                text_list = train_texts[i: i + self.tokenize_batch]
                input_ids = self.tokenizer(text_list).input_ids
                # 使用滑动窗口进行窗口截断
                for x in input_ids:
                    windows = self.slice_window_truncate(x)
                    train_windows += windows
            data_list = train_windows
            logger.info(f'Total training data num: {len(data_list)}')

        # 计算数据集的token数量
        logger.info('Calculating number of training token...')
        self.data_list = data_list
        total_token_num = 0
        for x in tqdm(data_list):
            total_token_num += len(x)
        logger.info(f'Total training token num: {total_token_num}')

    def slice_window_truncate(self, input_ids):
        """
        对input_ids，按照窗口大小，进行滑动截断。返回所有截断窗口。
        """
        windows = []
        for i in range(0, len(input_ids), self.window_step_size):
            window = input_ids[i: i+self.max_seq_length]
            # 小于min_seq_length的序列，则将其抛弃。
            if len(window) < self.min_seq_length and i > 0:
                continue
            windows.append(window)
        return windows

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data
