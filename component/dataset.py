import json
from loguru import logger
import ast
import astunparse
from typing import Dict
from torch.utils.data import Dataset


class SFTDataset(Dataset):
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
                token_ids = self.tokenizer.build_single_message("observation", "", value) + [self.tokenizer.eos_token_id]
                input_ids += token_ids
                target_mask += [0] * len(token_ids)
            else:
                token_ids = self.tokenizer.build_single_message(role, "", conv["content"]) + [self.tokenizer.eos_token_id]
                input_ids += token_ids
                if role == 'system' or role == 'user':
                    target_mask += [0] * len(token_ids)
                # role=assistant
                else:
                    # 不计算<|assistant|>的loss
                    target_mask += [0] + [1] * (len(token_ids)-1)

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


