import json
from loguru import logger
from torch.utils.data import Dataset


class UnifiedSFTDataset(Dataset):
    """
    统一的数据处理dataset
    """
    def __init__(self, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template: {self.template_name} for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        # setting system information
        if self.system_format is not None:
            system = data['system'].strip() if 'system' in data.keys() else self.system
            # system信息不为空
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human = self.user_format.format(content=human, stop_token=self.tokenizer.eos_token)
            assistant = self.assistant_format.format(content=assistant, stop_token=self.tokenizer.eos_token)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

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


class ChatGLM2SFTDataset(UnifiedSFTDataset):

    def __getitem__(self, index):
        # 每条数据格式为: [gMASK]sop [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        data = self.data_list[index]
        data = json.loads(data)

        input_ids = self.tokenizer.get_prefix_tokens()
        target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human = self.user_format.format(content=human, idx=i + 1)
            assistant = self.assistant_format.format(content=assistant)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False) + [self.tokenizer.eos_token_id]

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


class ChatGLM3SFTDataset(UnifiedSFTDataset):

    def __getitem__(self, index):
        # [gMASK]sop <|system|>xxx<|user|>xxx<|assistant|>xxx<eos>
        data = self.data_list[index]
        data = json.loads(data)
        system = data['system'].strip() if 'system' in data.keys() else self.system
        input_ids = self.tokenizer.get_prefix_tokens() + \
                    [self.tokenizer.get_command(f"<|system|>")] + \
                    self.tokenizer.encode(system, add_special_tokens=False)
        target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            input_tokens = [self.tokenizer.get_command(f"<|user|>")] + \
                           self.tokenizer.encode(human, add_special_tokens=False) + \
                           [self.tokenizer.get_command(f"<|assistant|>")]
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False) + [self.tokenizer.eos_token_id]

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
