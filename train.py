from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
)
import argparse
from loguru import logger
import os
from os.path import join
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from component.collator import SFTDataCollator
from component.dataset import SFTDataset, ChatGLM2SFTDataset
from component.argument import CustomizedArguments
from component.trainer import Trainer
from component.loss import TargetLMLoss


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/sft.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # train_args_file = 'train_args/finetune.json'
    # 读取训练的参数配置
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def init_components(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    training_args.ddp_find_unused_parameters = False if ddp else None

    # 初始化model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    # ChatGLMTokenizer不需要设置，仅设置其他tokenizer
    elif tokenizer.__class__.__name__ != 'ChatGLMTokenizer':
        assert tokenizer.eos_token_id is not None
        assert tokenizer.bos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # # 部分tokenizer没有pad_token_id
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    # # 部分tokenizer的pad_token_id与eos_token_id相同，如InternLM，会导致无法计算eos_token_id的loss。将pad_token_id设为unk_token_id
    # if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.unk_token_id is not None:
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    # # 如果两者相同，模型训练时不会计算eos_token_id的loss
    # if tokenizer.pad_token_id == tokenizer.eos_token_id:
    #     raise Exception('pad_token_id should not be equal to eos_token_id')

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    # 初始化损失函数
    loss_func = TargetLMLoss(ignore_index=-100)
    # 加载训练集
    if model.config.model_type == 'chatglm':
        train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length)
    else:
        train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length)
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # tokenizer=tokenizer,
        data_collator=data_collator,
        compute_loss=loss_func
    )
    return trainer


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
