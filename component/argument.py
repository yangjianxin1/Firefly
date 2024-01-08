from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集。如果task_type=pretrain，请指定文件夹，将扫描其下面的所有jsonl文件"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    # min_seq_length: int = field(default=1024, metadata={"help": "输小最大长度"})
    # window_step_size: int = field(default=1024, metadata={"help": "滑动窗口步长"})
    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    task_type: str = field(default="sft", metadata={"help": "预训练任务：[sft, pretrain]"})
    tokenize_num_workers: int = field(default=1, metadata={"help": ""})


@dataclass
class QLoRAArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    task_type: str = field(default="", metadata={"help": "预训练任务：[sft, pretrain]"})
    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})


@dataclass
class DPOArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    max_prompt_length: Optional[int] = field(metadata={"help": "max length of prompt"})

    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})

    # 定义template，单轮对话prompt的拼接格式为：{system}{conv_begin}{human_begin}你好{human_end}{assistant_begin}
    system: int = field(default='', metadata={"help": ""})
    conv_begin: int = field(default='', metadata={"help": ""})
    human_begin: int = field(default='', metadata={"help": ""})
    human_end: int = field(default='', metadata={"help": ""})
    assistant_begin: int = field(default='', metadata={"help": ""})
    assistant_end: int = field(default='', metadata={"help": ""})

    use_lora: bool = field(default=False, metadata={"help": "预训练任务：[sft, pretrain]"})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})


@dataclass
class LOMOArguments:
    """
    LOMO训练的自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    clip_grad_norm: float = field(metadata={"help": "Maximum gradient normalized value (for gradient clipping)."})
    clip_grad_value: float = field(default=None, metadata={"help": "Maximum gradient value (for gradient clipping)."})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})
