from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})


