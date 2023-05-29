import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class TargetLMLoss(_Loss):

    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels, target_mask):
        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, labels, self.ignore_index)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
