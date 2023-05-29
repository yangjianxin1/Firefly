import transformers
from typing import Tuple, Union
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from component.loss import TargetLMLoss
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BloomForCausalLM(transformers.BloomForCausalLM):
    """
    继承自BloomForCausalLM，区别在于只计算target部分的loss
    """
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        labels=None,
        target_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        return_loss=False,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if return_loss:
            loss_fn = TargetLMLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fn(lm_logits, input_ids, target_mask)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

