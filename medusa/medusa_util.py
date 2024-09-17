import types
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

def calculate_loss_contribution(
    loss_i,
    i,
    medusa_only_heads,
    medusa_decay_coefficient,
    medusa_heads_coefficient,
    medusa_scheduler_coefficient,
):
    if i == 0:
        return loss_i if not medusa_only_heads else 0
    else:
        return (
            loss_i
            * medusa_decay_coefficient**i
            * medusa_heads_coefficient
            * medusa_scheduler_coefficient
        )

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x

class Medusa(nn.Module):
    """This class implements the Medusa draft model from the paper: https://arxiv.org/abs/2401.10774
    Reference implementation: https://github.com/FasterDecoding/Medusa
    
    Differences from reference implementation:
    1. Currently this only supports generating proposals from top-1 tokens.
    2. We have an optional token_map which reduces draft vocab to most 
       frequently used tokens to give some additional speed-up by reducing 
       sampling overhead. This is disabled unless the checkpoint file has 
       explicit token_map tensor and config has an optional attribute 
       truncated_vocab_size < vocab_size. To use this technique, one has to find
       the top-k most frequent tokens in target dataset and add that as a tensor
       in the draft checkpoint (using key token_map). Also, the draft config
       needs to have truncated_vocab_size (=k) as an attribute."""

    def __init__(self, medusa_num_heads, medusa_num_layers, hidden_size, vocab_size) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(hidden_size=hidden_size,
                          num_layers=medusa_num_layers)
            for _ in range(medusa_num_heads)
        ])
        self.orig_vocab_size = vocab_size

        self.lm_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(medusa_num_heads)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        hiddent_states_per_head = [block(hidden_states) for block in self.blocks]
        logits_per_head = [lm_head(hs) for lm_head, hs in zip(self.lm_heads, hiddent_states_per_head)]
        return logits_per_head

def add_medusa_heads(
    model,
    medusa_num_heads=4,
    medusa_num_layers=0,
    medusa_return: bool = False,
    medusa_only_heads: bool = False,
):
    """
    Args:
        model (nn.Module): The base language model to be used.
        medusa_num_heads (int, optional): The number of additional tokens to predict. Defaults to 3.
        medusa_num_layers (int, optional): The number of ResBlock layers for each Medusa head. Defaults to 0.
        medusa_return (bool, optional): If True, returns the Medusa logits; otherwise, the forward pass will use the `lm_head`. Defaults to False.
        medusa_only_heads (bool, optional): If True, only the Medusa head weights will be updated during fine-tuning; otherwise, the entire model's weights will be updated. Defaults to False.
    """
    hidden_size = model.lm_head.weight.shape[-1]
    vocab_size = model.lm_head.weight.shape[0]
    model.config.medusa_num_layers = medusa_num_layers
    model.config.medusa_num_heads = medusa_num_heads
    model.medusa_num_heads = medusa_num_heads
    # Create a list of Medusa heads
    model.medusa_head = Medusa(medusa_num_heads, medusa_num_layers, hidden_size, vocab_size)

    # Ensure medusa_head's dtype and device align with the base_model
    model.medusa_head.to(model.dtype).to(model.device)

    for i in range(medusa_num_heads):
        # Initialize the weights of each medusa_head using the base model's weights
        model.medusa_head.lm_heads[i].weight.data[:] = model.lm_head.weight.data[:]
    # logging the model summary
    print(model)
    model.old_forward = model.forward

    def forward(
        model,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass of the MedusaModel.
        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        loss = 0
        medusa_logits = None
        # LOG.debug("medusa_return: %s", medusa_return)
        if not medusa_return:
            return model.old_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Pass input through the base model
        if medusa_only_heads:
            with torch.no_grad():
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states = outputs[0]
                # The lm_head will be frozen as well, so it's within the context of torch.no_grad()
                medusa_logits = [model.lm_head(hidden_states)]
        else:
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            medusa_logits = [model.lm_head(hidden_states)]

        medusa_logits.extend(model.medusa_head(hidden_states))
        medusa_logits = torch.stack(medusa_logits, dim=0)

        if labels is None:
            return_dict = (
                return_dict if return_dict is not None else model.config.use_return_dict
            )

            if not return_dict:
                output = (medusa_logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=medusa_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # Fix all the coefficients to 1 for now
        medusa_scheduler_coefficient = 1
        medusa_heads_coefficient = 1
        medusa_decay_coefficient = 0.8

        if model.training:
            loss = 0

            loss_fct = CrossEntropyLoss()
            for i in range(model.medusa_num_heads+1):
                medusa_logits_i = (
                    medusa_logits[i, :, : -(1 + i)]
                    .contiguous()
                    .view(-1, medusa_logits.shape[-1])
                )
                medusa_logits_i = medusa_logits_i.float()
                medusa_labels = (
                    labels[..., (1 + i) :]
                    .contiguous()
                    .view(-1)
                    .to(medusa_logits_i.device)
                )

                loss_i = loss_fct(medusa_logits_i, medusa_labels)

                loss += calculate_loss_contribution(
                    loss_i,
                    i,
                    medusa_only_heads,
                    medusa_decay_coefficient,
                    medusa_heads_coefficient,
                    medusa_scheduler_coefficient,
                )
        else:
            if model.config.pretraining_tp > 1:
                raise NotImplementedError
            else:
                loss = 0
                medusa_logits = [model.lm_head(hidden_states)]
                medusa_logits.extend(model.medusa_head(hidden_states))
                medusa_logits = torch.stack(medusa_logits, dim=0)
                
                loss_fct = CrossEntropyLoss()
                for i in range(model.medusa_num_heads+1):
                    medusa_logits_i = (
                        medusa_logits[i, :, : -(1 + i)]
                        .contiguous()
                        .view(-1, medusa_logits.shape[-1])
                    )
                    medusa_logits_i = medusa_logits_i.float()
                    medusa_labels = (
                        labels[..., (1 + i) :]
                        .contiguous()
                        .view(-1)
                        .to(medusa_logits_i.device)
                    )

                    loss_i = loss_fct(medusa_logits_i, medusa_labels)

                    loss += calculate_loss_contribution(
                        loss_i,
                        i,
                        medusa_only_heads,
                        medusa_decay_coefficient,
                        medusa_heads_coefficient,
                        medusa_scheduler_coefficient,
                    )

        return_dict = (
            return_dict if return_dict is not None else model.config.use_return_dict
        )

        if not return_dict:
            output = (medusa_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=medusa_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = types.MethodType(forward, model)