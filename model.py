import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPTNeoXModel, GPTNeoXPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class PrefixEncoder(nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(config.pre_seq_len,
                                          config.hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(config.hidden_size, config.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(config.prefix_hidden_size,
                          config.num_hidden_layers * 2 * config.hidden_size))
        else:
            self.embedding = nn.Embedding(
                config.pre_seq_len,
                config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class GPTNeoXPrefixForCausalLM(GPTNeoXPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size,
                                   config.vocab_size,
                                   bias=False)

        for param in self.gpt_neox.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(
            batch_size, -1).to(self.gpt_neox.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, self.pre_seq_len,
                                               self.n_layer * 2, self.n_head,
                                               self.n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(
            self.gpt_neox.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask),
                                   dim=1)

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                               labels.view(-1))

        if not return_dict:
            output = (lm_logits, ) + outputs[1:]
            return ((lm_loss, ) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# class BertPromptForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.bert = BertModel(config)
#         self.embeddings = self.bert.embeddings
#         self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

#         for param in self.bert.parameters():
#             param.requires_grad = False

#         self.pre_seq_len = config.pre_seq_len
#         self.n_layer = config.num_hidden_layers
#         self.n_head = config.num_attention_heads
#         self.n_embd = config.hidden_size // config.num_attention_heads

#         self.prefix_tokens = torch.arange(self.pre_seq_len).long()
#         self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

#     def get_prompt(self, batch_size):
#         prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
#         prompts = self.prefix_encoder(prefix_tokens)
#         return prompts

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         batch_size = input_ids.shape[0]
#         raw_embedding = self.embeddings(
#             input_ids=input_ids,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#         )
#         prompts = self.get_prompt(batch_size=batch_size)
#         inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
#         prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
#         attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

#         outputs = self.bert(
#             # input_ids,
#             attention_mask=attention_mask,
#             # token_type_ids=token_type_ids,
#             # position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             # past_key_values=past_key_values,
#         )

#         # pooled_output = outputs[1]
#         sequence_output = outputs[0]
#         sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
#         first_token_tensor = sequence_output[:, 0]
#         pooled_output = self.bert.pooler.dense(first_token_tensor)
#         pooled_output = self.bert.pooler.activation(pooled_output)

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
