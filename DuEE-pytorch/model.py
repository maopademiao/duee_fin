from pytorch_transformers import BertPreTrainedModel, BertModel
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class BertForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.entity_activate_fn = nn.Tanh()
        # self.entity_ffn = nn.Linear(config.hidden_size, config.hidden_size)
        # self.final_hidden_ffn = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, segment_ids,mask_ids, labels=None):
        vectors = self.bert(input_ids,segment_ids,mask_ids)

        h1 = self.dropout(vectors[0])

        logits = self.classifier(h1)


        return logits  # (loss), logits, (hidden_states), (attentions)

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.entity_activate_fn = nn.Tanh()
        # self.entity_ffn = nn.Linear(config.hidden_size, config.hidden_size)
        # self.final_hidden_ffn = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, segment_ids,mask_ids, labels=None):
        vectors = self.bert(input_ids,segment_ids,mask_ids)

        h1 = self.dropout(vectors[1])

        logits = self.classifier(h1)


        return logits  # (loss), logits, (hidden_states), (attentions)