import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class BertFcNetwork(nn.Module):

    def __init__(self, bert, bert_size, dropout, num_categories):
        super(BertFcNetwork, self).__init__()
        self.num_categories = num_categories
        self.bert = bert
        self.bert_size = bert_size
        self.sentence_transform = nn.Sequential(
            nn.Linear(bert_size, num_categories),
            nn.Dropout(dropout),
            torch.nn.ReLU()
        )
        # self.linear = nn.Linear(self.cnn_size, self.num_categories, bias=False)

    def forward(self, bert_token, bert_segment, bert_mask):
        _, bertout = self.bert(bert_token, bert_segment, bert_mask, output_all_encoded_layers=False)
        meanbertlayer = torch.mean(_, dim=1)
        out = self.sentence_transform(meanbertlayer)
        # out = self.linear(out)
        # out = torch.softmax(out, -1)
        return out
