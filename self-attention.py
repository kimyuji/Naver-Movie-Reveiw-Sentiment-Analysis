import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention_classifier(nn.Module):
    def __init__(self, num_vocab, embedding_dim, hidden_dim, num_layers, out_node=1, drop_percent=0):
        super(Attention_classifier, self).__init__()
        self.num_vocab = num_vocab
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_percent = drop_percent
        self.embedding = nn.Embedding(num_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=drop_percent, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_node)
        self.sigmoid = nn.Sigmoid()

    # self attention 
    def attention(self, lstm_output, hidden_state): 
        hidden_state = hidden_state.squeeze(0)
        attention_output = torch.bmm(lstm_output, hidden_state.unsqueeze(2)).squeeze(2)
        attention_score = F.softmax(attention_output,1)
        context_vector = torch.bmm(lstm_output.transpose(1,2), attention_score.unsqueeze(2)).squeeze(2)
        return context_vector

    def forward(self, x):
        x = self.embedding(x)
        out, (hidden, _) = self.lstm(x)
        hidden = hidden[-1: , :] # last lyaer의 hidden
        attention_output = self.attention(out, hidden)
        out = self.fc(attention_output)
        out = self.sigmoid(out)
        return out
