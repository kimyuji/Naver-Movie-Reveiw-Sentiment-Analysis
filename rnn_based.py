
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NSMC_classifier(nn.Module):
    def __init__(self, num_vocab, embedding_dim, hidden_dim, num_layers, out_node, drop_percent=0.2, model="RNN"):  
        super(NSMC_classifier, self).__init__()
        self.model = model
        self.num_vocab = num_vocab
        self.embed = nn.Embedding(num_vocab, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_percent)

        self.fc = nn.Linear(hidden_dim, out_node) # out_node = 1 (binary)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.embed(x)

    # choose a model 
        if self.model == "RNN":
            out, hidden = self.rnn(x)
        elif self.model == "LSTM":
            out, (hidden, cell_state) = self.lstm(x)
            out = self.dropout(out)
        elif self.model == "GRU":
            out, hidden = self.gru(x)
        else:
            print("choose a model in ['RNN', 'LSTM', 'GRU']")
            raise # raise error

        out = self.dropout(out)
        out = self.fc(out[:,-1,:]) 
        out = self.sigmoid(out)
        return out
