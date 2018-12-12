import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSpace(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
        num_layers=1, bias=True, dropout=0, bidirectional=False):

        assert 0 <= dropout < 1

        super().__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
            num_layers = num_layers,
            bias = bias,
            dropout = dropout,
            bidirectional=bidirectional)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden() # hidden, cell

    def forward(self, char_idxs):
        embeds = self.embeddings(char_idxs)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(char_idxs), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(char_idxs.size()[0], -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores

    def init_hidden(self):
        # (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


class GRUSpace(nn.Module):

    def __init__(self, eembedding_dim, hidden_dim, vocab_size, tagset_size,
        num_layers=1, bias=True, dropout=0, bidirectional=False):

        assert 0 <= dropout < 1

        super().__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim,
            num_layers = num_layers,
            bias = bias,
            dropout = dropout,
            bidirectional=bidirectional)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden() # hidden, cell

    def forward(self, char_idxs):
        embeds = self.embeddings(char_idxs)
        lstm_out, self.hidden = self.gru(
            embeds.view(len(char_idxs), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(char_idxs.size()[0], -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores

    def init_hidden(self):
        # (num_layers, minibatch_size, hidden_dim)
        return torch.zeros(1, 1, self.hidden_dim)