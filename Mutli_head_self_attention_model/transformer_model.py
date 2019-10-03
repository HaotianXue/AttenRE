import torch
import torch.nn as nn
import attention
import layers
import copy
import torch.nn.functional as F


class dds_Transformer(nn.Module):

    """
    Transformer model for distant supervised relation extraction
    """

    def __init__(self, d_w, d_e, num_classes, hidden_dim, word_emb_weight,
                 num_layers=4, num_heads=8, dropout=0.1, max_sen_len=100):
        super(dds_Transformer, self).__init__()
        self.max_sen_len = max_sen_len
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.pos_embedding1 = nn.Embedding(2 * self.max_sen_len, d_e)
        self.pos_embedding2 = nn.Embedding(2 * self.max_sen_len, d_e)
        c = copy.deepcopy
        d_model = d_w + 2 * d_e
        self_attn = attention.MultiHeadAttention(h=num_heads, d_model=d_model, dropout=dropout)
        ff = layers.PositionwiseFeedForward(d_model=d_model, d_ff=hidden_dim, dropout=dropout)
        word_attn = attention.WordAttention(d_model)  # (batch, sen, d_model) => (batch, d_model)
        sen_attn = attention.SentenceAttention(d_model)  # (batch, d_model) => (1, d_model)
        self.model = nn.Sequential(
            layers.Encoder(layers.EncoderLayer(d_model, c(self_attn), c(ff), dropout), num_layers),
            word_attn,
            sen_attn,
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        for p in self.model.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)

    def forward(self, x):  # x: (batch, 3, max_sen_len)
        x_text, pos1, pos2 = x[:, 0, :], x[:, 1, :], x[:, 2, :]  # each: (batch_size, max_sen_len)
        x_text = self.w2v(torch.from_numpy(x_text))  # (batch_size, max_sen_len, d_w)
        pos1 = self.pos_embedding1(torch.from_numpy(pos1))  # (batch_size, max_sen_len, d_e)
        pos2 = self.pos_embedding2(torch.from_numpy(pos2))  # (batch_size, max_sen_len, d_e)
        input_tensor = torch.cat((x_text, pos1, pos2), 2)  # (batch_size, max_sen_len, d_w+2*d_e)
        output = self.model(input_tensor)  # (1, num_classes)
        output = torch.squeeze(output, dim=0)  # (num_classes)
        return output


class Transformer(nn.Module):

    """
    Transformer model for supervised relation extraction
    """

    def __init__(self, d_w, d_e, num_classes, hidden_dim, word_emb_weight,
                 num_layers=4, num_heads=8, dropout=0.1, max_sen_len=100):
        super(Transformer, self).__init__()
        self.max_sen_len = max_sen_len
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.pos_embedding1 = nn.Embedding(2 * self.max_sen_len, d_e)
        self.pos_embedding2 = nn.Embedding(2 * self.max_sen_len, d_e)
        c = copy.deepcopy
        d_model = d_w + 2 * d_e
        self_attn = attention.MultiHeadAttention(h=num_heads, d_model=d_model, dropout=dropout)
        ff = layers.PositionwiseFeedForward(d_model=d_model, d_ff=hidden_dim, dropout=dropout)
        word_attn = attention.WordAttention(d_model)  # (batch, sen, d_model) => (batch, d_model)
        self.model = nn.Sequential(
            layers.Encoder(layers.EncoderLayer(d_model, c(self_attn), c(ff), dropout), num_layers),
            word_attn,
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        for p in self.model.parameters():
            if p.dim() > 1:  # dim: 维度数
                nn.init.xavier_uniform_(p)

    def forward(self, x):  # x: (batch, 3, max_sen_len)
        x_text, pos1, pos2 = x[:, 0, :], x[:, 1, :], x[:, 2, :]  # each: (batch_size, max_sen_len)
        x_text = self.w2v(torch.from_numpy(x_text))  # (batch_size, max_sen_len, d_w)
        pos1 = self.pos_embedding1(torch.from_numpy(pos1))  # (batch_size, max_sen_len, d_e)
        pos2 = self.pos_embedding2(torch.from_numpy(pos2))  # (batch_size, max_sen_len, d_e)
        input_tensor = torch.cat((x_text, pos1, pos2), 2)  # (batch_size, max_sen_len, d_w+2*d_e)
        output = self.model(input_tensor)  # (batch_size, num_classes)
        return output

