import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import clones


def scaled_dot_product(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # TODO: sen长度不同
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of heads
        :param d_model: dimension of model
        :param dropout: drop-out rate
        """
        super(MultiHeadAttention, self).__init__()
        # we assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model and split heads => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]  # (batch, h, sen, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = scaled_dot_product(query, key, value, mask=mask,
                                          dropout=self.dropout)  # (batch, h, sen, d_k)

        # 3) "Concat" using a view and apply a final linear.
        # x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = torch.reshape(x.transpose(1, 2), (nbatches, -1, self.h * self.d_k))
        return self.linears[-1](x)  # (batch, sen, d_model)


class WordAttention(nn.Module):
    """
    Simple attention layer
    """
    def __init__(self, hidden_dim):
        super(WordAttention, self).__init__()
        self.w1 = nn.Parameter(torch.randn(hidden_dim, requires_grad=True))

    def forward(self, input):
        # shape (batch, seq_len, hidden_dim) * (hidden_dim, 1) -> (batch, hidden_dim)
        # attention with masked softmax
        attn = torch.einsum('ijk,k->ij', [input, self.w1])  # (batch, seq_len)
        attn_max, _ = torch.max(attn, dim=1, keepdim=True)  # (batch, 1)
        attn_exp = torch.exp(attn - attn_max)  # (batch, seq_len)  used exp-normalize-trick here
        attn_exp = attn_exp * (attn != 0).float()  # (batch, seq_len)  因为句子不一样长，有的句子后面全是padding:0
        norm_attn = attn_exp / (torch.sum(attn_exp, dim=1, keepdim=True))  # (batch, seq_len)
        summary = torch.einsum("ijk,ij->ik", [input, norm_attn])  # (batch, hidden_dim)
        return summary


class SentenceAttention(nn.Module):
    def __init__(self, input_dim):
        super(SentenceAttention, self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim, requires_grad=True))

    def forward(self, input):
        # shape (batch, hidden_dim) * (hidden_dim, 1) -> (1, hidden_dim)
        attn = torch.matmul(input, self.w1)
        norm_attn = F.softmax(attn, 0)  # (batch_size)
        weighted = torch.mul(input, norm_attn.unsqueeze(-1).expand_as(input))  # 元素对应相乘(支持broadcast所以不expand也行)
        summary = weighted.sum(0).squeeze()  # (hidden_dim) 若sum不加keepdim=True的话不用squeeze就已经是(hidden_dim)了
        return summary
