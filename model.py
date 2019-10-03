import torch
import torch.nn as nn
import torch.nn.functional as F
import my_dataloader
import utils
import numpy as np


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


class Net(nn.Module):
    '''
    Deep Nerual Network for supervised relation classification
    '''
    def __init__(self, d_w, d_e, num_filter, num_classes, window_size, word_emb_weight):
        '''
        :param d_w: dimension for word embedding
        :param d_e: dimension for position embedding
        :param num_filter: number of filters for the conv layer
        :param window_size: kernel size <=> shape == (window_size, sen_length)
        :param word_emb_path: the path of word embedding file
        '''
        super(Net, self).__init__()
        self.max_sen_len = 100
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.pos_embedding1 = nn.Embedding(2*self.max_sen_len, d_e)
        self.pos_embedding2 = nn.Embedding(2*self.max_sen_len, d_e)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_filter,
                      kernel_size=(window_size, d_w+2*d_e),
                      stride=(1, 1),
                      padding=(0, 0)),  # out_shape: (batch_size, num_filter, max_sen_len-window_size+1, 1)
                      # padding=(1, 0)),  # out_shape: (batch_size, num_filter, max_sen_len, 1)
            # nn.Tanh(),
            nn.MaxPool2d(kernel_size=(self.max_sen_len-window_size+1, 1), stride=(1, 1)),  # out_shape: (batch_size, num_filter, 1, 1)
            nn.ReLU()
            # nn.Dropout2d(p=0.5)  # TODO: try drop p=0.5 or use batchNorm
        )
        self.conv_layer.apply(self.weights_init)
        self.linear_layer = nn.Sequential(
            nn.Linear(num_filter, int(num_filter / 2)),
            # nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(int(num_filter / 2), num_classes)  # out_shape: (batch_size, num_classes)
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        '''
        forward procedure for the conv net
        :param x: (x_text[index] :: a tuple, pos1[index] :: tensor, pos2[index] :: tensor)
        :return: a computation result that output by the forward procedure
        '''
        # Todo: 把x_text[index], pos1[index]和pos2[index]分别查embedding并合并为一个torch tensor matrix
        x_text, pos1, pos2 = x
        x_text_matrix = self.w2v(x_text)  # out_shape: (batch_size, max_sen_len, d_w)
        pos1_matrix = self.pos_embedding1(pos1)  # out_shape: (batch_size, max_sen_len, d_e)
        pos2_matrix = self.pos_embedding2(pos2)  # out_shape: (batch_size, max_sen_len, d_e)
        x = torch.cat((x_text_matrix, pos1_matrix, pos2_matrix), 2)  # out_shape: (batch_size, max_sen_len, d_w+2*d_e)
        x = torch.unsqueeze(x, 1)  # out_shape: (batch_size, in_channel=1, max_sen_len, d_w+2*d_e)
        x = self.conv_layer(x)  # out_shape: (batch_size, num_filter, 1, 1)
        batch_size, num_filter = x.shape[0], x.shape[1]
        assert x.shape[2] == 1 and x.shape[3] == 1
        x = x.view(batch_size, -1)
        assert x.shape[1] == num_filter
        out = self.linear_layer(x)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class RNN_net(nn.Module):
    '''
    Supervised RNN version
    '''
    def __init__(self, d_w, d_e, hidden_dim, num_layers, num_classes, word_emb_weight,
                 dropout_prob=0.2, activation_fn=nn.ReLU):
        super(RNN_net, self).__init__()
        self.max_sen_len = 100
        self.hidden_dim = hidden_dim
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.pos_embedding1 = nn.Embedding(2 * self.max_sen_len, d_e)
        self.pos_embedding2 = nn.Embedding(2 * self.max_sen_len, d_e)
        self.rnn_layer = nn.GRU(input_size=d_w+2*d_e,
                                hidden_size=hidden_dim,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)  # out_shape: (batch_size, seq_len, hidden_dim*num_directions)
        self.rnn_layer.apply(self.weights_init)
        self.word_attn = WordAttention(hidden_dim*2)  # out_shape: (batch_size, hidden_dim*num_directions)
        self.linear_layer = nn.Sequential(  # int_shape: (batch_size, hidden_size*num_directions)
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, int(hidden_dim / 4)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim / 4), num_classes)  # out_shape: (batch_size, num_classes)
        )
        self.linear_layer.apply(self.weights_init)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = sorted(x, key=lambda x: len(x[0]))
        x.reverse()  # the longest sequence should be placed in the first of the list

        seq_len = list()
        batch_in = list()
        for i, (sen, pos1, pos2) in enumerate(x):
            seq_len.append(len(sen))
            _word = self.w2v(torch.from_numpy(sen))  # shape: (sen_len, d_w)
            _pos1 = self.pos_embedding1(torch.from_numpy(pos1))  # shape: (sen_len, d_e)
            _pos2 = self.pos_embedding2(torch.from_numpy(pos2))  # shape: (sen_len, d_e)
            combined = torch.cat((_word, _pos1, _pos2), 1)  # shape: (sen_len, d_w+2*d_e)
            batch_in.append(combined)

        seq_len = torch.tensor(seq_len, dtype=torch.int)

        # 把句子都pad到最长句子的那个长度
        stacked_batch_in = nn.utils.rnn.pad_sequence(sequences=batch_in,
                                                     batch_first=True)  # out_shape: (batch_size, m_sen_len, d_w+2*d_e)
        packed_batch_in = nn.utils.rnn.pack_padded_sequence(stacked_batch_in,
                                                            seq_len,
                                                            batch_first=True)  # out_shape: (batch_size, m_sen_len, d_w+2*d_e)
        rnn_output, _ = self.rnn_layer(
            packed_batch_in)  # out_shape: (batch_size, m_sen_len, hidden_size*num_directions)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        unpacked = self.tanh(unpacked)
        output = self.word_attn(unpacked)  # out_shape: (batch_size, hidden_size*num_directions)
        output = self.tanh(output)
        output = self.linear_layer(output)  # shape: (batch_size, num_classes)
        return output

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # TODO: use orthogonal init
        if isinstance(m, nn.GRU) or isinstance(m, nn.LSTM) or isinstance(m, nn.RNN):
            ih = (param.data for name, param in m.named_parameters() if 'weight_ih' in name)
            hh = (param.data for name, param in m.named_parameters() if 'weight_hh' in name)
            b = (param.data for name, param in m.named_parameters() if 'bias' in name)
            for t in ih:
                nn.init.xavier_uniform_(t)
            for t in hh:
                nn.init.orthogonal_(t)
            for t in b:
                nn.init.constant_(t, 0)


class ddsNet(nn.Module):
    '''
    Deep Neural Network for distant supervised relation classification -- CNN
    '''
    def __init__(self, d_w, d_e, num_filter, num_classes, window_size, word_emb_weight):
        super(ddsNet, self).__init__()
        self.max_sen_len = 100
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.pos_embedding1 = nn.Embedding(2 * self.max_sen_len, d_e)
        self.pos_embedding2 = nn.Embedding(2 * self.max_sen_len, d_e)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_filter,
                      kernel_size=(window_size, d_w + 2 * d_e),
                      stride=(1, 1),
                      padding=(0, 0)),  # out_shape: (batch_size, num_filter, max_sen_len-window_size+1, 1)
            # padding=(1, 0)),  # out_shape: (batch_size, num_filter, max_sen_len, 1)
            # nn.Tanh(),
            nn.MaxPool2d(kernel_size=(self.max_sen_len - window_size + 1, 1), stride=(1, 1)),
            # out_shape: (batch_size, num_filter, 1, 1)
            nn.ReLU()
            # nn.Dropout2d(p=0.5)  # TODO: try drop p=0.5 or use batchNorm
        )
        self.conv_layer.apply(self.weights_init)
        self.linear_layer = nn.Sequential(
            nn.Linear(num_filter, int(num_filter / 2)),
            # nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(int(num_filter / 2), num_classes)  # out_shape: (batch_size, num_classes)
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        # x和y是同一个entity pair的句子以及所有可能的relation
        # x: [(sent2id, pos1vec, pos2vec)]    y: [1, 0, ..., 1, ..., 0]
        input2fc = None  # input to fully connection layer(linear layer)
        num_sen = len(x)
        for sent2id, pos1vec, pos2vec in x:
            # convert to torch array and add new dimension as batch_size (here batch_size=1)
            sent2id = torch.unsqueeze(torch.from_numpy(sent2id), 0)  # out_shape: (batch_size, max_sen_len)
            pos1vec = torch.unsqueeze(torch.from_numpy(pos1vec), 0)  # out_shape: (batch_size, max_sen_len)
            pos2vec = torch.unsqueeze(torch.from_numpy(pos2vec), 0)  # out_shape: (batch_size, max_sen_len)
            x_text_matrix = self.w2v(sent2id)  # out_shape: (batch_size, max_sen_len, d_w)
            pos1_matrix = self.pos_embedding1(pos1vec)  # out_shape: (batch_size, max_sen_len, d_e)
            pos2_matrix = self.pos_embedding2(pos2vec)  # out_shape: (batch_size, max_sen_len, d_e)
            x = torch.cat((x_text_matrix, pos1_matrix, pos2_matrix),
                          2)  # out_shape: (batch_size, max_sen_len, d_w+2*d_e)
            x = torch.unsqueeze(x, 1)  # out_shape: (batch_size, in_channel=1, max_sen_len, d_w+2*d_e)
            x = self.conv_layer(x)  # out_shape: (batch_size, num_filter, 1, 1)
            batch_size, num_filter = x.shape[0], x.shape[1]
            x = x.view(batch_size, -1)  # out_shape: (batch_size, num_filter)
            assert x.shape[1] == num_filter

            if input2fc is None:
                input2fc = x / num_sen  # shape: (batch_size, num_filter)
            else:
                input2fc += x / num_sen  # shape: (batch_size, num_filter)
            assert input2fc.shape[0] == batch_size and input2fc.shape[1] == num_filter

        out = self.linear_layer(input2fc)  # out_shape: (batch_size, num_classes)
        out = torch.squeeze(out)  # out_shape: (num_classes)
        return out

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# TODO: Do we have to initialize h0?
# TODO: Implement attention layer
class ddsRNN(nn.Module):
    '''
        Deep Neural Network for distant supervised relation classification -- RNN
    '''
    def __init__(self, d_w, d_e, hidden_dim, num_layers, num_classes, word_emb_weight,
                 dropout_prob=0.5, activation_fn=nn.ReLU):
        super(ddsRNN, self).__init__()
        self.max_sen_len = 100
        self.hidden_dim = hidden_dim
        self.w2v = nn.Embedding.from_pretrained(word_emb_weight, freeze=False)
        self.pos_embedding1 = nn.Embedding(2 * self.max_sen_len, d_e)
        self.pos_embedding2 = nn.Embedding(2 * self.max_sen_len, d_e)
        self.rnn_layer = nn.LSTM(input_size=d_w+2*d_e,
                                 hidden_size=hidden_dim,
                                 num_layers=num_layers,
                                 bias=True,
                                 batch_first=True,
                                 dropout=dropout_prob,
                                 bidirectional=True)  # out_shape: (batch_size, seq_len, hidden_size*num_directions)
        self.rnn_layer.apply(self.weights_init)
        self.word_attn = WordAttention(hidden_dim*2)  # out_shape: (batch_size, hidden_dim*num_directions)
        self.sent_attn = SentenceAttention(hidden_dim*2)  # out_shape: (hidden_dim*num_directions)
        self.linear_layer = nn.Sequential(  # int_shape: (hidden_size*num_directions)
            nn.Linear(hidden_dim*2, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, num_classes)  # out_shape: (num_classes)
        )
        self.linear_layer.apply(self.weights_init)

    def forward(self, x):
        x = sorted(x, key=lambda x: len(x[0]))
        x.reverse()  # the longest sequence should be placed in the first of the list

        seq_len = list()
        batch_in = list()
        for i, (sen, pos1, pos2) in enumerate(x):
            seq_len.append(len(sen))
            _word = self.w2v(torch.from_numpy(sen))  # shape: (sen_len, d_w)
            _pos1 = self.pos_embedding1(torch.from_numpy(pos1))  # shape: (sen_len, d_e)
            _pos2 = self.pos_embedding2(torch.from_numpy(pos2))  # shape: (sen_len, d_e)
            combined = torch.cat((_word, _pos1, _pos2), 1)  # shape: (sen_len, d_w+2*d_e)
            batch_in.append(combined)

        seq_len = torch.tensor(seq_len, dtype=torch.int)

        # 把句子都pad到最长句子的那个长度
        stacked_batch_in = nn.utils.rnn.pad_sequence(sequences=batch_in,
                                                     batch_first=True)  # out_shape: (batch_size, m_sen_len, d_w+2*d_e)
        packed_batch_in = nn.utils.rnn.pack_padded_sequence(stacked_batch_in,
                                                            seq_len,
                                                            batch_first=True)  # out_shape: (batch_size, m_sen_len, d_w+2*d_e)
        rnn_output, _ = self.rnn_layer(packed_batch_in)  # out_shape: (batch_size, m_sen_len, hidden_size*num_directions)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        '''
        # TODO: 尝试 average word vectors
        B, T, D = unpacked.shape  # B: batch_size, T: length of longest sequence, D: hidden_size*num_directions
        output = list()
        for i in range(B):
            last_index = unpacked_len[i] - 1
            last_out = unpacked[i, last_index, :]  # shape: (hidden_size*num_directions)
            last_out = torch.unsqueeze(last_out, 0)  # shape: (1, hidden_size*num_directions)
            output.append(last_out)
        output = torch.cat(output, 0)  # shape: (batch_size, hidden_size*num_directions)
        # 把batch中的每句句子预测出来的class结合到一起 => shape: (num_classes)
        output = torch.sum(output, 0) / B  # shape: (hidden_size*num_directions)
        output = torch.unsqueeze(output, 0)  # shape: (1, hidden_size*num_directions)
        '''
        output = self.word_attn(unpacked)
        output = self.sent_attn(output)
        output = self.linear_layer(output)  # shape: (num_classes)

        # output = torch.squeeze(output)  # shape: (num_classes)
        return output

    # method to initialize the model weights (in order to improve performance)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # TODO: use orthogonal init
        if isinstance(m, nn.GRU) or isinstance(m, nn.LSTM) or isinstance(m, nn.RNN):
            ih = (param.data for name, param in m.named_parameters() if 'weight_ih' in name)
            hh = (param.data for name, param in m.named_parameters() if 'weight_hh' in name)
            b = (param.data for name, param in m.named_parameters() if 'bias' in name)
            # nn.init.uniform(m.embed.weight.data, a=-0.5, b=0.5)
            for t in ih:
                nn.init.xavier_uniform(t)
            for t in hh:
                nn.init.orthogonal(t)
            for t in b:
                nn.init.constant(t, 0)

