from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch
from models.HALSTM import AttentionWordRNN,AttentionSentRNN,AttentionSentRNNv2,AttentionWordRNNv2


class plainEncoder(nn.Module):

    def __init__(self,
             vocab_size,
             embed_size,
             hidden_size,
             max_len,
             feature_base_dim,
             n_layers=1,
             dropout=0.5,
             bidirection=False,
             unit="RNN"):
        super(plainEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.max_len = max_len
        self.feature_base_dim = feature_base_dim
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.num_dir = 2 if bidirection else 1
        if unit == "RNN":
            self.rnn1 = nn.RNN(self.embed_size, self.hidden_size, n_layers,
                              dropout=dropout, bidirectional=bidirection,batch_first = True)
            self.rnn2 = nn.RNN(self.embed_size, self.hidden_size, n_layers,
                               dropout=dropout, bidirectional=bidirection,batch_first = True)
        if unit == "LSTM":
            self.rnn1 = nn.LSTM(self.embed_size, self.hidden_size, n_layers,
                               dropout=dropout, bidirectional=bidirection,batch_first = True)
            self.rnn2 = nn.LSTM(self.embed_size, self.hidden_size, n_layers,
                                dropout=dropout, bidirectional=bidirection,batch_first = True)
        if unit == "GRU":
            self.rnn1 = nn.GRU(self.embed_size, self.hidden_size, n_layers,
                              dropout=dropout, bidirectional=bidirection,batch_first = True)
            self.rnn2 = nn.GRU(self.embed_size, self.hidden_size, n_layers,
                               dropout=dropout, bidirectional=bidirection,batch_first = True)
        self.init_weights()
        # -> ngf x 1 x 1
        self.fc = nn.Sequential(
            nn.Linear(2*self.num_dir*self.hidden_size, self.feature_base_dim, bias=False),
            nn.LeakyReLU(0.2, True)
        )
    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, x1, x2, hidden=None):
        embedded1 = self.embed(x1)
        embedded2 = self.embed(x2)

        self.rnn1.flatten_parameters()
        outputs1, hidden1 = self.rnn1(embedded1, hidden)
        self.rnn2.flatten_parameters()
        outputs2, hidden2 = self.rnn2(embedded2, hidden)

        outputs = torch.cat((outputs1[:,-1,:], outputs2[:,-1,:]), 1)
        # print(outputs.shape)
        outputs = self.fc(outputs)
        return outputs, hidden


class HAttnEncoder(nn.Module):

    def __init__(self,
             vocab_size,
             embed_size,
             hidden_size,
             max_len,
             feature_base_dim,
             n_layers=1,
             dropout=0.5,
             bidirection=True,
             unit="RNN"):
        super(HAttnEncoderv3, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.max_sent_len = max_len
        self.feature_base_dim = feature_base_dim
        self.num_dir = 2 if bidirection else 1
        self.batch_first = True

        # Word-Level LSTM
        self.word_RNN1 = AttentionWordRNNv2(num_tokens=self.vocab_size,
                                         embed_size=self.embed_size,
                                         word_gru_hidden=self.hidden_size,
                                         bidirectional=bidirection,
                                         dropout=dropout,
                                         batch_first=self.batch_first)
        # Sentence-Level LSTM
        self.setence_RNN1 = AttentionSentRNNv2(sent_gru_hidden=self.hidden_size,
                                            word_gru_hidden=self.hidden_size,
                                            feature_base_dim=self.feature_base_dim,
                                            bidirectional=bidirection,
                                            dropout=dropout,
                                            batch_first=self.batch_first)
        # Word-Level LSTM
        self.word_RNN2 = AttentionWordRNNv2(num_tokens=self.vocab_size,
                                            embed_size=self.embed_size,
                                            word_gru_hidden=self.hidden_size,
                                            bidirectional=bidirection,
                                            dropout=dropout,
                                            batch_first=self.batch_first)
        # Sentence-Level LSTM
        self.setence_RNN2 = AttentionSentRNNv2(sent_gru_hidden=self.hidden_size,
                                               word_gru_hidden=self.hidden_size,
                                               feature_base_dim=self.feature_base_dim,
                                               bidirectional=bidirection,
                                               dropout=dropout,
                                               batch_first=self.batch_first)

        # -> ngf x 1 x 1
        self.fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size * self.num_dir, self.feature_base_dim, bias=False),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x1, x2, hidden=None):

        outputs1 = self.forward_once1(x1)
        outputs2 = self.forward_once2(x2)

        outputs = torch.cat((outputs1, outputs2), 1)

        outputs = self.fc(outputs)
        return outputs, hidden

    def forward_once1(self,x,state_word=None):
        batch,sent_len,word_num = x.shape

        x = x.view(batch*sent_len, -1)
        word_embed, state_word, _ = self.word_RNN1(x)
        all_word_embed = word_embed.view(batch, sent_len, -1)
        sent_embed, state_sent, _ = self.setence_RNN1(all_word_embed)
        return sent_embed

    def forward_once2(self,x,state_word=None):
        batch,sent_len,word_num = x.shape

        x = x.view(batch*sent_len, -1)
        word_embed, state_word, _ = self.word_RNN2(x)
        all_word_embed = word_embed.view(batch, sent_len, -1)
        sent_embed, state_sent, _ = self.setence_RNN2(all_word_embed)
        return sent_embed



class RNN_ENCODER(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 max_len,
                 feature_base_dim,
                 n_layers=1,
                 dropout=0.5,
                 bidirection=True,
                 unit="RNN"):
        super(RNN_ENCODER, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.feature_base_dim = feature_base_dim
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.num_dir = 2 if bidirection else 1
        if unit == "RNN":
            self.rnn1 = nn.RNN(self.embed_size, self.hidden_size, n_layers,
                               dropout=dropout, bidirectional=bidirection, batch_first=True)
            self.rnn2 = nn.RNN(self.embed_size, self.hidden_size, n_layers,
                               dropout=dropout, bidirectional=bidirection, batch_first=True)
        if unit == "LSTM":
            self.rnn1 = nn.LSTM(self.embed_size, self.hidden_size, n_layers,
                                dropout=dropout, bidirectional=bidirection, batch_first=True)
            self.rnn2 = nn.LSTM(self.embed_size, self.hidden_size, n_layers,
                                dropout=dropout, bidirectional=bidirection, batch_first=True)
        if unit == "GRU":
            self.rnn1 = nn.GRU(self.embed_size, self.hidden_size, n_layers,
                               dropout=dropout, bidirectional=bidirection, batch_first=True)
            self.rnn2 = nn.GRU(self.embed_size, self.hidden_size, n_layers,
                               dropout=dropout, bidirectional=bidirection, batch_first=True)
        self.init_weights()
        # -> ngf x 1 x 1
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * (self.max_len[0] + self.max_len[1]) * self.num_dir, self.feature_base_dim,
                      bias=False),
            nn.LeakyReLU(0.2, True)
        )

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (Variable(weight.new(self.n_layers * self.num_dir,
                                    bsz, self.hidden_size).zero_()),
                Variable(weight.new(self.n_layers * self.num_dir,
                                    bsz, self.hidden_size).zero_()))

    def forward(self, x1, x2, hidden=None):

        emb1 = self.drop(self.embed(x1))
        self.rnn1.flatten_parameters()
        outputs1, hidden = self.rnn1(emb1)

        emb2 = self.drop(self.embed(x2))
        self.rnn2.flatten_parameters()
        outputs2, hidden = self.rnn1(emb2)

        c_code1 = outputs1.transpose(1, 2)
        c_code2 = outputs2.transpose(1, 2)

        words_emb = torch.cat((c_code1, c_code2), 2)

        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.hidden_size * self.num_dir)
        # print (sent_emb.shape)

        return words_emb, sent_emb