import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from utils import mix_pos_neg_Trans


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x



class TransModel(nn.Module):
    def __init__(self, vocab_size=24):
        super().__init__()
        self.hidden_dim = 25
        self.emb_dim = 512

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.emb_dim, 61)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)

        self.esm_block = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.LeakyReLU(),
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
        )

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16*513, 512)

        self.block1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
        )
        self.out1 = nn.Sequential(
            nn.BatchNorm1d(1536),
            nn.LeakyReLU(),
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(),
            # nn.Linear(64,2)
        )

        self.out2 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64,2)
        )

    def forward(self, x, esm, fp, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x).permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask = mask).permute(1, 0, 2)
        # x = x.mean(dim=1)
        # x, _ = torch.max(x,dim=1)
        # x = self.pool_avg(x.transpose(1,2)).squeeze(2)
        x = self.pool_max(x.transpose(1,2)).squeeze(2)
        fp = F.relu(self.conv1(fp.unsqueeze(1)))
        fp = self.pool(fp)
        fp = fp.view(fp.size(0),-1)
        fp = F.relu(self.fc1(fp))
        esm = self.esm_block(esm)
        x = torch.cat((x,fp,esm),dim=1)
        x = self.out1(x)
        return self.out2(x)




