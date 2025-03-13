import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, key, value, choose):
        if choose == 0:
            query, key = self.W_q(query), self.W_k(key)
            features = query.unsqueeze(2) + key.unsqueeze(1)
            features = torch.tanh(features)
            scores = self.W_v(features).squeeze(-1)
            attn_weights = F.softmax(scores, dim=1)
        else:
            attn_weights = F.softmax(torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.size(2)), dim=-1)
        return torch.bmm(attn_weights, value)


class Encoder(nn.Module):
    ''' A encoder model using CNN + LSTM structure. '''

    def __init__(self, input_size, hidden_size, embedding_size, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.embedding_size = embedding_size

        # Convolutional Layers
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=input_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.Softplus(),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=input_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(input_size, self.embedding_size),
        )
        self.lstm = nn.LSTM(self.embedding_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.permute(0, 2, 1)
        x = self.layer3(x)
        lstm_out, (h_t, c_t) = self.lstm(x)
        return lstm_out, h_t, c_t

class Decoder(nn.Module):
    ''' Decoder using LSTM + Additive Attention. '''

    def __init__(self, hidden_size, attention_hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = AdditiveAttention(hidden_size, hidden_size, attention_hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, decoder_input, encoder_output):
        attn_output = self.attention(decoder_input, encoder_output, encoder_output, choose=0)
        lstm_out, (h_t, c_t) = self.lstm(attn_output)
        return lstm_out, h_t, c_t


class BMOP(nn.Module):
    ''' A sequence to sequence model with CNN, LSTM and Additive Attention mechanism. '''

    def __init__(self, model_cfg, d_model=128, hidden_size=128, output_size=27, attention_hidden_size=128, dropout=0.2,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.d_model = d_model

        # Projection layers for input/output sequence
        self.proj = nn.Linear(28, d_model)
        self.proj2 = nn.Linear(28, d_model)

        # Encoder and Decoder
        self.encoder = Encoder(input_size=d_model, hidden_size=hidden_size, embedding_size=d_model, dropout=dropout)
        self.decoder = Decoder(hidden_size=hidden_size, attention_hidden_size=attention_hidden_size,
                               output_size=output_size)

        # Final layers to predict corner offset and rotation offset
        self.proj_inverse_corner = nn.Linear(d_model, 27)
        self.proj_inverse_ry = nn.Linear(d_model, 1)

        # Loss and dropout
        self.dropout = nn.Dropout(p=dropout)

    def get_loss(self):

        predicted_box_corner_offset = self.forward_ret_dict['predicted_corner_offset']
        predicted_box_ry_offset = self.forward_ret_dict['predicted_ry_offset']
        gt_box_corner_offset = self.forward_ret_dict['gt_box_corner_offset']
        gt_box_ry_offset = self.forward_ret_dict['gt_box_ry_offset']

        corner_loss = torch.mean((predicted_box_corner_offset - gt_box_corner_offset) ** 2)
        ry_loss = torch.mean((predicted_box_ry_offset - gt_box_ry_offset) ** 2)

        return corner_loss + ry_loss

    def forward(self, batch_dict):

        box_corner_list = batch_dict['history_box_corner']

        src_seq = box_corner_list[:, 0:-1, :] - box_corner_list[:, 1:, :]  # (B, T-1, C)
        trg_seq = box_corner_list[:, 0:1, :] - box_corner_list[:, 1:2, :]  # (B, 1, C)

        src_seq_ = self.proj(src_seq)  # (B, T-1, d_model)
        trg_seq_ = self.proj2(trg_seq)  # (B, 1, d_model)

        enc_output, h_t, c_t = self.encoder(src_seq_)  # Encoder output
        dec_output, dec_h_t, dec_c_t = self.decoder(trg_seq_, enc_output)

        dec_output_corner = self.proj_inverse_corner(dec_output)  # (B, 1, 27)
        dec_output_ry = self.proj_inverse_ry(dec_output)  # (B, 1, 1)

        if self.training:
            ret_dict = {
                'predicted_corner_offset': dec_output_corner,
                'predicted_ry_offset': dec_output_ry,
                'gt_box_corner_offset': batch_dict['gt_box_corner_offset'],
                'gt_box_ry_offset': batch_dict['gt_box_ry_offset']
            }

            self.forward_ret_dict = ret_dict
            loss = self.get_loss()
            batch_dict.update({'corner_loss': loss})

        batch_dict.update({'predicted_box_corner_offset': dec_output_corner,
                           'predicted_box_ry_offset': dec_output_ry})

        return batch_dict

