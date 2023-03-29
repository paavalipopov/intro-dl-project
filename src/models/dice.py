# pylint: disable=
"""DICE model from https://github.com/UsmanMahmood27/DICE"""

import torch
from torch import nn


class DICE(nn.Module):
    def __init__(self, model_config):

        super().__init__()
        # self.encoder = encoder

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=100 // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(53 ** 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        self.n_regions = 53
        self.n_regions_after = 53

        self.n_heads = 1
        self.attention_embedding = 48 * self.n_heads

        self.upscale = 0.05
        self.upscale2 = 0.5

        self.HW = torch.nn.Hardswish()

        self.gta_embed = nn.Sequential(
            nn.Linear(
                self.n_regions * self.n_regions,
                round(self.upscale * self.n_regions * self.n_regions),
            ),
        )

        self.gta_norm = nn.Sequential(
            nn.BatchNorm1d(round(self.upscale * self.n_regions * self.n_regions)),
            nn.ReLU(),
        )

        self.gta_attend = nn.Sequential(
            nn.Linear(
                round(self.upscale * self.n_regions * self.n_regions),
                round(self.upscale2 * self.n_regions * self.n_regions),
            ),
            nn.ReLU(),
            nn.Linear(round(self.upscale2 * self.n_regions * self.n_regions), 1),
        )

        self.key_layer = nn.Sequential(
            nn.Linear(
                100,
                # self.lstm.output_dim,
                self.attention_embedding,
            ),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(
                100,
                # self.lstm.output_dim,
                self.attention_embedding,
            ),
        )

        self.query_layer = nn.Sequential(
            nn.Linear(
                100,
                # self.lstm.output_dim,
                self.attention_embedding,
            ),
        )

        self.multihead_attn = nn.MultiheadAttention(
            self.attention_embedding, self.n_heads
        )

    def gta_attention(self, x, node_axis=1):

        x_readout = x.mean(node_axis, keepdim=True)
        x_readout = x * x_readout

        a = x_readout.shape[0]
        b = x_readout.shape[1]
        x_readout = x_readout.reshape(-1, x_readout.shape[2])
        x_embed = self.gta_norm(self.gta_embed(x_readout))
        x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)
        x_graphattention = self.HW(x_graphattention.reshape(a, b))
        return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis)

    def multi_head_attention(self, outputs):

        key = self.key_layer(outputs)
        value = self.value_layer(outputs)
        query = self.query_layer(outputs)

        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        query = query.permute(1, 0, 2)

        attn_output, attn_output_weights = self.multihead_attn(key, value, query)
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output, attn_output_weights

    def forward(self, x):
        # x.shape: [Batch_size; time_len; n_channels]
        B, T, C = x.shape

        # # TODO: debug
        # print(f"DICE input shape: {x.shape}")

        # pass input to LSTM; treat each channel as an independent single-feature time series
        x = x.permute(0, 2, 1)
        x = x.reshape(B * C, T, 1)
        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output.reshape(B, C, T, -1)
        # lstm_output.shape: [Batch_size; n_channels; time_len; lstm_hidden_size]

        # # TODO: debug
        # print(f"LSTM output shape: {lstm_output.shape}")

        # pass lstm_output at each time point to multihead attention to reveal spatial correlations
        lstm_output = lstm_output.permute(2, 0, 1, 3)
        # lstm_output.shape: [time_len; Batch_size; n_channels; lstm_hidden_size]
        lstm_output = lstm_output.reshape(T * B, C, -1)
        _, attn_weights = self.multi_head_attention(lstm_output)

        # # TODO: debug
        # print(f"Raw attention output shape: {attn_weights.shape}")

        attn_weights = attn_weights.reshape(T, B, C, C)

        attn_weights = attn_weights.permute(1, 0, 2, 3)

        attn_weights = attn_weights.reshape(B, T, -1)

        # # TODO: debug
        # print(f"Reshaped attention output shape: {attn_weights.shape}")

        FC = self.gta_attention(attn_weights)

        FC = FC.reshape(B, -1)

        # # TODO: debug
        # print(f"FC shape: {FC.shape}")

        logits = self.classifier(FC)

        return logits
