# pylint: disable=no-member, invalid-name, missing-function-docstring, missing-class-docstring
"""Transformer model and its variations"""

import numpy as np
import torch
from torch import nn

# class PositionalEncoding(nn.Module):
#    """Positional encoding in the form of module"""

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


def positional_encoding(x, n=10000, scaled: bool = True):
    """Positional encoding in the form of function"""
    bs, ln, fs = x.shape
    # bs: batch size
    # ln: length in time
    # fs: number of channels

    if scaled:
        C = np.sqrt(fs)
    else:
        C = 1.0

    P = np.zeros((bs, ln, fs))
    for k in range(ln):
        for i in np.arange(int(fs / 2)):
            denominator = np.power(n, 2 * i / fs)
            P[:, k, 2 * i] = np.ones((bs)) * np.sin(k / denominator)
            P[:, k, 2 * i + 1] = np.ones((bs)) * np.cos(k / denominator)
        if fs % 2 == 1:
            i = int(fs / 2)
            denominator = np.power(n, 2 * i / fs)
            P[:, k, 2 * i] = np.ones((bs)) * np.sin(k / denominator)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    return C * x + torch.tensor(P, dtype=torch.float32).to(device)


class Transformer(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        head_hidden_size = int(model_config["head_hidden_size"])
        num_layers = int(model_config["num_layers"])
        num_heads = int(model_config["num_heads"])
        self.scaled = bool(model_config["scaled"])
        self.post = bool(model_config["post"])

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # if self.post:
        #     self.input_embed = nn.Sequential(
        #         nn.Linear(input_size, hidden_size),
        #         PositionalEncoding(hidden_size),
        #     )
        # else:
        #     self.input_embed = nn.Sequential(
        #         PositionalEncoding(input_size),
        #         nn.Linear(input_size, hidden_size),
        #     )

        self.input_embed = nn.Linear(input_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x.shape: [Batch_size; time_len; n_channels]

        if self.post:
            input_embed = self.input_embed(x)
            input_embed = positional_encoding(input_embed, scaled=self.scaled)
        else:
            input_embed = positional_encoding(x, scaled=self.scaled)
            input_embed = self.input_embed(input_embed)

        tf_output = self.transformer(input_embed)
        tf_output = tf_output[:, 0, :]

        fc_output = self.fc(tf_output)
        return fc_output


class MeanTransformer(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        head_hidden_size = int(model_config["head_hidden_size"])
        num_layers = int(model_config["num_layers"])
        num_heads = int(model_config["num_heads"])
        self.scaled = bool(model_config["scaled"])
        self.post = bool(model_config["post"])

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # if self.post:
        #     self.input_embed = nn.Sequential(
        #         nn.Linear(input_size, hidden_size),
        #         PositionalEncoding(hidden_size),
        #     )
        # else:
        #     self.input_embed = nn.Sequential(
        #         PositionalEncoding(input_size),
        #         nn.Linear(input_size, hidden_size),
        #     )

        self.input_embed = nn.Linear(input_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x.shape: [Batch_size; time_len; n_channels]

        if self.post:
            input_embed = self.input_embed(x)
            input_embed = positional_encoding(input_embed, scaled=self.scaled)
        else:
            input_embed = positional_encoding(x, scaled=self.scaled)
            input_embed = self.input_embed(x)

        tf_output = self.transformer(input_embed)
        tf_output = torch.mean(tf_output, 1)

        fc_output = self.fc(tf_output)
        return fc_output
