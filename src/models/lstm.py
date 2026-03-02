# pylint: disable=no-member, invalid-name, missing-function-docstring, missing-class-docstring
"""LSTM model and its variations"""

import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        self.hidden_size = int(model_config["hidden_size"])
        bidirectional = bool(model_config["bidirectional"])
        num_layers = int(model_config["num_layers"])

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                2 * self.hidden_size if bidirectional else self.hidden_size, output_size
            ),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        if self.bidirectional:
            out_forward = lstm_output[:, -1, : self.hidden_size]
            out_reverse = lstm_output[:, 0, self.hidden_size :]
            lstm_output = torch.cat((out_forward, out_reverse), 1)
        else:
            lstm_output = lstm_output[:, -1, :]

        fc_output = self.fc(lstm_output)
        return fc_output


class MeanLSTM(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        bidirectional = bool(model_config["bidirectional"])
        num_layers = int(model_config["num_layers"])

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_size),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        lstm_output = torch.mean(lstm_output, dim=1)
        logits = self.fc(lstm_output)

        return logits
