# pylint: disable=C0115,C0103,C0116,R1725,R0913
"""Models for experiments and functions for setting them up"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def model_factory(conf, model_config):
    if conf.model in ["mlp", "wide_mlp", "deep_mlp"]:
        return MLP(model_config)
    elif conf.model == "attention_mlp":
        return AttentionMLP(model_config)
    elif conf.model == "new_attention_mlp":
        return NewAttentionMLP(model_config)
    elif conf.model == "meta_mlp":
        return MetaMLP(model_config)
    elif conf.model == "window_mlp":
        return WindowMLP(model_config)
    elif conf.model == "pe_mlp":
        return PE_MLP(model_config)
    elif conf.model == "pe_att_mlp":
        return PE_Att_MLP(model_config)
    elif conf.model == "mlp_tf":
        return MLP_TF(model_config)

    elif conf.model == "lstm":
        return LSTM(model_config)
    elif conf.model == "noah_lstm":
        return NoahLSTM(model_config)

    elif conf.model == "transformer":
        return Transformer(model_config)
    elif conf.model == "mean_transformer":
        return MeanTransformer(model_config)
    elif conf.model == "pe_transformer":
        return PE_Transformer(model_config)

    # elif conf.model == "stdim":
    #     from src.stdim_encoder_trainer import EncoderTrainer

    #     encoder = EncoderTrainer(
    #         encoder_config=exp.model_config["encoder"],
    #         dataset=exp.encoder_dataset,
    #         logpath=exp.runpath,
    #         wandb_logger=exp.wandb_logger,
    #         batch_size=exp.batch_size,
    #         device=exp.device,
    #     ).train_encoder()

    #     return STDIM(encoder, Probe(exp.model_config["probe"]))

    raise NotImplementedError()


def positional_encoding(x, n=10000, scaled: bool = True):
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


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class MLP(nn.Module):
    def __init__(self, model_config):
        super(MLP, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        num_layers = int(model_config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53
        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, ln, -1)
        logits = fc_output.mean(1)
        return logits


class PE_MLP(nn.Module):
    def __init__(self, model_config):
        super(PE_MLP, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        num_layers = int(model_config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53
        x = positional_encoding(x)
        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, ln, -1)
        logits = fc_output.mean(1)
        return logits


class PE_Att_MLP(nn.Module):
    def __init__(self, model_config):
        super(PE_Att_MLP, self).__init__()

        input_size = int(model_config["input_size"])
        time_length = int(model_config["time_length"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        attention_size = int(model_config["attention_size"])
        num_layers = int(model_config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

        self.attn = nn.Sequential(
            nn.LayerNorm(time_length + 1),
            nn.Dropout(p=dropout),
            nn.Linear(time_length + 1, attention_size),
            nn.ReLU(),
            nn.LayerNorm(attention_size),
            nn.Dropout(p=dropout),
            nn.Linear(attention_size, time_length),
        )

    def get_attention(self, outputs):
        # calculate mean over time
        outputs_mean = outputs.mean(1).unsqueeze(1)

        # add output's mean to the end of each output time dimension
        # as a reference for attention layer
        # and pass it to attention layer
        weights_list = []
        for output, output_mean in zip(outputs, outputs_mean):
            result = torch.cat((output, output_mean)).swapaxes(0, 1)
            result_tensor = self.attn(result)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)
        # normalize weights and swap axes to the output shape
        normalized_weights = F.softmax(weights, dim=2).swapaxes(1, 2)
        return normalized_weights

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53
        x = positional_encoding(x)
        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, ln, -1)

        # get weights form attention layer
        normalized_weights = self.get_attention(fc_output)

        # sum outputs weight-wise
        logits = torch.einsum("ijk,ijk->ik", fc_output, normalized_weights)

        return logits


class MLP_TF(nn.Module):
    def __init__(self, model_config):
        super(MLP_TF, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])

        self.post = bool(model_config["post"])
        self.scaled = bool(model_config["scaled"])

        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        num_layers = int(model_config["num_layers"])

        dec_head_hidden_size = model_config["decoder"]["head_hidden_size"]
        dec_num_heads = model_config["decoder"]["num_heads"]
        dec_num_layers = model_config["decoder"]["num_layers"]

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )

        self.fc = nn.Sequential(*layers)

        dec_hidden_size = dec_head_hidden_size * dec_num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dec_hidden_size, nhead=dec_num_heads, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=dec_num_layers
        )
        dec_layers = [
            nn.Linear(hidden_size, dec_hidden_size),
            nn.ReLU(),
            transformer_encoder,
        ]
        self.decoder = nn.Sequential(*dec_layers)
        self.fc_out = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(dec_hidden_size, output_size)
        )

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53

        if self.post:
            fc_output = self.fc(x.view(-1, fs))
            fc_output = fc_output.view(bs, ln, -1)
            fc_output = positional_encoding(fc_output, scaled=self.scaled)
        else:
            x = positional_encoding(x, scaled=self.scaled)
            fc_output = self.fc(x.view(-1, fs))
            fc_output = fc_output.view(bs, ln, -1)

        tf_output = self.decoder(fc_output)
        tf_output = tf_output[:, 0, :]
        logits = self.fc_out(tf_output)

        return logits


class MetaMLP(nn.Module):
    def __init__(self, model_config):
        super(MetaMLP, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        num_layers = int(model_config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )

        self.fc = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53
        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, ln, -1)
        fc_output = fc_output.mean(1)
        logits = self.decoder(fc_output)
        return logits


class WindowMLP(nn.Module):
    def __init__(self, model_config):
        super(WindowMLP, self).__init__()

        # unpack config
        self.mode = model_config["mode"]

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        num_layers = int(model_config["num_layers"])

        self.decoder_type = model_config["decoder"]["type"]
        if self.decoder_type == "lstm":
            self.dec_hidden_size = model_config["decoder"]["hidden_size"]
            dec_num_layers = model_config["decoder"]["num_layers"]
            self.bidirectional = model_config["decoder"]["bidirectional"]
        elif self.decoder_type == "tf":
            dec_head_hidden_size = model_config["decoder"]["head_hidden_size"]
            dec_num_heads = model_config["decoder"]["num_heads"]
            dec_num_layers = model_config["decoder"]["num_layers"]
        else:
            raise NotImplementedError()

        # initialize MLP
        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )

        self.fc = nn.Sequential(*layers)

        # initialize decoder
        if self.decoder_type == "lstm":
            self.decoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=self.dec_hidden_size,
                num_layers=dec_num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
            self.fc_out = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(
                    2 * self.dec_hidden_size
                    if self.bidirectional
                    else self.dec_hidden_size,
                    output_size,
                ),
            )
        elif self.decoder_type == "tf":
            dec_hidden_size = dec_head_hidden_size * dec_num_heads

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dec_hidden_size, nhead=dec_num_heads, batch_first=True
            )
            transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=dec_num_layers
            )
            dec_layers = [
                nn.Linear(hidden_size, dec_hidden_size),
                nn.ReLU(),
                transformer_encoder,
            ]
            self.decoder = nn.Sequential(*dec_layers)
            self.fc_out = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(dec_hidden_size, output_size)
            )

        # set raw decoder for pretrained MLP
        if self.mode in ["FPT", "UFPT"]:
            self.raw_decoder = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        else:
            if self.mode != "NPT":
                raise NotImplementedError()

    def forward(self, x):
        bs, wn, ln, fs = x.shape
        # bs:  batch size
        # ws:  number of windows
        # ln:  window time length
        # fs:  number of channels, 53

        # in case of not-pretrained MLP the final model is trained from scratch
        if self.mode == "NPT":
            fc_output = self.fc(x.view(-1, fs))
            fc_output = fc_output.view(bs, wn, ln, -1)
            fc_output = fc_output.mean(2)

            if self.decoder_type == "lstm":
                lstm_output, _ = self.decoder(fc_output)

                if self.bidirectional:
                    out_forward = lstm_output[:, -1, : self.dec_hidden_size]
                    out_reverse = lstm_output[:, 0, self.dec_hidden_size :]
                    lstm_output = torch.cat((out_forward, out_reverse), 1)
                else:
                    lstm_output = lstm_output[:, -1, :]

                logits = self.fc_out(lstm_output)

            elif self.decoder_type == "tf":
                tf_output = self.decoder(fc_output)
                tf_output = tf_output[:, 0, :]
                logits = self.fc_out(tf_output)

            return logits

        # in case of pretrained MLP, in phase 1 MLP is pretrained,
        # in phase 2 the decoder (in case of FPT) or MLP+decoder (in case of UFPT) is trained
        # NOTE: trained params are set externally in the optimizer
        if self.mode in ["FPT", "UFPT"]:
            raise NotImplementedError()


class AttentionMLP(nn.Module):
    def __init__(self, model_config):
        super(AttentionMLP, self).__init__()

        input_size = int(model_config["input_size"])
        time_length = int(model_config["time_length"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        num_layers = int(model_config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

        self.attn = nn.Sequential(
            nn.Linear(time_length + 1, 3 * hidden_size),
            nn.ReLU(),
            nn.Linear(3 * hidden_size, time_length),
        )

    def get_attention(self, outputs):
        # calculate mean over time
        outputs_mean = outputs.mean(1).unsqueeze(1)

        # add output's mean to the end of each output time dimension
        # as a reference for attention layer
        # and pass it to attention layer
        weights_list = []
        for output, output_mean in zip(outputs, outputs_mean):
            result = torch.cat((output, output_mean)).swapaxes(0, 1)
            result_tensor = self.attn(result)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)
        # normalize weights and swap axes to the output shape
        normalized_weights = F.softmax(weights, dim=2).swapaxes(1, 2)
        return normalized_weights

    def forward(self, x):
        bs, ln, fs = x.shape
        fc_output = self.fc(x.reshape(-1, fs))
        fc_output = fc_output.reshape(bs, ln, -1)

        # get weights form attention layer
        normalized_weights = self.get_attention(fc_output)

        # sum outputs weight-wise
        logits = torch.einsum("ijk,ijk->ik", fc_output, normalized_weights)

        return logits


class NewAttentionMLP(nn.Module):
    def __init__(self, model_config):
        super(NewAttentionMLP, self).__init__()

        input_size = int(model_config["input_size"])
        time_length = int(model_config["time_length"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        attention_size = int(model_config["attention_size"])
        num_layers = int(model_config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

        self.attn = nn.Sequential(
            nn.LayerNorm(time_length + 1),
            nn.Dropout(p=dropout),
            nn.Linear(time_length + 1, attention_size),
            nn.ReLU(),
            nn.LayerNorm(attention_size),
            nn.Dropout(p=dropout),
            nn.Linear(attention_size, time_length),
        )

    def get_attention(self, outputs):
        # calculate mean over time
        outputs_mean = outputs.mean(1).unsqueeze(1)

        # add output's mean to the end of each output time dimension
        # as a reference for attention layer
        # and pass it to attention layer
        weights_list = []
        for output, output_mean in zip(outputs, outputs_mean):
            result = torch.cat((output, output_mean)).swapaxes(0, 1)
            result_tensor = self.attn(result)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)
        # normalize weights and swap axes to the output shape
        normalized_weights = F.softmax(weights, dim=2).swapaxes(1, 2)
        return normalized_weights

    def forward(self, x):
        bs, ln, fs = x.shape
        fc_output = self.fc(x.reshape(-1, fs))
        fc_output = fc_output.reshape(bs, ln, -1)

        # get weights form attention layer
        normalized_weights = self.get_attention(fc_output)

        # sum outputs weight-wise
        logits = torch.einsum("ijk,ijk->ik", fc_output, normalized_weights)

        return logits


class LSTM(nn.Module):
    def __init__(self, model_config):
        super(LSTM, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        fc_dropout = float(model_config["fc_dropout"])
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
            nn.Dropout(p=fc_dropout),
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_size),
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


class NoahLSTM(nn.Module):
    def __init__(self, model_config):
        super(NoahLSTM, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        fc_dropout = float(model_config["fc_dropout"])
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
            nn.Dropout(p=fc_dropout),
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_size),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        lstm_output = torch.mean(lstm_output, dim=1)
        logits = self.fc(lstm_output)

        return logits


class Transformer(nn.Module):
    def __init__(self, model_config):
        super(Transformer, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        fc_dropout = float(model_config["fc_dropout"])
        head_hidden_size = int(model_config["head_hidden_size"])
        num_layers = int(model_config["num_layers"])
        num_heads = int(model_config["num_heads"])

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            transformer_encoder,
        ]
        self.transformer = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x.shape = [Batch_size; time_len; n_channels]
        fc_output = self.transformer(x)
        fc_output = fc_output[:, 0, :]
        fc_output = self.fc(fc_output)
        return fc_output


class MeanTransformer(nn.Module):
    def __init__(self, model_config):
        super(MeanTransformer, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        fc_dropout = float(model_config["fc_dropout"])
        head_hidden_size = int(model_config["head_hidden_size"])
        num_layers = int(model_config["num_layers"])
        num_heads = int(model_config["num_heads"])

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            transformer_encoder,
        ]
        self.transformer = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x.shape = [Batch_size; time_len; n_channels]
        fc_output = self.transformer(x)
        fc_output = torch.mean(fc_output, 1)
        fc_output = self.fc(fc_output)
        return fc_output


class PE_Transformer(nn.Module):
    def __init__(self, model_config):
        super(PE_Transformer, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        fc_dropout = float(model_config["fc_dropout"])
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

        self.input_embed = nn.Linear(input_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x.shape = [Batch_size; time_len; n_channels]

        if self.post:
            input_embed = self.input_embed(x)
            input_embed = positional_encoding(input_embed, scaled=self.scaled)
        else:
            x = positional_encoding(x, scaled=self.scaled)
            input_embed = self.input_embed(x)

        tf_output = self.transformer(x)
        tf_output = tf_output[:, 0, :]

        fc_output = self.fc(tf_output)
        return fc_output


class STDIM(nn.Module):
    def __init__(self, encoder, probe):
        super().__init__()
        self.encoder = encoder
        self.probe = probe

    def forward(self, x):
        x = self.encoder(x)
        return self.probe(x)


class Probe(nn.Module):
    def __init__(self, probe_config):
        super().__init__()
        self.model = nn.Linear(
            in_features=int(probe_config["input_size"]),
            out_features=int(probe_config["output_size"]),
        )

    def forward(self, x):
        return self.model(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureOneCNN(nn.Module):
    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def __init__(self, encoder_config):
        super().__init__()

        self.feature_size = int(encoder_config["feature_size"])
        self.input_size = int(encoder_config["input_channels"])
        self.conv_output_size = int(encoder_config["conv_output_size"])

        init_ = lambda m: self.init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.flatten = Flatten()

        final_conv_size = 200 * self.conv_output_size
        # final_conv_shape = (200, self.conv_output_size)
        self.main = nn.Sequential(
            init_(nn.Conv1d(self.input_size, 64, 4, stride=1)),
            nn.ReLU(),
            init_(nn.Conv1d(64, 128, 4, stride=1)),
            nn.ReLU(),
            init_(nn.Conv1d(128, 200, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(final_conv_size, self.feature_size)),
        )

    @property
    def local_layer_depth(self):
        return self.main[4].out_channels

    @staticmethod
    def get_conv_output_size(conv_input_size):
        # https://www.baeldung.com/cs/convolutional-layer-size
        conv_output_size = conv_input_size - 4 + 1  # 1st layer
        conv_output_size = conv_output_size - 4 + 1  # 2nd layer
        conv_output_size = conv_output_size - 3 + 1  # 3rd layer

        return conv_output_size

    def forward(self, inputs, fmaps=False, five=False):
        f5 = self.main[:6](inputs)
        out = self.main[6:](f5)

        if five:
            return f5.permute(0, 2, 1)
        if fmaps:
            return {
                "f5": f5.permute(0, 2, 1),
                "out": out,
            }
        return out
