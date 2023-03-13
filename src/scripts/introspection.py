import os
import csv
import argparse
import json

from captum.attr import IntegratedGradients, NoiseTunnel, Saliency, visualization as viz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.settings import ASSETS_ROOT, LOGS_ROOT, UTCNOW
from src.ts_data import (
    load_ABIDE1,
    load_COBRE,
    load_FBIRN,
    load_OASIS,
    load_ABIDE1_869,
    load_UKB,
    load_BSNIP,
    TSQuantileTransformer,
)
from src.ts_model import (
    LSTM,
    MLP,
    Transformer,
    AttentionMLP,
    NewAttentionMLP,
)
from src.ts_model_tests import (
    AnotherLSTM,
    NewestAttentionMLP,
    EnsembleLogisticRegression,
    AnotherEnsembleLogisticRegression,
    MySVM,
    EnsembleSVM,
    No_Res_MLP,
    No_Ens_MLP,
    Transposed_MLP,
    UltimateAttentionMLP,
)

sns.set_theme(style="whitegrid", font_scale=2, rc={"figure.figsize": (18, 9)})


class Introspection:
    def __init__(self, raw_path: str, methods: list, num_subjects: int) -> None:
        self.raw_path = raw_path
        self.introspection_methods = methods
        self.num_subjects = num_subjects

        _, _, self._model, self._dataset = os.path.split(self.raw_path)[1].split("-")

        self.initialize_dataset()
        self.initialize_model()

        self.image_path = ASSETS_ROOT.joinpath(
            f"images/{UTCNOW}-{self._model}-{self._dataset}"
        )

    def initialize_dataset(self) -> None:
        if self._dataset == "oasis":
            features, _ = load_OASIS()
        elif self._dataset == "abide":
            features, _ = load_ABIDE1()
        elif self._dataset == "fbirn":
            features, _ = load_FBIRN()
        elif self._dataset == "cobre":
            features, _ = load_COBRE()
        elif self._dataset == "abide_869":
            features, _ = load_ABIDE1_869()
        elif self._dataset == "ukb":
            features, _ = load_UKB()
        elif self._dataset == "bsnip":
            features, _ = load_BSNIP()

        self.data_shape = features.shape
        self.features = np.swapaxes(features, 1, 2)  # [n_samples; seq_len; n_features]

    def initialize_model(self) -> None:
        config_file = os.path.join(self.raw_path, "runs.csv")

        df = pd.read_csv(config_file, delimiter=",")
        config = df.loc[df["test_score"].idxmax()].to_dict()
        config.pop("test_score")
        config.pop("test_accuracy")
        config.pop("test_loss")

        if self._model in [
            "mlp",
            "wide_mlp",
            "deep_mlp",
            "attention_mlp",
            "new_attention_mlp",
            "newest_attention_mlp",
            "nores_mlp",
            "noens_mlp",
            "trans_mlp",
            "ultimate_attention_mlp",
        ]:
            if self._model in ["mlp", "wide_mlp", "deep_mlp"]:
                model = MLP(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "attention_mlp":
                model = AttentionMLP(
                    input_size=self.data_shape[1],  # PRIOR
                    time_length=self.data_shape[2],
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "ultimate_attention_mlp":
                model = UltimateAttentionMLP(
                    input_size=self.data_shape[1],  # PRIOR
                    time_length=self.data_shape[2],
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    attention_size=int(config["attention_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "new_attention_mlp":
                model = NewAttentionMLP(
                    input_size=self.data_shape[1],  # PRIOR
                    time_length=self.data_shape[2],
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    attention_size=int(config["attention_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "nores_mlp":
                model = No_Res_MLP(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "noens_mlp":
                model = No_Ens_MLP(
                    input_size=self.data_shape[1] * self.data_shape[2],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "trans_mlp":
                model = Transposed_MLP(
                    input_size=self.data_shape[2],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )

        elif self._model in ["lstm", "another_lstm"]:
            if self._model == "lstm":
                model = LSTM(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    batch_first=True,
                    bidirectional=config["bidirectional"],
                    fc_dropout=config["fc_dropout"],
                )
            elif self._model == "another_lstm":
                model = AnotherLSTM(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    batch_first=True,
                    bidirectional=False,
                    fc_dropout=config["fc_dropout"],
                )

        elif self._model == "transformer":
            model = Transformer(
                input_size=self.data_shape[1],  # PRIOR
                output_size=2,  # PRIOR
                hidden_size=int(config["hidden_size"]) * int(config["num_heads"]),
                num_layers=int(config["num_layers"]),
                num_heads=int(config["num_heads"]),
                fc_dropout=config["fc_dropout"],
            )

        else:
            raise NotImplementedError()

        model_path = os.path.join(self.raw_path, "k_0/0000/model.best.pth")
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)
        self.model = model.eval()

    def introspection(self, i, feature):
        feature = feature.astype(np.float32)
        feature = torch.tensor(feature).unsqueeze(0)
        feature.requires_grad = True

        cutoff = (feature.shape[1] * feature.shape[2]) // 20  # 5%
        time_range = feature.shape[1]

        for method in self.introspection_methods:
            if method == "saliency":
                saliency = Saliency(self.model)
                self.model.zero_grad()
                grads0 = saliency.attribute(feature, target=0)
                self.model.zero_grad()
                grads1 = saliency.attribute(feature, target=1)
            elif method == "ig":
                ig = IntegratedGradients(self.model)
                self.model.zero_grad()
                grads0, _ = ig.attribute(
                    inputs=feature,
                    target=0,
                    baselines=torch.zeros_like(feature),
                    return_convergence_delta=True,
                )
                self.model.zero_grad()
                grads1, _ = ig.attribute(
                    inputs=feature,
                    target=1,
                    baselines=torch.zeros_like(feature),
                    return_convergence_delta=True,
                )
            elif method == "ignt":
                ig = IntegratedGradients(self.model)
                nt = NoiseTunnel(ig)
                self.model.zero_grad()
                grads0, _ = nt.attribute(
                    inputs=feature,
                    target=0,
                    baselines=torch.zeros_like(feature),
                    return_convergence_delta=True,
                    nt_type="smoothgrad_sq",
                    nt_samples=5,
                    stdevs=0.2,
                )
                self.model.zero_grad()
                grads1, _ = nt.attribute(
                    inputs=feature,
                    target=1,
                    baselines=torch.zeros_like(feature),
                    return_convergence_delta=True,
                    nt_type="smoothgrad_sq",
                    nt_samples=5,
                    stdevs=0.2,
                )
            else:
                print(f"Method '{method}' is undefined")
                return

            fig, axs = plt.subplots(1, 1, figsize=(21, 9))
            # transpose to [num_features; time_len; 1]
            _ = viz.visualize_image_attr(
                np.transpose(grads0.cpu().detach().numpy(), (2, 1, 0)),
                np.transpose(feature.cpu().detach().numpy(), (2, 1, 0)),
                method="heat_map",
                cmap="inferno",
                show_colorbar=False,
                plt_fig_axis=(fig, axs),
                use_pyplot=False,
            )
            plt.savefig(
                self.image_path.joinpath(f"{method}/colormap/{i:04d}.0.png"),
                format="png",
                dpi=300,
            )
            plt.close()

            fig, axs = plt.subplots(1, 1, figsize=(21, 9))
            _ = viz.visualize_image_attr(
                np.transpose(grads1.cpu().detach().numpy(), (2, 1, 0)),
                np.transpose(feature.cpu().detach().numpy(), (2, 1, 0)),
                method="heat_map",
                cmap="inferno",
                show_colorbar=False,
                plt_fig_axis=(fig, axs),
                use_pyplot=False,
            )
            plt.savefig(
                self.image_path.joinpath(f"{method}/colormap/{i:04d}.1.png"),
                format="png",
                dpi=300,
            )
            plt.close()

            # bar charts
            threshold0 = np.sort(grads0.detach().numpy().ravel())[
                -cutoff
            ]  # get the nth largest value
            idx = grads0 < threshold0
            grads0[idx] = 0

            threshold1 = np.sort(grads1.detach().numpy().ravel())[
                -cutoff
            ]  # get the nth largest value
            idx = grads1 < threshold1
            grads1[idx] = 0

            plt.bar(
                range(time_range),
                np.sum(grads0.cpu().detach().numpy(), axis=(0, 2)),
                align="center",
                color="blue",
            )
            plt.xlim([0, time_range])
            plt.grid(False)
            plt.axis("off")
            plt.savefig(
                self.image_path.joinpath(f"{method}/barchart/{i:04d}.0.png"),
                format="png",
                dpi=300,
            )
            plt.close()

            plt.bar(
                range(time_range),
                np.sum(grads1.cpu().detach().numpy(), axis=(0, 2)),
                align="center",
                color="red",
            )
            plt.xlim([0, time_range])
            plt.grid(False)
            plt.axis("off")
            plt.savefig(
                self.image_path.joinpath(f"{method}/barchart/{i:04d}.1.png"),
                format="png",
                dpi=300,
            )
            plt.close()

    def run_introspection(self):
        if "saliency" in self.introspection_methods:
            os.makedirs(self.image_path.joinpath("saliency/colormap"))
            os.makedirs(self.image_path.joinpath("saliency/barchart"))
        if "ig" in self.introspection_methods:
            os.makedirs(self.image_path.joinpath("ig/colormap"))
            os.makedirs(self.image_path.joinpath("ig/barchart"))
        if "ignt" in self.introspection_methods:
            os.makedirs(self.image_path.joinpath("ignt/colormap"))
            os.makedirs(self.image_path.joinpath("ignt/barchart"))

        for i, feature in zip(range(self.num_subjects), self.features):
            self.introspection(i, feature)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["saliency", "ig", "ignt"],
        required=True,
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    introspection = Introspection(args.path, args.methods, args.num_subjects)
    introspection.run_introspection()
