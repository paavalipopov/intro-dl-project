# pylint: disable=invalid-name, too-few-public-methods, no-member, unpacking-non-sequence
"""Introspection scripts"""
import os
import json

from captum.attr import IntegratedGradients, NoiseTunnel, Saliency, visualization as viz
import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
import torch

from src.utils import NpEncoder


# sns.set_theme(style="whitegrid", font_scale=2, rc={"figure.figsize": (18, 9)})


def introspector_factory(conf, model_config, dataloaders, model):
    """Introspector factory"""
    if conf.model in ["lstm", "mean_lstm", "transformer", "mean_transformer"]:
        return Introspector(vars(conf), model_config, dataloaders, model)
    if conf.model == "dice":
        raise NotImplementedError()

    raise ValueError(f"{conf.model} is not recognized")


class Introspector:
    """Basic introspector"""

    def __init__(self, conf, model_conf, dataloaders, model) -> None:

        self.config = conf
        self.methods = conf["methods"]
        self.save_path = conf["run_dir"]
        self.model_config = model_conf
        self.dataloaders = dataloaders
        self.model = model

        if "saliency" in self.methods:
            os.makedirs(f"{self.save_path}saliency/colormap", exist_ok=True)
            os.makedirs(f"{self.save_path}saliency/barchart", exist_ok=True)
        if "ig" in self.methods:
            os.makedirs(f"{self.save_path}ig/colormap", exist_ok=True)
            os.makedirs(f"{self.save_path}ig/barchart", exist_ok=True)
        if "ignt" in self.methods:
            os.makedirs(f"{self.save_path}ignt/colormap", exist_ok=True)
            os.makedirs(f"{self.save_path}ignt/barchart", exist_ok=True)

        # save model configs in the run's directory
        with open(f"{self.save_path}model_config.json", "w", encoding="utf8") as fp:
            json.dump(self.model_config, fp, indent=2, cls=NpEncoder)

    def run(self, cutoff):
        """Run introspection, save results"""
        for i, (feature, target) in enumerate(
            zip(self.dataloaders["features"], self.dataloaders["labels"])
        ):
            if i >= cutoff:
                break

            feature = feature.unsqueeze(0)
            feature.requires_grad = True

            for method in self.methods:
                # get data
                if method == "saliency":
                    saliency = Saliency(self.model)
                    self.model.zero_grad()
                    grads = saliency.attribute(feature, target=target)
                elif method == "ig":
                    ig = IntegratedGradients(self.model)
                    self.model.zero_grad()
                    grads, _ = ig.attribute(
                        inputs=feature,
                        target=target,
                        baselines=torch.zeros_like(feature),
                        return_convergence_delta=True,
                    )
                elif method == "ignt":
                    ig = IntegratedGradients(self.model)
                    nt = NoiseTunnel(ig)
                    self.model.zero_grad()
                    grads, _ = nt.attribute(
                        inputs=feature,
                        target=target,
                        baselines=torch.zeros_like(feature),
                        return_convergence_delta=True,
                        nt_type="smoothgrad_sq",
                        nt_samples=5,
                        stdevs=0.2,
                    )
                else:
                    raise ValueError(f"'{method}' methods is not recognized")

                # plot colormaps
                fig, axs = plt.subplots(1, 1, figsize=(13, 5))
                # data needs to be transposed to [num_features; time_len; 1]
                _ = viz.visualize_image_attr(
                    np.transpose(grads.cpu().detach().numpy(), (2, 1, 0)),
                    np.transpose(feature.cpu().detach().numpy(), (2, 1, 0)),
                    method="heat_map",
                    cmap="inferno",
                    show_colorbar=False,
                    plt_fig_axis=(fig, axs),
                    use_pyplot=False,
                )

                plt.savefig(
                    f"{self.save_path}/{method}/colormap/{i:04d}_target_{target}.png",
                    format="png",
                    dpi=300,
                )
                plt.close()

                # bar charts: summarizes gradients at each time point

                plt.bar(
                    range(feature.shape[1]),
                    np.sum(grads.cpu().detach().numpy(), axis=(0, 2)),
                    align="center",
                    color="blue",
                )
                plt.xlim([0, feature.shape[1]])
                plt.grid(False)
                plt.axis("off")
                plt.savefig(
                    f"{self.save_path}/{method}/barchart/{i:04d}_target_{target}.png",
                    format="png",
                    dpi=300,
                )
                plt.close()
