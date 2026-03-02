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
    if conf.model in ["lstm", "mean_lstm", "transformer", "mean_transformer", "dice"]:
        return Introspector(vars(conf), model_config, dataloaders, model)

    raise ValueError(f"{conf.model} is not recognized")


class Introspector:
    """Basic introspector"""

    def __init__(self, conf, model_conf, dataloaders, model) -> None:
        self.config = conf
        self.methods = conf["methods"]
        self.save_path = conf["run_dir"]
        self.model_config = model_conf
        self.model = model

        self.has_mask = dataloaders["mask"] is not None
        self.dataloaders = dataloaders

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
        print("\tPlotting generalized saliency maps")
        targets = torch.unique(self.dataloaders["labels"])
        for target in targets:
            filter_array = self.dataloaders["labels"] == target
            features = self.dataloaders["features"][filter_array]
            features.requires_grad = True

            if self.has_mask:
                mask = self.dataloaders["mask"][filter_array]
                mask = torch.mean(mask, axis=0).unsqueeze(0)

            for method in self.methods:
                # get grads
                grads = self.get_grads(method, features, target)

                # plot colormaps
                fig, axs = plt.subplots(1, 1, figsize=(13, 5))
                # data needs to be transposed to [num_features; time_len; 1]
                _ = viz.visualize_image_attr(
                    np.transpose(grads.cpu().detach().numpy(), (2, 1, 0)),
                    np.transpose(features.cpu().detach().numpy(), (2, 1, 0)),
                    method="heat_map",
                    cmap="inferno",
                    show_colorbar=False,
                    plt_fig_axis=(fig, axs),
                    use_pyplot=False,
                )

                if self.has_mask:
                    axs.imshow(
                        np.transpose(mask.cpu().detach().numpy(), (2, 1, 0)),
                        alpha=0.5,
                        cmap="gist_gray",
                    )
                plt.savefig(
                    f"{self.save_path}/{method}/colormap/general_{target}.eps",
                    format="eps",
                    # dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{self.save_path}/{method}/colormap/general_{target}.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # bar charts: summarizes gradients at each time point
                plt.bar(
                    range(features.shape[1]),
                    np.sum(grads.cpu().detach().numpy(), axis=(0, 2)),
                    align="center",
                    color="blue" if target == 0 else "red",
                )
                plt.xlim([0, features.shape[1]])
                plt.grid(False)
                plt.axis("off")
                plt.savefig(
                    f"{self.save_path}/{method}/barchart/general_{target}.eps",
                    format="eps",
                    # dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{self.save_path}/{method}/barchart/general_{target}.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        print("\tPlotting single saliency maps")
        counter = np.zeros_like(np.unique(self.dataloaders["labels"]))
        for i in range(self.dataloaders["labels"].shape[0]):
            target = self.dataloaders["labels"][i]
            if counter[target] >= cutoff:
                continue
            counter[target] += 1

            feature = self.dataloaders["features"][i].unsqueeze(0)
            feature.requires_grad = True
            if self.has_mask:
                mask = self.dataloaders["mask"][i].unsqueeze(0)

            for method in self.methods:
                # get grads
                grads = self.get_grads(method, feature, target)

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

                if self.has_mask:
                    axs.imshow(
                        np.transpose(mask.cpu().detach().numpy(), (2, 1, 0)),
                        alpha=0.5,
                        cmap="gist_gray",
                    )
                plt.savefig(
                    f"{self.save_path}/{method}/colormap/{i:04d}_target_{target}.eps",
                    format="eps",
                    # dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{self.save_path}/{method}/colormap/{i:04d}_target_{target}.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # bar charts: summarizes gradients at each time point

                plt.bar(
                    range(feature.shape[1]),
                    np.sum(grads.cpu().detach().numpy(), axis=(0, 2)),
                    align="center",
                    color="blue" if target == 0 else "red",
                )
                plt.xlim([0, feature.shape[1]])
                plt.grid(False)
                plt.axis("off")
                plt.savefig(
                    f"{self.save_path}/{method}/barchart/{i:04d}_target_{target}.eps",
                    format="eps",
                    # dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{self.save_path}/{method}/barchart/{i:04d}_target_{target}.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    def get_grads(self, method, features, target):
        """Returns gradients according to method"""
        if method == "saliency":
            saliency = Saliency(self.model)
            self.model.zero_grad()
            grads = saliency.attribute(features, target=target)
        elif method == "ig":
            ig = IntegratedGradients(self.model)
            self.model.zero_grad()
            grads, _ = ig.attribute(
                inputs=features,
                target=target,
                baselines=torch.zeros_like(features),
                return_convergence_delta=True,
            )
        elif method == "ignt":
            ig = IntegratedGradients(self.model)
            nt = NoiseTunnel(ig)
            self.model.zero_grad()
            grads, _ = nt.attribute(
                inputs=features,
                target=target,
                baselines=torch.zeros_like(features),
                return_convergence_delta=True,
                nt_type="smoothgrad_sq",
                nt_samples=5,
                stdevs=0.2,
            )
        else:
            raise ValueError(f"'{method}' methods is not recognized")

        return grads
