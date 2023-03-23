# pylint: disable=invalid-name, too-many-function-args
"""Functions for extracting dataset features and labels"""
import h5py
import numpy as np
import pandas as pd

from scipy import stats

from src.settings import DATA_ROOT


def data_factory(conf):
    """
    Return dict of {"main" dataset and additional test datasets as dicts of {"features", "labels"}}
    Features have shape [subjects, time, components]
    """
    assert conf.ds is not None
    assert conf.multiclass is not None
    assert conf.filter_indices is not None

    # load main dataset
    data = {}
    data["main"] = load_dataset(conf, conf.ds)

    data_shape = {}
    data_shape["main"] = data["main"]["features"].shape
    n_classes = np.unique(data["main"]["labels"]).shape[0]

    # z-score over time
    if conf.zscore:
        for ds, datum in data.items():
            datum["features"] = stats.zscore(datum["features"], axis=2)

            # filter out invalid data
            good_indices = []
            for i, ts in enumerate(datum["features"]):
                if np.sum(np.isnan(ts)) == 0:
                    good_indices += [i]

            datum["features"] = datum["features"][good_indices]
            datum["labels"] = datum["labels"][good_indices]

            data[ds] = datum
            data_shape[ds] = datum["features"].shape

    data_info = {
        "data_shape": data_shape,
        "n_classes": n_classes,
    }
    return data, data_info


def data_postfactory(conf, model_config, original_data):
    """
    Post-process the raw dataset data according to model_config (like sliding window)
    """
    if conf.model in ["lstm", "mean_lstm", "transformer", "mean_transformer"]:
        return original_data, conf.data_info
    if conf.model == "dice":
        # TODO: implement
        raise NotImplementedError("DICE model data postprocessing is not implemented")

    raise ValueError(f"'{conf.model}' model is not recognized")


def load_dataset(conf, dataset):
    """
    Return the dataset defined by 'dataset'.
    Shape: [subjects, time, components]
    """
    # transparency mask: used for introspection
    mask = None

    if dataset == "abide":
        data, labels = load_ABIDE1(filter_indices=conf.filter_indices)
    elif dataset == "cobre":
        data, labels = load_COBRE(filter_indices=conf.filter_indices)
    elif dataset == "synth1":
        data = np.load(f"{DATA_ROOT}/synth1/data.npz")
        labels = data["labels"]
        mask = data["masks"]
        data = data["data"]
    elif dataset == "synth2":
        data = np.load(f"{DATA_ROOT}/synth2/data.npz")
        labels = data["labels"]
        # mask = data["masks"]
        data = data["data"]

        ###
        mask = np.zeros_like(data)
        for i in range(mask.shape[0] // 2, mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    if k == 2 * j:
                        mask[i, j, k] = 1
    else:
        raise NotImplementedError(f"'{dataset}' dataset is not found")

    data = np.swapaxes(data, 1, 2)  # new: [subjects, time, components]
    if mask is not None:
        mask = np.swapaxes(mask, 1, 2)

    return {"features": data, "labels": labels, "mask": mask}


def load_ABIDE1(
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
    filter_indices: bool = True,
):
    """
    Return ABIDE1 data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv")
    - path to labels
    filter_indices: bool = True
    - whether ICA components should be filtered

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("ABIDE1_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (569, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 569 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    if filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        # filter the data: leave only correct components
        data = data[:, idx, :]
        # print(data.shape)
        # 53 - components - data.shape[1]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_COBRE(
    dataset_path: str = DATA_ROOT.joinpath("cobre/COBRE_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("cobre/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("cobre/labels_COBRE.csv"),
    filter_indices: bool = True,
):
    """
    Return COBRE data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("cobre/COBRE_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("cobre/correct_indices_GSP.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("cobre/labels_COBRE.csv")
    - path to labels
    filter_indices: bool = True
    - whether ICA components should be filtered

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("COBRE_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (157, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 157 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    if filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        # filter the data: leave only correct components
        data = data[:, idx, :]
        # print(data.shape)
        # 53 - components - data.shape[1]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels
