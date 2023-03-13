import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset


def dataloader_factory(conf, model_config, data, outer_k, trial, inner_k=None):
    "Return dataloaders"
    if conf.model == "stdim":
        dataloaders = stdim_dataloader(
            conf, model_config, data, outer_k, trial, inner_k
        )
    else:
        dataloaders = common_dataloader(conf, data, outer_k, trial, inner_k)

    return dataloaders


def common_dataloader(conf, data, outer_k, trial, inner_k):
    skf = StratifiedKFold(n_splits=conf.n_splits, shuffle=True, random_state=42)
    CV_folds = list(skf.split(data["main"]["features"], data["main"]["labels"]))

    # train/test split
    train_index, test_index = CV_folds[outer_k]
    X_train, X_test = (
        data["main"]["features"][train_index],
        data["main"]["features"][test_index],
    )
    y_train, y_test = (
        data["main"]["labels"][train_index],
        data["main"]["labels"][test_index],
    )

    if conf.mode in ["tune", "experiment"] or (
        conf.mode == "nested_exp" and inner_k is None
    ):
        # tune, experiment modes, and nested_exp mode in the experiment phase
        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=conf.data_shape[0] // conf.n_splits,
            random_state=42 if conf.mode == "tune" else 42 + trial,
            stratify=y_train,
        )
    else:
        # nested_exp mode in the tune phase
        # inner k-fold split of train data, train/test split
        inner_skf = StratifiedKFold(
            n_splits=conf.n_splits, shuffle=True, random_state=42
        )
        inner_CV_folds = list(inner_skf.split(X_train, y_train))

        n_observations = X_train.shape[0]

        train_index, test_index = inner_CV_folds[inner_k]

        X_train, X_test = X_train[train_index], X_train[test_index]
        y_train, y_test = y_train[train_index], y_train[test_index]

        # inner train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=n_observations // conf.n_splits,
            random_state=42,
            stratify=y_train,
        )

    # create torch tensors
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64),
    )
    valid_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.int64),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.int64),
    )

    # create dataloaders
    dataloaders = {}
    dataloaders["train"] = DataLoader(
        train_ds,
        batch_size=conf.batch_size,
        num_workers=0,
        shuffle=True,
    )
    dataloaders["valid"] = DataLoader(
        valid_ds,
        batch_size=conf.batch_size,
        num_workers=0,
        shuffle=False,
    )
    dataloaders["test"] = DataLoader(
        test_ds,
        batch_size=conf.batch_size,
        num_workers=0,
        shuffle=False,
    )

    if conf.mode == "experiment" or (conf.mode == "nested_exp" and inner_k is None):
        # dataloaders for extra test datasets
        for ds in data:
            if ds != "main":
                dataloaders[ds] = DataLoader(
                    TensorDataset(
                        torch.tensor(data[ds]["features"], dtype=torch.float32),
                        torch.tensor(data[ds]["labels"], dtype=torch.int64),
                    ),
                    batch_size=conf.batch_size,
                    num_workers=0,
                    shuffle=False,
                )

    return dataloaders


def stdim_dataloader(conf, model_config, data, outer_k, trial, inner_k=None):
    pass
