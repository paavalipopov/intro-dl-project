import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset


def dataloader_factory(conf, data, outer_k, trial, inner_k=None):
    """Return dataloaders"""
    if conf.model in ["lstm", "mean_lstm", "transformer", "mean_transformer"]:
        dataloaders = common_dataloader(conf, data, outer_k, trial, inner_k)
    elif conf.model == "dice":
        dataloaders = dice_dataloader(conf, data, outer_k, trial, inner_k)
    else:
        raise NotImplementedError(
            f"'{conf.model}' model is not familiar to dataloader_factory"
        )

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

    if conf.mode == "tune" and not conf.glob:
        # tune mode should be completely unaware of the test set
        # unless we are looking for globally optimal hyperparams
        inner_skf = StratifiedKFold(
            n_splits=conf.n_splits, shuffle=True, random_state=42
        )
        inner_CV_folds = list(inner_skf.split(X_train, y_train))

        train_index, test_index = inner_CV_folds[inner_k]
        X_train, X_test = X_train[train_index], X_train[test_index]
        y_train, y_test = y_train[train_index], y_train[test_index]

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=X_train.shape[0] // conf.n_splits,
        random_state=42 + trial if conf.mode == "exp" else 42,
        stratify=y_train,
    )

    # create dataloaders
    dataloaders = {}
    dataloaders["train"] = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.int64),
        ),
        batch_size=conf.batch_size,
        num_workers=0,
        shuffle=True,
    )
    dataloaders["valid"] = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.int64),
        ),
        batch_size=conf.batch_size,
        num_workers=0,
        shuffle=False,
    )
    dataloaders["test"] = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.int64),
        ),
        batch_size=conf.batch_size,
        num_workers=0,
        shuffle=False,
    )

    # dataloaders for extra test datasets
    #### not applicable for this project
    for ds in data:
        if ds != "main" and conf.mode != "tune":
            dataloaders[ds] = DataLoader(
                TensorDataset(
                    torch.tensor(data[ds]["features"], dtype=torch.float32),
                    torch.tensor(data[ds]["labels"], dtype=torch.int64),
                ),
                batch_size=conf.batch_size,
                num_workers=0,
                shuffle=False,
            )
    ####

    return dataloaders


def dice_dataloader(conf, model_config, data, outer_k, trial, inner_k=None):
    raise NotImplementedError()
