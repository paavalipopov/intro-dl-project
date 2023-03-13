from torch import optim


def optimizer_factory(conf, model, model_config):
    if conf.model == "stdim":
        optimizer = optim.Adam(
            model.probe.parameters(),
            lr=float(model_config["lr"]),
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(model_config["lr"]),
        )

    return optimizer
