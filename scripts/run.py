import sys
import time

from src.utils import get_argparser, get_resumed_params
from src.data import data_factory
from src.naming import name_factory
from src.model_config import model_config_factory
from src.logger import logger_factory
from src.dataloader import dataloader_factory
from src.model import model_factory
from src.optimizer import optimizer_factory
from src.criterion import criterion_factory
from src.trainer import trainer_factory


def start(conf):
    if conf.mode == "resume":
        assert conf.path is not None
        conf = get_resumed_params(conf)

    data, conf.data_shape, conf.n_classes, conf.extra_data_shape = data_factory(conf)

    if conf.mode == "tune":
        pass

    elif conf.mode == "experiment":
        pass

    elif conf.mode == "nested_exp":
        # outer CV: for testing tuned models
        for outer_k in range(conf.n_splits):
            # inner CV: for tuning models
            for trial in range(conf.num_inner_trials):
                model_config = model_config_factory(
                    conf, "tune", random_seed=int(time.time())
                )
                for inner_k in range(conf.n_splits):
                    (
                        conf.project_name,
                        conf.trial_name,
                        conf.trial_dir,
                    ) = name_factory(conf, "tune", outer_k, trial, inner_k)

                    dataloaders = dataloader_factory(
                        conf, model_config, data, outer_k, trial, inner_k
                    )
                    model = model_factory(conf, model_config)
                    optimizer = optimizer_factory(conf, model, model_config)
                    criterion = criterion_factory(conf)

                    logger, model_config["link"] = logger_factory(
                        conf, model_config, "tune"
                    )

                    trainer = trainer_factory(
                        conf,
                        dataloaders,
                        model,
                        optimizer,
                        criterion,
                        logger,
                    )
                    trainer.run()

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    parser = get_argparser(sys.argv)
    args = parser.parse_args()
    print(args)

    # start(args)
