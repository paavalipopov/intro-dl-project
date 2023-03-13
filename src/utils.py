import argparse
from apto.utils.misc import boolean_flag

models = [
    "mlp",
    "wide_mlp",
    "deep_mlp",
    "attention_mlp",
    "new_attention_mlp",
    "meta_mlp",
    "pe_mlp",
    "lstm",
    "noah_lstm",
    "transformer",
    "mean_transformer",
    "pe_transformer",
    "stdim",
]
datasets = [
    "oasis",
    "adni",
    "abide",
    "abide_869",
    "abide_roi",
    "fbirn",
    "fbirn_100",
    "fbirn_200",
    "fbirn_400",
    "fbirn_1000",
    "cobre",
    "bsnip",
    "hcp",
    "hcp_roi",
    "ukb",
    "ukb_age_bins",
    "time_fbirn",
]


def get_argparser(sys_argv):
    "Get params parser"
    resume = "resume" in sys_argv
    tune = "tune" in sys_argv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tune",
            "exp",
            "nested_exp",
            "resume",
        ],
        required=True,
        help="'tune' for model hyperparams tuning; \
            'experiment' for experiments with tuned model; \
                'resume' for resuming interrupted experiment",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=resume,
        help="Path to the interrupted experiment \
            (e.g., /Users/user/mlp_project/assets/logs/prefix-mode-model-ds), \
                used in 'resume' mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=models,
        required=not resume,
        help="Name of the model to run",
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=datasets,
        required=not resume,
        help="Name of the dataset to use for training",
    )

    parser.add_argument(
        "--test-ds",
        nargs="*",
        type=str,
        choices=datasets,
        help="Additional datasets for testing",
    )

    # some datasets have multiple classes; set to true if you want to load all classes
    boolean_flag(parser, "multiclass", default=False)

    # whehter dataset should be z-scored over time
    boolean_flag(parser, "scaled", default=False)

    # whehter ICA components should be filtered
    boolean_flag(parser, "filter-indices", default=True)

    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for the project name (body of the project name \
            is '$mode-$model-$dataset'): default: UTC time",
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=50 if tune else 10,
        help="Number of trials to run on each test fold",
    )
    parser.add_argument(
        "--num-inner-trials",
        type=int,
        default=50,
        help="Number of trials to run on each test fold for nested_exp runs",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="Number of splits for StratifiedKFold (affects the number of test folds)",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Max number of epochs (min 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (default: 30)",
    )

    return parser


def get_resumed_params(conf):
    # path = conf.path
    return None
