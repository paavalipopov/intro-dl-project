# Experiment Template

- `.github/workflow` - codestyle and CI
- `assets` - datasets, logs, etc
- `bin` - bash files to start pipelines
- `docker` - docker files
- `examples` - notebooks and full-featured examples
- `requirements` - python requirements
- `src` - code
- `tests` - tests

## How to reproduce?

```bash
bash bin/...  # download data
pip install -r ./requirements/...  # install dependencies, or use docker
bash bin/...  # run experiments
# use examples/... to analize results
```

## Examples
```
for model in mlp new_attention_mlp lstm transformer
do
    for dataset in oasis abide fbirn cobre abide_869 bsnip
    do
        PYTHONPATH=./ python src/scripts/ts_dl_experiments.py --mode tune --model $model --ds $dataset --max-epochs 200 --num-trials 10     
        PYTHONPATH=./ python src/scripts/ts_dl_experiments.py --mode experiment --model $model --ds $dataset --max-epochs 200 --num-trials 10     
    done
done
```

## Options for `src/scripts/ts_dl_experiments.py`

### Required
- `--mode`: 
    - `tune` - tune mode: run multiple experiments with different hyperparams
    - `experiment` - experiment mode: run experiments with best hyperparams found in the `tune` mode
    - `resume` - see below
- `--model`: some of the working models; check the sourse code for more info
    - `mlp`
    - `wide_mlp`
    - `deep_mlp`
    - `new_attention_mlp`
    - `lstm`
    - `noah_lstm`
    - `transformer`
    - `mean_transformer`
- `--ds`: dataset for the experiments
    - `oasis`
    - `adni`
    - `cobre`
    - `bsnip`

    - `abide` - ICA ABIDE1 (569 subjects)
    - `abide_869` - ICA ABIDE1 (869 subjects)
    - `abide_roi` - ROI Schaefer 200 ABIDE1

    - `ukb`

    - `fbirn` - ICA Fbirn
    - `time_fbirn` - Time and time-reversed ICA FBIRN
    - `fbirn_100` - ROI Schaefer 100 FBRIN
    - `fbirn_200` - ROI Schaefer 200 FBRIN
    - `fbirn_400` - ROI Schaefer 400 FBRIN
    - `fbirn_1000` - ROI Schaefer 1000 FBRIN

    - `hcp_roi`

### Optional
- `--test-ds`: additional datasets for tests
    - fraction of dataset from `--ds` is always used for tests, no need to use it here
    - options are the same as for `--ds`
    - for multiple datasets, write them in a space separated way

- `--prefix`: custom prefix for the project
    - default prefix is UTC time
    - appears in the name of logs directory and the name of WandB project
    - `tune`->`experiment` experiments should use the same prefix (unless it is default)
    - don't use `-` character in the prefix
    - don't use `resume` as a prefix

- `scaled`: whether dataset should be scaled first using `sklearn`'s `StandardScaler`

- `--max-epochs` - max epochs to use (default=30):
    - don't use less than 30 epochs
    - in `tune` mode the number of epochs is choosen randomly between `30` and `max-epochs`
    - in `experiment` mode the number of epochs is `max-epochs` 

- `--num-splits` - number of splits for `StratifiedKFold` cross-validation (default=5):
    - the `ds` dataset is split in `num-splits` equally sized folds; 
    - each fold is used as test dataset `num-splits` times (see below), the rest is train-val dataset

- `--num-trials` - number of trials for each fold (default=1):
    - for each trial, a new seed for `train_test_split` is used for splitting train-val dataset into train and val datasets
    - **important note**: if you provide the same `num-splits` and `num-trials` for different experiments on the same dataset, datasets splits will be the same

### Required for `resume` mode
- `--mode`: 
    - `resume` - resume mode: for resuming interrupted experiment
- `--path`:
    - path to the interrupted experiment (e.g., `/Users/user/mlp_project/assets/logs/prefix-mode-model-ds`)
- note that to resume experiments correctly you need to provide the same `-num-splits` and `--num-trials` as the ones used in the interrupted experiment (unless they are default)

## Running ST-DIM experiment
`PYTHONPATH=./ python src/stdim/ts_stdim_experiments.py --mode tune --ds fbirn --max-epochs 200`

## Running Window MLP experiment
`PYTHONPATH=./ python src/scripts/window_mlp_experiments.py --mode tune --model window_mlp --model-mode NPT --model-decoder lstm --ds fbirn --max-epochs 200 --num-trials 10 --prefix test`

PYTHONPATH=./ python src/scripts/window_mlp_experiments.py --mode tune --model window_mlp --model-mode NPT --model-decoder lstm --ds fbirn --max-epochs 200 --num-trials 10 --prefix test;
PYTHONPATH=./ python src/scripts/window_mlp_experiments.py --mode experiment --model window_mlp --model-mode NPT --model-decoder lstm --ds fbirn --max-epochs 200 --num-trials 10 --prefix test;
PYTHONPATH=./ python src/scripts/window_mlp_experiments.py --mode tune --model window_mlp --model-mode NPT --model-decoder tf --ds fbirn --max-epochs 200 --num-trials 10 --prefix test;
PYTHONPATH=./ python src/scripts/window_mlp_experiments.py --mode experiment --model window_mlp --model-mode NPT --model-decoder tf --ds fbirn --max-epochs 200 --num-trials 10 --prefix test

PYTHONPATH=./ python src/scripts/ts_dl_experiments.py --mode tune --model pe_transformer --ds fbirn --max-epochs 200 --num-trials 10;
PYTHONPATH=./ python src/scripts/ts_dl_experiments.py --mode experiment --model pe_transformer --ds fbirn --max-epochs 200 --num-trials 10;
PYTHONPATH=./ python src/scripts/ts_dl_experiments.py --mode tune --model pe_mlp --ds fbirn --max-epochs 200 --num-trials 10;
PYTHONPATH=./ python src/scripts/ts_dl_experiments.py --mode experiment --model pe_mlp --ds fbirn --max-epochs 200 --num-trials 10;

PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds fbirn --test-ds cobre bsnip --prefix fixed_cv; 
PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds cobre --test-ds fbirn bsnip --prefix fixed_cv;
PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds bsnip --test-ds fbirn bsnip --prefix fixed_cv;
PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds oasis --test-ds adni --prefix fixed_cv;
PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds adni --test-ds oasis --prefix fixed_cv;
PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds fbirn_200 --prefix fixed_cv;

PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds abide_869 --prefix fixed_cv;
PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds abide_roi --prefix fixed_cv;
PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds hcp --prefix fixed_cv;
PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds hcp_roi --prefix fixed_cv;
PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --ds time_fbirn --prefix fixed_cv;
