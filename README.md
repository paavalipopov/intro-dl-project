# Requirements

```bash
conda create -n introdl python=3.9
conda activate introdl
- if mac user:
    conda install pytorch torchvision torchaudio -c pytorch
- else:
    conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia
pip install -r requirements.txt
```
To download fMRI datasets, run
```
PYTHONPATH=./ python scripts/download_datasets.py
```

# Examples
## Training
```
for model in lstm mean_lstm transformer mean_transformer
do
    for dataset in cobre abide synth1 synth2
    do
        PYTHONPATH=./ python scripts/run_experiments.py --mode tune --model $model --ds $dataset --prefix test
        PYTHONPATH=./ python scripts/run_experiments.py --mode exp --model $model --ds $dataset --prefix test
    done
done
```
or
```
PYTHONPATH=./ python scripts/run_experiments.py --mode tune --model lstm --ds abide --prefix test
PYTHONPATH=./ python scripts/run_experiments.py --mode exp --model lstm --ds abide --prefix test

```
## Introspection
For model introspection to work you need to move (or copy) the `exp` folder of the desired model from the `assets/logs` to `assets/trained_models`.

```
for model in lstm mean_lstm transformer mean_transformer
do
    for dataset in cobre abide synth1 synth2
    do
        PYTHONPATH=./ python scripts/run_introspection.py --model $model --ds $dataset --prefix test
    done
done
```
or
```
PYTHONPATH=./ python scripts/run_introspection.py --model lstm --ds abide --prefix test
```
## Options for `scripts/run_experiments.py`

### Required
- `--mode`: 
    - `tune` - tune mode: run multiple experiments with different hyperparams
    - `exp` - experiment mode: run experiments with best hyperparams found in the `tune` mode
    - `resume` - see below
- `--model`: some of the working models; check the sourse code for more info
    - `lstm`
    - `mean_lstm`
    - `transformer`
    - `mean_transformer`
- `--ds`: dataset for the experiments
    - `cobre`
    - `abide`


### Optional
- `--prefix`: custom prefix for the project
    - default prefix is UTC time
    - appears in the name of logs directory and the name of WandB project
    - `tune`->`exp` experiments will use the same prefix (unless it is default)
    - don't use `-` character in the prefix
    - don't use `resume` or `tune` as a prefix

- `--multiclass`: some datasets have multiple classes (default: False); pass `--multiclass` if you want to load all classes

- `--zscore`: whether dataset should be zscored over time direction (default: False); pass `--zscore` if you want zscore the data

- `--filter-indices`: whehter ICA components in real-world fMRI data should be filtered (default: True); pass `--no-filter-indices` if you want to load all ICA components

- `--max-epochs` - max epochs to use (default: 200)

- `--n-splits` - number of splits for `StratifiedKFold` cross-validation (default=5):
    - the `ds` dataset is split in `num-splits` equally sized folds

- `--n-trials` - number of trials for each test fold (default=50 for `tune` and 10 for `exp`):
    - in `tune` it equals to number of different hyperparams sets tested on each fold
    - in `exp` mode, for each trial, a new seed for `train_test_split` is used for splitting train-val dataset into train and val datasets
    - **important note**: if you provide the same `num-splits` and `num-trials` for different experiments on the same dataset, datasets splits will be the same

- `--batch-size` - batch size (default: 64)
- `--patience` - patience for early stopping (default: 30)

### Required for `resume` mode
- `--mode`: 
    - `resume` - resume mode: for resuming interrupted experiment
- `--path`:
    - path to the interrupted experiment (e.g., `/Users/user/intro-dl-project/assets/logs/prefix-mode-model-ds`)

