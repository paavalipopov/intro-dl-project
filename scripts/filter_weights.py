"""Script for deleting unnecessary trined weights"""
import os
import json
import shutil

import pandas as pd
from src.settings import WEIGHTS_ROOT

for directory in os.listdir(WEIGHTS_ROOT):
    if not directory.startswith("."):
        directory = os.path.join(WEIGHTS_ROOT, directory)
        print(f"Filtering weights in {directory}")

        with open(f"{directory}/general_config.json", "r", encoding="utf8") as fp:
            config = json.load(fp)
        n_splits = config["n_splits"]
        n_trials = config["n_trials"]

        df = pd.read_csv(f"{directory}/runs.csv", delimiter=",", index_col=False)
        for k in range(n_splits):
            best_trial = df["test_score"][k * n_trials : (k + 1) * n_trials].idxmax()
            best_trial = best_trial - k * n_trials

            for trial in range(n_trials):
                if trial != best_trial:
                    try:
                        shutil.rmtree(f"{directory}/k_{k:02d}/trial_{trial:04d}")
                        print(
                            f"\tDeleted '{directory}/k_{k:02d}/trial_{trial:04d}' directory"
                        )
                    except FileNotFoundError:
                        print(
                            f"'\t{directory}/k_{k:02d}/trial_{trial:04d}' does not exist"
                        )
