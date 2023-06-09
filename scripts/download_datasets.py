# pylint: disable=line-too-long
"""Script for downloading fMRI datasets"""
import requests

from src.settings import DATA_ROOT

items = [
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/Data/ABIDE1_AllData.h5",
        DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    ),
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/IndicesAndLabels/labels_ABIDE1.csv",
        DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
    ),
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/IndicesAndLabels/correct_indices_GSP.csv",
        DATA_ROOT.joinpath("abide/correct_indices_GSP.csv"),
    ),
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/Data/COBRE_AllData.h5",
        DATA_ROOT.joinpath("cobre/COBRE_AllData.h5"),
    ),
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/IndicesAndLabels/labels_COBRE.csv",
        DATA_ROOT.joinpath("cobre/labels_COBRE.csv"),
    ),
    (
        "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master/IndicesAndLabels/correct_indices_GSP.csv",
        DATA_ROOT.joinpath("cobre/correct_indices_GSP.csv"),
    ),
]

for item in items:
    r = requests.get(item[0], allow_redirects=True, timeout=100)
    with open(item[1], "wb") as f:
        f.write(r.content)
