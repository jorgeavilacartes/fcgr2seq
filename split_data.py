import json
import pandas as pd 
from pathlib import Path
from random import shuffle
from parameters import PARAMETERS

TRAIN_SIZE = PARAMETERS["TRAIN_SIZE"]
paths = pd.read_csv("data/io/list_paths_io.csv")
paths_input = paths["input_path"].tolist()
paths_output = paths["output_path"].tolist()

idx = list(paths.index)
shuffle(idx)

n = paths.shape[0]
n_train = int(TRAIN_SIZE*n)
n_val = n_train + int((1-TRAIN_SIZE)*n/2)

list_train = idx[:n_train]
list_val = idx[n_train:n_val]
list_test = idx[n_val:]

datasets = dict()
datasets["train"] = {
                "input": paths.loc[list_train,"input_path"].tolist(), 
                "output": paths.loc[list_train,"output_path"].tolist()
                }
datasets["val"] = {
                "input": paths.loc[list_val,"input_path"].tolist(), 
                "output": paths.loc[list_val,"output_path"].tolist()
                }
datasets["test"] = {
                "input": paths.loc[list_test,"input_path"].tolist(),
                "output": paths.loc[list_test,"output_path"].tolist()
                }

PATH_SAVE = Path("data/train")
PATH_SAVE.mkdir(parents=True, exist_ok=True)
with open(PATH_SAVE.joinpath("datasets.json"),"w") as fp:
    json.dump(datasets,fp, indent=4)