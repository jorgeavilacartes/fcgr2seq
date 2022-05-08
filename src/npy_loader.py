from pathlib import Path
from typing import List
import numpy as np

class InputOutputLoader:

    def __call__(self, fcgr_path: Path, label_path: Path):
        "given an image path, return the input-output for the model"
        fcgr = np.load(fcgr_path)
        label = np.load(label_path)
        return fcgr, label