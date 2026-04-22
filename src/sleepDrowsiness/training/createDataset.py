from datasets import load_dataset
from pathlib import Path, PureWindowsPath
import polars as pl
import numpy as np
import mediapipe as mp
import xgboost as xgb
import matplotlib as mpl
import datetime as dt
import os

dataPath = Path("../data")
dataName = f"parquetData_{dt.datetime.now().strftime("%H:%M:S_%m/%d/%Y")}"


class HuggingFace:
    def __init__(self, dataset: str, device: str, streaming: bool):
        if type(self.type):
            self.currentDataset = dataset


def createDataset(dataSet: str, dataLink: str):
    print(dataSet, type(dataLink), type(dataPath), os.name)
