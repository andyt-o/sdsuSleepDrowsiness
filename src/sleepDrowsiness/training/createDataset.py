from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path, PureWindowsPath
from dotenv import load_dotenv
import polars as pl
import numpy as np
import mediapipe as mp
import xgboost as xgb
import matplotlib as mpl
import datetime as dt
import cv2
import os

load_dotenv()
HUGGING_FACE = os.getenv("HUGGING_FACE")

dataPath = Path("../data")
dataName = f"parquetData_{dt.datetime.now().strftime("%H:%M:S_%m/%d/%Y")}"


class HuggingFace:
    def __init__(self, dataset: str, device: str, streaming: bool):
        self.dataset = dataset
        self.device = device
        self.streaming = streaming
        self.localdata: DatasetDict = None

    def streamData(self, num: int, split: str = "trai"):
        self.localdata = load_dataset(
            self.dataset, streaming=self.streaming, token=HUGGING_FACE
        )

        try:
            self.localdata = self.localdata[split]
            print(f"Success: Loading from {split} worked.")
        except Exception:
            print(
                f"Error: Loading from {split} failed, loading the first available split."
            )
            self.localdata = self.localdata[next(iter(self.localdata.keys()))]

        for i in self.localdata:
            cv2.imshow(
                "currentIteration",
                cv2.cvtColor(
                    np.ascontiguousarray(np.array(i["image"])), cv2.COLOR_RGB2BGR
                ),
            )
            while True:
                print(i, self.localdata.features["label"].int2str(i["label"]))
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                elif key == ord("e"):
                    break


def createDataset(platform: str, dataset: str, streaming=False):
    if platform == 0:
        HF_DATASET = HuggingFace(dataset, os.name, streaming)
        if streaming:
            print("Streaming Data")
            HF_DATASET.streamData(100)
